"""Translate durable plugin notifications to an acknowledged host extension."""

from collections.abc import Awaitable, Callable
from typing import Any

from . import plugin_bindings, plugin_descriptors
from .bindings import ProactiveHostBindings, ProactiveLifecycle, UserTurnResult


def session_state(meta: dict[str, Any]) -> dict[str, Any]:
    """Pass only Proactive-owned metadata into this plugin's session options."""
    return {key: meta.get(key) for key in ("proactive_watch_check", "proactive_agent")}


def session_tools(connection: Any, session_id: str, meta: dict[str, Any], config: Any, *, utility: bool):
    """Own tool enablement here so the ACP host only forwards session inputs."""
    raw = config.plugins.get("proactive")
    if utility or not isinstance(raw, dict) or raw.get("enabled") is not True:
        return []
    action = reminder_action_tool(connection, session_id, meta)
    return ([action] if action is not None else []) + proactive_agent_tools(connection, session_id, meta)


def watch_tool(connection: Any, session_id: str, meta: dict[str, Any]):
    capabilities = meta.get("host_capabilities") or {}
    send = getattr(connection, "ext_method", None) or getattr(connection, "extMethod", None)
    if not isinstance(capabilities, dict) or capabilities.get("proactive_watches") != 1 or not callable(send):
        return None
    from .watch import RegisterWatchTool

    async def request(payload):
        return await send("session/proactive_watch", {"sessionId": session_id, "watch": payload})

    return RegisterWatchTool(request)


def reminder_action_tool(connection: Any, session_id: str, meta: dict[str, Any]):
    """Expose actions only in the assistant session selected by the host."""
    capabilities = meta.get("host_capabilities") or meta.get("hostCapabilities") or {}
    send = getattr(connection, "ext_method", None) or getattr(connection, "extMethod", None)
    if (not isinstance(capabilities, dict)
            or capabilities.get("proactive_actions") != 1 or not callable(send)):
        return None
    from .action_tool import ProactiveReminderTool

    async def request(payload):
        return await send("session/proactive_action", {"sessionId": session_id, **payload})

    return ProactiveReminderTool(request)


def proactive_agent_tools(connection: Any, session_id: str, meta: dict[str, Any]):
    """Bind domain tools to this host's acknowledged ACP extensions."""
    context = meta.get("proactive_agent")
    caps = meta.get("host_capabilities") or {}
    publish = notification_publisher(connection, session_id, meta)
    if not isinstance(context, dict) or caps.get("proactive_agent") != 1 or publish is None:
        return []
    if not isinstance(context.get("source_session_id"), str) or not context["source_session_id"].strip():
        raise ValueError("proactive_agent_requires_bound_source")
    # Scheduled runs return the existing structured check receipt to the scheduler.
    if context.get("trigger") == "scheduled":
        return []
    from .agent_mode import NotifyTool, TasksTool

    send = getattr(connection, "ext_method", None) or getattr(connection, "extMethod", None)
    tools = [NotifyTool(publish, context)]
    if callable(send):
        tools.append(TasksTool(lambda: send("session/proactive_tasks", {"sessionId": session_id})))
        tool = watch_tool(connection, session_id, meta)
        if tool is not None:
            tools.append(tool)
    return tools


def notification_publisher(
    connection: Any, session_id: str, meta: dict[str, Any],
) -> Callable[[str, dict[str, Any]], Awaitable[None]] | None:
    capabilities = meta.get("host_capabilities") or meta.get("hostCapabilities") or {}
    if not isinstance(capabilities, dict) or capabilities.get("proactive_reminders") != 1:
        return None
    send = getattr(connection, "ext_method", None) or getattr(connection, "extMethod", None)
    if not callable(send):
        return None

    async def publish(topic: str, payload: dict[str, Any]) -> None:
        if topic == "proactive.turn_completed":
            if capabilities.get("proactive_agent") != 1:
                raise ValueError("proactive_agent_not_supported")
            result = await send("session/proactive_trigger", {"sessionId": session_id, "turn": payload})
            if not isinstance(result, dict) or result.get("status") != "queued":
                raise ValueError("proactive_agent_not_acknowledged")
            return
        if topic == "proactive.watch_proposal":
            if capabilities.get("proactive_watches") != 1 or meta.get("proactive_watch_check"):
                raise ValueError("watch_not_supported")
            result = await send("session/proactive_watch", {
                "sessionId": session_id, "watch": payload["watch"],
                "proposalId": payload["id"],
            })
            if not isinstance(result, dict) or result.get("status") not in (
                "proposed", "draft", "pending_confirmation", "registered",
            ):
                raise ValueError("watch_proposal_not_acknowledged")
            return
        if topic != "proactive.reminder":
            raise ValueError("unsupported_notification_topic")
        result = await send(
            "session/proactive_reminder",
            {"sessionId": session_id, "notification": payload},
        )
        if (
            not isinstance(result, dict)
            or result.get("status") != "persisted"
            or result.get("notificationId") != payload.get("id")
        ):
            raise ValueError("notification_not_acknowledged")

    publish.supports_watch_proposals = (  # type: ignore[attr-defined]
        capabilities.get("proactive_watches") == 1 and not meta.get("proactive_watch_check")
    )
    publish.supports_agent_runs = capabilities.get("proactive_agent") == 1  # type: ignore[attr-defined]
    return publish
