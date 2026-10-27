"""Host watch registration and read-only check boundaries."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from box_agent.plugins.proactive.acp import watch_tool
from box_agent.kernel.hook_types import HookContext
from box_agent.plugins.proactive.watch import RegisterWatchTool, WatchCheckPlugin


@pytest.mark.asyncio
async def test_watch_requires_host_capability_and_durable_receipt():
    host = SimpleNamespace(ext_method=AsyncMock(return_value={"status": "registered", "watch_id": "w"}))
    for capabilities in ({}, [], "invalid", {"proactive_watches": 0}):
        assert watch_tool(host, "acp-session", {"host_capabilities": capabilities}) is None
    tool = watch_tool(host, "acp-session", {"host_capabilities": {"proactive_watches": 1}})
    assert (await tool.execute(subject="沙洲")).success
    host.ext_method.assert_awaited_once_with(
        "session/proactive_watch", {"sessionId": "acp-session", "watch": {"subject": "沙洲"}},
    )
    host.ext_method.return_value = {"status": "queued"}
    assert not (await tool.execute(subject="沙洲")).success


@pytest.mark.asyncio
async def test_rejected_watch_does_not_retry_or_claim_registration():
    request = AsyncMock(side_effect=RuntimeError("WATCH_REVIEW_REJECTED"))
    result = await RegisterWatchTool(request).execute(subject="沙洲")
    assert not result.success
    assert "不要换说法重试" in result.error
    assert request.await_count == 1


def plugin(check=True):
    state = {"proactive_watch_check": {"source_url": "http://travel.test:8126/"} if check else None}
    resources = SimpleNamespace(
        context=SimpleNamespace(options=SimpleNamespace(state=state)),
        state=dict(state), system_prompt="base",
    )
    return WatchCheckPlugin(resources)


@pytest.mark.asyncio
async def test_long_watch_summary_is_shortened_without_changing_evidence_or_action():
    import json

    check = plugin()
    model = SimpleNamespace(generate=AsyncMock())
    check.agent = SimpleNamespace(llm=model)
    original = {"outcome": "suggest", "suggestion": "长" * 101, "reason": "有网页证据",
                "execution_prompt": "核对原会话材料，不代下单。", "evidence_url": "http://shop.test/",
                "evidence_quote": "现货", "change_summary": "已补货", "user_impact": "可以购买"}
    result = json.loads(await check.format_check_result(json.dumps(original)))
    assert result["suggestion"] == "长" * 99 + "…"
    assert {k: v for k, v in result.items() if k != "suggestion"} == {
        k: v for k, v in original.items() if k != "suggestion"
    }
    model.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    '{"outcome":"silent","reason":"unchanged"}', 'not-json',
    '{"outcome":"suggest","suggestion":"' + "字" * 100 + '"}',
    '{"outcome":"suggest","suggestion":"' + "😀" * 50 + '"}',
])
async def test_other_decisions_and_summaries_within_host_limit_need_no_extra_call(content):
    check = plugin()
    check.agent = SimpleNamespace(llm=SimpleNamespace(generate=AsyncMock()))
    assert await check.format_check_result(content) == content
    check.agent.llm.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("summary,expected", [
    ("😀" * 51, "😀" * 49 + "…"),
    ("字" * 98 + "😀多", "字" * 98 + "…"),
    ("word " * 30, "word " * 19 + "word…"),
])
async def test_short_preview_marks_omission_without_splitting_unicode(summary, expected):
    import json

    check = plugin()
    result = json.loads(await check.format_check_result(json.dumps({
        "outcome": "suggest", "suggestion": summary,
    })))
    assert result == {"outcome": "suggest", "suggestion": expected}
    assert len(result["suggestion"].encode("utf-16-le")) <= 200


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,allowed", [
    ("memory_search", {"query": "旅游"}, True),
    ("managed_browser_navigate", {"url": "http://travel.test:8126/news/closure"}, True),
    ("managed_browser_snapshot", {}, True),
    ("mcp_config", {"action": "set_browser_mode", "mode": "headed"}, False),
    ("mcp_config", {"action": "set_browser_mode", "mode": "headless"}, True),
    ("managed_browser_navigate", {"url": "http://travel.test:8126/control"}, False),
    ("managed_browser_navigate", {"url": "https://elsewhere.test/"}, False),
    ("managed_browser_click", {"ref": "submit"}, False),
    ("bash", {"command": "echo unsafe"}, False),
    ("register_watch", {}, False),
])
async def test_check_allows_only_read_tools_and_registered_origin(name, args, allowed):
    result = await plugin().handle(HookContext(
        event="tool.before_execution", run_id="r", payload={"tool_name": name, "arguments": args},
    ))
    assert (result.action == "allow") is allowed


@pytest.mark.asyncio
async def test_check_disables_qa_writes_without_changing_normal_session():
    check = plugin()
    await check.prepare()
    assert "proactive_watch_check" not in check.resources.state
    assert check.resources.state["memory_runtime"] is None
    assert check.resources.state["memory_manager"] is None
    assert check.get_hooks()
    normal = plugin(False)
    await normal.prepare()
    assert normal.resources.state == {}
    assert normal.resources.system_prompt == "base"
    assert not normal.get_hooks()


@pytest.mark.asyncio
async def test_watch_draft_uses_normal_schedule_ui_without_claiming_saved():
    draft = {"name": "旅行关注", "prompt": "只读检查", "watch_draft_id": "host-draft"}
    request = AsyncMock(return_value={"status": "draft", "draft": draft})
    result = await RegisterWatchTool(request).execute(subject="沙洲")
    assert result.success
    assert result.raw_output == {"kind": "officev3_schedule_draft", "draft": draft}
    assert "尚未开始检查" in result.content
    request.return_value = {"status": "pending_confirmation"}
    result = await RegisterWatchTool(request).execute(subject="沙洲")
    assert result.success and result.raw_output is None
    request.return_value = {"status": "draft", "draft": {}}
    assert not (await RegisterWatchTool(request).execute(subject="沙洲")).success


def test_browser_and_memory_configuration_are_visible_only_in_watch_sessions():
    normal = SimpleNamespace(tools={}, local_tool_exposure=SimpleNamespace(require_tools=Mock()))
    plugin(False).install_agent(normal)
    normal.local_tool_exposure.require_tools.assert_not_called()
    planning = SimpleNamespace(
        tools={"register_watch": object()},
        local_tool_exposure=SimpleNamespace(require_tools=Mock()),
    )
    plugin(False).install_agent(planning)
    planning.local_tool_exposure.require_tools.assert_called_once_with(
        ("mcp_config", "memory_search", "memory_read", "proactive_memory_read"),
    )


@pytest.mark.asyncio
async def test_read_only_guard_is_installed_on_real_session_run(tmp_path, monkeypatch):
    from pathlib import Path
    from box_agent.agent_session import AgentSession
    from box_agent.events import ToolCallResult
    from box_agent.plugins.proactive import plugin_bindings, plugin_descriptors
    from box_agent.plugins.runtime import PluginRuntime
    from box_agent.session_context import HostBindings, SessionOptions
    from tests.test_agent_session import session_config
    from tests.test_hook_plugins import Model, Echo

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    tool = Echo()
    runtime = PluginRuntime(plugins=plugin_descriptors(), bindings=plugin_bindings())
    session = await AgentSession.open(
        config=session_config(tmp_path), runtime=runtime,
        host=HostBindings(llm_client=Model(), tools=[tool], system_prompt="system"),
        options=SessionOptions(workspace_dir=tmp_path, state={
            "proactive_watch_check": {"source_url": "http://travel.test/"},
        }),
    )
    try:
        session.agent.add_user_message("check")
        events = [event async for event in session.run_events(options=session.build_run_options(logger=None))]
        results = [event for event in events if isinstance(event, ToolCallResult)]
        assert results and not results[0].success
        assert not tool.calls
    finally:
        await session.aclose()
        await runtime.aclose()
