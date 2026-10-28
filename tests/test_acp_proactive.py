"""ACP completion and acknowledged background delivery, without CLI rendering."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from box_agent.acp import BoxACPAgent
from box_agent.plugins.proactive.acp import notification_publisher, session_state, session_tools
from box_agent.plugins.proactive.bindings import ProactiveLifecycle
from box_agent.memory import create_memory_backend
from tests.test_agent_session import session_config
from tests.test_memory_backends import backend_settings
from tests.test_proactive_plugin import Model, SILENT, isolated_profile, records, server


class Host:
    def __init__(self):
        self.updates = []
        self.notifications = []
        self.fail_delivery = False

    async def sessionUpdate(self, payload):
        self.updates.append(payload)

    async def extMethod(self, method, params):
        self.notifications.append((method, params))
        if self.fail_delivery:
            raise OSError("host storage unavailable")
        return {"status": "persisted", "notificationId": params["notification"]["id"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [None, False, True])
async def test_acp_selects_optional_plugins_through_shared_application_factory(
    tmp_path, monkeypatch, enabled,
):
    import box_agent.acp as acp
    from box_agent.session_assembly import create_application_runtime

    config = session_config(tmp_path)
    config.agent.enable_memory = False
    config.plugins = {} if enabled is None else {"proactive": {"enabled": enabled}}
    calls = []

    def factory(**extensions):
        runtime = create_application_runtime(**extensions)
        calls.append(runtime)
        return runtime

    monkeypatch.setattr(acp, "create_application_runtime", factory, raising=False)
    agent = BoxACPAgent(Host(), config, Model(), [], "system")
    try:
        assert calls == [agent._plugin_runtime]
        ids = {p.plugin_id for p in agent._plugin_runtime.host.resolve_dependencies()}
        proactive_ids = {"session.proactive", "session.proactive_watch", "session.proactive_agent"}
        assert ids & proactive_ids == (proactive_ids if enabled is True else set())
        assert "session.resources" in ids
        if enabled is not True:
            session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
                "proactive_agent": {"trigger": "turn_completed", "source_session_id": "original"},
                "proactive_watch_check": {"source_url": "http://shop.test/"},
            }))
            state = agent._sessions[session.sessionId]
            assert not ProactiveLifecycle.turn_observers(state)
            assert not any(name.startswith("proactive_") for name in state.agent.tools)
            result = await prompt(agent, session.sessionId, "普通问答", "ordinary-1")
            assert result.stopReason == "end_turn"
    finally:
        await agent.aclose()


@pytest.mark.parametrize("enabled,utility", [(False, False), (None, False), (True, True)])
def test_disabled_and_utility_sessions_do_not_expose_proactive_tools(enabled, utility):
    config = SimpleNamespace(plugins={"proactive": {"enabled": enabled}})
    host = SimpleNamespace(extMethod=lambda *args: None)
    meta = {"host_capabilities": {"proactive_actions": 1, "proactive_agent": 1,
                                  "proactive_reminders": 1, "proactive_watches": 1},
            "proactive_agent": {"trigger": "chat", "source_session_id": "assistant"}}
    assert session_tools(host, "s", meta, config, utility=utility) == []


@pytest.mark.parametrize("trigger,expected", [
    ("chat", {"proactive_reminder", "proactive_notify", "proactive_tasks", "register_watch"}),
    ("scheduled", {"proactive_reminder"}),
])
def test_host_capabilities_select_existing_proactive_tools(trigger, expected):
    config = SimpleNamespace(plugins={"proactive": {"enabled": True}})
    host = SimpleNamespace(extMethod=lambda *args: None)
    meta = {"host_capabilities": {"proactive_actions": 1, "proactive_agent": 1,
                                  "proactive_reminders": 1, "proactive_watches": 1},
            "proactive_agent": {"trigger": trigger, "source_session_id": "assistant"}}
    assert {tool.name for tool in session_tools(host, "s", meta, config, utility=False)} == expected


def test_plugin_state_passes_only_its_own_host_metadata():
    watch = {"source_url": "http://shop.test/"}
    mode = {"trigger": "scheduled", "source_session_id": "source"}
    assert session_state({"proactive_watch_check": watch, "proactive_agent": mode,
                          "other_host_field": "not_plugin_state"}) == {
        "proactive_watch_check": watch, "proactive_agent": mode,
    }


@pytest.fixture
async def adapter(tmp_path, server):
    config = session_config(tmp_path)
    config.agent = backend_settings(tmp_path)
    config.agent.enable_memory_extraction = False
    config.agent.goal_autopilot_enabled = False
    config.plugins = {"proactive": {"enabled": True}}
    host, model = Host(), Model()
    agent = BoxACPAgent(host, config, model, [], "system", memory_manager=create_memory_backend(config.agent))
    try:
        yield agent, host, model
    finally:
        await agent.aclose()


async def start_session(agent, tmp_path, *, capable=True, utility=False):
    return await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "task-A", "utility": utility,
        "host_capabilities": {"proactive_reminders": 1 if capable else 0},
    }))


@pytest.mark.asyncio
async def test_acp_transport_binding_stays_outside_shared_session_state(adapter, tmp_path):
    from dataclasses import fields

    from box_agent.session_context import HostBindings

    agent, _, _ = adapter
    created = await start_session(agent, tmp_path)
    state = agent._sessions[created.sessionId]
    observer = ProactiveLifecycle.turn_observers(state)[0]
    resources = observer.resources

    assert callable(observer.publish_notification)
    assert "publish_notification" not in {field.name for field in fields(HostBindings)}
    assert "publish_notification" not in resources.state
    assert "publish_notification" not in resources.context.options.state
    assert "publish_notification" not in vars(state)


@pytest.mark.asyncio
async def test_watch_check_runs_with_acp_memory_fallback_without_saving_fake_qa(adapter, tmp_path, server):
    agent, host, model = adapter
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "internal-check",
        "proactive_watch_check": {"source_url": "http://travel.test/"},
    }))
    state = agent._sessions[session.sessionId]
    assert state.memory_runtime is None
    result = await prompt(
        agent, session.sessionId,
        "本轮无需规划新任务，也不发布执行计划；直接完成这项已登记的只读检查。"
        "检索原旅行规划并检查新资讯。",
        "check-1",
    )
    assert result.stopReason == "end_turn"
    assert state.pending_plan_approval is None
    assert not host.notifications
    assert not any(path.endswith("/qa") or path.endswith("/qa/save") for path, _ in server.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("watch_check", [False, True])
async def test_watch_check_delivers_final_json_without_streamed_tool_preamble(
    adapter, tmp_path, watch_check,
):
    from box_agent.schema import FunctionCall, StreamEvent, ToolCall

    agent, host, model = adapter
    final = json.dumps({"outcome": "silent", "reason": "库存仍未变化"}, ensure_ascii=False)
    calls = 0

    async def generate_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StreamEvent(type="text", delta="我先检索登记的库存信息。")
            yield StreamEvent(type="finish", finish_reason="tool_calls", tool_calls=[
                ToolCall(id="lookup", type="function", function=FunctionCall(
                    name="tool_search", arguments={"tool_names": ["memory_search"]},
                )),
            ])
        else:
            yield StreamEvent(type="text", delta=final[:20])
            yield StreamEvent(type="text", delta=final[20:])
            yield StreamEvent(type="finish", finish_reason="stop")

    model.generate_stream = generate_stream
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "inventory-check" if watch_check else "ordinary-chat",
        **({"proactive_watch_check": {"source_url": "http://shop.test/"}} if watch_check else {}),
    }))
    result = await prompt(agent, session.sessionId, "核对库存", "t1")
    assert result.stopReason == "end_turn"
    assert calls == 2
    text = "".join(row.update.content.text for row in host.updates
                   if row.update.sessionUpdate == "agent_message_chunk")
    if watch_check:
        assert json.loads(text) == {"outcome": "silent", "reason": "库存仍未变化"}
    else:
        assert "我先检索登记的库存信息。" in text
        assert text.endswith(final)


async def prompt(agent, session_id, text, turn_id):
    return await agent.prompt(SimpleNamespace(
        sessionId=session_id, prompt=[{"text": text}], field_meta={"turn_id": turn_id},
    ))


@pytest.mark.asyncio
async def test_watch_summary_is_fitted_before_acp_delivery_without_changing_action(adapter, tmp_path):
    from box_agent.schema import LLMResponse, StreamEvent

    agent, host, model = adapter
    decision = {"outcome": "suggest", "suggestion": "库存更新" * 30,
                "execution_prompt": "核对商品页，不代下单。", "evidence_quote": "现货"}
    repairs = []

    async def generate(*args, **kwargs):
        if kwargs.get("call_kind") == "proactive_watch_summary":
            repairs.append(kwargs)
            return LLMResponse(content="银色相机已补货，建议核对发货时间。", finish_reason="stop")
        return LLMResponse(content='{"continue":false}', finish_reason="stop")

    async def generate_stream(*args, **kwargs):
        yield StreamEvent(type="text", delta=json.dumps(decision, ensure_ascii=False))
        yield StreamEvent(type="finish", finish_reason="stop")

    model.generate, model.generate_stream = generate, generate_stream
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "scheduled-check", "proactive_watch_check": {"source_url": "http://shop.test/"},
    }))
    await prompt(agent, session.sessionId, "检查商品页", "check-1")
    text = "".join(row.update.content.text for row in host.updates
                   if row.update.sessionUpdate == "agent_message_chunk")
    assert json.loads(text) == {**decision, "suggestion": "库存更新" * 24 + "库存更…"}
    assert not repairs


@pytest.mark.asyncio
async def test_acp_complete_qa_produces_durable_notification_outside_answer_stream(adapter, tmp_path, server):
    agent, host, model = adapter
    session = await start_session(agent, tmp_path)
    result = await prompt(agent, session.sessionId, "帮我拟季度汇报提纲", "t1")
    assert result.stopReason == "end_turn"
    state = agent._sessions[session.sessionId]
    await ProactiveLifecycle.drain_turn_observers(state)
    plugin = ProactiveLifecycle.turn_observers(state)[0]
    assert len(host.notifications) == 1
    method, payload = host.notifications[0]
    assert method == "session/proactive_reminder"
    assert payload["sessionId"] == session.sessionId
    assert payload["notification"]["source_session_id"] == "task-A"
    assert payload["notification"]["source_turn_id"] == "t1"
    assert payload["notification"]["execution_prompt"] == model.decision["execution_prompt"]
    assert records(plugin)[0]["delivery"] == "delivered"
    qa = json.loads(model.requests[-1]["messages"][1].content)["current_qa"]
    assert qa == {"user": "帮我拟季度汇报提纲", "assistant": "以下是新的汇报提纲。"}
    chunks = [row.update.content.text for row in host.updates
              if row.update.sessionUpdate == "agent_message_chunk"]
    assert "以下是新的汇报提纲。" in "".join(chunks)
    assert "主动提醒" not in "".join(chunks)


@pytest.mark.asyncio
async def test_rejected_duplicate_turn_does_not_cancel_pending_reminder(adapter, tmp_path):
    agent, host, _ = adapter
    entered, release = asyncio.Event(), asyncio.Event()
    original_send = host.extMethod

    async def delayed_send(method, params):
        entered.set()
        await release.wait()
        return await original_send(method, params)

    host.extMethod = delayed_send
    session = await start_session(agent, tmp_path)
    state = agent._sessions[session.sessionId]
    try:
        await prompt(agent, session.sessionId, "季度汇报", "t1")
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(ValueError, match="TURN_ALREADY_ACCEPTED"):
            await prompt(agent, session.sessionId, "重复发送", "t1")
    finally:
        release.set()
        await ProactiveLifecycle.drain_turn_observers(state)
    plugin = ProactiveLifecycle.turn_observers(state)[0]
    assert len(host.notifications) == 1
    assert records(plugin)[0]["delivery"] == "delivered"


@pytest.mark.asyncio
@pytest.mark.parametrize("capable,utility", [(False, False), (True, True)])
async def test_client_without_delivery_capability_and_utility_sessions_do_not_judge(
    adapter, tmp_path, capable, utility,
):
    agent, host, model = adapter
    session = await start_session(agent, tmp_path, capable=capable, utility=utility)
    await prompt(agent, session.sessionId, "hello", "t1")
    state = agent._sessions[session.sessionId]
    assert not ProactiveLifecycle.turn_observers(state)
    assert not model.requests and not host.notifications


@pytest.mark.asyncio
async def test_unacknowledged_delivery_is_retried_with_same_id_on_next_turn(adapter, tmp_path):
    agent, host, model = adapter
    session = await start_session(agent, tmp_path)
    state = agent._sessions[session.sessionId]
    host.fail_delivery = True
    await prompt(agent, session.sessionId, "季度汇报", "t1")
    await ProactiveLifecycle.drain_turn_observers(state)
    plugin = ProactiveLifecycle.turn_observers(state)[0]
    assert records(plugin)[0]["status"] == "suggest"
    assert records(plugin)[0]["delivery"] == "pending"
    host.fail_delivery = False
    model.decision = SILENT
    await prompt(agent, session.sessionId, "提纲已经完成", "t2")
    await ProactiveLifecycle.drain_turn_observers(state)
    assert len(host.notifications) == 2
    assert host.notifications[0] == host.notifications[1]
    assert records(plugin)[0]["delivery"] == "delivered"
    assert records(plugin)[1]["status"] == "silent"


@pytest.mark.asyncio
async def test_delivery_requires_matching_persistence_acknowledgement():
    class WrongAck:
        async def ext_method(self, method, payload):
            return {"status": "persisted", "notificationId": "another-id"}

    publisher = notification_publisher(WrongAck(), "s", {"host_capabilities": {"proactive_reminders": 1}})
    with pytest.raises(ValueError, match="not_acknowledged"):
        await publisher("proactive.reminder", {"id": "expected"})
    assert notification_publisher(WrongAck(), "s", {}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,capable,utility,expected", [
    (True, True, False, True),
    (True, False, False, False),
    (True, True, True, False),
    (False, True, False, False),
])
async def test_action_tool_is_only_registered_for_opted_in_host_session(
    adapter, tmp_path, enabled, capable, utility, expected,
):
    agent, _, _ = adapter
    agent._config.plugins["proactive"]["enabled"] = enabled
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "assistant", "utility": utility,
        "host_capabilities": {"proactive_actions": 1 if capable else 0},
    }))
    assert ("proactive_reminder" in agent._sessions[session.sessionId].agent.tools) is expected
