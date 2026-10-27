"""The proactive role uses the real Agent loop, while its plugin only launches it."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from box_agent.plugins.proactive.bindings import ProactiveLifecycle
from box_agent.kernel.hook_types import HookContext
from box_agent.plugins.proactive.agent_mode import NotifyTool, ProactiveModePlugin
from box_agent.plugins.proactive.memory import ProactiveMemoryReadTool
from box_agent.schema import FunctionCall, StreamEvent, ToolCall
from tests.test_acp_proactive import adapter, prompt
from tests.test_proactive_plugin import isolated_profile, records, server


def meta(trigger="chat"):
    return {"session_id": "proactive-assistant", "proactive_agent": {
        "trigger": trigger, "source_session_id": "proactive-assistant", "source_turn_id": "test-turn",
    }, "host_capabilities": {"proactive_reminders": 1, "proactive_agent": 1, "proactive_watches": 1}}


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["chat", "turn_completed", "scheduled", "legacy_scheduled"])
async def test_proactive_reads_historical_evidence_without_changing_core_memory_tools(
    adapter, tmp_path, server, trigger,
):
    from box_agent.tools.memory_tool import create_memory_tools

    agent, _, _ = adapter
    agent._base_tools.extend(create_memory_tools(agent._memory))
    path = "memory://items/item_report.md"
    server.item_reads[path] = {"content": "旧周报保留了100、200、300的原结论。"}
    metadata = ({"proactive_watch_check": {"source_url": "http://shop.test/"}}
                if trigger == "legacy_scheduled" else meta(trigger))
    metadata["memory"] = {"tenant_id": "tenant-test", "user_id": "user-test"}
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta=metadata))
    tools = agent._sessions[session.sessionId].agent.tools
    result = await tools["proactive_memory_read"].execute(path=path)
    assert result.success and "旧周报" in result.content
    read = next(data for url, data in reversed(server.calls)
                if url.endswith("/files/read") and data.get("path") == path)
    assert (read["tenant_id"], read["user_id"]) == ("tenant-test", "user-test")
    assert not (await tools["memory_read"].execute(path=path)).success

    ordinary = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={}))
    assert "proactive_memory_read" not in agent._sessions[ordinary.sessionId].agent.tools


@pytest.mark.asyncio
async def test_reading_evidence_does_not_authorize_core_memory_writes(adapter, tmp_path, server):
    from box_agent.tools.memory_tool import create_memory_tools
    from tests.test_memory_backends import core_file_response

    agent, _, _ = adapter
    agent._base_tools.extend(create_memory_tools(agent._memory))
    path = "memory://memory.md"
    server.item_reads[path] = core_file_response(path, "<mem>原始规则</mem>")
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta=meta()))
    tools = agent._sessions[session.sessionId].agent.tools
    assert (await tools["proactive_memory_read"].execute(path=path)).success
    result = await tools["memory_write"].execute(path="memory.md", content="<mem>修改规则</mem>")
    assert not result.success
    assert not any(url.endswith("/files/write") for url, _ in server.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["", "memory://", "https://other.test/", "/etc/passwd"])
async def test_evidence_reader_rejects_non_resource_paths_without_network_access(path):
    backend = SimpleNamespace(read_memory=AsyncMock())
    assert not (await ProactiveMemoryReadTool(backend).execute(path=path)).success
    backend.read_memory.assert_not_called()


@pytest.mark.asyncio
async def test_evidence_read_failure_is_not_reported_as_empty_evidence():
    backend = SimpleNamespace(read_memory=AsyncMock(side_effect=RuntimeError("private detail")))
    result = await ProactiveMemoryReadTool(backend).execute(path="memory://items/report")
    assert not result.success and "private detail" not in result.error
    backend.read_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_user_turn_dispatches_full_qa_without_separate_judge(adapter, tmp_path):
    agent, host, model = adapter
    host.extMethod = AsyncMock(return_value={"status": "queued"})
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "original", "host_capabilities": {"proactive_reminders": 1, "proactive_agent": 1},
    }))
    await prompt(agent, session.sessionId, "我想准备季度汇报", "complete-1")
    state = agent._sessions[session.sessionId]
    await ProactiveLifecycle.drain_turn_observers(state)
    method, payload = host.extMethod.call_args.args
    assert method == "session/proactive_trigger"
    assert payload["turn"]["user"] == "我想准备季度汇报"
    assert payload["turn"]["assistant"] == "以下是新的汇报提纲。"
    assert payload["turn"]["source_session_id"] == "original"
    assert not model.requests
    observer = ProactiveLifecycle.turn_observers(state)[0]
    assert not hasattr(observer, "judge")
    assert records(observer)[0]["status"] == "dispatched"


@pytest.mark.asyncio
async def test_assistant_runs_notify_tool_in_agent_loop_without_recursive_trigger(adapter, tmp_path):
    agent, host, model = adapter
    calls = 0

    async def stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StreamEvent(type="finish", finish_reason="tool_use", tool_calls=[ToolCall(
                id="notice-1", type="function", function=FunctionCall(name="proactive_notify", arguments={
                    "suggestion": "银色相机已补货（测试）", "target": "相机补货测试", "reason": "用户明确要求展示测试提醒",
                }),
            )])
        else:
            yield StreamEvent(type="text", delta="测试提醒已保存。")
            yield StreamEvent(type="finish", finish_reason="stop")

    model.generate_stream = stream
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta=meta()))
    state = agent._sessions[session.sessionId]
    assert {"proactive_notify", "proactive_tasks", "register_watch"} <= state.agent.tools.keys()
    assert not ProactiveLifecycle.turn_observers(state)
    assert "Proactive 模式" in state.agent.system_prompt
    result = await prompt(agent, session.sessionId, "请展示一条测试提醒", "t1")
    assert result.stopReason == "end_turn"
    assert calls == 2
    assert len(host.notifications) == 1
    method, payload = host.notifications[0]
    assert method == "session/proactive_reminder"
    assert payload["notification"]["source_session_id"] == "proactive-assistant"


@pytest.mark.asyncio
async def test_draft_tool_only_requests_confirmation_and_never_creates_task(adapter, tmp_path):
    agent, host, _ = adapter
    host.extMethod = AsyncMock(return_value={"status": "draft", "draft": {"watch_draft_id": "bound-draft"}})
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta=meta()))
    tool = agent._sessions[session.sessionId].agent.tools["register_watch"]
    result = await tool.execute(subject="补货测试", source_url="https://shop.test/", expires_on="2026-10-30",
                                plan_summary="用户不想反复刷新", check_condition="银色补货", check_interval_minutes=10)
    assert result.success
    assert "尚未开始" in result.content
    assert host.extMethod.call_args.args[0] == "session/proactive_watch"
    assert host.extMethod.await_count == 1


@pytest.mark.asyncio
async def test_notify_failure_does_not_claim_saved():
    tool = NotifyTool(AsyncMock(side_effect=OSError()), meta()["proactive_agent"])
    result = await tool.execute("测试提醒", "测试", "用户请求")
    assert not result.success
    assert "未确认" in result.error


@pytest.mark.asyncio
async def test_user_notice_stays_within_100_characters_without_appending_internal_reason():
    publish = AsyncMock()
    tool = NotifyTool(publish, meta()["proactive_agent"])
    assert (await tool.execute("提醒" * 50, "相机", "内部依据" * 80)).success
    assert publish.call_args.args[1]["suggestion"] == "提醒" * 50
    publish.reset_mock()
    result = await tool.execute("长" * 101, "相机", "内部依据")
    assert not result.success and "100字" in result.error
    publish.assert_not_called()


@pytest.mark.asyncio
async def test_watch_proposal_waits_for_assistant_consent_without_a_modal_tool_output(adapter, tmp_path):
    agent, host, _ = adapter
    host.extMethod = AsyncMock(return_value={"status": "proposed", "notificationId": "watch-proposal-1"})
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta=meta()))
    result = await agent._sessions[session.sessionId].agent.tools["register_watch"].execute(
        subject="补货", source_url="https://shop.test/", expires_on="2026-10-30",
        plan_summary="用户计划月底旅行，目前缺货", check_condition="银色补货", check_interval_minutes=360,
    )
    assert result.success
    assert "固定助理" in result.content and "尚未打开" in result.content
    assert result.raw_output is None


@pytest.mark.asyncio
async def test_scheduled_profile_is_read_only_and_does_not_write_fake_qa(adapter, tmp_path, server):
    agent, _, _ = adapter
    value = meta("scheduled")
    value["proactive_watch_check"] = {"source_url": "https://shop.test/"}
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta=value))
    state = agent._sessions[session.sessionId]
    assert "Proactive 模式" in state.agent.system_prompt
    assert not ProactiveLifecycle.turn_observers(state)
    assert state.memory_runtime is None
    assert "register_watch" not in state.agent.tools
    assert "proactive_notify" not in state.agent.tools


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments,allowed", [
    ("register_watch", {}, True), ("proactive_notify", {}, True),
    ("memory_search", {"query": "相机"}, True),
    ("create_scheduled_task", {}, False), ("write_file", {}, False),
    ("bash", {"command": "touch bypass"}, False),
    ("mcp_config", {"action": "set_browser_mode", "mode": "headed"}, False),
])
async def test_proactive_tools_cannot_bypass_confirmation_or_read_only_scope(name, arguments, allowed):
    resources = SimpleNamespace(context=SimpleNamespace(options=SimpleNamespace(state={"proactive_agent": meta()["proactive_agent"]})))
    policy = ProactiveModePlugin(resources)
    decision = await policy.handle(HookContext(event="tool.before_execution", run_id="test", payload={
        "tool_name": name, "arguments": arguments,
    }))
    assert (decision.action == "allow") is allowed
