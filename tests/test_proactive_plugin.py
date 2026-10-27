"""Proactive decisions are grounded, opt-in, cancellable and session-scoped."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from box_agent.agent_session import AgentSession
from box_agent.agent_service import AgentService
from box_agent.api import RunRequest, RunResult, RunStatus
from box_agent.memory import MemoryBackendError, create_memory_backend
from box_agent.plugins.proactive import plugin_bindings, plugin_descriptors
from box_agent.plugins.proactive.bindings import ProactiveHostBindings, ProactiveLifecycle
from box_agent.plugins.proactive.journal import Journal
from box_agent.plugins.proactive.judge import Decision, Judge
from box_agent.plugins.proactive.memory import Evidence, MemoryReader
from box_agent.plugins.proactive.runtime import Turn
from box_agent.plugins.runtime import PluginRuntime
from box_agent.schema import LLMResponse, StreamEvent
from box_agent.session_context import SessionOptions
from tests.test_agent_session import session_config
from tests.test_memory_backends import backend_settings

PATH = "memory://qa-chunks/old-material"
SUGGEST = dict(outcome="suggest", kind="reuse", target="季度汇报模板",
               suggestion="可以复用之前的季度汇报模板。", reason="模板结构与当前汇报一致",
               execution_prompt="请核对 reports/template.pptx 并复用季度汇报模板；如已复用则不要重复修改。",
               evidence_ids=[PATH])
SILENT = dict(outcome="silent", kind="none", target="", suggestion="",
              reason="本轮已经完成，无需补充动作", evidence_ids=[])
OLD_QA = {"user": "季度汇报模板在 reports/template.pptx，沿用蓝白配色。", "assistant": "已记录模板位置。"}
ITEM_PATH = "memory://items/item_weekly-report.md"
ITEM_BODY = "---\nid: item_weekly-report\nstatus: pending_confirmation\nupdated_at: 2026-10-14\n---\n## 当前情况\n旧周报使用 100、200、300、400，尚未按新数据更新。"


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("BOX_AGENT_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


class Model:
    def __init__(self, decision=None):
        self.decision = decision or SUGGEST
        self.requests = []

    async def generate(self, *args, **kwargs):
        kwargs = {**dict(zip(("messages", "tools"), args)), **kwargs}
        if kwargs.get("call_kind") != "proactive":
            return LLMResponse(content='{"continue":false}', finish_reason="stop")
        self.requests.append(kwargs)
        data = json.loads(kwargs["messages"][1].content)
        text = json.dumps(self.decision, ensure_ascii=False) if "current_qa" in data else "季度汇报 模板"
        return LLMResponse(content=text, finish_reason="stop")

    async def generate_stream(self, *args, **kwargs):
        yield StreamEvent(type="text", delta="以下是新的汇报提纲。")
        yield StreamEvent(type="finish", finish_reason="stop")


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(calls=[], empty=False, fail=False, search_payload=None,
                            item_payload={"chunks": []}, item_reads={}, fail_item_read=False)

    def handle(request):
        data = json.loads(request.content)
        state.calls.append((request.url.path, data))
        if state.fail:
            return httpx.Response(503)
        row = {"memory_id": "old-material", "result_type": "qa_source", "timestamp_ms": "1000",
               "content": json.dumps(OLD_QA, ensure_ascii=False)}
        if request.url.path == "/v1/memory/search":
            payload = state.item_payload
        elif request.url.path.endswith("/qa/search"):
            payload = (state.search_payload if state.search_payload is not None else
                       {"chunks": [] if state.empty else [row]})
        elif request.url.path.endswith("/fetch_recent"):
            payload = {"chunks": [row]}
        elif request.url.path.endswith("/files/read"):
            if state.fail_item_read and data["path"].startswith("memory://items/"):
                return httpx.Response(503)
            payload = state.item_reads.get(data["path"], {"content": "", "exists": False})
        else:
            payload = {"memory_id": "current"}
        return httpx.Response(200, json={"ok": True, "data": payload})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    return state


@pytest.fixture
async def open_session():
    """Register the plugin explicitly; no CLI or ACP adapter is wired by this fixture."""
    runtimes = []

    async def open_test_session(tmp_path, *, settings=None, model=None, profile="acp",
                                diagnostics=None, register=True, utility=False,
                                memory_user_id=None, publisher=None):
        config = session_config(tmp_path)
        config.agent = backend_settings(tmp_path)
        config.agent.enable_memory_extraction = False
        config.agent.memory_external.max_retries = 0
        if memory_user_id is not None:
            config.agent.memory_user_id = memory_user_id
        config.plugins = {"proactive": settings if settings is not None else {"enabled": True}}
        model = model or Model()
        output = []
        async def acknowledge(topic, payload):
            pass
        runtime = PluginRuntime(
            plugins=plugin_descriptors() if register else (),
            bindings=plugin_bindings() if register else (),
        )
        runtimes.append(runtime)
        session = await AgentSession.open(
            config=config, runtime=runtime, host=ProactiveHostBindings(
                llm_client=model, system_prompt="system", tools=[], output=output.append,
                memory_manager=create_memory_backend(config.agent),
                diagnostics=diagnostics,
                publish_notification=(publisher or acknowledge) if register else None,
            ),
            options=SessionOptions(profile=profile, workspace_dir=tmp_path, utility=utility),
        )
        return session, model, output

    try:
        yield open_test_session
    finally:
        for runtime in reversed(runtimes):
            await runtime.aclose()


async def run_completed_turn(session, *, user_message, session_id, turn_id=None):
    """Exercise the shared lifecycle directly, without a presentation adapter."""
    run_id = f"run-{uuid4().hex}"
    turn_id = turn_id or run_id
    async with ProactiveLifecycle.user_turn(
        session, user=user_message, session_id=session_id, turn_id=turn_id,
    ) as completed:
        handle = await AgentService().start(
            RunRequest(run_id, session_id, user_message), session=session,
            options=session.build_run_options(session_id=session_id, turn_id=turn_id, logger=None),
        )
        async with handle:
            async for _ in handle.events():
                pass
            completed.result = await handle.result()
            return completed.result.final_content


def records(plugin):
    with plugin.journal.connect() as db:
        return [json.loads(row[0]) for row in db.execute("SELECT payload FROM decisions ORDER BY updated")]


def plugin_for(session):
    observers = ProactiveLifecycle.turn_observers(session)
    return observers[0] if observers else None


@pytest.mark.asyncio
async def test_explicit_turn_observer_judges_full_qa_and_keeps_existing_save(tmp_path, server, open_session):
    session, model, output = await open_session(tmp_path)
    try:
        text = await run_completed_turn(session, user_message="帮我拟季度汇报提纲", session_id="s", turn_id="t")
        plugin = plugin_for(session)
        await plugin.drain()
        await session.memory_runtime.aclose()
        record = records(plugin)[0]
        assert record["status"] == "suggest"
        sent = json.loads(model.requests[-1]["messages"][1].content)
        assert sent["current_qa"] == {"user": "帮我拟季度汇报提纲", "assistant": text}
        assert all(req["tools"] is None for req in model.requests)
        assert any("[主动提醒] 可以复用" in line for line in output)
        saves = [data for path, data in server.calls if path.endswith("/save")]
        assert len(saves) == 1 and saves[0]["content"]["assistant"] == text
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("settings,utility", [({"enabled": False}, False), ({"enabled": True}, True)])
async def test_disabled_or_utility_plugin_does_not_call_model_or_create_state(
    tmp_path, server, open_session, settings, utility,
):
    progress = []
    session, model, output = await open_session(
        tmp_path, settings=settings, utility=utility,
        diagnostics=lambda event, data: progress.append((event, data)),
    )
    try:
        assert plugin_for(session) is None
        await run_completed_turn(session, user_message="hello", session_id="s")
        assert model.requests == []
        assert not any(event == "plugin.progress" for event, _ in progress)
        assert not (tmp_path / "home" / ".box-agent" / "proactive").exists()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["cli", "acp"])
async def test_default_hosts_do_not_enable_reminders_from_legacy_config(
    tmp_path, server, open_session, profile,
):
    progress = []
    session, model, output = await open_session(
        tmp_path, profile=profile, register=False,
        diagnostics=lambda event, data: progress.append((event, data)),
    )
    try:
        assert "session.proactive" not in session.plugin_session.plan.plugin_ids
        assert plugin_for(session) is None
        if profile == "cli":
            from box_agent.cli import _run_session_turn

            text = await _run_session_turn(session, user_message="hello", session_id="s", logger=None)
        else:
            text = await run_completed_turn(session, user_message="hello", session_id="s")
        await session.memory_runtime.aclose()
        assert text and model.requests == []
        assert not any("[主动提醒]" in line for line in output)
        assert not any(event == "plugin.progress" for event, _ in progress)
        assert not any(path.endswith(("/qa/search", "/fetch_recent")) for path, _ in server.calls)
        assert not (tmp_path / "home" / ".box-agent" / "proactive").exists()
        saves = [data for path, data in server.calls if path.endswith("/save")]
        assert len(saves) == 1 and saves[0]["content"]["assistant"] == text
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_nested_runs_produce_one_decision_for_original_user(tmp_path, server, open_session):
    session, model, _ = await open_session(tmp_path)
    try:
        async with ProactiveLifecycle.user_turn(session, user="原始问题", session_id="s", turn_id="outer"):
            await run_completed_turn(session, user_message="原始问题", session_id="s", turn_id="outer")
            await run_completed_turn(session, user_message="内部续跑", session_id="s", turn_id="outer")
            assert model.requests == []
        plugin = plugin_for(session)
        await plugin.drain()
        assert len(records(plugin)) == 1
        assert json.loads(model.requests[-1]["messages"][1].content)["current_qa"]["user"] == "原始问题"
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_new_input_cancels_pending_decision_but_does_not_cancel_memory_save(tmp_path, server, open_session):
    entered = asyncio.Event()
    progress = []

    class SlowModel(Model):
        async def generate(self, *args, **kw):
            if kw.get("call_kind") != "proactive":
                return await super().generate(*args, **kw)
            entered.set()
            await asyncio.Event().wait()

    session, _, output = await open_session(
        tmp_path, model=SlowModel(),
        diagnostics=lambda event, data: progress.append(data) if event == "plugin.progress" else None,
    )
    try:
        await run_completed_turn(session, user_message="hello", session_id="s")
        await asyncio.wait_for(entered.wait(), 1)
        plugin = plugin_for(session)
        ProactiveLifecycle.user_input_started(session)
        await plugin.drain()
        await session.memory_runtime.aclose()
        assert records(plugin)[0]["status"] == "cancelled"
        assert len(output) == 1  # startup only, no stale reminder
        assert [(item["operation"], item["phase"]) for item in progress] == [
            ("prepare_query", "start"), ("prepare_query", "cancelled"),
        ]
        assert any(path.endswith("/save") for path, _ in server.calls)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_no_new_evidence_is_not_reminded_twice_across_sessions(tmp_path, server, open_session):
    for sid in ("first", "second"):
        session, _, output = await open_session(tmp_path)
        try:
            await run_completed_turn(session, user_message="季度汇报", session_id=sid)
            plugin = plugin_for(session)
            await plugin.drain()
            assert records(plugin)[-1]["status"] == ("suggest" if sid == "first" else "silent")
            if sid == "second":
                assert records(plugin)[-1]["reason"] == "duplicate_suggestion"
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_silent_result_and_reason_are_visible_and_recorded(tmp_path, server, open_session):
    session, _, output = await open_session(tmp_path, model=Model(SILENT))
    try:
        await run_completed_turn(session, user_message="完成了", session_id="s")
        plugin = plugin_for(session)
        await plugin.drain()
        assert records(plugin)[0]["status"] == "silent"
        assert any("本轮不提醒" in line and SILENT["reason"] in line for line in output)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_qa_search_sends_scoped_query_and_preserves_source_without_path(tmp_path, server):
    backend = create_memory_backend(backend_settings(tmp_path))
    reader = MemoryReader(backend, limit=6, recent_limit=20)
    evidence, mode = await reader.search("季度汇报", user="new question", started_ms=2000)
    assert server.calls == [(endpoint, {
        "tenant_id": backend.tenant_id, "user_id": backend.user_id,
        "query": "季度汇报", "top_k": 12,
    }) for endpoint in ("/v1/memory/search", "/v1/memory/qa/search")]
    assert mode == "qa_search"
    assert len(evidence) == 1 and evidence[0].id == PATH
    assert json.loads(evidence[0].content) == OLD_QA


def set_item(server, *, path=ITEM_PATH, content=ITEM_BODY):
    server.item_payload = {"chunks": [{
        "result_type": "item_memory", "path": path,
        "content": "搜索摘要不是完整证据", "timestamp_ms": 9999999999999,
    }]}
    server.item_reads[path] = {
        "exists": True, "path": path, "resource_type": "item_memory", "content": content,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["suggest", "silent"])
async def test_item_search_reads_full_body_in_same_user_scope_before_judgment(tmp_path, server, open_session, outcome):
    set_item(server)
    model = Model({**SUGGEST, "evidence_ids": [ITEM_PATH]} if outcome == "suggest" else SILENT)
    progress = []
    session, _, _ = await open_session(
        tmp_path, model=model,
        diagnostics=lambda event, data: progress.append(data) if event == "plugin.progress" else None,
    )
    try:
        await run_completed_turn(session, user_message="纠正数据并重画图", session_id="s", turn_id="t")
        plugin = plugin_for(session)
        await plugin.drain()
        data = json.loads(model.requests[-1]["messages"][1].content)
        assert data["historical_evidence"][0]["content"] == ITEM_BODY
        assert data["historical_evidence"][0]["id"] == ITEM_PATH
        assert data["current_qa"]["user"] == "纠正数据并重画图"
        assert records(plugin)[0]["retrieval_mode"] == "item_and_qa_search"
        assert records(plugin)[0]["status"] == outcome
        assert data["historical_evidence"][1]["id"] == PATH
        assert any(path.endswith("/qa/search") for path, _ in server.calls)
        assert not any(path.endswith("/fetch_recent") for path, _ in server.calls)
        read = next(body for path, body in server.calls if body.get("path") == ITEM_PATH)
        assert read == {"tenant_id": "default", "user_id": "default", "path": ITEM_PATH}
        assert [(p["operation"], p["phase"]) for p in progress] == [
            (operation, phase) for operation in
            ("prepare_query", "memory_item_search", "memory_item_read", "memory_search", "judge_reminder")
            for phase in ("start", "result")
        ]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_item_reads_are_deduplicated_bounded_and_do_not_use_search_snippets(tmp_path, server):
    set_item(server)
    server.item_payload["chunks"] *= 2
    server.item_payload["chunks"].append({"result_type": "item_memory", "path": "memory://items/item_other.md"})
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=1, recent_limit=20)
    evidence, mode = await reader.search("周报", user="u", started_ms=2000)
    assert mode == "item_and_qa_search"
    assert len(evidence) == 1 and evidence[0].content == ITEM_BODY
    assert [path for path, _ in server.calls] == [
        "/v1/memory/search", "/v1/memory/files/read", "/v1/memory/qa/search",
    ]


@pytest.mark.asyncio
async def test_unrelated_items_do_not_hide_relevant_historical_qa(tmp_path, server):
    set_item(server, content="已完成客户留存分析，与季度产品汇报无关。")
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    evidence, mode = await reader.search("季度汇报", user="new question", started_ms=2000)
    assert mode == "item_and_qa_search"
    assert [entry.id for entry in evidence] == [ITEM_PATH, PATH]
    assert json.loads(evidence[1].content) == OLD_QA
    qa_call = next(body for path, body in server.calls if path.endswith("/qa/search"))
    assert qa_call == {"tenant_id": "default", "user_id": "default",
                       "query": "季度汇报", "top_k": 12}


@pytest.mark.asyncio
async def test_full_item_results_leave_room_for_qa_within_total_budget(tmp_path, server):
    set_item(server)
    other = "memory://items/item_other.md"
    server.item_payload["chunks"].append({"result_type": "item_memory", "path": other})
    server.item_reads[other] = {
        "exists": True, "path": other, "resource_type": "item_memory", "content": "其他旧事项",
    }
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=2, recent_limit=20)
    evidence, _ = await reader.search("季度汇报", user="new question", started_ms=2000)
    assert [entry.id for entry in evidence] == [ITEM_PATH, PATH]


@pytest.mark.asyncio
async def test_combined_retrieval_deduplicates_identical_content(tmp_path, server):
    set_item(server, content=json.dumps(OLD_QA, ensure_ascii=False))
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    evidence, _ = await reader.search("季度汇报", user="new question", started_ms=2000)
    assert [entry.id for entry in evidence] == [ITEM_PATH]


@pytest.mark.asyncio
async def test_items_do_not_disable_recent_qa_fallback(tmp_path, server):
    set_item(server)
    server.empty = True
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    evidence, mode = await reader.search("季度汇报", user="new question", started_ms=2000)
    assert mode == "item_and_recent_fallback"
    assert [entry.id for entry in evidence] == [ITEM_PATH, PATH]


@pytest.mark.asyncio
async def test_invalid_qa_search_with_items_is_still_an_error(tmp_path, server):
    set_item(server)
    server.search_payload = {"chunks": None}
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    with pytest.raises(MemoryBackendError, match="invalid_qa_search_response"):
        await reader.search("季度汇报", user="new question", started_ms=2000)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"chunks": None}, {"chunks": {}}, {"chunks": [None]}, []])
async def test_malformed_item_search_is_error_not_qa_fallback(tmp_path, server, payload):
    server.item_payload = payload
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    with pytest.raises(MemoryBackendError, match="invalid_item_search_response"):
        await reader.search("周报", user="u", started_ms=2000)
    assert len(server.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["https://example.com/item.md", "memory://user.md",
                                 "memory://items/../user.md", None])
async def test_item_search_rejects_non_item_paths_before_read(tmp_path, server, path):
    set_item(server, path=path)
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    with pytest.raises(MemoryBackendError, match="invalid_item_path"):
        await reader.search("周报", user="u", started_ms=2000)
    assert len(server.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"exists": "true"}, {"path": "memory://items/item_other.md"},
                                   {"resource_type": "qa_chunk"}, {"content": None},
                                   {"content": "x" * 24001}])
async def test_invalid_item_body_never_falls_back_to_search_snippet(tmp_path, server, change):
    set_item(server)
    server.item_reads[ITEM_PATH].update(change)
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    with pytest.raises(MemoryBackendError):
        await reader.search("周报", user="u", started_ms=2000)
    assert len(server.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"exists": False}, {"content": "  "}])
async def test_missing_or_empty_item_body_falls_back_to_qa(tmp_path, server, change):
    set_item(server)
    server.item_reads[ITEM_PATH].update(change)
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    evidence, mode = await reader.search("周报", user="u", started_ms=2000)
    assert mode == "qa_search" and evidence[0].id == PATH


@pytest.mark.asyncio
async def test_item_read_failure_is_reported_without_judgment_or_qa_fallback(tmp_path, server, open_session):
    set_item(server)
    server.fail_item_read = True
    progress = []
    session, model, output = await open_session(
        tmp_path, diagnostics=lambda event, data: progress.append(data) if event == "plugin.progress" else None,
    )
    try:
        await run_completed_turn(session, user_message="纠正数据", session_id="s", turn_id="t")
        plugin = plugin_for(session)
        await plugin.drain()
        assert records(plugin)[0]["status"] == "error"
        assert len(model.requests) == 1  # Query preparation only.
        assert progress[-1]["operation"] == "memory_item_read" and progress[-1]["phase"] == "error"
        assert not any(path.endswith(("/qa/search", "/fetch_recent")) for path, _ in server.calls)
        assert any("判断失败" in text for text in output)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_item_timestamp_only_refresh_keeps_reminder_fingerprint(tmp_path, server):
    set_item(server)
    reader = MemoryReader(create_memory_backend(backend_settings(tmp_path)), limit=6, recent_limit=20)
    first, _ = await reader.search("周报", user="u", started_ms=2000)
    server.item_reads[ITEM_PATH]["content"] = ITEM_BODY.replace("2026-10-14", "2026-10-15")
    second, _ = await reader.search("周报", user="u", started_ms=2000)
    assert first[0].digest == second[0].digest
    server.item_reads[ITEM_PATH]["content"] += "\n收到新报价。"
    third, _ = await reader.search("周报", user="u", started_ms=2000)
    assert third[0].digest != second[0].digest


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"chunks": None}, {"chunks": {}}, []])
async def test_malformed_qa_search_is_error_without_recent_fallback(tmp_path, server, payload):
    server.search_payload = payload
    backend = create_memory_backend(backend_settings(tmp_path))
    reader = MemoryReader(backend, limit=6, recent_limit=20)
    with pytest.raises(MemoryBackendError, match="invalid_qa_search_response"):
        await reader.search("query", user="user", started_ms=2000)
    assert len(server.calls) == 2
    assert server.calls[-1][0] == "/v1/memory/qa/search"


@pytest.mark.asyncio
async def test_empty_qa_search_uses_bounded_readonly_fallback_not_fake_search(tmp_path, server):
    server.empty = True
    backend = create_memory_backend(backend_settings(tmp_path))
    reader = MemoryReader(backend, limit=6, recent_limit=20)
    evidence, mode = await reader.search("模板", user="current", started_ms=2000)
    assert mode == "recent_fallback" and evidence[0].id == PATH
    assert server.calls[-1][1] == {"tenant_id": "default", "user_id": "default", "scope": "user", "limit": 20}
    assert not any(path.endswith(("/save", "/write", "/edit")) for path, _ in server.calls)


def test_current_qa_and_snippets_without_content_cannot_be_historical_evidence(tmp_path):
    reader = MemoryReader(None, limit=6, recent_limit=20)
    rows = [
        {"path": PATH, "content": {"user": "current", "assistant": "answer"}},
        {"path": PATH, "content": "concurrent", "timestamp_ms": "2000"},
        {"path": PATH, "snippet": "insufficient"},
    ]
    assert reader._normalize(rows, user="current", started_ms=2000) == []


@pytest.mark.asyncio
async def test_unknown_evidence_cannot_generate_suggestion():
    model = Model({**SUGGEST, "evidence_ids": ["memory://invented"]})
    with pytest.raises(ValueError, match="unknown_evidence"):
        await Judge(model).decide("u", "a", [Evidence(PATH, PATH, "old", "digest")], [], session_id="s", turn_id="t")


@pytest.mark.asyncio
async def test_provider_error_is_not_silence_and_secret_is_not_recorded(tmp_path, server, open_session):
    class FailingModel(Model):
        async def generate(self, *args, **kw):
            if kw.get("call_kind") != "proactive":
                return await super().generate(*args, **kw)
            raise RuntimeError("secret-api-key")

    session, _, output = await open_session(tmp_path, model=FailingModel())
    try:
        await run_completed_turn(session, user_message="hello", session_id="s")
        plugin = plugin_for(session)
        await plugin.drain()
        record = records(plugin)[0]
        assert record["status"] == "error" and record["reason"] == "RuntimeError"
        assert "secret-api-key" not in json.dumps(record) + str(output)
        assert any("判断失败" in line for line in output)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_timeout_finishes_without_blocking_main_answer(tmp_path, server, open_session):
    class SlowModel(Model):
        async def generate(self, *args, **kw):
            if kw.get("call_kind") != "proactive":
                return await super().generate(*args, **kw)
            await asyncio.Event().wait()

    session, _, output = await open_session(tmp_path, model=SlowModel())
    try:
        plugin = plugin_for(session)
        plugin.config.timeout_seconds = 0.02
        answer = await run_completed_turn(session, user_message="hello", session_id="s")
        assert answer
        await asyncio.wait_for(plugin.drain(), 1)
        assert records(plugin)[0]["status"] == "error"
        assert records(plugin)[0]["reason"] == "TimeoutError"
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,stop,answer", [
    (RunStatus.CANCELLED, "cancelled", "partial"),
    (RunStatus.WAITING_FOR_USER, "waiting_for_user", "question"),
    (RunStatus.COMPLETED, "max_steps", "partial"),
    (RunStatus.COMPLETED, "end_turn", ""),
])
async def test_incomplete_runs_do_not_trigger_model(tmp_path, server, open_session, status, stop, answer):
    session, model, _ = await open_session(tmp_path)
    try:
        plugin = plugin_for(session)
        turn = Turn("u", "s", "t", int(time.time() * 1000), plugin.generation,
                    RunResult("r", status, stop, answer))
        plugin.schedule(turn)
        await plugin.drain()
        assert records(plugin)[0]["status"] == "skipped"
        assert not model.requests
    finally:
        await session.aclose()


def test_journal_claim_is_idempotent_across_instances_and_new_evidence_can_remind(tmp_path):
    first, second = Journal(tmp_path / "state.db"), Journal(tmp_path / "state.db")
    assert first.claim("s", "t") and not second.claim("s", "t")
    record = dict(status="suggest", reason="reuse", decision=SUGGEST,
                  evidence=[dict(id=PATH, digest="old")])
    assert first.finish("s", "t", record)["status"] == "suggest"
    second.claim("s", "t2")
    assert second.finish("s", "t2", record)["status"] == "silent"
    second.claim("s", "t3")
    assert second.finish("s", "t3", {**record, "evidence": [dict(id=PATH, digest="changed")]})["status"] == "suggest"


def test_changed_update_is_not_suppressed_by_unchanged_historical_memory(tmp_path):
    journal = Journal(tmp_path / "state.db")
    record = dict(status="suggest", reason="new confirmed data", evidence=[dict(id=PATH, digest="old")],
                  decision={**SUGGEST, "kind": "update", "target": "用户增长周报",
                            "suggestion": "将4月新增人数改为960人。"})
    journal.claim("source-a", "t1")
    assert journal.finish("source-a", "t1", record)["status"] == "suggest"
    corrected = {**record, "decision": {**record["decision"], "suggestion": "将4月新增人数改为9600人。"}}
    journal.claim("source-b", "t2")
    assert journal.finish("source-b", "t2", corrected)["status"] == "suggest"
    journal.claim("source-b", "t3")
    assert journal.finish("source-b", "t3", corrected)["reason"] == "duplicate_suggestion"


def test_legacy_reminder_blocks_same_action_but_allows_changed_update(tmp_path):
    import hashlib

    journal = Journal(tmp_path / "state.db")
    decision = {**SUGGEST, "kind": "update", "suggestion": "改成960人。"}
    legacy_key = hashlib.sha256(json.dumps(
        [decision["kind"], decision["target"].strip().casefold(), ["old"]], ensure_ascii=False,
    ).encode()).hexdigest()
    with journal.connect() as db:
        db.execute("INSERT INTO reminders VALUES (?, ?, ?)",
                   (legacy_key, time.time(), json.dumps(decision, ensure_ascii=False)))
    record = dict(status="suggest", decision=decision, evidence=[dict(id=PATH, digest="old")])
    journal.claim("s", "same")
    assert journal.finish("s", "same", record)["status"] == "silent"
    journal.claim("s", "changed")
    record["decision"] = {**decision, "suggestion": "改成9600人。"}
    assert journal.finish("s", "changed", record)["status"] == "suggest"


def test_terminal_control_sequences_are_not_rendered():
    from box_agent.plugins.proactive.runtime import _safe_text
    assert _safe_text("\x1b[31mHello\x1b[0m\r\x07") == "Hello"
    assert _safe_text("\x1b]52;c;malicious\x07safe") == "safe"


@pytest.mark.parametrize("payload", [{**SUGGEST, "evidence_ids": []}, {**SILENT, "suggestion": "act"}])
def test_invalid_decision_shape_is_rejected(payload):
    with pytest.raises(ValueError):
        Decision.model_validate(payload)


@pytest.mark.asyncio
async def test_immediate_new_input_still_records_cancellation(tmp_path, server, open_session):
    session, model, output = await open_session(tmp_path)
    try:
        plugin = plugin_for(session)
        plugin.schedule(Turn("u", "s", "t", 2000, plugin.generation,
                             RunResult("r", RunStatus.COMPLETED, "end_turn", "answer")))
        plugin.invalidate()
        await plugin.drain()
        assert records(plugin)[0]["status"] == "cancelled"
        assert not model.requests
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_memory_failure_does_not_fall_back_or_masquerade_as_no_evidence(tmp_path, server):
    backend = create_memory_backend(backend_settings(tmp_path))
    backend.config.max_retries = 0
    reader = MemoryReader(backend, limit=6, recent_limit=20)
    server.fail = True
    with pytest.raises(Exception):
        await reader.search("query", user="user", started_ms=2000)
    assert len(server.calls) == 1
    assert server.calls[0][0] == "/v1/memory/search"


@pytest.mark.asyncio
async def test_oversized_qa_is_explicitly_skipped_not_silently_truncated(tmp_path, server, open_session):
    session, model, _ = await open_session(tmp_path)
    try:
        plugin = plugin_for(session)
        plugin.schedule(Turn("u" * 24001, "s", "t", 2000, plugin.generation,
                             RunResult("r", RunStatus.COMPLETED, "end_turn", "answer")))
        await plugin.drain()
        assert records(plugin)[0]["reason"] == "qa_exceeds_budget"
        assert not model.requests
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_invalid_config_degrades_without_breaking_normal_chat(tmp_path, server, open_session):
    session, model, output = await open_session(tmp_path, settings={"enabled": True, "timeout_seconds": -1})
    try:
        assert plugin_for(session) is None
        text = await run_completed_turn(session, user_message="hello", session_id="s")
        assert text and not model.requests
        assert any("未启用" in line for line in output)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_evidence_namespace_isolated_by_memory_user(tmp_path, server, open_session):
    session, _, _ = await open_session(tmp_path)
    try:
        first_path = plugin_for(session).journal.path
        other, _, _ = await open_session(tmp_path, memory_user_id="different-user")
        try:
            assert plugin_for(other).journal.path != first_path
        finally:
            await other.aclose()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("content,finish", [('not json', 'stop'), (json.dumps(SUGGEST), 'length')])
async def test_malformed_or_truncated_judge_output_is_rejected(content, finish):
    class BadModel:
        async def generate(self, **kwargs):
            return LLMResponse(content=content, finish_reason=finish)

    with pytest.raises(ValueError):
        await Judge(BadModel()).decide("u", "a", [Evidence(PATH, PATH, "old", "digest")], [], session_id="s", turn_id="t")


@pytest.mark.asyncio
async def test_silent_empty_reason_gets_one_format_repair_without_changing_verdict():
    class RepairModel:
        calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            value = {**SILENT, "reason": ""} if self.calls == 1 else SILENT
            return LLMResponse(content=json.dumps(value), finish_reason="stop")

    model = RepairModel()
    decision = await Judge(model).decide("u", "a", [Evidence(PATH, PATH, "old", "digest")], [], session_id="s", turn_id="t")
    assert decision.outcome == "silent" and decision.reason == SILENT["reason"]
    assert model.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_plugin_emits_structured_search_and_judgment_stages(tmp_path, server, open_session, fallback):
    server.empty = fallback
    progress = []

    def show(event, data):
        if event == "plugin.progress":
            if data["operation"] == "memory_search" and data["phase"] == "start":
                assert not any(path.endswith("/qa/search") for path, _ in server.calls)
            progress.append(data)

    session, _, _ = await open_session(tmp_path, diagnostics=show)
    try:
        await run_completed_turn(session, user_message="帮我拟季度汇报提纲", session_id="s", turn_id="t")
        plugin = plugin_for(session)
        await plugin.drain()
        operations = ["prepare_query", "memory_item_search", "memory_search"]
        if fallback:
            operations.append("memory_fetch_recent")
        operations.append("judge_reminder")
        assert [(item["operation"], item["phase"]) for item in progress] == [
            (operation, phase) for operation in operations for phase in ("start", "result")
        ]
        assert all(item["session_id"] == "s" and item["turn_id"] == "t" for item in progress)
        search_start = next(item for item in progress
                            if item["operation"] == "memory_search" and item["phase"] == "start")
        assert search_start["arguments"]["query"] == "季度汇报 模板"
        details = [detail for item in progress for detail in item.get("details", [])]
        assert any(PATH in detail and OLD_QA["user"] in detail for detail in details)
        assert progress[-1]["summary"] == "判断完成：有值得提醒的行动"
        assert any("不是搜索命中" in item["summary"] for item in progress) is fallback
        assert records(plugin)[0]["retrieval_mode"] == ("recent_fallback" if fallback else "qa_search")
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_search_failure_closes_progress_step_without_claiming_success(tmp_path, server, open_session):
    progress = []
    session, _, output = await open_session(
        tmp_path, diagnostics=lambda event, data: progress.append(data) if event == "plugin.progress" else None,
    )
    try:
        server.fail = True
        plugin = plugin_for(session)
        plugin.schedule(Turn("u", "s", "t", 2000, plugin.generation,
                             RunResult("r", RunStatus.COMPLETED, "end_turn", "answer")))
        await plugin.drain()
        assert progress[-1]["operation"] == "memory_item_search"
        assert progress[-1]["phase"] == "error"
        assert not any(item["operation"] in {"memory_fetch_recent", "judge_reminder"} for item in progress)
        assert any("判断失败" in text for text in output)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_display_failure_does_not_change_decision_or_memory_retrieval(tmp_path, server, open_session):
    def broken_display(event, data):
        raise OSError("output unavailable")

    session, _, _ = await open_session(tmp_path, diagnostics=broken_display)
    try:
        await run_completed_turn(session, user_message="季度汇报", session_id="s")
        plugin = plugin_for(session)
        await plugin.drain()
        assert records(plugin)[0]["status"] == "suggest"
    finally:
        await session.aclose()
