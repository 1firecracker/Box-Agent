"""Post-answer watch proposals never alter the normal Agent's tools or prompt."""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from box_agent.plugins.proactive.acp import notification_publisher
from box_agent.plugins.proactive.bindings import ProactiveLifecycle
from box_agent.plugins.proactive.judge import JUDGE_PROMPT, Judge
from tests.test_acp_proactive import adapter, prompt
from tests.test_proactive_plugin import (
    Model, SILENT, isolated_profile, open_session, plugin_for, records,
    run_completed_turn, server,
)


def test_watch_prompt_encourages_follow_up_beyond_completed_deliverables_and_websites():
    assert "本轮回答或文档已经完成，不代表用户的事情已经结束" in JUDGE_PROMPT
    assert "跟进邮件变化、网站的最新信息、商品/机票/酒店的实时价格" in JUDGE_PROMPT
    assert "来源或必要信息尚不清楚时，可以提出建议并请用户补充" in JUDGE_PROMPT
    assert "不要编造接入能力，也不要声称任务已启动" in JUDGE_PROMPT
    assert "必须用户保存后才开始检查" in JUDGE_PROMPT
    assert "且本轮问答含可读取的明确网页 URL" not in JUDGE_PROMPT
    assert "只提议对该网页来源的后台只读检查" not in JUDGE_PROMPT


def proposal():
    future = (date.today() + timedelta(days=3)).isoformat()
    return dict(
        outcome="suggest", kind="watch", target="沙洲开放", suggestion="建议关注出行前的开放公告。",
        execution_prompt="", reason="未来出行依赖景点开放状态", evidence_ids=["current_qa"],
        watch=dict(subject="沙洲开放", source_url="http://travel.test/news",
                   planned_visit_date=future, expires_on=future, plan_summary="用户计划前往沙洲。"),
    )


def generic_proposal():
    value = proposal()
    value.update(target="产品接口发布", suggestion="建议关注新接口发布，便于更新集成方案。")
    value["watch"] = dict(
        subject="产品接口发布", source_url="https://product.test/releases",
        expires_on=(date.today() + timedelta(days=7)).isoformat(),
        plan_summary="用户准备集成批量接口，目前产品公告说明尚未发布。",
        check_condition="批量接口正式发布并有可用文档，需要更新集成方案。",
        check_interval_minutes=720,
    )
    return value


@pytest.mark.asyncio
async def test_generic_watch_can_propose_without_travel_date():
    value = generic_proposal()
    decision = await Judge(Model(value)).decide(
        "我要集成批量接口 https://product.test/releases", "接口尚未发布", [], [],
        allow_watch=True, session_id="s", turn_id="t",
    )
    assert decision.watch.planned_visit_date is None
    assert decision.watch.check_interval_minutes == 720
    assert decision.watch.check_condition == value["watch"]["check_condition"]
    assert decision.execution_prompt == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("patch", [
    {"check_condition": None}, {"check_condition": "   "},
    {"check_interval_minutes": 1}, {"check_interval_minutes": "720"},
    {"expires_on": "2000-01-01"},
    {"expires_on": (date.today() + timedelta(days=91)).isoformat()},
    {"source_url": "https://invented.test/releases"},
    {"plan_summary": " "},
])
async def test_generic_watch_rejects_missing_scope_unbounded_period_or_invented_source(patch):
    value = generic_proposal()
    value["watch"].update(patch)
    with pytest.raises(ValueError):
        await Judge(Model(value)).decide(
            "https://product.test/releases", "尚未发布", [], [], allow_watch=True,
            session_id="s", turn_id="t",
        )


@pytest.mark.asyncio
async def test_generic_watch_delivery_is_bound_deduplicated_and_keeps_distinct_conditions(tmp_path, server, open_session):
    publish = AsyncMock()
    publish.supports_watch_proposals = True
    model = Model(generic_proposal())
    session, _, _ = await open_session(tmp_path, model=model, publisher=publish)
    plugin = plugin_for(session)
    plugin.reader.search = AsyncMock(return_value=([], "none"))
    try:
        for turn in ("t1", "t2", "t3"):
            if turn == "t3":
                model.decision["watch"]["check_condition"] = "旧接口宣布停用日期，需调整迁移安排。"
            await run_completed_turn(session, user_message="接口集成 https://product.test/releases",
                                     session_id="original", turn_id=turn)
            await plugin.drain()
        assert publish.await_count == 2
        topic, payload = publish.call_args_list[0].args
        assert topic == "proactive.watch_proposal"
        assert payload["source_session_id"] == "original"
        assert payload["watch"] == generic_proposal()["watch"]
        assert records(plugin)[1]["reason"] == "duplicate_suggestion"
        assert "register_watch" not in session.agent.tools
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_first_plan_can_propose_watch_without_memory_and_deduplicates_wording(tmp_path, server, open_session):
    publish = AsyncMock()
    publish.supports_watch_proposals = True
    model = Model(proposal())
    session, _, _ = await open_session(tmp_path, model=model, publisher=publish)
    plugin = plugin_for(session)
    plugin.reader.search = AsyncMock(return_value=([], "none"))
    try:
        for turn in ("t1", "t2"):
            await run_completed_turn(session, user_message="规划沙洲出游 http://travel.test/news",
                                     session_id="original", turn_id=turn)
            await plugin.drain()
            model.decision["suggestion"] = "可以定期核对这个沙洲的公告。"
        publish.assert_awaited_once()
        topic, payload = publish.call_args.args
        assert topic == "proactive.watch_proposal"
        assert payload["source_session_id"] == "original"
        assert payload["source_turn_id"] == "t1"
        assert payload["watch"]["source_url"] == "http://travel.test/news"
        assert records(plugin)[0]["delivery"] == "delivered"
        assert records(plugin)[1]["reason"] == "duplicate_suggestion"
        assert "register_watch" not in session.agent.tools
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unsupported", "unknown_url", "past_date"])
async def test_watch_proposal_rejects_missing_capability_or_ungrounded_scope(case):
    value = proposal()
    if case == "past_date":
        value["watch"]["planned_visit_date"] = "2000-01-01"
    user = "http://travel.test/news" if case != "unknown_url" else "ordinary question"
    with pytest.raises(ValueError):
        await Judge(Model(value)).decide(user, "answer", [], [], allow_watch=case != "unsupported",
                                         session_id="s", turn_id="t")


@pytest.mark.asyncio
async def test_capable_acp_normal_agent_has_no_watch_instruction_or_tool(adapter, tmp_path):
    agent, host, model = adapter
    model.decision = proposal()
    host.extMethod = AsyncMock(return_value={"status": "draft", "draft": {"watch_draft_id": "d"}})
    session = await agent.newSession(SimpleNamespace(cwd=str(tmp_path), field_meta={
        "session_id": "original", "host_capabilities": {"proactive_reminders": 1, "proactive_watches": 1},
    }))
    state = agent._sessions[session.sessionId]
    assert "register_watch" not in state.agent.tools
    assert "持续关注：" not in state.agent.system_prompt
    await prompt(agent, session.sessionId, "做旅游攻略 http://travel.test/news", "t1")
    await ProactiveLifecycle.drain_turn_observers(state)
    host.extMethod.assert_awaited_once()
    method, payload = host.extMethod.call_args.args
    assert method == "session/proactive_watch"
    assert payload["sessionId"] == session.sessionId
    assert payload["watch"] == proposal()["watch"]
    for row in host.updates:
        raw = getattr(row.update, "rawOutput", None)
        assert not isinstance(raw, dict) or raw.get("kind") != "officev3_schedule_draft"


@pytest.mark.asyncio
async def test_watch_delivery_requires_host_capability_and_acknowledgement():
    host = SimpleNamespace(ext_method=AsyncMock(return_value={"status": "queued"}))
    publisher = notification_publisher(host, "s", {"host_capabilities": {"proactive_reminders": 1}})
    with pytest.raises(ValueError, match="not_supported"):
        await publisher("proactive.watch_proposal", {"id": "p", "watch": proposal()["watch"]})
    host.ext_method.assert_not_awaited()
    publisher = notification_publisher(host, "s", {"host_capabilities": {
        "proactive_reminders": 1, "proactive_watches": 1,
    }})
    with pytest.raises(ValueError, match="not_acknowledged"):
        await publisher("proactive.watch_proposal", {"id": "p", "watch": proposal()["watch"]})


@pytest.mark.asyncio
async def test_ordinary_qa_can_stay_silent_with_watch_capability(tmp_path, server, open_session):
    publish = AsyncMock()
    publish.supports_watch_proposals = True
    session, _, _ = await open_session(tmp_path, model=Model(SILENT), publisher=publish)
    try:
        plugin = plugin_for(session)
        plugin.reader.search = AsyncMock(return_value=([], "none"))
        await run_completed_turn(session, user_message="你好", session_id="s", turn_id="t")
        await plugin.drain()
        assert records(plugin)[0]["status"] == "silent"
        publish.assert_not_awaited()
    finally:
        await session.aclose()
