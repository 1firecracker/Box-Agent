"""Prepared actions are explicit, host-gated, and cannot target arbitrary sessions."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from box_agent.plugins.proactive.acp import reminder_action_tool
from box_agent.plugins.proactive.action_tool import ProactiveReminderTool
from box_agent.plugins.proactive.judge import Decision
from tests.test_proactive_plugin import SUGGEST, SILENT


def test_suggestion_requires_a_prepared_instruction_and_silent_cannot_carry_one():
    assert Decision(**SUGGEST).execution_prompt == SUGGEST["execution_prompt"]
    with pytest.raises(ValidationError):
        Decision(**{**SUGGEST, "execution_prompt": " "})
    with pytest.raises(ValidationError):
        Decision(**{**SILENT, "execution_prompt": "execute"})


@pytest.mark.asyncio
async def test_action_tool_only_forwards_action_and_persisted_reminder_identity():
    request = AsyncMock(return_value={"status": "running"})
    tool = ProactiveReminderTool(request)
    result = await tool.invoke({"action": "accept", "reminder_id": "notice-1"})
    assert result.success and '"running"' in result.content
    request.assert_awaited_once_with({"action": "accept", "reminder_id": "notice-1"})
    for invalid in ({"action": "accept"}, {"action": "accept", "reminder_id": "notice-1", "prompt": "other"},
                    {"action": "accept", "reminder_id": "notice-1", "session_id": "other"}):
        assert not (await tool.invoke(invalid)).success
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_old_hosts_do_not_expose_actions_and_transport_errors_do_not_trigger_retries():
    host = type("Host", (), {"extMethod": AsyncMock(return_value={"reminders": []})})()
    assert reminder_action_tool(host, "s", {}) is None
    assert reminder_action_tool(host, "s", {"host_capabilities": {"proactive_reminders": 1}}) is None
    tool = reminder_action_tool(host, "s", {"host_capabilities": {"proactive_actions": 1}})
    assert (await tool.execute("list")).success
    host.extMethod.assert_awaited_once_with("session/proactive_action", {"sessionId": "s", "action": "list", "reminder_id": ""})
    host.extMethod.side_effect = TimeoutError()
    assert not (await tool.execute("accept", "notice-1")).success
    assert host.extMethod.await_count == 2
