"""A narrow host-owned action, never an arbitrary cross-session send tool."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from ...tools.base import Tool, ToolResult


class ProactiveReminderTool(Tool):
    def __init__(self, request: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]):
        self.request = request

    @property
    def name(self) -> str:
        return "proactive_reminder"

    @property
    def description(self) -> str:
        return (
            "查看助理会话收到的主动建议及执行状态，或接受/忽略一条建议。"
            "用户说接受这条建议时，先用 list 查到真实 reminder_id，再用 accept。"
            "多条建议含糊不清时先请用户选择。accept 会由客户端弹出确认，"
            "仅在用户确认后，把预先保存的指令送到来源会话后台执行。"
            "工具不接受自定义 Prompt 或目标会话。返回 running 只是开始执行，"
            "不能声称已完成；用 status 查看结果。不得自行接受或批量执行建议。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object", "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["list", "accept", "ignore", "status"]},
                "reminder_id": {"type": "string", "maxLength": 256},
            },
            "required": ["action"],
        }

    async def execute(self, action: str, reminder_id: str = "") -> ToolResult:
        if action not in ("list", "accept", "ignore", "status"):
            return ToolResult(success=False, error="Unsupported reminder action")
        if action != "list" and not reminder_id.strip():
            return ToolResult(success=False, error="先调用 list 获取真实 reminder_id。")
        try:
            result = await self.request({"action": action, "reminder_id": reminder_id})
        except Exception:
            return ToolResult(success=False, error="客户端未确认操作结果；请查询状态，勿自动重试执行。")
        return ToolResult(success=True, content=json.dumps(result, ensure_ascii=False))
