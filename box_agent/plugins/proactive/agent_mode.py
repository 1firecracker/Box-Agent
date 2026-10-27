"""Proactive tools and policy over the ordinary Box-Agent execution loop."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from importlib.resources import files

from ...kernel.hook_types import BeforeToolDecision
from ...plugins.hooks import HookSpec
from ...tools.base import Tool, ToolResult
from ...user_paths import state_path
from .journal import Journal
from .memory import add_evidence_reader
from .watch import WatchCheckPlugin


AGENT_PROMPT = files(__package__).joinpath("prompts/agent.md").read_text(encoding="utf-8").strip()


class NotifyTool(Tool):
    name = "proactive_notify"
    description = (
        "向固定助理投递一条持久化主动提醒或测试提醒。路由和来源由宿主绑定。"
        "suggestion 用100字以内说明具体建议，不附长理由或网址；依据写在 reason，需要后续执行时附完整 execution_prompt。"
        "只发布建议，不执行，不表示用户已接受。工具成功才代表提醒已保存。"
    )
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "suggestion": {"type": "string", "minLength": 1, "maxLength": 100},
        "target": {"type": "string", "minLength": 1, "maxLength": 120},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "execution_prompt": {"type": "string", "maxLength": 4000},
    }, "required": ["suggestion", "target", "reason"]}

    def __init__(self, publish, context):
        self.publish = publish
        self.context = context
        self.journal = None

    async def execute(self, suggestion: str, target: str, reason: str, execution_prompt: str = ""):
        if not suggestion.strip() or not target.strip() or not reason.strip():
            return ToolResult(success=False, error="提醒必须包含具体建议、目标和依据。")
        if len(suggestion) > 100:
            return ToolResult(success=False, error="提醒正文不得超过100字；请精简正文，完整依据保留在 reason。")
        source = self.context["source_session_id"]
        turn = self.context.get("source_turn_id") or f"assistant-chat-{time.time_ns()}"
        digest = hashlib.sha256(json.dumps([source, turn, target, suggestion], ensure_ascii=False).encode()).hexdigest()
        payload = {"version": 1, "id": digest, "source_session_id": source,
                   "source_turn_id": turn, "target": target,
                   "suggestion": suggestion, "created_at": time.time()}
        if execution_prompt.strip():
            payload["execution_prompt"] = execution_prompt
        journal_turn = f"notify-{digest}"
        if self.journal:
            if self.journal.claim(source, journal_turn):
                record = self.journal.finish(source, journal_turn, {
                    "status": "suggest", "reason": reason, "evidence": [],
                    "decision": {"kind": "follow_up", "target": target, "suggestion": suggestion,
                                 "reason": reason, "evidence_ids": [], "execution_prompt": execution_prompt},
                    "delivery": "pending", "notification": payload,
                })
                if record["status"] == "silent":
                    return ToolResult(success=True, content="相同建议已经记录，无新变化，不重复提醒。")
            elif not any(key == journal_turn for key, _ in self.journal.pending_deliveries(source)):
                return ToolResult(success=True, content="这条提醒已经处理，不重复投递。")
        try:
            await self.publish("proactive.reminder", payload)
        except Exception:
            return ToolResult(success=False, error="客户端未确认提醒已保存，请勿声称投递成功。")
        if self.journal:
            self.journal.mark_delivery(source, journal_turn, "delivered")
        return ToolResult(success=True, content=json.dumps({"status": "persisted", "reminder_id": digest}))


class TasksTool(Tool):
    name = "proactive_tasks"
    description = "查询 Proactive 自己的关注任务和近期提醒，用于避免重复。创建须用 register_watch 草稿并等待用户保存。"
    parameters = {"type": "object", "additionalProperties": False, "properties": {}}

    def __init__(self, request):
        self.request = request

    async def execute(self):
        try:
            result = await self.request()
            return ToolResult(success=True, content=json.dumps(result, ensure_ascii=False))
        except Exception:
            return ToolResult(success=False, error="无法读取关注任务，不能假定没有任务。")


class ProactiveModePlugin(WatchCheckPlugin):
    """Configure one Box-Agent run; no model loop or timer lives here."""

    guard_id = "proactive.read-only"

    def __init__(self, resources):
        super().__init__(resources)
        self.mode = resources.context.options.state.get("proactive_agent")
        self.check = self.mode if self.mode and self.mode.get("trigger") != "scheduled" else None

    async def prepare(self):
        self.resources.state.pop("proactive_agent", None)
        if not self.mode:
            return
        add_evidence_reader(self.resources)
        self.resources.system_prompt += f"\n\n{AGENT_PROMPT}\n今天：{date.today().isoformat()}"
        backend = self.resources.memory_manager
        if backend is not None and hasattr(backend, "tenant_id"):
            identity = "\0".join((backend.config.base_url, backend.tenant_id, backend.user_id))
            namespace = hashlib.sha256(identity.encode()).hexdigest()[:20]
            journal = Journal(state_path(f"proactive/{namespace}/decisions.sqlite3"))
            for tool in self.resources.tools:
                if isinstance(tool, NotifyTool):
                    tool.journal = journal
        if self.mode.get("trigger") != "chat":
            self.resources.state.update(memory_runtime=None, memory_manager=None, memory_extractor=None)

    def get_hooks(self):
        return (HookSpec("proactive-capabilities", ("tool.before_execution",), "before_tool", self),) if self.check else ()

    def install_agent(self, agent):
        super().install_agent(agent)
        if self.mode:
            agent.local_tool_exposure.require_tools(tuple(
                name for name in ("proactive_notify", "proactive_tasks", "register_watch", "proactive_reminder")
                if name in agent.tools
            ))

    async def handle(self, context):
        name = context.payload.get("tool_name", "")
        args = context.payload.get("arguments", {})
        safe = {"memory_search", "memory_read", "proactive_memory_read", "tool_search", "web_search", "web_fetch",
                "managed_browser_navigate", "managed_browser_snapshot", "managed_browser_take_screenshot",
                "proactive_notify", "register_watch", "proactive_tasks", "proactive_reminder"}
        if name in safe:
            return BeforeToolDecision.allow()
        if name == "get_skill" and args.get("skill_name") in {"browser-use", "memory-guide"}:
            return BeforeToolDecision.allow()
        if name == "mcp_config" and (args.get("action") == "inspect_browser" or
                (args.get("action") == "set_browser_mode" and args.get("mode") == "headless")):
            return BeforeToolDecision.allow()
        return BeforeToolDecision.deny("主动跟进只能读取信息、发布建议和准备需用户确认的关注草稿。", "PROACTIVE_READ_ONLY")
