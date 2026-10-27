"""Background lifecycle owned entirely by the proactive plugin."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from ...memory import MemsenseMemoryBackend
from ...user_paths import state_path
from .config import ProactiveConfig
from .journal import Journal
from .judge import Judge
from .memory import MemoryReader

logger = logging.getLogger(__name__)


def _safe_text(text: str) -> str:
    # Model/memory text is untrusted, including terminal control sequences.
    text = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(c for c in text if c == "\n" or (c.isprintable() and c not in "\u202a\u202b\u202d\u202e\u202c"))


@dataclass
class Turn:
    user: str
    session_id: str
    turn_id: str
    started_ms: int
    generation: int
    result: Any = None
    progress_step: str | None = None


class ProactivePlugin:
    def __init__(self, resources, *, publish_notification=None):
        self.resources = resources
        self.publish_notification = publish_notification
        self.enabled = False
        self.active: Turn | None = None
        self.generation = 0
        self.tasks: set[asyncio.Task] = set()
        self.output = resources.context.host.output or (lambda text: None)

    async def prepare(self) -> None:
        context = self.resources.context
        raw = context.config.plugins.get("proactive")
        if (raw is None or context.options.utility or self.publish_notification is None
                or context.options.state.get("proactive_watch_check")
                or context.options.state.get("proactive_agent")):
            return
        try:
            self.config = ProactiveConfig.model_validate(raw)
            if not self.config.enabled:
                return
            backend = self.resources.memory_manager
            if not isinstance(backend, MemsenseMemoryBackend):
                raise ValueError("memsense_backend_required")
            self.agent_runs = getattr(self.publish_notification, "supports_agent_runs", False) is True
            if not self.agent_runs:
                # Compatibility with older hosts which do not advertise Agent execution.
                self.reader = MemoryReader(backend, limit=self.config.evidence_limit,
                                           recent_limit=self.config.recent_limit)
                self.judge = Judge(self.resources.llm_client)
            self.allow_watch = bool(getattr(self.publish_notification, "supports_watch_proposals", False))
            identity = "\0".join((backend.config.base_url, backend.tenant_id, backend.user_id))
            namespace = hashlib.sha256(identity.encode()).hexdigest()[:20]
            self.journal = Journal(state_path(f"proactive/{namespace}/decisions.sqlite3"))
            self.enabled = True
            self.emit("[主动提醒] 已启用：完整问答结束后启动 Proactive 模式 Box-Agent。" if self.agent_runs
                      else "[主动提醒] 已启用：回答结束后只读检索 MemSense；不会自动执行建议。")
        except Exception as exc:
            self.emit(f"[主动提醒] 未启用（{type(exc).__name__}）；正常对话不受影响。")
            logger.warning("proactive initialization failed: %s", type(exc).__name__)

    def emit(self, text: str) -> None:
        try:
            self.output(_safe_text(text))
        except Exception:
            logger.warning("proactive output failed")

    def _progress(self, turn: Turn, operation: str, phase: str, data: dict) -> None:
        if turn.generation != self.generation and phase != "cancelled":
            return
        turn.progress_step = operation if phase == "start" else None
        self.resources.context.emit(
            "plugin.progress", source="主动提醒插件", operation=operation, phase=phase,
            session_id=turn.session_id, turn_id=turn.turn_id, **data,
        )

    def invalidate(self) -> None:
        """A new user input invalidates all unfinished work for this session."""
        self.generation += 1
        for task in self.tasks:
            task.cancel()

    def begin_turn(self, *, user: str, session_id: str, turn_id: str) -> tuple[Turn, bool]:
        if self.active is not None:
            return self.active, False
        self.invalidate()
        self.active = Turn(user, session_id, turn_id, int(time.time() * 1000), self.generation)
        return self.active, True

    def end_turn(self, token: tuple[Turn, bool], *, result: Any, failed: bool) -> None:
        turn, owner = token
        if result is not None:
            turn.result = result
        if owner:
            self.active = None
            if not failed:
                self.schedule(turn)

    def schedule(self, turn: Turn) -> None:
        if not self.enabled or turn.generation != self.generation:
            return
        task = asyncio.create_task(self._process(turn), name=f"proactive:{turn.turn_id}")
        self.tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self.tasks.discard(done)
            if done.cancelled():
                # Cancellation can arrive before the coroutine first runs, when
                # its try/finally has not yet been entered.
                try:
                    if self.journal.claim(turn.session_id, turn.turn_id):
                        self.journal.finish(turn.session_id, turn.turn_id, {
                            "status": "cancelled", "reason": "new_input_or_session_closed", "evidence": [],
                        })
                except Exception:
                    logger.warning("proactive cancellation could not be recorded")

        task.add_done_callback(finished)

    async def drain(self) -> None:
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)

    async def close(self) -> None:
        self.invalidate()
        await self.drain()

    async def _process(self, turn: Turn) -> None:
        record: dict = {"status": "error", "reason": "unknown", "evidence": []}
        claimed = False
        try:
            claimed = self.journal.claim(turn.session_id, turn.turn_id)
            if not claimed:
                return
            result = turn.result
            if result is None or result.status.value != "completed" or result.stop_reason != "end_turn":
                record.update(status="skipped", reason="turn_not_completed")
            elif not result.final_content.strip():
                record.update(status="skipped", reason="empty_answer")
            elif len(turn.user) + len(result.final_content) > self.config.max_qa_chars:
                record.update(status="skipped", reason="qa_exceeds_budget")
            else:
                record = await asyncio.wait_for(self._judge_turn(turn), self.config.timeout_seconds)
        except asyncio.CancelledError:
            record.update(status="cancelled", reason="new_input_or_session_closed")
        except Exception as exc:
            # Never persist exception messages: provider errors may contain secrets.
            record.update(status="error", reason=type(exc).__name__)
        finally:
            if claimed:
                if turn.generation != self.generation:
                    record.update(status="cancelled", reason="new_input_or_session_closed")
                try:
                    record = self.journal.finish(turn.session_id, turn.turn_id, record)
                    if turn.progress_step:
                        phase = {"error": "error", "cancelled": "cancelled"}.get(record["status"], "result")
                        summary = {"suggest": "判断完成：有值得提醒的行动",
                                   "silent": "判断完成：本轮不提醒",
                                   "cancelled": "新输入或会话关闭，已取消本轮判断"}.get(
                                       record["status"], f"判断未完成（{record['reason']}）")
                        self._progress(turn, turn.progress_step, phase, {"summary": summary})
                    if record["status"] != "cancelled":
                        self._display(record)
                    if record["status"] == "suggest":
                        await self._deliver(turn.session_id, turn.turn_id, record)
                except Exception as exc:
                    self.emit(f"[主动提醒] 记录失败（{type(exc).__name__}），未发布建议。")

    async def _judge_turn(self, turn: Turn) -> dict:
        if self.agent_runs:
            await self.publish_notification("proactive.turn_completed", {
                "source_session_id": turn.session_id, "source_turn_id": turn.turn_id,
                "user": turn.user, "assistant": turn.result.final_content,
            })
            return {"status": "dispatched", "reason": "box_agent_proactive_queued", "evidence": []}
        # Retrying an unacknowledged delivery uses the same ID; the host deduplicates.
        for previous_turn, previous in self.journal.pending_deliveries(turn.session_id):
            await self._deliver(turn.session_id, previous_turn, previous)
        context = {"session_id": turn.session_id, "turn_id": turn.turn_id}
        assistant = turn.result.final_content
        self._progress(turn, "prepare_query", "start", {"summary": "根据本轮完整问答整理检索词"})
        query = await self.judge.query(turn.user, assistant, **context)
        self._progress(turn, "prepare_query", "result", {"summary": "检索词已生成"})
        evidence, mode = await self.reader.search(
            query, user=turn.user, started_ms=turn.started_ms,
            progress=lambda operation, phase, data: self._progress(turn, operation, phase, data),
        )
        record = {"status": "silent", "reason": "no_historical_evidence", "query": query,
                  "retrieval_mode": mode, "evidence": [item.payload() for item in evidence]}
        if not evidence and not self.allow_watch:
            return record
        self._progress(turn, "judge_reminder", "start", {
            "summary": f"结合 {len(evidence)} 条候选记忆和已有提醒，判断是否值得打扰",
        })
        decision = await self.judge.decide(
            turn.user, assistant, evidence, self.journal.recent(),
            allow_watch=self.allow_watch, **context,
        )
        record.update(status=decision.outcome, reason=decision.reason, decision=decision.model_dump(mode="json"))
        if decision.outcome == "suggest":
            notification_id = hashlib.sha256(
                f"{self.journal.path.parent.name}\0{turn.session_id}\0{turn.turn_id}".encode()
            ).hexdigest()
            record.update(delivery="pending", notification={
                "version": 1, "id": notification_id, "source_session_id": turn.session_id,
                "source_turn_id": turn.turn_id, "suggestion": decision.suggestion,
                "execution_prompt": decision.execution_prompt,
                "target": decision.target, "created_at": time.time(),
            })
            if decision.kind == "watch":
                record["notification_topic"] = "proactive.watch_proposal"
                record["notification"]["watch"] = decision.watch.model_dump(mode="json", exclude_unset=True)
        return record

    async def _deliver(self, session_id: str, turn_id: str, record: dict) -> None:
        try:
            await asyncio.wait_for(
                self.publish_notification(
                    record.get("notification_topic", "proactive.reminder"), record["notification"],
                ),
                timeout=10,
            )
        except asyncio.CancelledError:
            raise  # Keep pending; an acknowledgement may have been lost.
        except Exception as exc:
            self.journal.mark_delivery(session_id, turn_id, "pending", type(exc).__name__)
            logger.warning("proactive delivery pending: %s", type(exc).__name__)
        else:
            self.journal.mark_delivery(session_id, turn_id, "delivered")

    def _display(self, record: dict) -> None:
        status, reason = record["status"], record["reason"]
        if status == "dispatched":
            self.emit("[主动提醒] 已交给 Proactive 模式 Box-Agent 后台处理。")
            return
        labels = {"no_historical_evidence": "没有找到可用的历史依据",
                  "duplicate_suggestion": "同一事项已提醒，没有新的依据",
                  "turn_not_completed": "本轮未正常完成", "empty_answer": "本轮没有最终回答",
                  "qa_exceeds_budget": "问答过长，本轮跳过判断"}
        if status == "suggest":
            decision = record["decision"]
            self.emit(f"\n[主动提醒] {decision['suggestion']}\n依据：{reason}\n来源："
                      + "、".join(decision["evidence_ids"]) + "\n（仅建议，尚未执行）\n")
        elif status == "error":
            self.emit(f"\n[主动提醒] 本轮判断失败（{reason}），不是‘无需提醒’。\n")
        else:
            self.emit(f"\n[主动提醒] 本轮不提醒：{labels.get(reason, reason).rstrip('。')}。\n")
