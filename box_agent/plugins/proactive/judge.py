"""Tool-free model decisions with validated evidence references."""

from __future__ import annotations

import json
from datetime import date
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ...schema import Message
from .memory import Evidence


class WatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=120)
    source_url: str = Field(min_length=1, max_length=1000)
    # Retained for existing travel drafts; generic watches need no travel date.
    planned_visit_date: date | None = None
    expires_on: date
    plan_summary: str = Field(min_length=1, max_length=4000)
    check_condition: str | None = Field(default=None, min_length=1, max_length=1000)
    check_interval_minutes: Literal[10, 30, 60, 180, 360, 720, 1440] = 720

    @model_validator(mode="after")
    def validate_scope(self):
        if not self.subject.strip() or not self.plan_summary.strip():
            raise ValueError("watch_requires_concrete_context")
        if self.check_condition is not None and not self.check_condition.strip():
            raise ValueError("watch_requires_check_condition")
        if self.planned_visit_date is None and self.check_condition is None:
            raise ValueError("watch_requires_check_condition")
        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["suggest", "silent"]
    kind: Literal["reuse", "update", "follow_up", "watch", "none"]
    target: str = Field(max_length=120)
    suggestion: str = Field(max_length=300)
    execution_prompt: str = Field(default="", max_length=4000)
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(max_length=6)
    watch: WatchProposal | None = None

    @model_validator(mode="after")
    def validate_suggestion(self):
        if self.outcome == "suggest" and self.kind == "watch":
            if (not self.watch or not self.target.strip() or not self.suggestion.strip()
                    or self.execution_prompt or self.evidence_ids != ["current_qa"]):
                raise ValueError("watch_requires_current_qa_and_structured_proposal")
        elif self.outcome == "suggest":
            if (self.kind == "none" or not self.target.strip() or not self.suggestion.strip()
                    or not self.execution_prompt.strip() or not self.evidence_ids):
                raise ValueError("suggestion_requires_action_and_evidence")
        elif self.suggestion or self.execution_prompt or self.kind != "none" or self.target or self.evidence_ids:
            raise ValueError("silent_requires_empty_action")
        if self.kind != "watch" and self.watch is not None:
            raise ValueError("unexpected_watch")
        return self


_PROMPTS = files(__package__).joinpath("prompts")
QUERY_PROMPT = _PROMPTS.joinpath("query.md").read_text(encoding="utf-8").strip()
JUDGE_PROMPT = _PROMPTS.joinpath("judge.md").read_text(encoding="utf-8").strip()


class Judge:
    def __init__(self, llm):
        self.llm = llm

    async def _generate(self, prompt: str, data: dict, *, session_id: str, turn_id: str) -> str:
        response = await self.llm.generate(
            messages=[Message(role="system", content=prompt),
                      Message(role="user", content=json.dumps(data, ensure_ascii=False))],
            tools=None, thinking_enabled=False, session_id=session_id, turn_id=turn_id,
            call_kind="proactive",
        )
        if response.tool_calls or response.finish_reason not in ("stop", "end_turn"):
            raise ValueError("incomplete_judge_response")
        return response.content.strip()

    async def query(self, user: str, assistant: str, **context) -> str:
        text = await self._generate(QUERY_PROMPT, {"user": user, "assistant": assistant}, **context)
        if not text or len(text) > 200:
            raise ValueError("invalid_search_query")
        return text

    async def decide(self, user: str, assistant: str, evidence: list[Evidence],
                     previous: list[dict], *, allow_watch: bool = False, **context) -> Decision:
        data = {
            "current_qa": {"user": user, "assistant": assistant},
            "historical_evidence": [item.payload() for item in evidence],
            "previous_suggestions": previous,
            "watch_proposals_enabled": allow_watch,
            "today": date.today().isoformat(),
        }
        text = await self._generate(JUDGE_PROMPT, data, **context)
        try:
            decision = Decision.model_validate_json(text)
        except ValidationError as exc:
            # One bounded formatting repair, never a retry-until-suggest loop.
            data["invalid_previous_output"] = text[:2000]
            data["validation_errors"] = [
                {"field": error["loc"], "type": error["type"]} for error in exc.errors()
            ]
            text = await self._generate(
                JUDGE_PROMPT + "\n上次输出未通过结构校验。请修正格式并补齐原因，保留基于证据的判断。",
                data, **context,
            )
            decision = Decision.model_validate_json(text)
        available = {item.id for item in evidence}
        if decision.kind == "watch":
            watch = decision.watch
            if not allow_watch or watch is None:
                raise ValueError("watch_not_supported")
            if (watch.source_url not in user + "\n" + assistant
                    or not watch.source_url.startswith(("http://", "https://"))
                    or watch.expires_on < date.today()
                    or (watch.expires_on - date.today()).days > 90
                    or (watch.planned_visit_date is not None and (
                        watch.planned_visit_date < date.today()
                        or watch.expires_on < watch.planned_visit_date))):
                raise ValueError("watch_requires_known_source_and_future_dates")
            available.add("current_qa")
        if not set(decision.evidence_ids) <= available:
            raise ValueError("unknown_evidence_id")
        return decision
