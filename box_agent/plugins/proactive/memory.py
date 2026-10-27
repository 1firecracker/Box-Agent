"""Read-only MemSense retrieval; normal QA saving remains with MemoryRuntime."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from ...memory import MemsenseMemoryBackend, MemoryBackendError
from ...tools.base import Tool, ToolResult


class ProactiveMemoryReadTool(Tool):
    """Read historical evidence with the session's existing MemSense identity."""

    name = "proactive_memory_read"
    description = "只读 memory_search 返回的 memory:// 历史证据全文；不修改核心记忆或历史记录。"
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "path": {"type": "string", "minLength": 10, "maxLength": 1000},
    }, "required": ["path"]}

    def __init__(self, backend: MemsenseMemoryBackend):
        self.backend = backend

    async def execute(self, path: str) -> ToolResult:
        if (not isinstance(path, str) or not path.startswith("memory://")
                or not 9 < len(path) <= 1000):
            return ToolResult(success=False, error="请使用检索返回的 memory:// 证据路径。")
        try:
            content = await self.backend.read_memory(path)
        except Exception:
            return ToolResult(success=False, error="历史记忆证据读取失败，请勿假定内容为空。")
        return ToolResult(success=True, content=content or "该记忆证据没有可读正文。")


def add_evidence_reader(resources: Any) -> None:
    config = getattr(resources.context, "config", None)
    raw = config.plugins.get("proactive") if config is not None else None
    backend = getattr(resources, "memory_manager", None)
    if (not isinstance(raw, dict) or raw.get("enabled") is not True
            or not isinstance(backend, MemsenseMemoryBackend)):
        return
    if not any(tool.name == ProactiveMemoryReadTool.name for tool in resources.tools):
        resources.tools.append(ProactiveMemoryReadTool(backend))


@dataclass(frozen=True)
class Evidence:
    id: str
    path: str
    content: str
    digest: str

    def payload(self) -> dict[str, str]:
        return asdict(self)


class MemoryReader:
    def __init__(self, backend: MemsenseMemoryBackend, *, limit: int, recent_limit: int):
        self.backend = backend
        self.limit = limit
        self.recent_limit = recent_limit

    async def search(self, query: str, *, user: str, started_ms: int,
                     progress: Callable[[str, str, dict[str, Any]], None] | None = None,
                     ) -> tuple[list[Evidence], str]:
        def report(operation: str, phase: str, **data: Any) -> None:
            if progress is not None:
                try:
                    progress(operation, phase, data)
                except Exception:
                    pass  # Optional presentation must not change retrieval.

        items = await self._search_items(query, report)

        # Items are asynchronously maintained summaries. A nonempty result does
        # not prove relevance or coverage of newly saved QA.
        report("memory_search", "start", summary="补充检索 MemSense 历史问答",
               arguments={"query": query, "top_k": self.limit * 2})
        data = await self.backend._post("proactive_qa_search", "/v1/memory/qa/search", {
            "tenant_id": self.backend.tenant_id, "user_id": self.backend.user_id,
            "query": query, "top_k": self.limit * 2,
        })
        if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
            raise MemoryBackendError("invalid_qa_search_response")
        evidence = self._normalize(data["chunks"], user=user, started_ms=started_ms)
        mode = "qa_search"
        report("memory_search", "result",
               summary=(f"取得 {min(len(evidence), self.limit)} 条历史候选，交给模型判断关联"
                        if evidence else "未找到可用历史依据"),
               details=self._preview(evidence[:self.limit]))
        if not evidence:
            # Only a valid search without usable evidence falls back to recent
            # QA. Errors are not empty results; relevance still needs judging.
            report("memory_fetch_recent", "start",
                   summary="搜索无可用结果，改读近期问答；不是搜索命中",
                   arguments={"limit": self.recent_limit})
            data = await self.backend._post("proactive_recent", "/v1/memory/fetch_recent", {
                "tenant_id": self.backend.tenant_id, "user_id": self.backend.user_id,
                "scope": "user", "limit": self.recent_limit,
            })
            if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
                raise MemoryBackendError("invalid_recent_response")
            evidence = self._normalize(data["chunks"], user=user, started_ms=started_ms)
            mode = "recent_fallback"
            report("memory_fetch_recent", "result",
                   summary=f"选取 {min(len(evidence), self.limit)} 条近期问答候选；关联性待判断",
                   details=self._preview(evidence[:self.limit]))
        if items:
            mode = f"item_and_{mode}" if evidence else "item_search"
            # Interleave both sources so a full Item result cannot crowd out
            # fresh QA. Preserve the total evidence budget and source paths.
            merged: list[Evidence] = []
            seen_ids: set[str] = set()
            seen_digests: set[str] = set()
            for index in range(max(len(items), len(evidence))):
                for source in (items, evidence):
                    if index >= len(source):
                        continue
                    entry = source[index]
                    if entry.id in seen_ids or entry.digest in seen_digests:
                        continue
                    seen_ids.add(entry.id)
                    seen_digests.add(entry.digest)
                    merged.append(entry)
                    if len(merged) == self.limit:
                        return merged, mode
            return merged, mode
        return evidence[:self.limit], mode

    async def _search_items(self, query: str, report: Callable[..., None]) -> list[Evidence]:
        report("memory_item_search", "start", summary="检索 MemSense 相关事项",
               arguments={"query": query, "top_k": self.limit * 2})
        identity = {"tenant_id": self.backend.tenant_id, "user_id": self.backend.user_id}
        data = await self.backend._post("proactive_item_search", "/v1/memory/search", {
            **identity, "query": query, "top_k": self.limit * 2,
        })
        if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
            raise MemoryBackendError("invalid_item_search_response")
        paths: list[str] = []
        for row in data["chunks"][:self.limit * 2]:
            if not isinstance(row, dict):
                raise MemoryBackendError("invalid_item_search_response")
            if row.get("result_type") != "item_memory":
                continue
            path = row.get("path")
            if not isinstance(path, str) or not re.fullmatch(r"memory://items/item_[A-Za-z0-9_-]+\.md", path):
                raise MemoryBackendError("invalid_item_path")
            if path not in paths:
                paths.append(path)
            if len(paths) == self.limit:
                break
        report("memory_item_search", "result", summary=f"找到 {len(paths)} 条候选事项，按需读取正文")
        if not paths:
            return []

        report("memory_item_read", "start", summary="读取候选 Item 正文",
               arguments={"count": len(paths)})
        evidence: list[Evidence] = []
        for path in paths:
            data = await self.backend._post("proactive_item_read", "/v1/memory/files/read", {
                **identity, "path": path,
            })
            if not isinstance(data, dict) or not isinstance(data.get("exists"), bool):
                raise MemoryBackendError("invalid_item_read_response")
            if not data["exists"]:
                continue  # A search entry may have been deleted before the read.
            if data.get("path") != path or data.get("resource_type") != "item_memory":
                raise MemoryBackendError("invalid_item_read_response")
            content = data.get("content")
            if not isinstance(content, str):
                raise MemoryBackendError("invalid_item_read_response")
            if not content.strip():
                continue
            if len(content) > 24000:
                raise MemoryBackendError("item_content_exceeds_budget")
            # Items are mutable summaries: a recent update does not make their
            # historical content unusable. Judge checks it against the current QA.
            # A metadata-only timestamp refresh must not bypass reminder dedup.
            digest_content = content
            if content.startswith("---\n") and "\n---\n" in content[4:]:
                frontmatter, body = content[4:].split("\n---\n", 1)
                frontmatter = re.sub(r"(?m)^updated_at:.*\n?", "", frontmatter)
                digest_content = frontmatter + "\n---\n" + body
            digest = hashlib.sha256(digest_content.encode()).hexdigest()
            evidence.append(Evidence(path, path, content, digest))
        report("memory_item_read", "result", summary=f"读到 {len(evidence)} 条 Item 正文，交给模型判断",
               details=self._preview(evidence))
        return evidence

    @staticmethod
    def _preview(evidence: list[Evidence]) -> list[str]:
        """Show at most three short source previews, never entire historical QA."""
        details = []
        for item in evidence[:3]:
            text = item.content
            try:
                qa = json.loads(text)
                if isinstance(qa, dict):
                    text = str(qa.get("user", text))
            except ValueError:
                pass
            details.append(f"{item.path} · {text[:100]}")
        return details

    def _normalize(self, rows: list[Any], *, user: str, started_ms: int) -> list[Evidence]:
        result: list[Evidence] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise MemoryBackendError("invalid_evidence")
            # Do not mistake this turn's concurrently saved QA for past evidence.
            timestamp = row.get("timestamp_ms")
            if timestamp is not None and int(timestamp) >= started_ms:
                continue
            content = row.get("content")
            if not content:
                continue  # A title or search snippet alone is not evidence.
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except (ValueError, TypeError):
                    parsed = content
            else:
                parsed = content
            if isinstance(parsed, dict) and str(parsed.get("user", "")).strip() == user.strip():
                continue
            content = parsed if isinstance(parsed, str) else json.dumps(parsed, ensure_ascii=False)
            path = row.get("path") or (
                f"memory://qa-chunks/{row['memory_id']}" if row.get("memory_id") else ""
            )
            if not isinstance(path, str) or not path.startswith("memory://") or path in seen:
                continue
            seen.add(path)
            digest = hashlib.sha256(content.encode()).hexdigest()
            excerpt = content[:4000] + ("\n[excerpt truncated]" if len(content) > 4000 else "")
            result.append(Evidence(path, path, excerpt, digest))
        return result
