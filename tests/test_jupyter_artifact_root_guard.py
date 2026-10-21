"""Regression coverage for execute_code cwd write confinement."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from box_agent.tools.jupyter_tool import JupyterSandboxTool, SandboxEnvironment


@pytest.mark.asyncio
async def test_execute_code_blocks_parent_writes_and_allows_cwd_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def skip_unrelated_package_verification(
        _self: SandboxEnvironment,
        _on_progress=None,
    ) -> None:
        return None

    monkeypatch.setattr(
        SandboxEnvironment,
        "_verify_packages",
        skip_unrelated_package_verification,
    )
    session_root = tmp_path / "session"
    session_root.mkdir()
    (session_root / "input.txt").write_text("source", encoding="utf-8")
    allowed_tmp = tmp_path / "kernel-tmp"
    allowed_tmp.mkdir()
    monkeypatch.setattr(
        "box_agent.tools.jupyter_tool.tempfile.gettempdir",
        lambda: str(allowed_tmp),
    )
    tool = JupyterSandboxTool(
        workspace_dir=str(session_root),
        runtime_env={"BOX_AGENT_SANDBOX_PYTHON": sys.executable},
    )
    assert "durable writes are confined to the session cwd" in tool.description
    session_id = "cwd-guard"

    try:
        blocked = await tool.execute(
            code=(
                "from pathlib import Path\n"
                "target = Path('../research/exact-page-reads.json')\n"
                "target.parent.mkdir(parents=True, exist_ok=True)\n"
                "target.write_text('{}', encoding='utf-8')"
            ),
            session_id=session_id,
        )

        assert blocked.success is False
        assert "EXECUTE_CODE_WRITE_OUTSIDE_WORKSPACE" in (blocked.error or "")
        assert not (tmp_path / "research" / "exact-page-reads.json").exists()

        allowed = await tool.execute(
            code=(
                "from pathlib import Path\n"
                "source = Path('input.txt').read_text(encoding='utf-8')\n"
                "target = Path('research/result.txt')\n"
                "target.parent.mkdir(parents=True, exist_ok=True)\n"
                "target.write_text(source, encoding='utf-8')\n"
                "print(target)"
            ),
            session_id=session_id,
        )

        assert allowed.success is True
        assert (session_root / "research" / "result.txt").read_text(
            encoding="utf-8"
        ) == "source"

        if sys.platform in {"darwin", "linux"}:
            outside = tmp_path / "outside"
            outside.mkdir()
            (outside / "keep.txt").write_text("keep", encoding="utf-8")
            for operation in (
                "os.unlink('keep.txt', dir_fd=directory)",
                "os.mkdir('created', dir_fd=directory)",
                "os.replace('research/result.txt', 'replacement.txt', dst_dir_fd=directory)",
            ):
                denied = await tool.execute(
                    code=(
                        f"import os\ndirectory = os.open({str(outside)!r}, os.O_RDONLY)\n"
                        f"try:\n    {operation}\nfinally:\n    os.close(directory)"
                    ),
                    session_id=session_id,
                )
                assert not denied.success
                assert "EXECUTE_CODE_WRITE_OUTSIDE_WORKSPACE" in denied.error
            assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
            assert not (outside / "created").exists()
            assert not (outside / "replacement.txt").exists()
            assert (session_root / "research/result.txt").read_text(encoding="utf-8") == "source"
    finally:
        await JupyterSandboxTool.shutdown_all()
