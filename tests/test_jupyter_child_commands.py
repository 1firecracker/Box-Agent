"""Real kernels share Bash's child command checks without rejecting probes."""

import asyncio
import os
from pathlib import Path
import sys

import pytest
import pytest_asyncio

from box_agent.tools.base import ToolInvocationContext
from box_agent.tools.chat_references import ChatReferences
from box_agent.tools.engine.execution import invoke_tool_once
from box_agent.tools.jupyter_tool import JupyterSandboxTool, SandboxEnvironment


@pytest_asyncio.fixture
async def kernel(tmp_path, monkeypatch):
    async def ready(*args, **kwargs):
        pass

    monkeypatch.setattr(SandboxEnvironment, "_verify_packages", ready)
    target = tmp_path / "published.txt"
    target.write_text("published")
    guard = ChatReferences(tmp_path)
    guard.retain(target)
    tool = JupyterSandboxTool(workspace_dir=str(tmp_path),
                              runtime_env={"BOX_AGENT_SANDBOX_PYTHON": sys.executable})

    async def run(code, session_id="regression"):
        return await invoke_tool_once(tool, {"code": code, "session_id": session_id},
                                      ToolInvocationContext(chat_references=guard))

    try:
        yield target, run
    finally:
        await JupyterSandboxTool.shutdown_all()


@pytest.mark.asyncio
async def test_read_only_child_commands_and_python_help_preserve_references(kernel):
    target, run = kernel
    result = await run("import subprocess, sys, os\n"
                       "print(subprocess.check_output([sys.executable, '--help'], text=True)[:30])\n"
                       "print(subprocess.check_output([sys.executable, '-c', 'print(42)'], text=True))\n"
                       "print(subprocess.check_output('command -v python3; ls -d .', shell=True, text=True))")
    assert result.success, result.error
    assert "42" in result.content
    assert target.read_text() == "published"
    # Persistent kernel calls restore their environment and install a fresh bridge.
    again = await run("print(subprocess.check_output([sys.executable, '-c', 'print(43)'], text=True))")
    assert again.success, again.error
    assert "43" in again.content


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [
    "import os; os.unlink(''.join(['published', '.txt']))",
    "import os; os.rename(''.join(['published', '.txt']), 'moved.txt')",
    "import subprocess; subprocess.run(['rm', 'published.txt'], check=True)",
])
async def test_child_deletion_and_move_are_blocked_before_referenced_file_changes(kernel, child):
    target, run = kernel
    result = await run("import subprocess, sys\n"
                       f"p = subprocess.run([sys.executable, '-c', {child!r}], capture_output=True, text=True)\n"
                       "print(p.returncode, p.stderr)")
    assert result.success, result.error
    assert "CHAT_REFERENCE_PROTECTED" in result.content
    assert target.read_text() == "published"


@pytest.mark.asyncio
async def test_child_edits_atomic_replace_and_temporary_cleanup_remain_allowed(kernel):
    target, run = kernel
    child = ("import os, tempfile\nfrom pathlib import Path\n"
             "with open('published.txt', 'a') as f: f.write(' appended')\n"
             "Path('new.txt').write_text('updated')\nos.replace('new.txt', 'published.txt')\n"
             "with tempfile.TemporaryDirectory() as d: Path(d, 'scratch').write_text('scratch')\n"
             "print('allowed')")
    result = await run("import subprocess, sys\n"
                       f"print(subprocess.check_output([sys.executable, '-c', {child!r}], text=True))")
    assert result.success, result.error
    assert "allowed" in result.content
    assert target.read_text() == "updated"


@pytest.mark.asyncio
async def test_kernel_producer_registers_outputs_with_existing_chat_references(kernel):
    from box_agent.artifact_metadata import read_metadata, read_scope

    target, run = kernel
    helper = Path(__file__).resolve().parents[1] / "box_agent/skills/document-skills/pptx/scripts/artifact_metadata.py"
    result = await run(
        "import runpy\nfrom pathlib import Path\n"
        f"metadata = runpy.run_path({str(helper)!r})\n"
        "Path('new-deck').mkdir()\n"
        "metadata['write_scope']('new-deck', {'schema_version': 1, 'default': 'intermediate'})\n"
        "Path('new-deck/new.png').write_bytes(b'preview')\n"
        "metadata['write_metadata']('new-deck/new.png', {'type': 'intermediate_asset'})"
    )
    assert result.success, result.error
    output = target.parent / "new-deck/new.png"
    assert read_metadata(output, target.parent)["type"] == "intermediate_asset"
    assert read_scope(output.parent, target.parent) == {"schema_version": 1, "default": "intermediate"}
    assert not (output.parent / ".new.png.artifact.json").exists()
    assert not (output.parent / ".artifact-delivery.json").exists()
    assert target.read_text() == "published"


@pytest.mark.asyncio
@pytest.mark.skipif(os.unlink not in os.supports_dir_fd or os.rename not in os.supports_dir_fd,
                    reason="descriptor-relative mutations unavailable")
@pytest.mark.parametrize("operation", [
    "os.unlink('published.txt', dir_fd=directory)",
    "os.rename('published.txt', 'moved.txt', src_dir_fd=directory, dst_dir_fd=directory)",
])
async def test_kernel_descriptor_relative_mutations_preserve_chat_references(kernel, operation):
    target, run = kernel
    result = await run(
        "import os\nfrom pathlib import Path\n"
        "directory = os.open('.', os.O_RDONLY)\n"
        "Path('nested').mkdir()\nos.chdir('nested')\n"
        f"try:\n    {operation}\nfinally:\n    os.close(directory)"
    )
    assert not result.success
    assert "CHAT_REFERENCE_PROTECTED" in result.error
    assert target.read_text() == "published"
    assert not (target.parent / "moved.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("args", ["[sys.executable, '-I', '-c', 'print(1)']",
                                   "['rm', 'published.txt']"])
async def test_child_cannot_disable_checks_or_delete_referenced_files(kernel, args):
    target, run = kernel
    result = await run(f"import subprocess, sys\nsubprocess.run({args}, check=True)")
    assert not result.success
    assert "CHAT_REFERENCE_PROTECTED" in result.error
    assert target.read_text() == "published"


@pytest.mark.asyncio
async def test_kernel_guard_does_not_block_other_sessions(kernel):
    target, run = kernel
    for name in ("slow", "fast"):
        assert (await run("import subprocess, sys", name)).success
    slow = asyncio.create_task(run("subprocess.run([sys.executable, '-c', 'import time; time.sleep(3)'])", "slow"))
    await asyncio.sleep(0.2)
    fast = await run("print(subprocess.check_output([sys.executable, '-c', 'print(99)'], text=True))", "fast")
    assert fast.success, fast.error
    assert "99" in fast.content
    assert not slow.done(), "Another session must finish while the slow child is running"
    assert (await slow).success
    assert target.read_text() == "published"
