"""Publication and cleanup remain correct across profile paths and workspaces."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from box_agent.artifact_metadata import metadata_path
from box_agent.artifact_publication import apply_delivery_policy, retained_chat_file_in
from box_agent.tools.chat_references import ChatReferences, bind_chat_references, chat_mutation_error
from box_agent.tools.engine.artifact_results import _detect_artifacts
from box_agent.tools.jupyter_tool import _kernel_write_guard_setup_code
from box_agent.tools.python_mutation_guard import PythonMutationGuard


def test_profile_workspace_deliverables_are_published_and_retained(tmp_path):
    workspace = tmp_path / '.box-agent/sandbox/sessions/chat'
    workspace.mkdir(parents=True)
    report = workspace / 'report.txt'
    report.write_text('deliverable')
    assert len(_detect_artifacts('call', 'bash', '[report.txt]', str(workspace))) == 1
    result = apply_delivery_policy({'type': 'artifact', 'abs_path': str(report)}, str(workspace))
    assert result['type'] == 'artifact'
    guard = ChatReferences(workspace)
    guard.retain(report)
    assert report in guard.paths
    assert chat_mutation_error(report, 'delete', workspace) is not None
    assert not _detect_artifacts('call', 'bash', f'[{metadata_path(report, workspace)}]', str(workspace))


@pytest.mark.parametrize('entry', ['guard', 'child', 'kernel'])
def test_other_workspace_preserves_external_attachment_and_dependencies(tmp_path, monkeypatch, entry):
    first, second, external = (tmp_path / name for name in ('first', 'second', 'external'))
    for folder in (first, second, external):
        folder.mkdir()
    report, image = external / 'report.html', external / 'image.png'
    report.write_text('<img src="image.png">')
    image.write_bytes(b'image')
    ChatReferences(first).retain(report)
    for target in (report, image, external):
        if entry == 'guard':
            with bind_chat_references(ChatReferences(second)):
                assert chat_mutation_error(target, 'delete', second) is not None
        else:
            operation = f'from pathlib import Path; Path({str(target)!r}).' + ('rmdir()' if target == external else 'unlink()')
            if entry == 'child':
                guard = PythonMutationGuard({**os.environ, 'BOX_AGENT_WORKSPACE_DIR': str(second)})
                try:
                    result = subprocess.run([sys.executable, '-c', operation], env=guard.environment,
                                            capture_output=True, text=True, timeout=10)
                finally:
                    guard.close()
            else:
                # Explicit pytest temp roots can be outside the OS temp root.
                # Permit the fixture sandbox so this exercises retention itself.
                monkeypatch.setattr('box_agent.tools.jupyter_tool.SANDBOX_BASE_DIR', tmp_path)
                result = subprocess.run([sys.executable, '-c', _kernel_write_guard_setup_code(second) + '\n' + operation],
                                        capture_output=True, text=True, timeout=10)
            assert result.returncode != 0
            assert 'CHAT_REFERENCE_PROTECTED' in result.stderr
        assert target.exists()
    assert set(external.iterdir()) == {report, image}


def test_confirmed_deletion_releases_all_workspace_caches_for_reused_path(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    report, other = tmp_path / 'report.txt', tmp_path / 'other.txt'
    report.write_text('old report')
    other.write_text('preserve')
    ChatReferences(first).retain(report)
    ChatReferences(first).retain(other)
    guard = ChatReferences(second)
    guard.retain(report)
    report.unlink()
    guard.deleted(frozenset({report}))
    report.write_text('unrelated replacement')
    for workspace in (first, second):
        assert chat_mutation_error(report, 'delete', workspace) is None
    assert chat_mutation_error(other, 'delete', second) is not None


@pytest.mark.parametrize('directory', [False, True])
def test_publication_history_does_not_block_unrelated_cleanup(tmp_path, directory):
    store = tmp_path / '.box-agent/artifact-metadata/files'
    store.mkdir(parents=True)
    for index in range(10001):
        original = str(tmp_path / f'old-{index}.png')
        digest = hashlib.sha256(original.encode()).hexdigest()
        (store / f'{digest}.json').write_text(json.dumps({
            'path': original, 'metadata': {'type': 'intermediate_asset'},
        }))
    scratch = tmp_path / 'scratch'
    if directory:
        scratch.mkdir()
        (scratch / 'temporary.txt').write_text('temporary')
        operation = f'import os; os.unlink({str(scratch / "temporary.txt")!r}); os.rmdir({str(scratch)!r})'
    else:
        scratch.write_text('temporary')
        operation = f'import os; os.unlink({str(scratch)!r})'
    assert retained_chat_file_in(scratch, tmp_path) is None
    result = subprocess.run([sys.executable, '-c', _kernel_write_guard_setup_code(tmp_path) + '\n' + operation],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not scratch.exists()


def test_shared_retention_cache_cannot_be_deleted_as_cleanup(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    report = tmp_path / 'report.txt'
    report.write_text('report')
    ChatReferences(workspace).retain(report)
    profile = Path.home() / '.box-agent'
    denied = chat_mutation_error(profile, 'delete', workspace)
    assert denied is not None
    assert 'CHAT_REFERENCE_PROTECTED' in denied.error


def test_shared_cache_rebuilds_missing_workspace_cache_from_session_log(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / 'report.txt'
    path.write_text('report')
    ChatReferences(tmp_path).retain(path)
    metadata_path(path, tmp_path).unlink()
    log = SimpleNamespace(events=[{'type': 'chat/reference', 'data': {'paths': [str(path)]}}])
    ChatReferences(tmp_path, session_log=log)
    assert metadata_path(path, tmp_path).exists()


def test_kernel_metadata_publication_and_temp_cleanup_ignore_publication_history(tmp_path):
    store = tmp_path / '.box-agent/artifact-metadata/files'
    store.mkdir(parents=True)
    for index in range(5000):
        original = str(tmp_path / f'old-{index}.png')
        digest = hashlib.sha256(original.encode()).hexdigest()
        (store / f'{digest}.json').write_text(json.dumps({
            'path': original, 'metadata': {'type': 'intermediate_asset'},
        }))
    module = Path(__file__).resolve().parents[1] / 'box_agent/artifact_metadata.py'
    guard = PythonMutationGuard({**os.environ, 'BOX_AGENT_WORKSPACE_DIR': str(tmp_path)})
    try:
        setup = _kernel_write_guard_setup_code(tmp_path, command_guard=guard.kernel_setup_code())
        program = (setup + f'\nimport runpy, tempfile\n'
                   f"metadata = runpy.run_path({str(module)!r})\n"
                   f"metadata['write_metadata']({str(tmp_path / 'preview.png')!r}, {{'type': 'intermediate_asset'}})\n"
                   "with tempfile.TemporaryDirectory(): pass\n")
        result = subprocess.run([sys.executable, '-c', program], env=os.environ.copy(), cwd=tmp_path,
                                capture_output=True, text=True, timeout=20)
    finally:
        guard.close()
    assert result.returncode == 0, result.stderr
    assert json.loads(metadata_path(tmp_path / 'preview.png', tmp_path).read_text())['metadata']['type'] == 'intermediate_asset'


def test_kernel_producer_does_not_need_write_access_to_shared_profile(tmp_path, monkeypatch):
    first, second, profile = (tmp_path / name for name in ('first', 'second', 'profile'))
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv('BOX_AGENT_HOME', str(profile))
    report = second / 'report.txt'
    report.write_text('report')
    ChatReferences(first).retain(report)
    module = Path(__file__).resolve().parents[1] / 'box_agent/artifact_metadata.py'
    guard = PythonMutationGuard({**os.environ, 'BOX_AGENT_WORKSPACE_DIR': str(second)})
    try:
        monkeypatch.setattr('box_agent.tools.jupyter_tool.tempfile.gettempdir', lambda: str(second / 'tmp'))
        setup = _kernel_write_guard_setup_code(second, command_guard=guard.kernel_setup_code())
        result = subprocess.run([sys.executable, '-c', setup + f'\nimport runpy\n'
                                 f"runpy.run_path({str(module)!r})['write_metadata']({str(report)!r}, {{'type': 'artifact'}})"],
                                env=os.environ.copy(), cwd=second, capture_output=True, text=True, timeout=10)
    finally:
        guard.close()
    assert result.returncode == 0, result.stderr
    assert chat_mutation_error(report, 'delete', second) is not None


def test_released_shared_targets_do_not_accumulate_directory_cleanup_work(tmp_path, monkeypatch):
    for index in range(6):
        report = tmp_path / f'report-{index}.txt'
        report.write_text('report')
        guard = ChatReferences(tmp_path)
        guard.retain(report)
        report.unlink()
        guard.deleted(frozenset({report}))
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr('box_agent.artifact_publication._REFERENCE_SCAN_MAX_ENTRIES', 3)
    assert retained_chat_file_in(scratch, tmp_path) is None


@pytest.mark.parametrize('failure', ['unreadable', 'redirected'])
def test_unavailable_shared_retention_rejects_cross_workspace_deletion(tmp_path, monkeypatch, failure):
    from box_agent.artifact_metadata import retention_directory

    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    report = tmp_path / 'report.txt'
    report.write_text('preserve')
    ChatReferences(first).retain(report)
    directory = retention_directory()
    if failure == 'unreadable':
        marker = next(directory.glob('*/*.json'))
        original_open = Path.open

        def unreadable(path, *args, **kwargs):
            if path == marker:
                raise PermissionError('unreadable shared reference')
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', unreadable)
    else:
        backup = tmp_path / 'redirected-reference-cache'
        directory.rename(backup)
        try:
            directory.symlink_to(backup, target_is_directory=True)
        except OSError:
            pytest.skip('directory symlinks unavailable')
    denied = chat_mutation_error(report, 'delete', second)
    assert denied is not None
    assert 'CHAT_REFERENCE_PROTECTED' in denied.error
    assert report.exists()


def test_producer_workspace_does_not_keep_another_chats_released_retention(tmp_path):
    from box_agent.artifact_metadata import write_metadata

    first, second, producer = (tmp_path / name for name in ('first', 'second', 'producer'))
    for directory in (first, second, producer):
        directory.mkdir()
    report = tmp_path / 'report.txt'
    report.write_text('old report')
    ChatReferences(first).retain(report)
    write_metadata(report, {'type': 'artifact'}, producer)
    report.unlink()
    ChatReferences(second).deleted(frozenset({report}))
    report.write_text('unrelated replacement')
    assert chat_mutation_error(report, 'delete', producer) is None
