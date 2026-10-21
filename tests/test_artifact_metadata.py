import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from box_agent.artifact_publication import metadata_path, read_metadata, write_metadata


def test_new_metadata_is_internal_and_legacy_metadata_remains_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'report.html'
    output.write_text('report')
    legacy = tmp_path / '.report.html.artifact.json'
    legacy.write_text(json.dumps({'type': 'artifact', 'description': 'old', 'retain_for_chat': True}))
    assert read_metadata(legacy)['description'] == 'old'
    write_metadata(output, {'type': 'artifact', 'description': 'new'})
    expected = tmp_path / '.box-agent/artifact-metadata/files' / (hashlib.sha256(str(output.resolve()).encode()).hexdigest() + '.json')
    assert metadata_path(output) == expected
    assert json.loads(expected.read_text()) == {'path': str(output), 'metadata': {'type': 'artifact', 'description': 'new', 'retain_for_chat': True}}
    assert json.loads(legacy.read_text())['description'] == 'old'


def test_parallel_writes_keep_same_named_files_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / '.git').mkdir()
    paths = [tmp_path / str(index) / 'image.png' for index in range(20)]
    for path in paths:
        path.parent.mkdir()
        path.write_bytes(b'image')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda path: write_metadata(path, {'type': 'artifact', 'description': path.parent.name}), paths))
    assert [read_metadata(metadata_path(path))['description'] for path in paths] == [str(index) for index in range(20)]
    assert not list(tmp_path.rglob('.*.artifact.json'))


@pytest.mark.parametrize('redirect', ['.box-agent', '.box-agent/artifact-metadata', '.box-agent/artifact-metadata/files'])
def test_storage_symlink_is_rejected_without_writes_outside_workspace(tmp_path, monkeypatch, redirect):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    redirected = tmp_path / redirect
    redirected.parent.mkdir(parents=True, exist_ok=True)
    redirected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        write_metadata(tmp_path / 'image.png', {'type': 'artifact'})
    assert list(outside.iterdir()) == []


def test_explicit_workspace_stores_external_attachment_and_protects_after_restart(tmp_path):
    from box_agent.tools.chat_references import ChatReferences
    from box_agent.artifact_publication import retained_chat_file_in
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    external = tmp_path / 'external/report.txt'
    external.parent.mkdir()
    external.write_text('attachment')
    guard = ChatReferences(workspace)
    guard.retain(external)
    assert not external.with_name('.report.txt.artifact.json').exists()
    assert list((workspace / '.box-agent/artifact-metadata/files').glob('*.json'))
    assert retained_chat_file_in(external, workspace) == external


@pytest.mark.parametrize('target', ['record', 'store'])
def test_central_cache_cannot_be_deleted_to_bypass_retention(tmp_path, target):
    from box_agent.tools.chat_references import ChatReferences, chat_mutation_error
    from box_agent.artifact_publication import retained_chat_file_in
    attachment = tmp_path / 'report.txt'
    attachment.write_text('report')
    ChatReferences(tmp_path).retain(attachment)
    marker = metadata_path(attachment, tmp_path)
    deletion = marker if target == 'record' else tmp_path / '.box-agent/artifact-metadata'
    assert retained_chat_file_in(deletion, tmp_path) == attachment
    assert chat_mutation_error(deletion, 'delete') is not None


def test_recreated_kernel_guards_central_cache_and_original(tmp_path):
    import subprocess
    import sys
    from box_agent.tools.chat_references import ChatReferences
    from box_agent.tools.jupyter_tool import _kernel_write_guard_setup_code
    path = tmp_path / 'report.txt'
    path.write_text('report')
    ChatReferences(tmp_path).retain(path)
    for target in (path, metadata_path(path, tmp_path)):
        program = _kernel_write_guard_setup_code(tmp_path) + '\nimport os\nos.unlink(' + repr(str(target)) + ')'
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'CHAT_REFERENCE_PROTECTED' in result.stderr
        assert target.exists()


def test_metadata_is_ignored_in_nested_git_workspace_without_gitignore(tmp_path):
    import subprocess
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    workspace = root / 'nested'
    workspace.mkdir()
    output = workspace / 'report.txt'
    output.write_text('report')
    from box_agent.artifact_metadata import write_metadata as write
    exclude = root / '.git/info/exclude'
    exclude.write_text('# preserve user exclusions\ncustom')
    write(output, {'type': 'artifact'}, workspace)
    assert exclude.read_text().startswith('# preserve user exclusions\ncustom\n')
    assert not (root / '.gitignore').exists()
    result = subprocess.run(['git', '-C', str(root), 'status', '--short', '--untracked-files=all'], capture_output=True, text=True, check=True)
    assert '.box-agent' not in result.stdout
    assert 'nested/report.txt' in result.stdout


@pytest.mark.asyncio
async def test_bash_propagates_own_workspace_without_global_environment_change(tmp_path):
    import os
    import json
    import sys
    import shlex
    from box_agent.tools.bash_tool import BashTool
    original = os.environ.get('BOX_AGENT_WORKSPACE_DIR')
    tool = BashTool(workspace_dir=str(tmp_path), runtime_env={'BOX_AGENT_WORKSPACE_DIR': '/unrelated'})
    result = await tool.execute(command=shlex.join([sys.executable, '-c', 'import os; print(os.environ["BOX_AGENT_WORKSPACE_DIR"])']))
    assert result.success, result.error
    assert str(tmp_path.resolve()) in result.content
    assert os.environ.get('BOX_AGENT_WORKSPACE_DIR') == original


def test_nested_standalone_intermediate_is_hidden_after_copy_and_restart(tmp_path, monkeypatch):
    import shutil
    from box_agent.tools.engine.artifact_results import _detect_regex_artifacts
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'renderer/private.png'
    source.parent.mkdir()
    source.write_bytes(b'private')
    write_metadata(source, {'type': 'intermediate_asset'})
    assert _detect_regex_artifacts('first', 'bash', '[renderer/private.png]', None, str(tmp_path))[0] == []
    copied = tmp_path / 'public.png'
    shutil.copyfile(source, copied)
    source.unlink()
    assert _detect_regex_artifacts('restart', 'bash', '[public.png]', None, str(tmp_path))[0] == []
    assert not list(tmp_path.rglob('.*.artifact.json'))


def test_legacy_scope_and_intermediate_fingerprint_are_read_without_rewriting(tmp_path):
    from box_agent.artifact_publication import delivery_scope, intermediate_fingerprints
    source = tmp_path / 'working.png'
    source.write_bytes(b'private')
    marker = tmp_path / '.working.png.artifact.json'
    marker.write_text('{"type":"intermediate_asset"}')
    scope = tmp_path / '.artifact-delivery.json'
    scope.write_text('{"schema_version":1,"default":"intermediate"}')
    assert delivery_scope(source, tmp_path) == tmp_path
    fingerprints = intermediate_fingerprints([marker], tmp_path)
    assert hashlib.sha256(b'private').hexdigest() in fingerprints[7]
    assert marker.read_text() == '{"type":"intermediate_asset"}'
    assert scope.read_text() == '{"schema_version":1,"default":"intermediate"}'


def test_scope_writes_are_central_and_preserve_explicit_placement_after_restart(tmp_path):
    from box_agent.artifact_metadata import write_scope
    from box_agent.artifact_publication import delivery_scope
    from box_agent.artifacts import make_artifact
    scope = tmp_path / 'delivery'
    scope.mkdir()
    output = scope / 'report.pdf'
    output.write_bytes(b'pdf')
    write_scope(scope, {'schema_version': 1, 'default': 'intermediate'}, tmp_path)
    write_metadata(output, {'type': 'artifact', 'description': 'published'}, tmp_path)
    assert delivery_scope(output, tmp_path) == scope
    artifact = make_artifact('restart', output, tmp_path)
    assert (artifact.placement, artifact.description) == ('primary', 'published')
    assert list(scope.iterdir()) == [output]


def test_git_worktree_records_use_local_shared_exclusion(tmp_path):
    import subprocess
    from box_agent.artifact_metadata import write_metadata as write
    root, worktree = tmp_path / 'repo', tmp_path / 'worktree'
    root.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True)
    git('init', '-q')
    git('-c', 'user.name=Metadata Test', '-c', 'user.email=test@example.com', 'commit', '--allow-empty', '-qm', 'initial')
    git('worktree', 'add', '--detach', '-q', str(worktree))
    output = worktree / 'report.txt'
    output.write_text('report')
    write(output, {'type': 'artifact'}, worktree)
    result = subprocess.run(['git', '-C', str(worktree), 'status', '--short', '--untracked-files=all'], check=True, capture_output=True, text=True)
    assert '.box-agent' not in result.stdout
    assert 'report.txt' in result.stdout
    assert not (worktree / '.gitignore').exists()


def test_python_child_checks_external_cache_with_explicit_workspace(tmp_path):
    import os
    import subprocess
    import sys
    from box_agent.tools.chat_references import ChatReferences
    from box_agent.tools.python_mutation_guard import PythonMutationGuard
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    external = tmp_path / 'external.txt'
    external.write_text('attachment')
    ChatReferences(workspace).retain(external)
    guard = PythonMutationGuard({**os.environ, 'BOX_AGENT_WORKSPACE_DIR': str(workspace)})
    try:
        result = subprocess.run([sys.executable, '-c', 'import os; os.unlink(' + repr(str(external)) + ')'], env=guard.environment, capture_output=True, text=True)
    finally:
        guard.close()
    assert result.returncode != 0
    assert 'CHAT_REFERENCE_PROTECTED' in result.stderr
    assert external.read_text() == 'attachment'


def test_kernel_propagates_workspace_to_producer_and_restores_environment(tmp_path):
    import os
    import subprocess
    import sys
    from box_agent.tools.jupyter_tool import _kernel_write_guard_setup_code, _KERNEL_WRITE_GUARD_DISABLE_CODE
    from box_agent.tools.python_mutation_guard import PythonMutationGuard
    guard = PythonMutationGuard({**os.environ, 'BOX_AGENT_WORKSPACE_DIR': str(tmp_path)})
    try:
        setup = _kernel_write_guard_setup_code(tmp_path, command_guard=guard.kernel_setup_code())
        program = ('import os\nos.environ["BOX_AGENT_WORKSPACE_DIR"] = "old"\n' + setup
                   + '\nassert os.environ["BOX_AGENT_WORKSPACE_DIR"] == ' + repr(str(tmp_path))
                   + '\n' + _KERNEL_WRITE_GUARD_DISABLE_CODE
                   + '\nassert os.environ["BOX_AGENT_WORKSPACE_DIR"] == "old"')
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
    finally:
        guard.close()
    assert result.returncode == 0, result.stderr


def test_explicit_workspace_does_not_fall_back_to_another_workspace_environment(tmp_path, monkeypatch):
    from box_agent.artifact_metadata import read_metadata as read, write_metadata as write
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    output = second / 'report.txt'
    output.write_text('report')
    write(output, {'type': 'intermediate_asset'}, first)
    monkeypatch.setenv('BOX_AGENT_WORKSPACE_DIR', str(first))
    assert read(output, second) == {}


def test_unreadable_central_retention_cache_rejects_deletion(tmp_path, monkeypatch):
    from box_agent.tools.chat_references import ChatReferences, chat_mutation_error
    path = tmp_path / 'report.txt'
    path.write_text('report')
    ChatReferences(tmp_path).retain(path)
    marker = metadata_path(path, tmp_path)
    original = Path.open
    def unreadable(self, *args, **kwargs):
        if self == marker:
            raise PermissionError('unreadable central cache')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', unreadable)
    denied = chat_mutation_error(path, 'delete', tmp_path)
    assert denied is not None
    assert 'CHAT_REFERENCE_PROTECTED' in denied.error
    assert path.exists()


def test_bash_reference_check_uses_workspace_for_external_attachment(tmp_path):
    from box_agent.tools.chat_references import ChatReferences, bash_chat_mutation_error
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    external = tmp_path / 'external.txt'
    external.write_text('attachment')
    ChatReferences(workspace).retain(external)
    denied = bash_chat_mutation_error('rm ' + str(external), workspace)
    assert denied is not None
    assert 'CHAT_REFERENCE_PROTECTED' in denied.error


def test_parent_publication_preserves_nested_standalone_retention_with_environment(tmp_path, monkeypatch):
    from box_agent.artifact_metadata import read_metadata as read, write_metadata as write
    from box_agent.artifact_publication import retained_chat_file_in
    monkeypatch.chdir(tmp_path)
    target = tmp_path / 'nested/report.txt'
    target.parent.mkdir()
    target.write_text('report')
    write(target, {'retain_for_chat': True})
    monkeypatch.setenv('BOX_AGENT_WORKSPACE_DIR', str(tmp_path))
    assert retained_chat_file_in(target, tmp_path) == target
    write(target, {'type': 'artifact'})
    assert read(target, tmp_path) == {'type': 'artifact', 'retain_for_chat': True}
    monkeypatch.delenv('BOX_AGENT_WORKSPACE_DIR')
    assert retained_chat_file_in(target, tmp_path) == target


def test_confirmed_deletion_releases_nested_standalone_cache_before_path_reuse(tmp_path, monkeypatch):
    from box_agent.artifact_publication import retain_chat_file
    from box_agent.tools.chat_references import ChatReferences, chat_mutation_error
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'nested/report.txt'
    path.parent.mkdir()
    path.write_text('old report')
    retain_chat_file(path, path.parent)
    guard = ChatReferences(tmp_path)
    guard.retain(path)
    path.unlink()
    guard.deleted(frozenset({path}))
    path.write_text('unrelated new file')
    assert chat_mutation_error(path, 'delete', tmp_path) is None


@pytest.mark.parametrize('environment_root', [False, True])
def test_external_standalone_dependencies_are_retained_from_runtime_workspace(tmp_path, monkeypatch, environment_root):
    from box_agent.artifact_metadata import read_metadata as read, write_metadata as write
    from box_agent.tools.chat_references import ChatReferences, chat_mutation_error
    workspace, external = tmp_path / 'workspace', tmp_path / 'external'
    workspace.mkdir()
    external.mkdir()
    book, source = external / 'book.bin', external / 'source.dat'
    book.write_bytes(b'book')
    source.write_bytes(b'source')
    write(book, {'type': 'artifact', 'dependencies': [str(source)], 'retain_for_chat': True}, external)
    old_record = metadata_path(book, external).read_bytes()
    if environment_root:
        monkeypatch.setenv('BOX_AGENT_WORKSPACE_DIR', str(workspace))
    else:
        monkeypatch.delenv('BOX_AGENT_WORKSPACE_DIR', raising=False)
    assert read(book, workspace)['dependencies'] == [str(source)]
    guard = ChatReferences(workspace)
    guard.retain(book)
    assert {book, source}.issubset(guard.paths)
    assert chat_mutation_error(source, 'delete', workspace) is not None
    assert metadata_path(book, external).read_bytes() == old_record
    assert not list(external.glob('.*.artifact.json'))


def test_explicit_metadata_precedes_external_fallback_with_unrelated_environment(tmp_path, monkeypatch):
    from box_agent.artifact_metadata import read_metadata as read, write_metadata as write
    workspace, external, unrelated = (tmp_path / name for name in ('workspace', 'external', 'unrelated'))
    for directory in (workspace, external, unrelated):
        directory.mkdir()
    book = external / 'book.bin'
    book.write_bytes(b'book')
    write(book, {'type': 'intermediate_asset', 'description': 'old', 'retain_for_chat': True}, external)
    monkeypatch.setenv('BOX_AGENT_WORKSPACE_DIR', str(unrelated))
    write(book, {'type': 'artifact', 'description': 'current'}, workspace)
    assert read(book, workspace) == {'type': 'artifact', 'description': 'current', 'retain_for_chat': True}
    assert not (unrelated / '.box-agent').exists()


def test_confirmed_external_deletion_releases_target_local_cache_only(tmp_path):
    from box_agent.artifact_publication import retain_chat_file
    from box_agent.tools.chat_references import ChatReferences, chat_mutation_error
    workspace, external = tmp_path / 'workspace', tmp_path / 'external'
    workspace.mkdir()
    external.mkdir()
    book, other = external / 'book.bin', external / 'other.bin'
    for path in (book, other):
        path.write_bytes(b'old')
        retain_chat_file(path, external)
    other_record = metadata_path(other, external).read_bytes()
    guard = ChatReferences(workspace)
    guard.retain(book)
    book.unlink()
    guard.deleted(frozenset({book}))
    assert not metadata_path(book, external).exists()
    book.write_bytes(b'new unrelated file')
    assert chat_mutation_error(book, 'delete', workspace) is None
    assert metadata_path(other, external).read_bytes() == other_record
