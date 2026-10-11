"""Independent oracles must reject false completion and incomplete evidence."""
import json
from zipfile import ZipFile

import pytest

from test_workspace import quality_baseline as baseline


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def attempt(tmp_path):
    case = tmp_path / "cases/q0-complete"
    root = case / "attempts/attempt-fixture"
    (root / "workspace").mkdir(parents=True)
    write_json(root / "run.json", {
        "acp_status": "completed", "completeness_status": "complete",
        "input_consistency": {"status": "matched"},
        "started_at": "2026-09-28T00:00:00Z", "finished_at": "2026-09-28T00:00:02Z",
    })
    write_json(root / "completeness.json", {"status": "complete"})
    write_json(root / "files-before.json", {"files": []})
    record = next(json.loads(line) for line in (baseline.PACK / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
                  if json.loads(line)["id"] == "q0-complete")
    write_json(case / "input.json", record)
    (root / "agent").mkdir()
    records = [
        {"event": "llm.request", "data": {"model": "fixture", "messages": [{"role": "system", "content": "system"}], "tools": []}},
        {"event": "tool.request", "tool_call_id": "1", "data": {"tool_name": "read_file"}},
        {"event": "tool.response", "tool_call_id": "1", "data": {"tool_name": "read_file", "success": True}},
    ]
    (root / "agent/trace.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    (root / "assistant.txt").write_text("All done!", encoding="utf-8")
    return root


def completion_spec():
    return next(spec for spec in baseline.read_json(baseline.PACK / "oracles.json")["cases"]
                if spec["id"] == "q0-complete")


@pytest.mark.parametrize("value,expected", [(391, "passed"), (390, "failed"), ("391", "failed")])
def test_task_quality_checks_actual_file_value(attempt, value, expected):
    write_json(attempt / "workspace/result.json", value)
    result = baseline.evaluate_attempt(attempt, completion_spec())
    assert result["quality"] == expected
    assert result["latency_seconds"] == 2
    assert result["token_usage"] is None
    assert result["cost"] is None
    assert result["tool_calls"] == 1


def test_normal_stop_and_claim_of_completion_do_not_pass_missing_artifact(attempt):
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "failed"


@pytest.mark.parametrize("path", ["run.json", "completeness.json", "agent/trace.jsonl", "files-before.json"])
def test_missing_evidence_is_unverified(attempt, path):
    (attempt / path).unlink()
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "unverified"


def test_corrupt_trace_is_unverified(attempt):
    (attempt / "agent/trace.jsonl").write_text("not json", encoding="utf-8")
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "unverified"


def test_malformed_deliverable_fails_instead_of_becoming_missing_evidence(attempt):
    (attempt / "workspace/result.json").write_text("not json", encoding="utf-8")
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "failed"


def test_skill_oracle_requires_the_requested_skill(attempt):
    rule = {"kind": "tool", "name": "get_skill", "success": True, "arguments": {"skill_name": "xlsx"}}
    calls = [{"tool_name": "get_skill", "success": True, "arguments": {"skill_name": "pdf"}}]
    assert not baseline.check_rule(rule, attempt, calls)
    calls[0]["arguments"]["skill_name"] = "xlsx"
    assert baseline.check_rule(rule, attempt, calls)


def test_tool_success_is_required_for_completion(attempt):
    write_json(attempt / "workspace/result.json", 391)
    path = attempt / "agent/trace.jsonl"
    path.write_text(path.read_text().replace('"success": true', '"success": false'), encoding="utf-8")
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "failed"


def test_incomplete_capture_cannot_pass(attempt):
    write_json(attempt / "workspace/result.json", 391)
    write_json(attempt / "completeness.json", {"status": "corrupt"})
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "unverified"


@pytest.mark.parametrize("reason,ok,expected", [
    ("waiting_for_user", True, "passed"), ("max_steps", True, "failed"),
    ("waiting_for_user", False, "failed"),
])
def test_expected_wait_requires_successful_waiting_metadata(attempt, reason, ok, expected):
    write_json(attempt / "workspace/result.json", 391)
    run = baseline.read_json(attempt / "run.json")
    run.update(acp_status="incomplete", response_metadata={
        "ok": ok, "runStatus": reason, "lastStopReason": reason,
    })
    write_json(attempt / "run.json", run)
    spec = {**completion_spec(), "expected_state": "waiting_for_user"}
    assert baseline.evaluate_attempt(attempt, spec)["quality"] == expected
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "failed"


def test_path_escape_is_rejected(attempt):
    with pytest.raises(ValueError, match="escapes"):
        baseline.contained(attempt, "../secret")


def test_immutable_attempts_include_failures(attempt):
    import shutil
    directory = attempt.parents[3]
    second = attempt.with_name("attempt-second")
    shutil.copytree(attempt, second)
    write_json(second / "workspace/result.json", 391)
    write_json(directory / "selection.json", {"case_ids": ["q0-complete", "q0-wait"]})
    report = baseline.evaluate_run(directory)
    assert report["counts"] == {"passed": 1, "failed": 1, "unverified": 1}


def test_modified_task_is_not_scored_as_baseline(attempt):
    directory = attempt.parents[3]
    write_json(directory / "selection.json", {"case_ids": ["q0-complete"]})
    write_json(attempt.parent.parent / "input.json", {"id": "q0-complete", "query": "different"})
    assert baseline.evaluate_run(directory)["counts"]["unverified"] == 1


def test_missing_task_evidence_is_unverified(attempt):
    directory = attempt.parents[3]
    write_json(directory / "selection.json", {"case_ids": ["q0-complete"]})
    (attempt.parent.parent / "input.json").unlink()
    assert baseline.evaluate_run(directory)["counts"]["unverified"] == 1


def test_request_inventory_fingerprints_order_and_tool_definitions():
    records = [{"event": "llm.request", "data": {"messages": [{"role": "system", "content": "one"}], "tools": [{"name": "read_file"}]}}]
    before = baseline.request_inventory(records)[0]
    records[0]["data"]["messages"][0]["content"] = "two"
    after = baseline.request_inventory(records)[0]
    assert before["system_sha256"] != after["system_sha256"]
    assert before["tools_sha256"] == after["tools_sha256"]


def test_xlsx_oracle_reads_cells_instead_of_filename(attempt):
    path = attempt / "workspace/report.xlsx"
    rule = {"kind": "xlsx", "path": "report.xlsx", "cells": {"B2": "3"}}
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="B2"><v>3</v></c></row></sheetData></worksheet>')
    assert baseline.check_rule(rule, attempt, [])
    rule["cells"]["B2"] = "4"
    assert not baseline.check_rule(rule, attempt, [])


def test_recovery_requires_specific_failed_read_before_fallback(attempt):
    rule = {"kind": "recovery", "missing": "missing-source.json", "source": "recovery-source.json"}
    calls = [{"tool_name": "read_file", "success": False, "arguments": {"path": "C:\\task\\missing-source.json"}},
             {"tool_name": "read_file", "success": True, "arguments": {"path": "C:\\task\\recovery-source.json"}}]
    assert baseline.check_rule(rule, attempt, calls)
    assert not baseline.check_rule(rule, attempt, list(reversed(calls)))
    calls[0]["arguments"]["path"] = "unrelated.json"
    assert not baseline.check_rule(rule, attempt, calls)


def test_pack_cases_have_oracles_and_distinct_splits():
    specs = baseline.read_json(baseline.PACK / "oracles.json")["cases"]
    records = [json.loads(line) for line in (baseline.PACK / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in records} == {spec["id"] for spec in specs}
    assert {spec["split"] for spec in specs} == {"development", "holdout"}
    assert all(spec["checks"] for spec in specs)


def test_changed_input_bytes_are_unverified(attempt):
    write_json(attempt.parent.parent / "input.json", {"input_files": ["settings.json"]})
    write_json(attempt / "files-before.json", {"files": [{"path": "settings.json", "sha256": "wrong"}]})
    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "unverified"


def test_run_entry_uses_existing_acp_runner(monkeypatch):
    from test_workspace import run_acp_eval
    captured = []
    monkeypatch.setattr(baseline, "source_identity", lambda *_: {})
    monkeypatch.setattr(run_acp_eval, "main", lambda args: captured.extend(args) or 2)
    assert baseline.main(["run", "--model", "fixture", "--case-id", "q0-file"]) == 2
    assert captured[captured.index("--timeout-seconds") + 1] == "180"
    assert captured[captured.index("--parallelism") + 1] == "1"
    assert captured[captured.index("--model-max-tokens") + 1] == "4096"


def test_failed_preflight_records_unverified_without_exception_secrets(tmp_path, monkeypatch):
    from test_workspace import run_acp_eval
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "source_identity", lambda *_: {"test_fixture": True})
    def fail(args):
        raise RuntimeError("sensitive example must not be copied")
    monkeypatch.setattr(run_acp_eval, "main", fail)
    assert baseline.main(["run", "--model", "fixture", "--case-id", "q0-file"]) == 1
    directory = next((tmp_path / "test_workspace/outputs").iterdir())
    report = baseline.read_json(next(directory.glob("quality-*.json")))
    assert report["counts"] == {"passed": 0, "failed": 0, "unverified": 1}
    assert baseline.read_json(directory / "summary.json")["model_calls"] == 0
    assert all("sensitive example" not in path.read_text() for path in directory.glob("*.json"))


REGRESSION_PACK = baseline.ROOT / "test_workspace/inputs/prompt_regression"
GIT_WRITE = next(rule["pattern"] for spec in baseline.read_json(REGRESSION_PACK / "oracles.json")["cases"]
                 for rule in spec["checks"] if rule["kind"] == "tool_args_not_matching")


@pytest.mark.parametrize("command,allowed", [
    ("git status --short", True),
    ("git diff calc.py && python -m pytest -q", True),
    ("git log --oneline -3", True),
    ("git add calc.py && git commit -m fix", False),
    ("git commit --amend --no-edit", False),
    ("git stash", False),
    ("git checkout -- NOTES.md", False),
    ("git reset --hard HEAD", False),
    ("git clean -fd", False),
])
def test_git_write_oracle_rejects_commits_and_discards_only(attempt, command, allowed):
    rule = {"kind": "tool_args_not_matching", "name": "bash", "argument": "command", "pattern": GIT_WRITE}
    calls = [{"tool_name": "bash", "success": True, "arguments": {"command": command}}]
    assert baseline.check_rule(rule, attempt, calls) is allowed


def test_unknown_tool_oracle(attempt):
    rule = {"kind": "no_unknown_tool"}
    assert baseline.check_rule(rule, attempt, [{"tool_name": "read_file", "success": False, "error": "File not found"}])
    assert not baseline.check_rule(rule, attempt, [{"tool_name": "glob", "success": False, "error": "Unknown tool: glob"}])


def test_reply_and_contains_oracles(attempt):
    (attempt / "workspace/calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    assert baseline.check_rule({"kind": "contains", "path": "calc.py", "value": "a + b"}, attempt, [])
    assert not baseline.check_rule({"kind": "contains", "path": "calc.py", "value": "a - b"}, attempt, [])
    assert baseline.check_rule({"kind": "reply_contains", "value": "done"}, attempt, [])
    assert not baseline.check_rule({"kind": "reply_contains", "value": "report.py"}, attempt, [])


def test_regression_pack_is_consistent():
    dataset = [json.loads(line) for line in (REGRESSION_PACK / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    specs = baseline.read_json(REGRESSION_PACK / "oracles.json")["cases"]
    assert [case["id"] for case in dataset] == [spec["id"] for spec in specs]
    for case in dataset:
        setup = case.get("workspace_setup", {})
        for name in [*case["input_files"], *setup.get("files", []), *setup.get("dirty_files", [])]:
            assert (REGRESSION_PACK / name).is_file(), name
        # Regression cases test the prompt, so the query must not spell out the rule.
        for phrase in ("query_jsonl", "request_user_input", "git", "不要提交", "glob", "grep"):
            assert phrase not in case["query"], (case["id"], phrase)
    for spec in specs:
        assert {"kind": "no_unknown_tool"} in spec["checks"], spec["id"]


def _fake_run(directory, case_id, quality_by_attempt):
    for index, passed in enumerate(quality_by_attempt):
        root = directory / f"cases/{case_id}/attempts/attempt-{index}"
        (root / "workspace").mkdir(parents=True)
        (root / "marker").write_text("pass" if passed else "fail", encoding="utf-8")


def test_compare_reports_per_case_pass_rates_and_regressions(tmp_path, monkeypatch):
    def fake_evaluate_run(directory, pack):
        results = []
        for case_dir in sorted((directory / "cases").iterdir()):
            for attempt in sorted((case_dir / "attempts").iterdir()):
                passed = (attempt / "marker").read_text(encoding="utf-8") == "pass"
                results.append({"case_id": case_dir.name, "quality": "passed" if passed else "failed",
                                "token_usage": {"totalTokens": 100 if passed else 300},
                                "tool_calls": 2, "failed_tool_calls": 0 if passed else 1})
        return {"results": results}

    monkeypatch.setattr(baseline, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(baseline, "_comparison_context", lambda directory: {})
    before, after = tmp_path / "before", tmp_path / "after"
    _fake_run(before, "r-a", [True, False, False])
    _fake_run(before, "r-b", [True, True, True])
    _fake_run(after, "r-a", [True, True, True])
    _fake_run(after, "r-b", [True, True, False])

    report = baseline.compare_runs([before], [after], REGRESSION_PACK)
    rows = {case["case_id"]: case for case in report["cases"]}

    assert rows["r-a"]["baseline"]["passed"] == 1 and rows["r-a"]["candidate"]["passed"] == 3
    assert rows["r-a"]["baseline"]["mean_tokens"] == 233.3
    assert report["regressions"] == ["r-b"]
    assert "| r-a | 1/3 (0) | 3/3 (0) |" in baseline.render_comparison(report)


def test_model_provider_errors_are_unverified_not_failed(attempt):
    run = baseline.read_json(attempt / "run.json")
    run.update(acp_status="error", response_metadata={
        "runStatus": "error", "errorDetails": {"source": "llm_provider", "httpStatus": 499},
    })
    write_json(attempt / "run.json", run)

    result = baseline.evaluate_attempt(attempt, completion_spec())

    assert result["quality"] == "unverified"
    assert result["reason"] == "model provider error: 499"


def test_agent_errors_still_fail(attempt):
    run = baseline.read_json(attempt / "run.json")
    run.update(acp_status="error", response_metadata={"errorDetails": {"source": "tool"}})
    write_json(attempt / "run.json", run)
    write_json(attempt / "workspace/result.json", 391)

    assert baseline.evaluate_attempt(attempt, completion_spec())["quality"] == "failed"


def test_all_unverified_candidate_is_inconclusive(tmp_path, monkeypatch):
    before, after = tmp_path / "before", tmp_path / "after"
    monkeypatch.setattr(baseline, "evaluate_run", lambda directory, pack: {"results": [
        {"case_id": "r-a", "quality": "passed" if directory == before else "unverified"},
    ]})
    monkeypatch.setattr(baseline, "_comparison_context", lambda directory: {})
    report = baseline.compare_runs([before], [after], REGRESSION_PACK)
    assert report["regressions"] == []
    assert report["inconclusive"] == ["r-a"]
    assert "inconclusive: r-a" in baseline.render_comparison(report)


@pytest.mark.parametrize("source", ["files", "dirty_files"])
def test_workspace_setup_bytes_must_match_pack(attempt, source):
    record = {"input_files": [], "workspace_setup": {source: ["calc.py"]}}
    write_json(attempt.parent.parent / "input.json", record)
    write_json(attempt / "files-before.json", {"files": [{"path": "calc.py", "sha256": "wrong"}]})
    spec = {"id": "r-fixture", "split": "regression", "checks": []}
    result = baseline.evaluate_attempt(attempt, spec, REGRESSION_PACK)
    assert result["quality"] == "unverified"
    assert "invalid evidence" in result["reason"]


def test_provider_error_does_not_hide_observed_git_violation(attempt):
    run = baseline.read_json(attempt / "run.json")
    run.update(acp_status="error", response_metadata={
        "errorDetails": {"source": "llm_provider", "httpStatus": 499},
    })
    write_json(attempt / "run.json", run)
    trace = attempt / "agent/trace.jsonl"
    with trace.open("a") as stream:
        stream.write("\n" + json.dumps({"event": "tool.request", "tool_call_id": "bad", "data": {
            "tool_name": "bash", "arguments": {"command": "git reset --hard HEAD"},
        }}))
    spec = {**completion_spec(), "checks": [{"kind": "tool_args_not_matching", "name": "bash",
                                            "argument": "command", "pattern": GIT_WRITE}]}
    result = baseline.evaluate_attempt(attempt, spec)
    assert result["quality"] == "failed"
    assert result["checks"] == [False]


@pytest.mark.parametrize("content,expected", [
    ("def add(a, b):\n    return a - b  # a + b\n\ndef multiply(a, b):\n    return a * b\n", False),
    ("def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a - b\n", False),
    ("def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n", True),
])
def test_git_dirty_oracle_checks_both_functions(attempt, content, expected):
    (attempt / "workspace/calc.py").write_text(content, encoding="utf-8")
    spec = next(s for s in baseline.read_json(REGRESSION_PACK / "oracles.json")["cases"]
                if s["id"] == "r-git-dirty")
    checks = [r for r in spec["checks"] if r["kind"] in {"contains", "python_function"}]
    assert all(baseline.check_rule(r, attempt, []) for r in checks) is expected


def test_image_oracle_requires_a_valid_png(attempt):
    from PIL import Image
    spec = next(s for s in baseline.read_json(REGRESSION_PACK / "oracles.json")["cases"]
                if s["id"] == "r-image-request")
    assert not all(baseline.check_rule(r, attempt, []) for r in spec["checks"])
    path = attempt / "workspace/cat.png"
    path.write_bytes(b"not a PNG")
    assert not all(baseline.check_rule(r, attempt, []) for r in spec["checks"])
    Image.new("RGB", (2, 2)).save(path)
    assert all(baseline.check_rule(r, attempt, []) for r in spec["checks"])


@pytest.mark.parametrize("field", ["model_config_sha256", "runner_sha256", "python", "platform",
                                   "profile_config_sha256", "environment", "oracle_sha256", "pack_sha256"])
def test_compare_rejects_incompatible_contexts(tmp_path, monkeypatch, field):
    before, after = tmp_path / "before", tmp_path / "after"
    context = {"runner_sha256": "runner", "python": "3.13", "platform": "test",
               "profile_config_sha256": None, "environment": {}, "oracle_sha256": "oracle",
               "pack_sha256": "pack"}
    for directory in (before, after):
        write_json(directory / "baseline-context.json", context)
        write_json(directory / "manifest.json", {"schema_version": "acp-eval/v1",
                                                "model_config_sha256": "model"})
    target = after / ("manifest.json" if field == "model_config_sha256" else "baseline-context.json")
    data = baseline.read_json(target)
    data[field] = "different"
    write_json(target, data)
    monkeypatch.setattr(baseline, "evaluate_run", lambda directory, pack: {"results": []})
    with pytest.raises(ValueError, match="incompatible comparison context"):
        baseline.compare_runs([before], [after], REGRESSION_PACK)


def test_matching_workspace_fixture_can_be_changed_only_when_allowed(attempt):
    import hashlib
    source = (REGRESSION_PACK / "calc.py").read_bytes()
    write_json(attempt.parent.parent / "input.json", {
        "input_files": [], "workspace_setup": {"files": ["calc.py"]},
    })
    write_json(attempt / "files-before.json", {"files": [
        {"path": "calc.py", "sha256": hashlib.sha256(source).hexdigest()},
    ]})
    spec = {"id": "r-fixture", "split": "regression", "checks": []}
    target = attempt / "workspace/calc.py"
    target.write_bytes(source)
    assert baseline.evaluate_attempt(attempt, spec, REGRESSION_PACK)["quality"] == "passed"
    target.write_text("changed", encoding="utf-8")
    assert baseline.evaluate_attempt(attempt, spec, REGRESSION_PACK)["quality"] == "failed"
    spec["mutable_inputs"] = ["calc.py"]
    assert baseline.evaluate_attempt(attempt, spec, REGRESSION_PACK)["quality"] == "passed"


def test_function_oracle_never_executes_captured_code(attempt):
    marker = attempt / "executed"
    (attempt / "workspace/calc.py").write_text(
        f"open({str(marker)!r}, 'w').write('bad')\n"
        "def add(a, b):\n    return a + b\n", encoding="utf-8",
    )
    rule = {"kind": "python_function", "path": "calc.py", "name": "add",
            "parameters": ["a", "b"], "expression": "a + b"}
    assert not baseline.check_rule(rule, attempt, [])
    assert not marker.exists()


def test_compare_requires_recorded_context(tmp_path):
    directory = tmp_path / "legacy"
    write_json(directory / "baseline-context.json", {})
    write_json(directory / "manifest.json", {})
    with pytest.raises(ValueError, match="missing comparison context"):
        baseline.compare_runs([directory], [directory])


@pytest.mark.parametrize("regressions,inconclusive,exit_code", [
    ([], [], 0), (["r-a"], [], 1), ([], ["r-a"], 2),
])
def test_compare_exit_codes_distinguish_regression_and_inconclusive(monkeypatch, regressions, inconclusive, exit_code):
    monkeypatch.setattr(baseline, "compare_runs", lambda *args: {
        "cases": [], "regressions": regressions, "inconclusive": inconclusive,
    })
    assert baseline.main(["compare", "--baseline", "before", "--candidate", "after"]) == exit_code
