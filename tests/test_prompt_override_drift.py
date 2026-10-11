"""Outdated prompt overrides are reported, never silently used or replaced."""

import pytest

import box_agent.config as config_module
from box_agent.config import Config, PROMPT_TEMPLATE_FILES, prompt_override_drift


@pytest.fixture
def override_home(tmp_path, monkeypatch):
    home = tmp_path / "box-home"
    (home / "config").mkdir(parents=True)
    monkeypatch.setenv("BOX_AGENT_HOME", str(home))
    monkeypatch.setattr(config_module, "_warned_prompt_overrides", set())
    return home / "config"


def _packaged(filename):
    return (Config.get_package_dir() / "config" / filename).read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", PROMPT_TEMPLATE_FILES)
def test_packaged_templates_never_report_drift(filename):
    assert prompt_override_drift(Config.get_package_dir() / "config" / filename, filename) is None


def test_up_to_date_override_is_silent(override_home, capsys):
    (override_home / "system_prompt.md").write_text(_packaged("system_prompt.md") + "\n额外的团队规则\n", encoding="utf-8")

    path = Config.find_config_file("system_prompt.md")

    assert path == override_home / "system_prompt.md"
    assert capsys.readouterr().err == ""


def test_outdated_override_is_used_but_reported_once(override_home, capsys):
    stale = _packaged("system_prompt.md").replace("{PLANNING_TOOLS}", "≥3 步用 `todo_write`。") + "\n{LEGACY_SLOT}\n"
    (override_home / "system_prompt.md").write_text(stale, encoding="utf-8")

    first = Config.find_config_file("system_prompt.md")
    second = Config.find_config_file("system_prompt.md")
    err = capsys.readouterr().err

    assert first == second == override_home / "system_prompt.md"  # resolution unchanged
    assert err.count("prompt override") == 1
    assert "missing {PLANNING_TOOLS}" in err and "unknown {LEGACY_SLOT}" in err
    drift = prompt_override_drift(first, "system_prompt.md")
    assert drift["missing"] == ["{PLANNING_TOOLS}"] and drift["unknown"] == ["{LEGACY_SLOT}"]


def test_editing_the_override_reports_again(override_home, capsys):
    target = override_home / "code_prompt.md"
    target.write_text("no placeholders\n", encoding="utf-8")
    Config.find_config_file("code_prompt.md")
    target.write_text("still none, edited\n", encoding="utf-8")
    import os
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000_000))
    Config.find_config_file("code_prompt.md")

    assert capsys.readouterr().err.count("prompt override") == 2


def test_non_prompt_files_are_not_checked(override_home, capsys):
    (override_home / "config.yaml").write_text("{UNKNOWN_SLOT}\n", encoding="utf-8")

    assert Config.find_config_file("config.yaml") == override_home / "config.yaml"
    assert capsys.readouterr().err == ""


def test_doctor_reports_outdated_override(override_home):
    from box_agent.cli import _doctor_prompt_status

    assert _doctor_prompt_status()["status"] == "ok"
    (override_home / "analysis_prompt.md").write_text("old analysis prompt\n", encoding="utf-8")

    status = _doctor_prompt_status()

    assert status["status"] == "warning"
    assert "analysis_prompt.md" in status["message"]
    assert status["overrides"][0]["missing"]
