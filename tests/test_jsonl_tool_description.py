"""Model-facing query examples must be valid calls to the real tool."""
import json

import pytest

from box_agent.tools.file.jsonl_tool import JsonlQueryTool


async def test_query_jsonl_documented_call_filters_and_projects_nested_fields(tmp_path):
    tool = JsonlQueryTool(workspace_dir=str(tmp_path))
    example_text = tool.description.split("Example: ", 1)[1]
    arguments, _ = json.JSONDecoder().raw_decode(example_text)
    (tmp_path / arguments["path"]).write_text(
        json.dumps({"id": 7, "data": {"error": "other"}}) + "\n"
        + json.dumps({"id": 42, "data": {"error": "target"}}) + "\n",
        encoding="utf-8",
    )

    result = await tool.execute(**arguments)

    assert result.success is True
    records = json.loads(result.content)["records"]
    assert len(records) == 1
    assert records[0]["data"] == {"/id": 42, "/data/error": "target"}


@pytest.mark.parametrize("invalid", ["id", "#id", " /id", "data.error"])
@pytest.mark.parametrize("parameter", ["fields", "where"])
async def test_query_jsonl_keeps_rejecting_undocumented_pointer_syntax(tmp_path, invalid, parameter):
    tool = JsonlQueryTool(workspace_dir=str(tmp_path))
    arguments = {parameter: [invalid] if parameter == "fields" else {invalid: 42}}

    result = await tool.execute(path="events.jsonl", **arguments)

    assert result.success is False
    assert "beginning with '/'" in result.error
