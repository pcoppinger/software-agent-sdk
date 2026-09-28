"""Model-facing editor commands share the existing editor's history and rules."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor import FileEditorCommands


def _tools(workspace: Path):
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm")
    state = ConversationState.create(
        id=uuid4(),
        agent=Agent(llm=llm, tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace)),
    )
    return {tool.name: tool for tool in FileEditorCommands.create(state)}


def test_command_schemas_are_flat_and_require_operation_inputs(tmp_path: Path):
    tools = _tools(tmp_path)
    assert set(tools) == {
        "file_view",
        "file_create",
        "file_replace",
        "file_insert",
        "file_undo",
    }
    expected = {
        "file_view": {"path"},
        "file_create": {"path", "file_text"},
        "file_replace": {"path", "old_str", "new_str"},
        "file_insert": {"path", "insert_line", "new_str"},
        "file_undo": {"path"},
    }
    for name, tool in tools.items():
        schema = tool.to_openai_tool()["function"].get("parameters")
        assert schema is not None
        assert set(schema["required"]) == expected[name]
        assert "command" not in schema["properties"]

    with pytest.raises(ValidationError):
        tools["file_replace"].action_from_arguments(
            {"path": str(tmp_path / "a.txt"), "old_str": "old"}
        )


def test_wrappers_share_history_and_editor_rules(tmp_path: Path):
    tools = _tools(tmp_path)
    path = tmp_path / "a.txt"
    filename = str(path)

    def call(name: str, **arguments):
        tool = tools[name]
        return tool(tool.action_from_arguments({"path": filename, **arguments}))

    assert not call("file_create", file_text="one\ntwo\n").is_error
    assert not call("file_replace", old_str="two", new_str="three").is_error
    assert path.read_text() == "one\nthree\n"
    assert not call("file_undo").is_error
    assert path.read_text() == "one\ntwo\n"
    assert not call("file_insert", insert_line=1, new_str="middle\n").is_error
    assert "middle" in call("file_view").text
    assert call("file_create", file_text="again").is_error


def test_editor_history_is_per_conversation(tmp_path: Path):
    first = _tools(tmp_path)
    second = _tools(tmp_path)
    path = str(tmp_path / "a.txt")
    create = first["file_create"]
    assert not create(
        create.action_from_arguments({"path": path, "file_text": "one\n"})
    ).is_error
    undo = second["file_undo"]
    assert undo(undo.action_from_arguments({"path": path})).is_error
    assert Path(path).read_text() == "one\n"


def test_all_wrappers_lock_the_same_file_resource(tmp_path: Path):
    tools = _tools(tmp_path)
    path = str(tmp_path / "a.txt")
    resources = set()
    for name, tool in tools.items():
        arguments: dict[str, Any] = {"path": path}
        if name == "file_create":
            arguments["file_text"] = "content"
        if name == "file_replace":
            arguments.update(old_str="old", new_str="new")
        if name == "file_insert":
            arguments.update(insert_line=0, new_str="new")
        action = tool.action_from_arguments(arguments)
        resources.add(tool.declared_resources(action).keys)
    assert resources == {(f"file:{Path(path).resolve()}",)}


def test_registered_toolset_resolves_five_commands(tmp_path: Path):
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm")
    state = ConversationState.create(
        id=uuid4(),
        agent=Agent(llm=llm, tools=[Tool(name="file_editor_commands")]),
        workspace=LocalWorkspace(working_dir=str(tmp_path)),
    )
    tools = resolve_tool(Tool(name="file_editor_commands"), state)
    assert {tool.name for tool in tools} == {
        "file_view",
        "file_create",
        "file_replace",
        "file_insert",
        "file_undo",
    }
