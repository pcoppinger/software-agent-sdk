"""Tests for the read-only repository viewer."""

from pathlib import Path
from uuid import uuid4

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.repository_view import RepositoryViewAction, RepositoryViewTool


def conversation_state(workspace: Path) -> ConversationState:
    return ConversationState.create(
        id=uuid4(),
        agent=Agent(llm=LLM(model="test-model"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace)),
    )


def test_repository_view_reads_file_and_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.go"
    source.write_text("package sample\n\nfunc Example() {}\n")
    tool = RepositoryViewTool.create(conversation_state(tmp_path))[0]

    viewed = tool(RepositoryViewAction(path=str(source), view_range=[1, 1]))
    listed = tool(RepositoryViewAction(path=str(tmp_path)))

    assert not viewed.is_error
    assert "package sample" in viewed.text
    assert "func Example" not in viewed.text
    assert not listed.is_error
    assert "source.go" in listed.text


def test_repository_view_schema_has_no_mutation_surface(tmp_path: Path) -> None:
    tool = RepositoryViewTool.create(conversation_state(tmp_path))[0]
    function = tool.to_openai_tool()["function"]
    assert "parameters" in function
    parameters = function["parameters"]

    assert set(parameters["properties"]) == {"path", "summary", "view_range"}
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is True
    with pytest.raises(Exception):
        RepositoryViewAction.model_validate(
            {"command": "create", "path": str(tmp_path / "new.txt")}
        )


def test_repository_view_rejects_workspace_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret")
    try:
        tool = RepositoryViewTool.create(conversation_state(tmp_path))[0]
        result = tool(RepositoryViewAction(path=str(outside)))
        assert result.is_error
        assert "secret" not in result.text
    finally:
        outside.unlink()
