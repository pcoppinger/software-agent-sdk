"""The Teams-facing navigation tools expose one operation per schema."""

from pathlib import Path
from typing import cast
from uuid import uuid4

from jsonschema import Draft202012Validator

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor.wrappers import FileWriteCommands
from openhands.tools.file_navigation import (
    FileReadAction,
    FileReadTool,
    FindFilesAction,
    FindFilesTool,
    ListFilesAction,
    ListFilesObservation,
    ListFilesTool,
    SearchFileContentsAction,
    SearchFileContentsTool,
)
from openhands.tools.glob.definition import GlobObservation
from openhands.tools.preset.default import register_default_tools
from openhands.tools.repository_search.definition import RepositorySearchObservation


def _state(workspace: Path) -> ConversationState:
    return ConversationState.create(
        id=uuid4(),
        agent=Agent(llm=LLM(model="test-model"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace)),
    )


def test_navigation_operations_are_distinct_and_workspace_confined(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source = source_dir / "widget.go"
    source.write_text("package widget\nfunc Widget() {}\n")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret")
    (source_dir / "outside.txt").symlink_to(outside)
    try:
        state = _state(tmp_path)
        read = FileReadTool.create(state)[0]
        listing = ListFilesTool.create(state)[0]
        find = FindFilesTool.create(state)[0]
        search = SearchFileContentsTool.create(state)[0]

        read_result = read(FileReadAction(path="src/widget.go", start_line=2))
        assert not read_result.is_error
        assert "func Widget" in read_result.text
        assert "package widget" not in read_result.text
        assert read(FileReadAction(path="src")).is_error
        assert read(FileReadAction(path=str(outside))).is_error
        assert read(FileReadAction(path="../" + outside.name)).is_error
        assert read(FileReadAction(path="src/outside.txt")).is_error

        root_listing = cast(ListFilesObservation, listing(ListFilesAction()))
        assert root_listing.entries == ["src/"]
        assert cast(
            ListFilesObservation, listing(ListFilesAction(directory="src"))
        ).entries == [
            "outside.txt",
            "widget.go",
        ]
        assert listing(ListFilesAction(directory=str(outside.parent))).is_error

        found = cast(GlobObservation, find(FindFilesAction(filename_glob="*.go")))
        assert not found.is_error
        assert found.files == [str(source)]
        assert (
            str(outside)
            not in cast(
                GlobObservation, find(FindFilesAction(filename_glob="*.txt"))
            ).files
        )
        assert find(FindFilesAction(filename_glob="*.go", directory="..")).is_error

        matched = cast(
            RepositorySearchObservation,
            search(SearchFileContentsAction(regex="func Widget", path="src")),
        )
        assert not matched.is_error
        assert len(matched.matches) == 1
        assert matched.matches[0].path == "src/widget.go"
        assert search(
            SearchFileContentsAction(regex="secret", path=str(outside))
        ).is_error
    finally:
        outside.unlink()


def test_navigation_schemas_have_only_operation_specific_parameters(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    expected = {
        FileReadTool: {"path", "start_line", "end_line", "summary"},
        ListFilesTool: {"directory", "start_after", "limit", "summary"},
        FindFilesTool: {"filename_glob", "directory", "summary"},
        SearchFileContentsTool: {
            "regex",
            "path",
            "filename_glob",
            "max_results",
            "summary",
        },
    }
    for tool_class, fields in expected.items():
        tool = tool_class.create(state)[0]
        schema = tool.to_openai_tool()["function"]["parameters"]
        Draft202012Validator.check_schema(schema)
        assert set(schema["properties"]) == fields
        assert schema["additionalProperties"] is False
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True


def test_write_bundle_does_not_expose_overloaded_view(tmp_path: Path) -> None:
    tools = FileWriteCommands.create(_state(tmp_path))
    assert [tool.name for tool in tools] == [
        "file_create",
        "replace_text_in_file",
        "insert_file_text",
        "undo_file_edit",
        "file_delete",
        "file_move",
    ]


def test_default_registration_resolves_all_new_tools(tmp_path: Path) -> None:
    register_default_tools(enable_browser=False)
    state = _state(tmp_path)
    for name in ("file_read", "list_files", "find_files", "search_file_contents"):
        resolved = resolve_tool(Tool(name=name), state)
        assert len(resolved) == 1
        assert resolved[0].name == name
    writes = resolve_tool(Tool(name="file_write_commands"), state)
    assert [tool.name for tool in writes] == [
        "file_create",
        "replace_text_in_file",
        "insert_file_text",
        "undo_file_edit",
        "file_delete",
        "file_move",
    ]
