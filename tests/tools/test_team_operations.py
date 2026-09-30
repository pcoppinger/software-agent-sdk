"""Structured Teams operations have narrow schemas and real backing behavior."""

import hashlib
import subprocess
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
from openhands.tools.checklist_operations import (
    ChecklistOperations,
    ReplaceChecklistAction,
    ViewChecklistAction,
)
from openhands.tools.command_operations import (
    CommandObservation,
    CommandOperations,
    RunCommandAction,
    StopCommandAction,
    WaitForCommandAction,
)
from openhands.tools.file_editor.wrappers import (
    CreateFileAction,
    DeleteFileAction,
    FileWriteCommands,
    InsertFileTextAction,
    MoveFileAction,
    ReplaceTextInFileAction,
    UndoFileEditAction,
)
from openhands.tools.preset.default import register_default_tools
from openhands.tools.task_tracker.definition import TaskItem, TaskTrackerObservation
from openhands.tools.team_context import (
    ListChangedFilesAction,
    ReadEvidenceAction,
    ReadEvidenceTool,
    ReadFileDiffAction,
    RepositoryDiffOperations,
)


def _state(workspace: Path) -> ConversationState:
    return ConversationState.create(
        id=uuid4(),
        agent=Agent(llm=LLM(model="test-model"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace)),
    )


def test_write_operations_share_editor_and_require_exact_mutation_digest(
    tmp_path: Path,
) -> None:
    tools = {tool.name: tool for tool in FileWriteCommands.create(_state(tmp_path))}
    path = tmp_path / "note.txt"
    created = tools["file_create"](CreateFileAction(path=str(path), content="alpha\n"))
    assert not created.is_error
    assert tools["file_create"](
        CreateFileAction(path=str(path), content="other")
    ).is_error
    replaced = tools["replace_text_in_file"](
        ReplaceTextInFileAction(path=str(path), old_text="alpha", new_text="beta")
    )
    assert not replaced.is_error
    inserted = tools["insert_file_text"](
        InsertFileTextAction(path=str(path), after_line=1, content="gamma\n")
    )
    assert not inserted.is_error
    assert not tools["undo_file_edit"](UndoFileEditAction(path=str(path))).is_error
    assert path.read_text() == "beta\n"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert tools["file_move"](
        MoveFileAction(
            source_path=str(path),
            destination_path=str(tmp_path / "moved.txt"),
            expected_sha256="0" * 64,
        )
    ).is_error
    assert not tools["file_move"](
        MoveFileAction(
            source_path=str(path),
            destination_path=str(tmp_path / "moved.txt"),
            expected_sha256=digest,
        )
    ).is_error
    moved = tmp_path / "moved.txt"
    assert tools["file_delete"](
        DeleteFileAction(path=str(moved), expected_sha256="0" * 64)
    ).is_error
    assert not tools["file_delete"](
        DeleteFileAction(path=str(moved), expected_sha256=digest)
    ).is_error
    assert not moved.exists()


def test_checklist_and_command_operations_are_single_purpose(tmp_path: Path) -> None:
    state = _state(tmp_path)
    checklist = {tool.name: tool for tool in ChecklistOperations.create(state)}
    assert (
        cast(
            TaskTrackerObservation, checklist["view_checklist"](ViewChecklistAction())
        ).task_list
        == []
    )
    planned = checklist["replace_checklist"](
        ReplaceChecklistAction(items=[TaskItem(title="Build", notes="", status="todo")])
    )
    assert not planned.is_error
    assert (
        cast(TaskTrackerObservation, checklist["view_checklist"](ViewChecklistAction()))
        .task_list[0]
        .title
        == "Build"
    )

    commands = {tool.name: tool for tool in CommandOperations.create(state)}
    result = cast(
        CommandObservation,
        commands["run_command"](
            RunCommandAction(command="printf ready", timeout_seconds=5)
        ),
    )
    assert not result.is_error
    assert result.exit_code == 0
    assert "ready" in result.text
    assert result.active_command_id is None


def test_command_wait_and_stop_target_only_the_active_command(tmp_path: Path) -> None:
    commands = {tool.name: tool for tool in CommandOperations.create(_state(tmp_path))}
    started = cast(
        CommandObservation,
        commands["run_command"](RunCommandAction(command="sleep 2", timeout_seconds=1)),
    )
    assert started.exit_code == -1
    assert started.active_command_id is not None
    assert commands["wait_for_command"](
        WaitForCommandAction(command_id="not-the-active-command")
    ).is_error
    finished = cast(
        CommandObservation,
        commands["wait_for_command"](
            WaitForCommandAction(
                command_id=started.active_command_id, timeout_seconds=5
            )
        ),
    )
    assert finished.exit_code == 0
    assert finished.active_command_id is None
    assert commands["stop_command"](
        StopCommandAction(command_id=started.active_command_id)
    ).is_error


def test_evidence_and_diff_are_bound_to_admitted_inputs(tmp_path: Path) -> None:
    state = _state(tmp_path)
    content = b"line one\nline two\n"
    digest = hashlib.sha256(content).hexdigest()
    evidence_path = tmp_path / "evidence" / digest[:2] / digest
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(content)
    evidence = ReadEvidenceTool.create(
        state, str(tmp_path / "evidence"), {"e-1": digest}
    )[0]
    assert (
        "line two" in evidence(ReadEvidenceAction(evidence_id="e-1", start_line=2)).text
    )
    assert evidence(ReadEvidenceAction(evidence_id="e-other")).is_error
    evidence_path.write_bytes(b"tampered")
    assert evidence(ReadEvidenceAction(evidence_id="e-1")).is_error

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    baseline = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
        .decode()
        .strip()
    )
    tracked.write_text("after\n")
    (tmp_path / "new.txt").write_text("new\n")
    diff = {
        tool.name: tool for tool in RepositoryDiffOperations.create(state, baseline)
    }
    changed = diff["list_changed_files"](ListChangedFilesAction())
    assert "tracked.txt" in changed.text
    assert "new.txt" in changed.text
    assert (
        "-before" in diff["read_file_diff"](ReadFileDiffAction(path="tracked.txt")).text
    )
    tracked.unlink()
    assert (
        "-before" in diff["read_file_diff"](ReadFileDiffAction(path="tracked.txt")).text
    )
    assert diff["read_file_diff"](ReadFileDiffAction(path="../outside.txt")).is_error


def test_new_tool_schemas_are_narrow_and_registered(tmp_path: Path) -> None:
    register_default_tools(enable_browser=False)
    state = _state(tmp_path)
    for name in ("command_operations", "checklist_operations", "file_write_commands"):
        for tool in resolve_tool(Tool(name=name), state):
            schema = tool.to_openai_tool()["function"].get("parameters")
            assert isinstance(schema, dict)
            Draft202012Validator.check_schema(schema)
            assert schema.get("additionalProperties") is False
            assert "command" not in schema["properties"] or tool.name == "run_command"
