"""Command-specific interfaces backed by one conversation-scoped file editor."""

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import Field, model_validator

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.schema import Schema
from openhands.tools.file_editor.definition import (
    FileEditorAction,
    FileEditorObservation,
)
from openhands.tools.file_editor.impl import FileEditorExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


class FileViewAction(Action):
    path: str = Field(description="Absolute path to a file or directory to view.")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional first line to read, starting at 1.",
    )
    end_line: int | None = Field(
        default=None,
        description="Optional last line to read; use -1 for the end of the file.",
    )
    view_range: list[int] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Legacy inclusive line range accepted from saved tool calls.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> "FileViewAction":
        if self.view_range is not None and (
            self.start_line is not None or self.end_line is not None
        ):
            raise ValueError("Use start_line/end_line or view_range, not both")
        if self.end_line is not None and self.end_line != -1 and self.end_line < 1:
            raise ValueError("end_line must be positive or -1")
        return self

    @property
    def inclusive_range(self) -> list[int] | None:
        if self.view_range is not None:
            return self.view_range
        if self.start_line is None and self.end_line is None:
            return None
        return [
            self.start_line or 1,
            self.end_line if self.end_line is not None else -1,
        ]


class FileCreateAction(Action):
    path: str = Field(description="Absolute path of the new file.")
    file_text: str = Field(description="Complete content of the new file.")


class FileReplaceAction(Action):
    path: str = Field(description="Absolute path of the file to edit.")
    old_str: str = Field(description="Exact, unique text to replace.")
    new_str: str = Field(
        description="Replacement text; use an empty string to delete the old text."
    )


class FileInsertAction(Action):
    path: str = Field(description="Absolute path of the file to edit.")
    insert_line: int = Field(
        ge=0, description="Insert after this one-based line; 0 inserts at the start."
    )
    new_str: str = Field(description="Text to insert.")


class FileUndoAction(Action):
    path: str = Field(description="Absolute path of the file whose last edit to undo.")


class _CommandExecutor(ToolExecutor[Action, FileEditorObservation]):
    """Translate one flat action into the existing editor's command contract."""

    def __init__(self, editor: FileEditorExecutor, command: str):
        self.editor = editor
        self.command = command

    def __call__(
        self,
        action: Action,
        conversation: "LocalConversation | None" = None,
    ) -> FileEditorObservation:
        if isinstance(action, FileViewAction):
            arguments = {"path": action.path, "view_range": action.inclusive_range}
        else:
            arguments = action.model_dump(exclude={"kind"})
        editor_action = FileEditorAction.model_validate(
            {"command": self.command, **arguments}
        )
        return self.editor(editor_action, conversation)


class _FileCommandMixin:
    """Give every wrapper the same path-level concurrency lock as file_editor."""

    command: ClassVar[str]
    action_model: ClassVar[type[Action]]
    tool_description: ClassVar[str]

    def declared_resources(self, action: Action) -> DeclaredResources:
        path = Path(getattr(action, "path")).resolve()
        return DeclaredResources(keys=(f"file:{path}",), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        editor: FileEditorExecutor | None = None,
    ) -> Sequence[ToolDefinition[Action, FileEditorObservation]]:
        working_dir = conv_state.workspace.working_dir
        if editor is None:
            editor = FileEditorExecutor(workspace_root=working_dir)
        read_only = cls.command == "view"
        tool_class = cast(type[ToolDefinition[Action, FileEditorObservation]], cls)
        return [
            tool_class(
                action_type=cls.action_model,
                observation_type=FileEditorObservation,
                description=(
                    f"{cls.tool_description} Use absolute paths. "
                    f"Current working directory: {working_dir}."
                ),
                annotations=ToolAnnotations(
                    title=tool_class.name,
                    readOnlyHint=read_only,
                    destructiveHint=not read_only,
                    idempotentHint=read_only,
                    openWorldHint=False,
                ),
                executor=_CommandExecutor(editor, cls.command),
            )
        ]


class FileViewTool(
    _FileCommandMixin, ToolDefinition[FileViewAction, FileEditorObservation]
):
    command: ClassVar[str] = "view"
    action_model: ClassVar[type[Action]] = FileViewAction
    tool_description: ClassVar[str] = (
        "Read a file with line numbers, or list a directory. "
        "Use optional start_line and end_line integers to limit the view."
    )

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["properties"].pop("view_range", None)
        schema["additionalProperties"] = False
        return schema


class FileCreateTool(
    _FileCommandMixin, ToolDefinition[FileCreateAction, FileEditorObservation]
):
    command: ClassVar[str] = "create"
    action_model: ClassVar[type[Action]] = FileCreateAction
    tool_description: ClassVar[str] = (
        "Create a new plain-text file; fails if the file already exists."
    )


class FileReplaceTool(
    _FileCommandMixin, ToolDefinition[FileReplaceAction, FileEditorObservation]
):
    command: ClassVar[str] = "str_replace"
    action_model: ClassVar[type[Action]] = FileReplaceAction
    tool_description: ClassVar[str] = (
        "Replace one exact, unique text span in a file. "
        "Include enough surrounding text to make old_str unique."
    )


class FileInsertTool(
    _FileCommandMixin, ToolDefinition[FileInsertAction, FileEditorObservation]
):
    command: ClassVar[str] = "insert"
    action_model: ClassVar[type[Action]] = FileInsertAction
    tool_description: ClassVar[str] = (
        "Insert text after a specified line in an existing file."
    )


class FileUndoTool(
    _FileCommandMixin, ToolDefinition[FileUndoAction, FileEditorObservation]
):
    command: ClassVar[str] = "undo_edit"
    action_model: ClassVar[type[Action]] = FileUndoAction
    tool_description: ClassVar[str] = (
        "Undo the last edit to this file in the current conversation."
    )


class FileEditorCommands(ToolDefinition[Action, FileEditorObservation]):
    """Resolve five distinct tools sharing one FileEditorExecutor per conversation."""

    name = "file_editor_commands"

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> list[ToolDefinition[Any, FileEditorObservation]]:
        working_dir = conv_state.workspace.working_dir
        editor = FileEditorExecutor(workspace_root=working_dir)
        tools: list[ToolDefinition[Any, FileEditorObservation]] = []
        for tool_class in (
            FileViewTool,
            FileCreateTool,
            FileReplaceTool,
            FileInsertTool,
            FileUndoTool,
        ):
            tools.extend(tool_class.create(conv_state, editor=editor))
        return tools


class FileWriteCommands(ToolDefinition[Action, Observation]):
    """Resolve write-only wrappers sharing one editor for conversation undo state."""

    name = "file_write_commands"

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> list[ToolDefinition[Any, Observation]]:
        editor = FileEditorExecutor(workspace_root=conv_state.workspace.working_dir)
        tools: list[ToolDefinition[Any, Observation]] = []
        for tool_class in (
            CreateFileTool,
            ReplaceTextInFileTool,
            InsertFileTextTool,
            UndoFileEditTool,
        ):
            tools.extend(tool_class.create(conv_state, editor=editor))
        for tool_class in (DeleteFileTool, MoveFileTool):
            tools.extend(tool_class.create(conv_state))
        return tools


class CreateFileAction(Action):
    path: str = Field(description="Path of a new file in this workspace.")
    content: str = Field(description="Complete text of the new file.")


class ReplaceTextInFileAction(Action):
    path: str = Field(description="Path of the existing file to edit.")
    old_text: str = Field(min_length=1, description="Exact, unique text to replace.")
    new_text: str = Field(description="Replacement text; empty removes old_text.")


class InsertFileTextAction(Action):
    path: str = Field(description="Path of the existing file to edit.")
    after_line: int = Field(
        ge=0, description="One-based line after which to insert; 0 is the start."
    )
    content: str = Field(description="Text to insert.")


class UndoFileEditAction(Action):
    path: str = Field(
        description="Path whose last edit in this conversation is to be undone."
    )


class _ModernEditExecutor(ToolExecutor[Action, FileEditorObservation]):
    def __init__(self, editor: FileEditorExecutor, command: str, root: Path):
        self.editor = editor
        self.command = command
        self.root = root.resolve()

    def __call__(
        self, action: Action, conversation: "LocalConversation | None" = None
    ) -> FileEditorObservation:
        supplied = Path(getattr(action, "path"))
        path = (supplied if supplied.is_absolute() else self.root / supplied).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            return FileEditorObservation.from_text(
                text="File path must remain inside the workspace",
                command=self.command,
                is_error=True,
            )
        arguments: dict[str, Any] = {"path": str(path)}
        if isinstance(action, CreateFileAction):
            arguments["file_text"] = action.content
        elif isinstance(action, ReplaceTextInFileAction):
            arguments.update(old_str=action.old_text, new_str=action.new_text)
        elif isinstance(action, InsertFileTextAction):
            arguments.update(insert_line=action.after_line, new_str=action.content)
        return self.editor(
            FileEditorAction.model_validate({"command": self.command, **arguments}),
            conversation,
        )


class _ModernEditMixin(_FileCommandMixin):
    def declared_resources(self, action: Action) -> DeclaredResources:
        executor = cast(_ModernEditExecutor, getattr(self, "executor"))
        supplied = Path(getattr(action, "path"))
        path = (
            supplied if supplied.is_absolute() else executor.root / supplied
        ).resolve()
        return DeclaredResources(keys=(f"file:{path}",), declared=True)

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = ToolDefinition._get_tool_schema(
            cast(ToolDefinition[Action, FileEditorObservation], self),
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        editor: FileEditorExecutor | None = None,
    ) -> Sequence[ToolDefinition[Action, FileEditorObservation]]:
        if editor is None:
            editor = FileEditorExecutor(workspace_root=conv_state.workspace.working_dir)
        working_dir = conv_state.workspace.working_dir
        tool_class = cast(type[ToolDefinition[Action, FileEditorObservation]], cls)
        return [
            tool_class(
                action_type=cls.action_model,
                observation_type=FileEditorObservation,
                description=f"{cls.tool_description} Workspace: {working_dir}.",
                annotations=ToolAnnotations(
                    title=tool_class.name,
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=_ModernEditExecutor(
                    editor, cls.command, Path(conv_state.workspace.working_dir)
                ),
            )
        ]


class CreateFileTool(
    _ModernEditMixin, ToolDefinition[CreateFileAction, FileEditorObservation]
):
    name = "file_create"
    command: ClassVar[str] = "create"
    action_model: ClassVar[type[Action]] = CreateFileAction
    tool_description: ClassVar[str] = (
        "Create one new text file; fail if it already exists."
    )


class ReplaceTextInFileTool(
    _ModernEditMixin, ToolDefinition[ReplaceTextInFileAction, FileEditorObservation]
):
    name = "replace_text_in_file"
    command: ClassVar[str] = "str_replace"
    action_model: ClassVar[type[Action]] = ReplaceTextInFileAction
    tool_description: ClassVar[str] = (
        "Replace one exact, unique span in an existing text file."
    )


class InsertFileTextTool(
    _ModernEditMixin, ToolDefinition[InsertFileTextAction, FileEditorObservation]
):
    name = "insert_file_text"
    command: ClassVar[str] = "insert"
    action_model: ClassVar[type[Action]] = InsertFileTextAction
    tool_description: ClassVar[str] = "Insert text after one line in an existing file."


class UndoFileEditTool(
    _ModernEditMixin, ToolDefinition[UndoFileEditAction, FileEditorObservation]
):
    name = "undo_file_edit"
    command: ClassVar[str] = "undo_edit"
    action_model: ClassVar[type[Action]] = UndoFileEditAction
    tool_description: ClassVar[str] = (
        "Undo the most recent editor change to this file in this conversation."
    )


class DeleteFileAction(Action):
    path: str = Field(description="Existing file path in this workspace.")
    expected_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the exact file to remove."
    )


class MoveFileAction(Action):
    source_path: str = Field(description="Existing source file in this workspace.")
    destination_path: str = Field(description="New destination path in this workspace.")
    expected_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the exact source file to move.",
    )


class FileMutationObservation(Observation):
    source_path: str
    destination_path: str | None = None


class _FileMutationExecutor(ToolExecutor[Action, FileMutationObservation]):
    def __init__(self, root: Path):
        self.root = root.resolve()

    def __call__(
        self,
        action: Action,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> FileMutationObservation:
        assert isinstance(action, (DeleteFileAction, MoveFileAction))
        raw_source = (
            action.path if isinstance(action, DeleteFileAction) else action.source_path
        )
        source_input = Path(raw_source)
        source_input = (
            source_input if source_input.is_absolute() else self.root / source_input
        )
        source = source_input.resolve()
        destination: Path | None = None
        destination_input: Path | None = None
        if isinstance(action, MoveFileAction):
            raw_destination = Path(action.destination_path)
            destination_input = (
                raw_destination
                if raw_destination.is_absolute()
                else self.root / raw_destination
            )
            destination = destination_input.resolve()
        fields = {
            "source_path": str(source),
            "destination_path": str(destination) if destination else None,
        }
        if (
            self.root not in source.parents
            or not source.is_file()
            or source_input.is_symlink()
        ):
            return FileMutationObservation.from_text(
                text="Source must be a regular file inside the workspace",
                is_error=True,
                **fields,
            )
        if destination is not None and (
            self.root not in destination.parents
            or destination.exists()
            or destination_input is not None
            and destination_input.is_symlink()
            or not destination.parent.is_dir()
        ):
            return FileMutationObservation.from_text(
                text=(
                    "Destination must be a new path in an existing workspace directory"
                ),
                is_error=True,
                **fields,
            )
        try:
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != action.expected_sha256:
                return FileMutationObservation.from_text(
                    text="File changed; expected_sha256 does not match",
                    is_error=True,
                    **fields,
                )
            if destination is None:
                source.unlink()
                return FileMutationObservation.from_text(text="File deleted", **fields)
            os.link(source, destination)
            source.unlink()
            return FileMutationObservation.from_text(text="File moved", **fields)
        except OSError as error:
            return FileMutationObservation.from_text(
                text=f"File mutation failed: {error}", is_error=True, **fields
            )


class _FileMutationMixin:
    action_model: ClassVar[type[Action]]
    tool_description: ClassVar[str]

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = ToolDefinition._get_tool_schema(
            cast(ToolDefinition[Action, FileMutationObservation], self),
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:
        assert isinstance(action, (DeleteFileAction, MoveFileAction))
        executor = cast(_FileMutationExecutor, getattr(self, "executor"))
        paths = (
            [action.path]
            if isinstance(action, DeleteFileAction)
            else [action.source_path, action.destination_path]
        )
        resolved = []
        for path in paths:
            supplied = Path(path)
            resolved.append(
                (
                    supplied if supplied.is_absolute() else executor.root / supplied
                ).resolve()
            )
        return DeclaredResources(
            keys=tuple(f"file:{path}" for path in resolved), declared=True
        )

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence[ToolDefinition[Action, FileMutationObservation]]:
        tool_class = cast(type[ToolDefinition[Action, FileMutationObservation]], cls)
        return [
            tool_class(
                action_type=cls.action_model,
                observation_type=FileMutationObservation,
                description=(
                    f"{cls.tool_description} Workspace: "
                    f"{conv_state.workspace.working_dir}."
                ),
                annotations=ToolAnnotations(
                    title=tool_class.name,
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=_FileMutationExecutor(Path(conv_state.workspace.working_dir)),
            )
        ]


class DeleteFileTool(
    _FileMutationMixin, ToolDefinition[DeleteFileAction, FileMutationObservation]
):
    name = "file_delete"
    action_model: ClassVar[type[Action]] = DeleteFileAction
    tool_description: ClassVar[str] = (
        "Delete one file only when its current SHA-256 matches expected_sha256."
    )


class MoveFileTool(
    _FileMutationMixin, ToolDefinition[MoveFileAction, FileMutationObservation]
):
    name = "file_move"
    action_model: ClassVar[type[Action]] = MoveFileAction
    tool_description: ClassVar[str] = (
        "Move one file to a new path only when its SHA-256 matches expected_sha256."
    )


register_tool(FileViewTool.name, FileViewTool)
register_tool(FileCreateTool.name, FileCreateTool)
register_tool(FileReplaceTool.name, FileReplaceTool)
register_tool(FileInsertTool.name, FileInsertTool)
register_tool(FileUndoTool.name, FileUndoTool)
register_tool(FileEditorCommands.name, FileEditorCommands)
register_tool(FileWriteCommands.name, FileWriteCommands)
register_tool(CreateFileTool.name, CreateFileTool)
register_tool(ReplaceTextInFileTool.name, ReplaceTextInFileTool)
register_tool(InsertFileTextTool.name, InsertFileTextTool)
register_tool(UndoFileEditTool.name, UndoFileEditTool)
register_tool(DeleteFileTool.name, DeleteFileTool)
register_tool(MoveFileTool.name, MoveFileTool)
