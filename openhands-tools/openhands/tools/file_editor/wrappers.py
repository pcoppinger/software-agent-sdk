"""Command-specific interfaces backed by one conversation-scoped file editor."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
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
    view_range: list[int] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Optional inclusive line range [start, end]; use -1 for the end.",
    )


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
        editor_action = FileEditorAction.model_validate(
            {"command": self.command, **action.model_dump(exclude={"kind"})}
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
        "Optionally request an inclusive line range."
    )


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


register_tool(FileViewTool.name, FileViewTool)
register_tool(FileCreateTool.name, FileCreateTool)
register_tool(FileReplaceTool.name, FileReplaceTool)
register_tool(FileInsertTool.name, FileInsertTool)
register_tool(FileUndoTool.name, FileUndoTool)
register_tool(FileEditorCommands.name, FileEditorCommands)
