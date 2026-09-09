"""Executor for read-only repository viewing."""

from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor
from openhands.tools.file_editor import FileEditorObservation
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError
from openhands.tools.repository_view.definition import RepositoryViewAction


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class RepositoryViewExecutor(ToolExecutor):
    """Expose only the FileEditor view operation within one workspace."""

    def __init__(self, working_dir: str):
        self.workspace_root = Path(working_dir).resolve()
        if not self.workspace_root.is_dir():
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")
        self.editor = FileEditor(workspace_root=str(self.workspace_root))

    def __call__(
        self,
        action: RepositoryViewAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> FileEditorObservation:
        requested = Path(action.path)
        if not requested.is_absolute():
            requested = self.workspace_root / requested
        resolved = requested.resolve()
        if (
            resolved != self.workspace_root
            and self.workspace_root not in resolved.parents
        ):
            return FileEditorObservation.from_text(
                text="View path must remain inside the current workspace",
                command="view",
                path=str(resolved),
                is_error=True,
            )
        try:
            return self.editor(
                command="view",
                path=str(resolved),
                view_range=action.view_range,
            )
        except ToolError as error:
            return FileEditorObservation.from_text(
                text=error.message,
                command="view",
                path=str(resolved),
                is_error=True,
            )
