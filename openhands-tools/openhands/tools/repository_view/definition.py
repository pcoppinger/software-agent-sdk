"""Workspace-confined read-only repository viewing."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)
from openhands.sdk.tool.schema import Schema
from openhands.tools.file_editor import FileEditorObservation


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class RepositoryViewAction(Action):
    """View one file or a shallow directory listing without mutation authority."""

    path: str = Field(
        description=(
            "Workspace-relative or absolute file/directory path. It must remain "
            "inside the current workspace."
        )
    )
    view_range: list[int] | None = Field(
        default=None,
        description=(
            "Optional one-based inclusive [start, end] line range for a file; "
            "use -1 as end to read through EOF."
        ),
    )


TOOL_DESCRIPTION = """Read-only repository file and directory viewer.
* Views text files with line numbers and lists directories to a shallow depth
* Optionally limits a file view to a one-based inclusive line range
* Rejects paths outside the current workspace, including symlink escapes
* Cannot create, replace, insert, undo, or otherwise modify files
"""  # noqa: E501


class RepositoryViewTool(ToolDefinition[RepositoryViewAction, FileEditorObservation]):
    """Create a workspace-confined read-only viewer."""

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["properties"]["view_range"].update(minItems=2, maxItems=2)
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:
        if not isinstance(action, RepositoryViewAction):
            raise TypeError(
                f"Expected RepositoryViewAction, got {type(action).__name__}"
            )
        normalized_path = Path(action.path).resolve()
        return DeclaredResources(
            keys=(f"file:{normalized_path}",),
            declared=True,
        )

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["RepositoryViewTool"]:
        from openhands.tools.repository_view.impl import RepositoryViewExecutor

        working_dir = conv_state.workspace.working_dir
        executor = RepositoryViewExecutor(working_dir=working_dir)
        description = (
            f"{TOOL_DESCRIPTION}\n\n"
            f"Your current workspace is: {executor.workspace_root}"
        )
        return [
            cls(
                description=description,
                action_type=RepositoryViewAction,
                observation_type=FileEditorObservation,
                annotations=ToolAnnotations(
                    title="repository_view",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(RepositoryViewTool.name, RepositoryViewTool)
