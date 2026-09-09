"""Workspace-confined repository content search."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class RepositorySearchAction(Action):
    """Search text files and return the matching lines."""

    pattern: str = Field(description="Case-insensitive regular expression to search")
    path: str | None = Field(
        default=None,
        description=(
            "Optional workspace-relative or absolute directory. It must remain "
            "inside the current workspace."
        ),
    )
    include: str | None = Field(
        default=None,
        description='Optional filename pattern such as "*.go" or "*_test.go"',
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum matching lines to return",
    )


class RepositoryMatch(BaseModel):
    """One repository search match."""

    path: str = Field(description="Workspace-relative file path")
    line: int = Field(ge=1, description="One-based line number")
    text: str = Field(description="Matching source line")


class RepositorySearchObservation(Observation):
    """Structured repository search result."""

    matches: list[RepositoryMatch]
    pattern: str
    search_path: str
    include_pattern: str | None = None
    truncated: bool = False


TOOL_DESCRIPTION = """Read-only repository content search.
* Returns matching source lines as workspace-relative `path:line:text` records
* Searches case-insensitively with regular expressions
* Can restrict the search to a directory and filename pattern
* Never reads outside the current workspace and never modifies files
* Returns at most 100 matching lines; narrow the pattern when results are truncated
"""  # noqa: E501


class RepositorySearchTool(
    ToolDefinition[RepositorySearchAction, RepositorySearchObservation]
):
    """Create a workspace-confined search executor."""

    def declared_resources(self, action: Action) -> DeclaredResources:
        if not isinstance(action, RepositorySearchAction):
            raise TypeError(
                f"Expected RepositorySearchAction, got {type(action).__name__}"
            )
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["RepositorySearchTool"]:
        from openhands.tools.repository_search.impl import RepositorySearchExecutor

        working_dir = conv_state.workspace.working_dir
        executor = RepositorySearchExecutor(working_dir=working_dir)
        description = (
            f"{TOOL_DESCRIPTION}\n\n"
            f"Your current workspace is: {executor.workspace_root}"
        )
        return [
            cls(
                description=description,
                action_type=RepositorySearchAction,
                observation_type=RepositorySearchObservation,
                annotations=ToolAnnotations(
                    title="repository_search",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(RepositorySearchTool.name, RepositorySearchTool)
