"""Single-purpose, workspace-confined tools for repository navigation."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from openhands.tools.file_editor import FileEditorObservation
from openhands.tools.glob.definition import GlobAction, GlobObservation
from openhands.tools.glob.impl import GlobExecutor
from openhands.tools.repository_search.definition import (
    RepositorySearchAction,
    RepositorySearchObservation,
)
from openhands.tools.repository_search.impl import RepositorySearchExecutor
from openhands.tools.repository_view.definition import RepositoryViewAction
from openhands.tools.repository_view.impl import RepositoryViewExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


def _inside_workspace(root: Path, value: str) -> Path | None:
    requested = Path(value)
    resolved = (requested if requested.is_absolute() else root / requested).resolve()
    if resolved == root or root in resolved.parents:
        return resolved
    return None


def _annotations(name: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=name,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


class FileReadAction(Action):
    path: str = Field(description="File path, absolute or relative to the workspace.")
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_lines(self) -> "FileReadAction":
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must not precede start_line")
        return self


class _FileReadExecutor(ToolExecutor[FileReadAction, FileEditorObservation]):
    def __init__(self, root: Path):
        self.root = root
        self.viewer = RepositoryViewExecutor(working_dir=str(root))

    def __call__(
        self,
        action: FileReadAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> FileEditorObservation:
        path = _inside_workspace(self.root, action.path)
        if path is None or not path.is_file():
            return FileEditorObservation.from_text(
                text="File must exist inside the current workspace",
                command="view",
                path=action.path,
                is_error=True,
            )
        return self.viewer(
            RepositoryViewAction(
                path=str(path),
                start_line=action.start_line,
                end_line=action.end_line,
            )
        )


class FileReadTool(ToolDefinition[FileReadAction, FileEditorObservation]):
    """Read a file, never a directory or a path outside the workspace."""

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:
        assert isinstance(action, FileReadAction)
        assert self.meta is not None
        root = Path(self.meta["workspace_root"])
        path = _inside_workspace(root, action.path)
        return DeclaredResources(
            keys=(f"file:{path}",) if path is not None else (),
            declared=True,
        )

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["FileReadTool"]:
        root = Path(conv_state.workspace.working_dir).resolve()
        return [
            cls(
                action_type=FileReadAction,
                observation_type=FileEditorObservation,
                description=(
                    "Read one text file with line numbers. Optional start_line and "
                    f"end_line are one-based inclusive bounds. Workspace: {root}."
                ),
                annotations=_annotations(cls.name),
                executor=_FileReadExecutor(root),
                meta={"workspace_root": str(root)},
            )
        ]


class ListFilesAction(Action):
    directory: str = Field(
        default=".",
        description="Directory path, absolute or relative to the workspace.",
    )
    start_after: str | None = Field(
        default=None, description="Return names alphabetically after this entry."
    )
    limit: int = Field(default=100, ge=1, le=100)


class ListFilesObservation(Observation):
    directory: str
    entries: list[str]
    truncated: bool = False


class _ListFilesExecutor(ToolExecutor[ListFilesAction, ListFilesObservation]):
    def __init__(self, root: Path):
        self.root = root

    def __call__(
        self,
        action: ListFilesAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ListFilesObservation:
        path = _inside_workspace(self.root, action.directory)
        if path is None or not path.is_dir():
            return ListFilesObservation.from_text(
                text="Directory must exist inside the current workspace",
                directory=action.directory,
                entries=[],
                is_error=True,
            )
        try:
            children = sorted(path.iterdir(), key=lambda child: child.name)
            if action.start_after is not None:
                children = [
                    child for child in children if child.name > action.start_after
                ]
            entries = [
                f"{child.name}/" if child.is_dir() else child.name
                for child in children[: action.limit + 1]
            ]
        except OSError as error:
            return ListFilesObservation.from_text(
                text=f"Cannot list directory: {error}",
                directory=str(path),
                entries=[],
                is_error=True,
            )
        truncated = len(entries) > action.limit
        entries = entries[: action.limit]
        return ListFilesObservation.from_text(
            text="\n".join(entries) if entries else "Directory is empty",
            directory=str(path),
            entries=entries,
            truncated=truncated,
        )


class ListFilesTool(ToolDefinition[ListFilesAction, ListFilesObservation]):
    """List immediate file and directory names without reading their contents."""

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["ListFilesTool"]:
        root = Path(conv_state.workspace.working_dir).resolve()
        return [
            cls(
                action_type=ListFilesAction,
                observation_type=ListFilesObservation,
                description=(
                    "List only the immediate names in one directory. "
                    "Directory entries end with '/'; this tool does not recurse. "
                    f"Workspace: {root}."
                ),
                annotations=_annotations(cls.name),
                executor=_ListFilesExecutor(root),
            )
        ]


class FindFilesAction(Action):
    filename_glob: str = Field(description='Filename glob, for example "**/*.go".')
    directory: str = Field(
        default=".",
        description="Directory to search, absolute or relative to the workspace.",
    )

    @model_validator(mode="after")
    def relative_pattern(self) -> "FindFilesAction":
        if (
            Path(self.filename_glob).is_absolute()
            or ".." in Path(self.filename_glob).parts
        ):
            raise ValueError("filename_glob must not be an absolute or parent path")
        return self


class _FindFilesExecutor(ToolExecutor[FindFilesAction, GlobObservation]):
    def __init__(self, root: Path):
        self.root = root
        self.glob = GlobExecutor(working_dir=str(root))

    def __call__(
        self,
        action: FindFilesAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> GlobObservation:
        path = _inside_workspace(self.root, action.directory)
        if path is None or not path.is_dir():
            return GlobObservation.from_text(
                text="Directory must exist inside the current workspace",
                files=[],
                pattern=action.filename_glob,
                search_path=action.directory,
                is_error=True,
            )
        result = self.glob(GlobAction(pattern=action.filename_glob, path=str(path)))
        files = [
            file
            for file in result.files
            if _inside_workspace(self.root, file) is not None
        ]
        if result.is_error:
            text = result.text
        else:
            text = "\n".join(files) if files else "No matching files"
        return GlobObservation.from_text(
            text=text,
            files=files,
            pattern=action.filename_glob,
            search_path=str(path),
            truncated=result.truncated,
            is_error=result.is_error,
        )


class FindFilesTool(ToolDefinition[FindFilesAction, GlobObservation]):
    """Find file paths by glob; do not read file contents."""

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        executor = self.executor
        if (
            isinstance(executor, _FindFilesExecutor)
            and executor.glob.is_parallel_safe()
        ):
            return DeclaredResources(keys=(), declared=True)
        return DeclaredResources(keys=(), declared=False)

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["FindFilesTool"]:
        root = Path(conv_state.workspace.working_dir).resolve()
        return [
            cls(
                action_type=FindFilesAction,
                observation_type=GlobObservation,
                description=(
                    "Find files by filename glob inside the workspace; returns "
                    f"paths, not contents. Workspace: {root}."
                ),
                annotations=_annotations(cls.name),
                executor=_FindFilesExecutor(root),
            )
        ]


class SearchFileContentsAction(Action):
    regex: str = Field(description="Case-insensitive regular expression.")
    path: str = Field(
        default=".",
        description="File or directory to search, relative to the workspace.",
    )
    filename_glob: str | None = Field(
        default=None,
        description='Optional filename glob, for example "*.go".',
    )
    max_results: int = Field(default=50, ge=1, le=100)


class _SearchFileContentsExecutor(
    ToolExecutor[SearchFileContentsAction, RepositorySearchObservation]
):
    def __init__(self, root: Path):
        self.search = RepositorySearchExecutor(working_dir=str(root))

    def __call__(
        self,
        action: SearchFileContentsAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> RepositorySearchObservation:
        return self.search(
            RepositorySearchAction(
                pattern=action.regex,
                path=action.path,
                include=action.filename_glob,
                max_results=action.max_results,
            )
        )


class SearchFileContentsTool(
    ToolDefinition[SearchFileContentsAction, RepositorySearchObservation]
):
    """Search file contents by regular expression, never by filename glob."""

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(
            add_security_risk_prediction=add_security_risk_prediction,
            action_type=action_type,
        )
        schema["additionalProperties"] = False
        return schema

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence["SearchFileContentsTool"]:
        root = Path(conv_state.workspace.working_dir).resolve()
        return [
            cls(
                action_type=SearchFileContentsAction,
                observation_type=RepositorySearchObservation,
                description=(
                    "Search file contents with a case-insensitive regular "
                    "expression; returns matching path, line, and text. "
                    f"Workspace: {root}."
                ),
                annotations=_annotations(cls.name),
                executor=_SearchFileContentsExecutor(root),
            )
        ]


register_tool(FileReadTool.name, FileReadTool)
register_tool(ListFilesTool.name, ListFilesTool)
register_tool(FindFilesTool.name, FindFilesTool)
register_tool(SearchFileContentsTool.name, SearchFileContentsTool)
