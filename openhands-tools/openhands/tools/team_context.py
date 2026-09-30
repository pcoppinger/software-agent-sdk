"""Invocation-bound evidence and repository-diff readers for Teams agents."""

import difflib
import hashlib
import re
import subprocess
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


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


_MAX_RESPONSE_BYTES = 256 << 10
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


def _annotation(name: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=name,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


class ReadEvidenceAction(Action):
    evidence_id: str = Field(description="Evidence ID admitted in this invocation.")
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_range(self) -> "ReadEvidenceAction":
        if self.start_line and self.end_line and self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        return self


class ContextReadObservation(Observation):
    source: str
    truncated: bool = False


class _EvidenceExecutor(ToolExecutor[ReadEvidenceAction, ContextReadObservation]):
    def __init__(self, root: Path, allowed: dict[str, str]):
        self.root = root.resolve()
        self.allowed = allowed

    def __call__(
        self,
        action: ReadEvidenceAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ContextReadObservation:
        digest = self.allowed.get(action.evidence_id)
        if digest is None or not _HEX_DIGEST.fullmatch(digest):
            return ContextReadObservation.from_text(
                text="Evidence ID is not admitted for this invocation",
                source=action.evidence_id,
                is_error=True,
            )
        path = self.root / digest[:2] / digest
        try:
            content = path.read_bytes()
        except OSError as error:
            return ContextReadObservation.from_text(
                text=f"Evidence unavailable: {error}",
                source=action.evidence_id,
                is_error=True,
            )
        if hashlib.sha256(content).hexdigest() != digest:
            return ContextReadObservation.from_text(
                text="Evidence checksum mismatch",
                source=action.evidence_id,
                is_error=True,
            )
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return ContextReadObservation.from_text(
                text="Evidence is not UTF-8 text",
                source=action.evidence_id,
                is_error=True,
            )
        start = (action.start_line or 1) - 1
        end = action.end_line or len(lines)
        selected = "\n".join(
            f"{index + 1}: {line}"
            for index, line in enumerate(lines)
            if start <= index < end
        )
        encoded = selected.encode("utf-8")
        truncated = len(encoded) > _MAX_RESPONSE_BYTES
        if truncated:
            selected = encoded[:_MAX_RESPONSE_BYTES].decode("utf-8", "ignore")
        return ContextReadObservation.from_text(
            text=selected or "No lines in requested range",
            source=action.evidence_id,
            truncated=truncated,
        )


class ReadEvidenceTool(ToolDefinition[ReadEvidenceAction, ContextReadObservation]):
    name = "read_evidence"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = super()._get_tool_schema(add_security_risk_prediction, action_type)
        schema["additionalProperties"] = False
        return schema

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",  # noqa: ARG003
        evidence_root: str,
        allowed: dict[str, str],
    ) -> Sequence["ReadEvidenceTool"]:
        if not Path(evidence_root).is_absolute() or not all(
            _HEX_DIGEST.fullmatch(digest) for digest in allowed.values()
        ):
            raise ValueError("Invalid invocation evidence binding")
        return [
            cls(
                action_type=ReadEvidenceAction,
                observation_type=ContextReadObservation,
                description=(
                    "Read admitted evidence by ID with verified SHA-256; "
                    "optional one-based line range."
                ),
                annotations=_annotation(cls.name),
                executor=_EvidenceExecutor(Path(evidence_root), allowed),
            )
        ]


class ListChangedFilesAction(Action):
    pass


class ReadFileDiffAction(Action):
    path: str = Field(description="File path relative to this invocation's repository.")


class _DiffExecutor(ToolExecutor[Action, ContextReadObservation]):
    def __init__(self, root: Path, baseline_commit: str):
        self.root = root.resolve()
        self.baseline_commit = baseline_commit

    def _git(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            check=False,
            timeout=20,
        )

    def __call__(
        self,
        action: Action,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ContextReadObservation:
        source = self.baseline_commit
        try:
            if isinstance(action, ListChangedFilesAction):
                changed = self._git(
                    "diff", "--name-only", "-z", self.baseline_commit, "--"
                )
                untracked = self._git(
                    "ls-files", "--others", "--exclude-standard", "-z"
                )
                if changed.returncode or untracked.returncode:
                    raise RuntimeError("Git could not enumerate changed files")
                paths = sorted(
                    set(changed.stdout.split(b"\0") + untracked.stdout.split(b"\0"))
                    - {b""}
                )
                text = "\n".join(path.decode("utf-8", "replace") for path in paths)
            else:
                assert isinstance(action, ReadFileDiffAction)
                requested = Path(action.path)
                path = (
                    requested if requested.is_absolute() else self.root / requested
                ).resolve()
                if self.root not in path.parents:
                    raise ValueError("File path must be in the bound repository")
                relative = str(path.relative_to(self.root))
                source = relative
                diff = self._git(
                    "diff", "--no-ext-diff", self.baseline_commit, "--", relative
                )
                if diff.returncode:
                    raise RuntimeError("Git could not read file diff")
                text = diff.stdout.decode("utf-8", "replace")
                if not text:
                    tracked = self._git("ls-files", "--error-unmatch", "--", relative)
                    if tracked.returncode and path.is_file():
                        current = path.read_text(encoding="utf-8").splitlines(
                            keepends=True
                        )
                        text = "".join(
                            difflib.unified_diff(
                                [], current, fromfile="/dev/null", tofile=relative
                            )
                        )
            encoded = text.encode("utf-8")
            truncated = len(encoded) > _MAX_RESPONSE_BYTES
            if truncated:
                text = encoded[:_MAX_RESPONSE_BYTES].decode("utf-8", "ignore")
            return ContextReadObservation.from_text(
                text=text or "No changes",
                source=source,
                truncated=truncated,
            )
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            return ContextReadObservation.from_text(
                text=f"Diff unavailable: {error}",
                source=source,
                is_error=True,
            )


class _DiffToolBase:
    action_model: ClassVar[type[Action]]
    description_text: ClassVar[str]

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        schema = ToolDefinition._get_tool_schema(
            cast(ToolDefinition[Action, ContextReadObservation], self),
            add_security_risk_prediction,
            action_type,
        )
        schema["additionalProperties"] = False
        return schema

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        baseline_commit: str,
        executor: _DiffExecutor | None = None,
    ) -> Sequence[ToolDefinition[Action, ContextReadObservation]]:
        if not _GIT_COMMIT.fullmatch(baseline_commit):
            raise ValueError("Invalid bound Git baseline")
        executor = executor or _DiffExecutor(
            Path(conv_state.workspace.working_dir), baseline_commit
        )
        tool_class = cast(type[ToolDefinition[Action, ContextReadObservation]], cls)
        return [
            tool_class(
                action_type=cls.action_model,
                observation_type=ContextReadObservation,
                description=cls.description_text,
                annotations=_annotation(tool_class.name),
                executor=executor,
            )
        ]


class ListChangedFilesTool(
    _DiffToolBase, ToolDefinition[ListChangedFilesAction, ContextReadObservation]
):
    name = "list_changed_files"
    action_model: ClassVar[type[Action]] = ListChangedFilesAction
    description_text: ClassVar[str] = (
        "List changed and untracked files against the invocation's bound Git baseline."
    )


class ReadFileDiffTool(
    _DiffToolBase, ToolDefinition[ReadFileDiffAction, ContextReadObservation]
):
    name = "read_file_diff"
    action_model: ClassVar[type[Action]] = ReadFileDiffAction
    description_text: ClassVar[str] = (
        "Read one file's diff against the invocation's bound Git baseline."
    )


class RepositoryDiffOperations(ToolDefinition[Action, ContextReadObservation]):
    name = "repository_diff_operations"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        baseline_commit: str,
    ) -> list[ToolDefinition[Any, ContextReadObservation]]:
        executor = _DiffExecutor(
            Path(conv_state.workspace.working_dir), baseline_commit
        )
        tools: list[ToolDefinition[Any, ContextReadObservation]] = []
        for tool_class in (ListChangedFilesTool, ReadFileDiffTool):
            tools.extend(tool_class.create(conv_state, baseline_commit, executor))
        return tools


register_tool(ReadEvidenceTool.name, ReadEvidenceTool)
register_tool(ListChangedFilesTool.name, ListChangedFilesTool)
register_tool(ReadFileDiffTool.name, ReadFileDiffTool)
register_tool(RepositoryDiffOperations.name, RepositoryDiffOperations)
