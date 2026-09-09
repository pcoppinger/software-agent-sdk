"""Implementation of workspace-confined repository content search."""

import fnmatch
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor
from openhands.tools.repository_search.definition import (
    RepositoryMatch,
    RepositorySearchAction,
    RepositorySearchObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class RepositorySearchExecutor(
    ToolExecutor[RepositorySearchAction, RepositorySearchObservation]
):
    """Search workspace text files without exposing command execution."""

    _MAX_LINE_CHARACTERS = 500

    def __init__(self, working_dir: str):
        self.workspace_root = Path(working_dir).resolve()
        if not self.workspace_root.is_dir():
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

    def __call__(
        self,
        action: RepositorySearchAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> RepositorySearchObservation:
        try:
            pattern = re.compile(action.pattern, re.IGNORECASE)
        except re.error as error:
            return self._error(action, self.workspace_root, f"Invalid regex: {error}")

        search_path = self._resolve_search_path(action.path)
        if search_path is None or not search_path.is_dir():
            return self._error(
                action,
                search_path or self.workspace_root,
                "Search path must be a directory inside the current workspace",
            )

        matches: list[RepositoryMatch] = []
        truncated = False
        for root, directories, filenames in os.walk(search_path, followlinks=False):
            directories[:] = sorted(
                name for name in directories if not name.startswith(".")
            )
            for filename in sorted(filenames):
                if filename.startswith(".") or (
                    action.include and not fnmatch.fnmatch(filename, action.include)
                ):
                    continue
                file_path = Path(root, filename)
                resolved_file = file_path.resolve()
                if (
                    not self._inside_workspace(resolved_file)
                    or not resolved_file.is_file()
                ):
                    continue
                try:
                    lines = resolved_file.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                except OSError:
                    continue
                relative = resolved_file.relative_to(self.workspace_root).as_posix()
                for line_number, line in enumerate(lines, 1):
                    if pattern.search(line) is None:
                        continue
                    if len(matches) == action.max_results:
                        truncated = True
                        break
                    matches.append(
                        RepositoryMatch(
                            path=relative,
                            line=line_number,
                            text=line[: self._MAX_LINE_CHARACTERS],
                        )
                    )
                if truncated:
                    break
            if truncated:
                break

        text = self._format(matches, action.pattern, truncated)
        return RepositorySearchObservation.from_text(
            text=text,
            matches=matches,
            pattern=action.pattern,
            search_path=str(search_path),
            include_pattern=action.include,
            truncated=truncated,
        )

    def _resolve_search_path(self, value: str | None) -> Path | None:
        candidate = self.workspace_root if value is None else Path(value)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        return resolved if self._inside_workspace(resolved) else None

    def _inside_workspace(self, path: Path) -> bool:
        return path == self.workspace_root or self.workspace_root in path.parents

    def _error(
        self,
        action: RepositorySearchAction,
        search_path: Path,
        message: str,
    ) -> RepositorySearchObservation:
        return RepositorySearchObservation.from_text(
            text=message,
            matches=[],
            pattern=action.pattern,
            search_path=str(search_path),
            include_pattern=action.include,
            is_error=True,
        )

    def _format(
        self,
        matches: list[RepositoryMatch],
        pattern: str,
        truncated: bool,
    ) -> str:
        if not matches:
            return f"No matching lines found for pattern '{pattern}'"
        output = "\n".join(
            f"{match.path}:{match.line}:{match.text}" for match in matches
        )
        if truncated:
            output += "\n[Results truncated; narrow the search pattern or path.]"
        return output
