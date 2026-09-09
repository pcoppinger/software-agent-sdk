from pathlib import Path

from openhands.tools.repository_search import (
    RepositorySearchAction,
    RepositorySearchExecutor,
)


def test_returns_workspace_relative_matching_lines(tmp_path: Path) -> None:
    source = tmp_path / "service.go"
    source.write_text("package service\nfunc StartRole() {}\nfunc StopRole() {}\n")
    executor = RepositorySearchExecutor(str(tmp_path))

    result = executor(RepositorySearchAction(pattern="startrole", include="*.go"))

    assert not result.is_error
    assert [match.model_dump() for match in result.matches] == [
        {"path": "service.go", "line": 2, "text": "func StartRole() {}"}
    ]
    assert result.text == "service.go:2:func StartRole() {}"


def test_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    executor = RepositorySearchExecutor(str(tmp_path))

    result = executor(RepositorySearchAction(pattern="anything", path=".."))

    assert result.is_error
    assert result.matches == []
    assert "inside the current workspace" in result.text


def test_excludes_hidden_and_symlink_escaped_files(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-search-fixture.txt"
    outside.write_text("needle\n")
    (tmp_path / ".hidden.txt").write_text("needle\n")
    (tmp_path / "escape.txt").symlink_to(outside)
    (tmp_path / "visible.txt").write_text("needle\n")
    executor = RepositorySearchExecutor(str(tmp_path))

    result = executor(RepositorySearchAction(pattern="needle"))

    assert [match.path for match in result.matches] == ["visible.txt"]


def test_bounds_results_and_reports_truncation(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("\n".join(["match"] * 4))
    executor = RepositorySearchExecutor(str(tmp_path))

    result = executor(RepositorySearchAction(pattern="match", max_results=2))

    assert [match.line for match in result.matches] == [1, 2]
    assert result.truncated
    assert result.text.endswith(
        "[Results truncated; narrow the search pattern or path.]"
    )
