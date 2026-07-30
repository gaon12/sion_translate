"""Fail when generated corpora or translation queues enter the Git index."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATA_FILES = {
    PurePosixPath("data/.gitkeep"),
    PurePosixPath("data/data.txt"),
}


def tracked_paths() -> list[PurePosixPath]:
    """Return paths from Git's index without relying on shell quoting."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(raw.decode("utf-8", errors="surrogateescape"))
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def forbidden_data_paths(paths: list[PurePosixPath]) -> list[PurePosixPath]:
    """Find tracked files under directories reserved for local text data."""

    violations: list[PurePosixPath] = []
    for path in paths:
        if path.parts[:1] == ("data",) and path not in ALLOWED_DATA_FILES:
            violations.append(path)
        elif path.parts[:1] == ("translation_queue",):
            violations.append(path)
    return sorted(violations, key=str)


def main() -> int:
    violations = forbidden_data_paths(tracked_paths())
    if not violations:
        print("Git index contains no generated corpus or translation queue files.")
        return 0

    print("Refusing tracked corpus artifacts:", file=sys.stderr)
    for path in violations:
        print(f"  - {path}", file=sys.stderr)
    print(
        "Keep aggregate documentation outside data/, or sanitize it before publishing.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
