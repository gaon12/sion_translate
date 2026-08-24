"""Provide inter-process exclusive locks for artifacts and training runs.

Tokenizer and dataset creation follow a create-if-missing policy. Without one
lock around the complete operation, two processes can both observe a missing
artifact and publish different generations into the same path. That may leave a
mixed tokenizer/dataset state rather than an obvious failed process.

Locks are attached to open file descriptors instead of file existence. A stale
file from a crashed process would otherwise block work forever, while the kernel
always releases an operating-system lock when its process exits.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, Callable, Iterator

LOCK_FILENAME = ".sion_artifacts.lock"
TRAINING_RUN_LOCK_FILENAME = ".sion_training_run.lock"

# Lock a fixed byte beyond the holder-information region. Windows byte-range
# locks prevent writes to the locked range, so locking and writing the same byte
# would make the process block itself.
_LOCK_OFFSET = 1 << 30
# Pad the holder record to a fixed width so updates never require truncation.
_HOLDER_WIDTH = 128

if sys.platform == "win32":  # pragma: no cover - platform-specific branch
    import msvcrt

    def _try_acquire(handle: IO[str]) -> bool:
        try:
            handle.seek(_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _release(handle: IO[str]) -> None:
        try:
            handle.seek(_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:  # pragma: no cover - platform-specific branch
    import fcntl

    def _try_acquire(handle: IO[str]) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _release(handle: IO[str]) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def _describe_holder(path: Path) -> str:
    try:
        recorded = path.read_text(encoding="utf-8")[:_HOLDER_WIDTH].strip()
    except OSError:
        recorded = ""
    return recorded or "unknown process"


@contextmanager  # pyright: ignore[reportDeprecated]
def _exclusive_lock(
    root: str | Path,
    *,
    filename: str,
    conflict_message: Callable[[Path, str], str],
    timeout: float = 0.0,
    poll_interval: float = 1.0,
) -> Iterator[Path]:
    """Lock ``root`` exclusively while a caller creates or changes it.

    ``timeout=0`` fails immediately. Waiting is not the default because another
    tokenizer build may take hours; reporting its owner is safer than appearing
    to hang silently. The lock file records the owner's host and process ID so a
    conflict message identifies the process to inspect.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / filename
    deadline = time.monotonic() + max(0.0, timeout)
    handle = open(lock_path, "r+", encoding="utf-8") if lock_path.exists() else None
    if handle is None:
        lock_path.touch()
        handle = open(lock_path, "r+", encoding="utf-8")  # noqa: SIM115 - context owns it
    try:
        while True:
            if _try_acquire(handle):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(conflict_message(root, _describe_holder(lock_path)))
            time.sleep(poll_interval)
        handle.seek(0)
        handle.truncate()
        handle.write(f"host={socket.gethostname()} pid={os.getpid()} started={time.time():.0f}\n")
        handle.flush()
        yield lock_path
    finally:
        _release(handle)
        handle.close()


def _artifact_conflict_message(root: Path, holder: str) -> str:
    return (
        f"artifact root is locked by another process: {root}\n"
        f"  current holder: {holder}\n"
        "  Running two jobs against the same artifacts directory can mix "
        "different tokenizer and dataset generations. Wait for the current "
        "job to finish or give this job a separate artifact path."
    )


def _training_run_conflict_message(root: Path, holder: str) -> str:
    return (
        f"training output directory is locked by another process: {root}\n"
        f"  current holder: {holder}\n"
        "  Running two jobs in the same training.output_dir can mix checkpoints, "
        "logs, and exports. Wait for the current job to finish or give this run "
        "a separate training.output_dir."
    )


@contextmanager  # pyright: ignore[reportDeprecated]
def artifact_lock(
    root: str | Path,
    *,
    timeout: float = 0.0,
    poll_interval: float = 1.0,
) -> Iterator[Path]:
    """Reserve ``root`` exclusively for the complete artifact build."""

    with _exclusive_lock(
        root,
        filename=LOCK_FILENAME,
        conflict_message=_artifact_conflict_message,
        timeout=timeout,
        poll_interval=poll_interval,
    ) as lock_path:
        yield lock_path


@contextmanager  # pyright: ignore[reportDeprecated]
def artifact_locks(
    roots: Iterable[str | Path],
    *,
    timeout: float = 0.0,
    poll_interval: float = 1.0,
) -> Iterator[tuple[Path, ...]]:
    """Acquire a canonical set of artifact roots in one deterministic order."""

    canonical_by_key: dict[str, Path] = {}
    for root in roots:
        canonical = Path(root).resolve()
        canonical_by_key.setdefault(os.path.normcase(str(canonical)), canonical)
    ordered = tuple(canonical_by_key[key] for key in sorted(canonical_by_key))
    with ExitStack() as scope:
        for root in ordered:
            scope.enter_context(
                artifact_lock(
                    root,
                    timeout=timeout,
                    poll_interval=poll_interval,
                )
            )
        yield ordered


@contextmanager  # pyright: ignore[reportDeprecated]
def training_run_lock(
    root: str | Path,
    *,
    timeout: float = 0.0,
    poll_interval: float = 1.0,
) -> Iterator[Path]:
    """Reserve ``training.output_dir`` for the complete training run."""

    with _exclusive_lock(
        root,
        filename=TRAINING_RUN_LOCK_FILENAME,
        conflict_message=_training_run_conflict_message,
        timeout=timeout,
        poll_interval=poll_interval,
    ) as lock_path:
        yield lock_path


__all__ = [
    "LOCK_FILENAME",
    "TRAINING_RUN_LOCK_FILENAME",
    "artifact_lock",
    "artifact_locks",
    "training_run_lock",
]
