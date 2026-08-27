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

import errno
import os
import socket
import stat
import sys
import time
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Iterator, cast

LOCK_FILENAME = ".sion_artifacts.lock"
TRAINING_RUN_LOCK_FILENAME = ".sion_training_run.lock"

# Lock a fixed byte beyond the holder-information region. Windows byte-range
# locks prevent writes to the locked range, so locking and writing the same byte
# would make the process block itself.
_LOCK_OFFSET = 1 << 30
# Pad the holder record to a fixed width so updates never require truncation.
_HOLDER_WIDTH = 128

_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class _PathIdentity:
    """Identity of one directory component held by the secure traversal."""

    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _SecureRoot:
    """An absolute lock root and its platform-owned directory handle."""

    path: Path
    descriptor: int
    ancestors: tuple[_PathIdentity, ...]


def _absolute_path(path: Path) -> Path:
    """Normalize ``.`` and ``..`` without resolving a link in the input path."""

    return Path(os.path.abspath(path))


def _is_reparse_stat(value: os.stat_result) -> bool:
    return bool(
        stat.S_ISLNK(value.st_mode)
        or getattr(value, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _validated_directory_identity(path: Path, value: os.stat_result) -> _PathIdentity:
    if _is_reparse_stat(value):
        raise ValueError(f"Lock roots cannot traverse a symbolic link or reparse point: {path}")
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"Every existing lock-root component must be a directory: {path}")
    device, inode = _identity(value)
    return _PathIdentity(path=path, device=device, inode=inode)


def _validate_regular_lock_stat(path: Path, value: os.stat_result) -> tuple[int, int]:
    if _is_reparse_stat(value):
        raise ValueError(f"Lock file cannot be a symbolic link or reparse point: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Lock file must be a regular file: {path}")
    if int(value.st_nlink) != 1:
        raise ValueError(f"Lock file cannot have multiple hard links: {path}")
    return _identity(value)


def _validate_root_chain(root: _SecureRoot) -> None:
    """Reject a namespace swap before a lock file can be changed."""

    for expected in root.ancestors:
        try:
            actual = os.lstat(expected.path)
        except OSError as error:
            raise RuntimeError(
                f"Lock-root component changed during acquisition: {expected.path}"
            ) from error
        observed = _validated_directory_identity(expected.path, actual)
        if (observed.device, observed.inode) != (expected.device, expected.inode):
            raise RuntimeError(f"Lock-root component changed during acquisition: {expected.path}")


def _validate_lock_filename(filename: str) -> None:
    candidate = Path(filename)
    if not filename or filename in {".", ".."} or candidate.name != filename:
        raise ValueError(f"lock filename must be one plain path component: {filename!r}")


if sys.platform == "win32":  # pragma: no cover - platform-specific branch
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3
    _OPEN_ALWAYS = 4
    _FILE_ATTRIBUTE_DIRECTORY = 0x0010
    _FILE_ATTRIBUTE_NORMAL = 0x0080
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _INVALID_HANDLE_VALUE = cast(int, ctypes.c_void_p(-1).value)

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _create_file_w = cast(Any, _kernel32.CreateFileW)
    _create_file_w.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _create_file_w.restype = wintypes.HANDLE
    _get_file_information = cast(Any, _kernel32.GetFileInformationByHandle)
    _get_file_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    _get_file_information.restype = wintypes.BOOL
    _close_handle = cast(Any, _kernel32.CloseHandle)
    _close_handle.argtypes = [wintypes.HANDLE]
    _close_handle.restype = wintypes.BOOL

    def _windows_error(operation: str, path: Path) -> OSError:
        error_code = ctypes.get_last_error()
        return OSError(
            error_code,
            f"{operation}: {path}: {ctypes.FormatError(error_code).strip()}",
            str(path),
        )

    def _windows_handle_information(
        handle: int,
        path: Path,
    ) -> tuple[int, int, tuple[int, int]]:
        information = _ByHandleFileInformation()
        if not _get_file_information(handle, ctypes.byref(information)):
            raise _windows_error("Cannot inspect lock handle", path)
        file_index = (int(information.file_index_high) << 32) | int(information.file_index_low)
        return (
            int(information.file_attributes),
            int(information.number_of_links),
            (int(information.volume_serial_number), file_index),
        )

    def _open_windows_directory(path: Path) -> tuple[int, tuple[int, int]]:
        handle = cast(
            int,
            _create_file_w(
                str(path),
                _FILE_READ_ATTRIBUTES,
                # Omitting FILE_SHARE_DELETE pins this exact directory entry
                # while its descendants and lock file are opened.
                _FILE_SHARE_READ | _FILE_SHARE_WRITE,
                None,
                _OPEN_EXISTING,
                _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
                None,
            ),
        )
        if handle == _INVALID_HANDLE_VALUE:
            raise _windows_error("Cannot open lock-root directory", path)
        try:
            attributes, _links, identity = _windows_handle_information(handle, path)
            if attributes & _REPARSE_POINT_ATTRIBUTE:
                raise ValueError(
                    f"Lock roots cannot traverse a symbolic link or reparse point: {path}"
                )
            if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
                raise ValueError(f"Every existing lock-root component must be a directory: {path}")
            return handle, identity
        except BaseException:
            _close_handle(handle)
            raise

    @contextmanager  # pyright: ignore[reportDeprecated]
    def _secure_lock_root(path: Path) -> Iterator[_SecureRoot]:
        """Create a Windows root while pinning every non-reparse ancestor.

        Windows has no Python ``openat`` directory traversal. Native directory
        handles opened without delete sharing keep already-validated ancestors
        from being renamed while each missing child is created.
        """

        absolute = _absolute_path(path)
        if not absolute.anchor:
            raise ValueError(f"lock root must have an absolute filesystem anchor: {absolute}")
        current = Path(absolute.anchor)
        handles: list[int] = []
        identities: list[_PathIdentity] = []
        try:
            for index, part in enumerate((absolute.anchor, *absolute.parts[1:])):
                if index:
                    current /= part
                    try:
                        before = os.lstat(current)
                    except FileNotFoundError:
                        try:
                            os.mkdir(current)
                        except FileExistsError:
                            pass
                    else:
                        _validated_directory_identity(current, before)
                handle, native_identity = _open_windows_directory(current)
                handles.append(handle)
                path_stat = os.lstat(current)
                identity = _validated_directory_identity(current, path_stat)
                if (identity.device, identity.inode) != native_identity:
                    raise RuntimeError(f"Lock-root component changed during acquisition: {current}")
                identities.append(identity)
            root = _SecureRoot(
                path=absolute,
                descriptor=handles[-1],
                ancestors=tuple(identities),
            )
            _validate_root_chain(root)
            yield root
        finally:
            for handle in reversed(handles):
                _close_handle(handle)

    def _open_lock_file(root: _SecureRoot, filename: str) -> tuple[IO[str], tuple[int, int]]:
        lock_path = root.path / filename
        try:
            existing = os.lstat(lock_path)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _validate_regular_lock_stat(lock_path, existing)

        native_handle = cast(
            int,
            _create_file_w(
                str(lock_path),
                _GENERIC_READ | _GENERIC_WRITE,
                # Prevent unlink/replacement after identity validation while
                # still allowing every contender to open and byte-range lock it.
                _FILE_SHARE_READ | _FILE_SHARE_WRITE,
                None,
                _OPEN_ALWAYS,
                _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
                None,
            ),
        )
        if native_handle == _INVALID_HANDLE_VALUE:
            try:
                raced = os.lstat(lock_path)
            except OSError:
                raced = None
            if raced is not None and (_is_reparse_stat(raced) or not stat.S_ISREG(raced.st_mode)):
                _validate_regular_lock_stat(lock_path, raced)
            raise _windows_error("Cannot open lock file", lock_path)
        try:
            attributes, number_of_links, native_identity = _windows_handle_information(
                native_handle,
                lock_path,
            )
            if attributes & _REPARSE_POINT_ATTRIBUTE:
                raise ValueError(
                    f"Lock file cannot be a symbolic link or reparse point: {lock_path}"
                )
            if attributes & _FILE_ATTRIBUTE_DIRECTORY:
                raise ValueError(f"Lock file must be a regular file: {lock_path}")
            if number_of_links != 1:
                raise ValueError(f"Lock file cannot have multiple hard links: {lock_path}")
            descriptor = msvcrt.open_osfhandle(native_handle, os.O_RDWR)
        except BaseException:
            _close_handle(native_handle)
            raise

        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.lstat(lock_path)
            descriptor_identity = _validate_regular_lock_stat(lock_path, descriptor_stat)
            path_identity = _validate_regular_lock_stat(lock_path, path_stat)
            if descriptor_identity != native_identity or path_identity != native_identity:
                raise RuntimeError(f"Lock file changed during secure open: {lock_path}")
            handle = os.fdopen(
                descriptor,
                "r+",
                encoding="utf-8",
                errors="replace",
                newline="",
            )
        except BaseException:
            os.close(descriptor)
            raise
        return handle, native_identity

    def _validate_open_lock(
        root: _SecureRoot,
        filename: str,
        handle: IO[str],
        expected_identity: tuple[int, int],
    ) -> None:
        _validate_root_chain(root)
        lock_path = root.path / filename
        descriptor_identity = _validate_regular_lock_stat(lock_path, os.fstat(handle.fileno()))
        path_identity = _validate_regular_lock_stat(lock_path, os.lstat(lock_path))
        if descriptor_identity != expected_identity or path_identity != expected_identity:
            raise RuntimeError(f"Lock file changed during acquisition: {lock_path}")

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

    _POSIX_DIRECTORY_FLAGS = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    _POSIX_LOCK_FLAGS = (
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )

    def _open_posix_directory_at(parent_descriptor: int, part: str, path: Path) -> int:
        try:
            descriptor = os.open(part, _POSIX_DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        except OSError as error:
            try:
                observed = os.stat(part, dir_fd=parent_descriptor, follow_symlinks=False)
            except OSError:
                raise error from None
            _validated_directory_identity(path, observed)
            raise error
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(part, dir_fd=parent_descriptor, follow_symlinks=False)
            descriptor_identity = _validated_directory_identity(path, descriptor_stat)
            path_identity = _validated_directory_identity(path, path_stat)
            if (descriptor_identity.device, descriptor_identity.inode) != (
                path_identity.device,
                path_identity.inode,
            ):
                raise RuntimeError(f"Lock-root component changed during acquisition: {path}")
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @contextmanager  # pyright: ignore[reportDeprecated]
    def _secure_lock_root(path: Path) -> Iterator[_SecureRoot]:
        """Create a POSIX root through ``openat`` and ``O_NOFOLLOW`` handles."""

        absolute = _absolute_path(path)
        anchor = Path(absolute.anchor or os.sep)
        descriptor = os.open(anchor, _POSIX_DIRECTORY_FLAGS)
        identities: list[_PathIdentity] = []
        try:
            anchor_descriptor_stat = os.fstat(descriptor)
            anchor_path_stat = os.lstat(anchor)
            anchor_identity = _validated_directory_identity(anchor, anchor_descriptor_stat)
            observed_anchor = _validated_directory_identity(anchor, anchor_path_stat)
            if (anchor_identity.device, anchor_identity.inode) != (
                observed_anchor.device,
                observed_anchor.inode,
            ):
                raise RuntimeError(f"Lock-root anchor changed during acquisition: {anchor}")
            identities.append(observed_anchor)

            current = anchor
            for part in absolute.parts[1:]:
                current /= part
                try:
                    child = _open_posix_directory_at(descriptor, part, current)
                except FileNotFoundError:
                    try:
                        os.mkdir(part, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = _open_posix_directory_at(descriptor, part, current)
                child_stat = os.fstat(child)
                identities.append(_validated_directory_identity(current, child_stat))
                os.close(descriptor)
                descriptor = child

            root = _SecureRoot(
                path=absolute,
                descriptor=descriptor,
                ancestors=tuple(identities),
            )
            _validate_root_chain(root)
            yield root
        finally:
            os.close(descriptor)

    def _posix_lock_path_stat(root: _SecureRoot, filename: str) -> os.stat_result | None:
        try:
            return os.stat(filename, dir_fd=root.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def _open_lock_file(root: _SecureRoot, filename: str) -> tuple[IO[str], tuple[int, int]]:
        lock_path = root.path / filename
        existing = _posix_lock_path_stat(root, filename)
        if existing is not None:
            _validate_regular_lock_stat(lock_path, existing)
        try:
            descriptor = os.open(
                filename,
                _POSIX_LOCK_FLAGS,
                0o666,
                dir_fd=root.descriptor,
            )
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raced = _posix_lock_path_stat(root, filename)
                if raced is not None:
                    _validate_regular_lock_stat(lock_path, raced)
            raise
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = _posix_lock_path_stat(root, filename)
            if path_stat is None:
                raise RuntimeError(f"Lock file disappeared during secure open: {lock_path}")
            descriptor_identity = _validate_regular_lock_stat(lock_path, descriptor_stat)
            path_identity = _validate_regular_lock_stat(lock_path, path_stat)
            if descriptor_identity != path_identity:
                raise RuntimeError(f"Lock file changed during secure open: {lock_path}")
            handle = os.fdopen(
                descriptor,
                "r+",
                encoding="utf-8",
                errors="replace",
                newline="",
            )
        except BaseException:
            os.close(descriptor)
            raise
        return handle, descriptor_identity

    def _validate_open_lock(
        root: _SecureRoot,
        filename: str,
        handle: IO[str],
        expected_identity: tuple[int, int],
    ) -> None:
        _validate_root_chain(root)
        lock_path = root.path / filename
        path_stat = _posix_lock_path_stat(root, filename)
        if path_stat is None:
            raise RuntimeError(f"Lock file disappeared during acquisition: {lock_path}")
        descriptor_identity = _validate_regular_lock_stat(lock_path, os.fstat(handle.fileno()))
        path_identity = _validate_regular_lock_stat(lock_path, path_stat)
        if descriptor_identity != expected_identity or path_identity != expected_identity:
            raise RuntimeError(f"Lock file changed during acquisition: {lock_path}")

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


def _describe_holder(handle: IO[str]) -> str:
    try:
        handle.seek(0)
        recorded = handle.read(_HOLDER_WIDTH).strip()
    except (OSError, UnicodeError):
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

    _validate_lock_filename(filename)
    requested_root = Path(root)
    deadline = time.monotonic() + max(0.0, timeout)
    with _secure_lock_root(requested_root) as secure_root:
        lock_path = secure_root.path / filename
        handle, expected_identity = _open_lock_file(secure_root, filename)
        try:
            while True:
                if _try_acquire(handle):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(conflict_message(secure_root.path, _describe_holder(handle)))
                time.sleep(poll_interval)
            # Recheck the complete namespace and descriptor/path identity after
            # waiting. No lock file is truncated until these checks succeed.
            _validate_open_lock(secure_root, filename, handle, expected_identity)
            handle.seek(0)
            handle.truncate()
            handle.write(
                f"host={socket.gethostname()} pid={os.getpid()} started={time.time():.0f}\n"
            )
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
        # ``resolve`` would silently follow an attacker-controlled link before
        # the secure acquisition code can reject it. Lexical normalization still
        # deduplicates ``.`` and ``..`` without changing path identity.
        canonical = _absolute_path(Path(root))
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
