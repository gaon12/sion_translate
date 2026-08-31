"""Enforce hard process deadlines for operations that may block native code.

Thread-based timeouts cannot run while the owning interpreter is stuck in a
native extension, a filesystem call, or a deadlocked GIL holder. This module
therefore arms a clean helper interpreter before the risky operation starts.
The helper terminates the owning process when its absolute deadline expires.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager


_WATCHDOG_START_SECONDS = 5.0
_WATCHDOG_REAP_SECONDS = 5.0


# The helper intentionally imports no project modules and never touches CUDA.
# Reading one byte from stdin cancels the deadline. An EOF also cancels it when
# the owner exits normally, which prevents a stale helper from targeting a PID
# that the operating system might later reuse.
_WATCHDOG_CODE = r"""
import os
import signal
import sys
import threading
import time

parent_pid = int(sys.argv[1])
deadline_ns = int(sys.argv[2])
operation = sys.argv[3]
cancelled = threading.Event()

def read_cancellation():
    try:
        os.read(0, 1)
    finally:
        cancelled.set()

reader = threading.Thread(target=read_cancellation, daemon=True)
reader.start()
os.write(1, b"R")
remaining_seconds = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
if cancelled.wait(remaining_seconds):
    reader.join(1.0)
    os._exit(0)

# Terminate before diagnostics. A full or disconnected stderr pipe must never
# delay the paid-compute safety boundary.
try:
    hard_stop_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(parent_pid, hard_stop_signal)
except ProcessLookupError:
    pass
except OSError as error:
    try:
        os.write(2, f"FATAL: deadline watchdog could not stop pid {parent_pid}: {error}\n".encode())
    except OSError:
        pass
else:
    message = (
        "FATAL: a protected operation exceeded its hard deadline; terminated the "
        f"owning process (pid={parent_pid}, operation={operation}).\n"
    )
    try:
        os.write(2, message.encode("utf-8", errors="replace")[:4096])
    except OSError:
        pass
os._exit(124)
"""


def _validate_deadline(operation: str, timeout_seconds: float) -> tuple[str, float]:
    """Return bounded watchdog arguments or reject an unsafe request."""

    safe_operation = " ".join(str(operation).splitlines()).strip()[:512]
    if not safe_operation:
        raise ValueError("hard-deadline operation must not be empty")
    if isinstance(timeout_seconds, bool):
        raise ValueError("hard-deadline timeout must be a finite positive number")
    normalized_timeout = float(timeout_seconds)
    if not math.isfinite(normalized_timeout) or normalized_timeout <= 0.0:
        raise ValueError("hard-deadline timeout must be a finite positive number")
    return safe_operation, normalized_timeout


def _reap_failed_watchdog(watchdog: subprocess.Popen[bytes]) -> None:
    """Stop a helper that did not prove that its deadline was armed."""

    if watchdog.stdin is not None:
        watchdog.stdin.close()
    if watchdog.poll() is None:
        watchdog.kill()
    try:
        watchdog.wait(timeout=_WATCHDOG_REAP_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("hard-deadline watchdog could not be reaped") from error
    else:
        # Kill and reap before closing the read side. A startup reader may own
        # the pipe's I/O lock until process exit closes the writer.
        if watchdog.stdout is not None:
            watchdog.stdout.close()


def _start_watchdog(
    *,
    operation: str,
    timeout_seconds: float,
) -> subprocess.Popen[bytes]:
    """Arm an external watchdog and wait for its startup acknowledgement."""

    safe_operation, normalized_timeout = _validate_deadline(operation, timeout_seconds)
    deadline_ns = time.monotonic_ns() + int(normalized_timeout * 1_000_000_000)
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW  # pyright: ignore[reportAttributeAccessIssue]
    try:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                _WATCHDOG_CODE,
                str(os.getpid()),
                str(deadline_ns),
                safe_operation,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Fatal diagnostics must remain visible even when the owner dies.
            stderr=None,
            bufsize=0,
            close_fds=True,
            creationflags=creation_flags,
        )
    except OSError as error:
        raise RuntimeError(
            "could not start the hard-deadline watchdog; refusing to begin "
            "an operation that could outlive its paid compute budget"
        ) from error
    if watchdog.stdin is None or watchdog.stdout is None:
        _reap_failed_watchdog(watchdog)
        raise RuntimeError("hard-deadline watchdog has incomplete control pipes")

    ready: list[bytes] = []
    startup_errors: list[BaseException] = []
    ready_pipe = watchdog.stdout

    def read_ready_byte() -> None:
        try:
            ready.append(ready_pipe.read(1))
        except BaseException as error:
            startup_errors.append(error)

    reader = threading.Thread(
        target=read_ready_byte,
        name="sion-hard-deadline-start",
        daemon=True,
    )
    reader.start()
    reader.join(_WATCHDOG_START_SECONDS)
    if reader.is_alive() or startup_errors or ready != [b"R"]:
        _reap_failed_watchdog(watchdog)
        reader.join(_WATCHDOG_REAP_SECONDS)
        detail = f": {startup_errors[0]}" if startup_errors else ""
        raise RuntimeError("hard-deadline watchdog did not confirm that it was armed" + detail)
    ready_pipe.close()
    return watchdog


def _cancel_watchdog(watchdog: subprocess.Popen[bytes]) -> None:
    """Cancel and reap a helper after the protected operation has ended."""

    cancellation_error: OSError | None = None
    cancellation_pipe = watchdog.stdin
    if cancellation_pipe is not None and not cancellation_pipe.closed:
        try:
            cancellation_pipe.write(b"\x00")
            cancellation_pipe.flush()
        except OSError as error:
            cancellation_error = error
        finally:
            cancellation_pipe.close()
    try:
        return_code = watchdog.wait(timeout=_WATCHDOG_REAP_SECONDS)
    except subprocess.TimeoutExpired as error:
        watchdog.kill()
        try:
            watchdog.wait(timeout=_WATCHDOG_REAP_SECONDS)
        except subprocess.TimeoutExpired as reap_error:
            raise RuntimeError(
                "hard-deadline watchdog could not be reaped after cancellation"
            ) from reap_error
        raise RuntimeError("hard-deadline watchdog did not acknowledge cancellation") from error
    if return_code != 0 or cancellation_error is not None:
        detail = f": {cancellation_error}" if cancellation_error is not None else ""
        raise RuntimeError(
            f"hard-deadline watchdog exited unexpectedly with code {return_code}{detail}"
        )


@contextmanager
def hard_process_deadline(
    operation: str,
    *,
    timeout_seconds: float,
) -> Generator[None, None, None]:
    """Terminate this process if one protected operation misses its deadline."""

    watchdog = _start_watchdog(
        operation=operation,
        timeout_seconds=timeout_seconds,
    )
    try:
        yield
    finally:
        _cancel_watchdog(watchdog)


def hard_terminate_current_process(message: str) -> None:
    """Fail closed when a background mutation cannot be stopped safely."""

    # Production termination deliberately precedes all output: stderr may
    # itself be the blocked resource. The reason remains an argument so tests
    # and alternate supervisors can capture it without changing call sites.
    del message
    try:
        os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))
    except OSError:
        pass
    os._exit(124)
