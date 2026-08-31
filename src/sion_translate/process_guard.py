"""Keep paid Linux worker processes attached to their owning launcher.

Linux does not terminate a child automatically when its parent disappears.  A
detached ``torchrun`` process can therefore keep GPUs allocated after its shell,
scheduler, or container supervisor has died.  This module arms Linux's
``PR_SET_PDEATHSIG`` contract before executing another Python module.  The
launcher records its exact PID for torchrun workers so a worker also detects the
small race in which its parent exits before the worker finishes starting.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import runpy
import signal
import sys
from collections.abc import Sequence
from typing import Callable, cast


EXPECTED_PARENT_PID_ENVIRONMENT = "SION_EXPECTED_GUARDIAN_PID"
INHERITED_SIGNAL_MASK_ENVIRONMENT = "SION_INHERITED_SIGNAL_MASK"
_PR_SET_PDEATHSIG = 1
# Windows type stubs omit these POSIX-only names. Runtime lookup keeps the real
# platform values on Linux and macOS while the fallbacks are used only where the
# corresponding POSIX operation is unavailable.
_LINUX_SIGKILL = cast(int, getattr(signal, "SIGKILL", 9))
_POSIX_SIG_SETMASK = cast(int, getattr(signal, "SIG_SETMASK", 2))
_ROLES = ("launcher", "worker")

_PthreadSigmask = Callable[[int, set[int]], object]


def _set_linux_parent_death_signal(signum: int) -> None:
    """Ask the Linux kernel to signal this process when its parent exits."""

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signum, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _die_after_parent_race(expected_parent_pid: int, observed_parent_pid: int) -> None:
    """Stop immediately rather than run expensive work without its guardian."""

    os.kill(os.getpid(), _LINUX_SIGKILL)
    # SIGKILL cannot be caught.  This branch exists only for test doubles or a
    # broken platform shim which returned without delivering the signal.
    raise RuntimeError(
        "the guarded parent changed before startup completed: "
        f"expected {expected_parent_pid}, observed {observed_parent_pid}"
    )


def arm_linux_parent_death_signal(expected_parent_pid: int | None = None) -> bool:
    """Arm SIGKILL-on-parent-death and close both sides of the setup race.

    ``expected_parent_pid`` is supplied by the parent before spawning whenever
    possible.  Comparing both before and after ``prctl`` catches a parent that
    dies immediately before this function or while the kernel contract is being
    installed.  Non-Linux platforms deliberately do nothing.
    """

    if sys.platform != "linux":
        return False

    observed_parent_pid = os.getppid()
    expected = observed_parent_pid if expected_parent_pid is None else expected_parent_pid
    if expected <= 0:
        raise ValueError("expected_parent_pid must be positive")
    if observed_parent_pid != expected:
        _die_after_parent_race(expected, observed_parent_pid)

    _set_linux_parent_death_signal(_LINUX_SIGKILL)
    observed_parent_pid = os.getppid()
    if observed_parent_pid != expected:
        _die_after_parent_race(expected, observed_parent_pid)
    return True


def _restore_inherited_signal_mask() -> None:
    """Restore the mask that easy_run held before its guarded spawn."""

    raw_signal_mask = os.environ.pop(INHERITED_SIGNAL_MASK_ENVIRONMENT, None)
    if raw_signal_mask is None:
        return
    pthread_sigmask = cast(
        _PthreadSigmask | None,
        getattr(signal, "pthread_sigmask", None),
    )
    if pthread_sigmask is None:
        raise RuntimeError(
            f"{INHERITED_SIGNAL_MASK_ENVIRONMENT} was supplied on a platform "
            "without pthread_sigmask"
        )
    raw_signal_numbers = [] if not raw_signal_mask else raw_signal_mask.split(",")
    if any(not raw_signum.isdecimal() for raw_signum in raw_signal_numbers):
        raise RuntimeError(
            f"{INHERITED_SIGNAL_MASK_ENVIRONMENT} must contain comma-separated signal numbers"
        )
    inherited_signal_mask = {int(raw_signum) for raw_signum in raw_signal_numbers}
    if any(signum <= 0 for signum in inherited_signal_mask):
        raise RuntimeError(
            f"{INHERITED_SIGNAL_MASK_ENVIRONMENT} must contain positive signal numbers"
        )
    pthread_sigmask(_POSIX_SIG_SETMASK, inherited_signal_mask)


def _expected_parent_pid() -> int:
    """Read the exact parent PID published by the guarded process owner."""

    raw_parent_pid = os.environ.get(EXPECTED_PARENT_PID_ENVIRONMENT)
    if raw_parent_pid is None:
        raise RuntimeError(
            f"{EXPECTED_PARENT_PID_ENVIRONMENT} is missing; guarded processes must be "
            "started by easy_run or the guarded launcher"
        )
    try:
        parent_pid = int(raw_parent_pid)
    except ValueError as error:
        raise RuntimeError(
            f"{EXPECTED_PARENT_PID_ENVIRONMENT} must contain a positive process ID"
        ) from error
    if parent_pid <= 0:
        raise RuntimeError(f"{EXPECTED_PARENT_PID_ENVIRONMENT} must contain a positive process ID")
    return parent_pid


def _parse_arguments(argv: Sequence[str] | None) -> tuple[str, str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a Python module under the sion parent-death guard."
    )
    parser.add_argument("role", choices=_ROLES)
    parser.add_argument("module", help="Fully qualified Python module to execute")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    module_arguments = list(parsed.arguments)
    if module_arguments[:1] == ["--"]:
        module_arguments.pop(0)
    return str(parsed.role), str(parsed.module), module_arguments


def main(argv: Sequence[str] | None = None) -> None:
    """Arm the guard, publish this guardian PID, and execute the target module."""

    _role, module, module_arguments = _parse_arguments(argv)
    expected_parent_pid = _expected_parent_pid()
    arm_linux_parent_death_signal(expected_parent_pid)
    _restore_inherited_signal_mask()

    # torchrun inherits this exact PID and passes it unchanged to every worker.
    # A guarded worker replaces it with its own PID before running application
    # code, allowing explicitly guarded descendants to form the same chain.
    os.environ[EXPECTED_PARENT_PID_ENVIRONMENT] = str(os.getpid())
    sys.argv = [module, *module_arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
