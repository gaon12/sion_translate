from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from sion_translate.process_deadline import hard_process_deadline


def _blocked_deadline_worker(started_path_text: str) -> None:
    """Enter a deadline in a disposable process that the watchdog may kill."""

    started_path = Path(started_path_text)
    with hard_process_deadline("blocked deadline test", timeout_seconds=0.5):
        started_path.write_text("started", encoding="utf-8")
        while True:
            time.sleep(60.0)


@pytest.mark.parametrize("timeout_seconds", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_hard_process_deadline_rejects_invalid_timeouts(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        with hard_process_deadline("invalid timeout test", timeout_seconds=timeout_seconds):
            pass


def test_hard_process_deadline_rejects_an_empty_operation() -> None:
    with pytest.raises(ValueError, match="operation must not be empty"):
        with hard_process_deadline("\n  ", timeout_seconds=1.0):
            pass


def test_hard_process_deadline_cancels_after_normal_completion() -> None:
    completed: list[str] = []

    with hard_process_deadline("normal completion test", timeout_seconds=5.0):
        completed.append("done")

    assert completed == ["done"]


def test_hard_process_deadline_terminates_a_blocked_owner(tmp_path: Path) -> None:
    started_path = tmp_path / "started.txt"
    process = multiprocessing.get_context("spawn").Process(
        target=_blocked_deadline_worker,
        args=(str(started_path),),
    )
    started_at = time.monotonic()
    process.start()
    process.join(10.0)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        pytest.fail("hard-deadline watchdog did not terminate the blocked process")

    assert started_path.read_text(encoding="utf-8") == "started"
    assert process.exitcode not in {None, 0}
    assert time.monotonic() - started_at < 10.0
