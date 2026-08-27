"""Inter-process exclusion for artifact roots.

If two jobs simultaneously decide that a tokenizer is missing, they can leave a
mixed tokenizer and dataset generation instead of an obvious failure. A normal
fingerprint check only sees that combination as new, so the corruption needs an
exclusive build lock rather than after-the-fact detection.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from sion_translate.locking import (
    LOCK_FILENAME,
    TRAINING_RUN_LOCK_FILENAME,
    artifact_lock,
    artifact_locks,
    training_run_lock,
)
from sion_translate.training.distributed import DistributedContext


def test_the_lock_creates_the_root_if_it_is_missing(tmp_path) -> None:
    root = tmp_path / "artifacts"
    with artifact_lock(root) as lock_path:
        assert root.is_dir()
        assert lock_path.name == LOCK_FILENAME


def test_the_holder_is_recorded_for_the_error_message(tmp_path) -> None:
    with artifact_lock(tmp_path) as lock_path:
        recorded = lock_path.read_text(encoding="utf-8")
    assert f"pid={os.getpid()}" in recorded
    assert "host=" in recorded


def test_the_lock_is_reentrant_across_sequential_uses(tmp_path) -> None:
    """The lock rejects overlapping work but allows sequential reuse."""
    for _ in range(3):
        with artifact_lock(tmp_path):
            pass


def test_multiple_artifact_locks_are_deduplicated_and_canonically_ordered(tmp_path) -> None:
    first = tmp_path / "z-root"
    second = tmp_path / "a-root"

    with artifact_locks((first, second, first / ".." / first.name)) as roots:
        assert roots == tuple(sorted((first.resolve(), second.resolve()), key=str))
        assert all((root / LOCK_FILENAME).is_file() for root in roots)


def test_shared_raw_root_serializes_jobs_with_different_dataset_roots(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    first_dataset_root = tmp_path / "dataset-a"
    second_dataset_root = tmp_path / "dataset-b"
    script = textwrap.dedent(
        f"""
        import sys
        from sion_translate.locking import artifact_locks
        try:
            with artifact_locks(({str(raw_root)!r}, {str(second_dataset_root)!r})):
                sys.exit(0)
        except RuntimeError as error:
            print(error)
            sys.exit(3)
        """
    )

    with artifact_locks((raw_root, first_dataset_root)):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            cwd=os.getcwd(),
        )

    assert result.returncode == 3, result.stderr
    assert str(raw_root) in result.stdout


def test_training_run_lock_uses_a_separate_lock_file(tmp_path) -> None:
    with artifact_lock(tmp_path) as artifact_path:
        with training_run_lock(tmp_path) as run_path:
            assert artifact_path.name == LOCK_FILENAME
            assert run_path.name == TRAINING_RUN_LOCK_FILENAME


def test_a_second_training_process_is_refused_for_the_same_output_dir(tmp_path) -> None:
    script = textwrap.dedent(
        f"""
        import sys
        from sion_translate.locking import training_run_lock
        try:
            with training_run_lock({str(tmp_path)!r}):
                sys.exit(0)
        except RuntimeError as error:
            print(error)
            sys.exit(3)
        """
    )

    with training_run_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            cwd=os.getcwd(),
        )

    assert result.returncode == 3, result.stderr
    assert "training output directory is locked by another process" in result.stdout
    assert "training.output_dir" in result.stdout
    assert f"pid={os.getpid()}" in result.stdout


def test_training_run_lock_is_released_after_the_scope(tmp_path) -> None:
    with training_run_lock(tmp_path):
        pass
    with training_run_lock(tmp_path):
        pass


def test_coordinated_run_lock_is_owned_only_by_rank_zero(
    tmp_path,
    monkeypatch,
) -> None:
    import sion_translate.cli.train as train_module

    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    monkeypatch.setattr(train_module, "broadcast_bool", lambda value, _context: value)

    with train_module.coordinated_training_run_lock(tmp_path, context) as lock_path:
        assert lock_path == tmp_path / TRAINING_RUN_LOCK_FILENAME
        assert lock_path.is_file()


def test_coordinated_peer_follows_rank_zero_lock_failure(tmp_path, monkeypatch) -> None:
    import sion_translate.cli.train as train_module

    context = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")
    monkeypatch.setattr(train_module, "broadcast_bool", lambda _value, _context: True)

    with pytest.raises(RuntimeError, match="Rank 0.*run lock"):
        with train_module.coordinated_training_run_lock(tmp_path, context):
            raise AssertionError("a peer must not enter after rank 0 failed")

    assert not (tmp_path / TRAINING_RUN_LOCK_FILENAME).exists()


def test_a_second_process_is_refused_while_the_lock_is_held(tmp_path) -> None:
    """Use another process because an OS lock may be reentrant in one process."""
    script = textwrap.dedent(
        f"""
        import sys
        from sion_translate.locking import artifact_lock
        try:
            with artifact_lock({str(tmp_path)!r}):
                sys.exit(0)
        except RuntimeError:
            sys.exit(3)
        """
    )
    with artifact_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
    assert result.returncode == 3, result.stderr


def test_the_refusal_names_the_holder_and_the_remedy(tmp_path) -> None:
    script = textwrap.dedent(
        f"""
        from sion_translate.locking import artifact_lock
        try:
            with artifact_lock({str(tmp_path)!r}):
                pass
        except RuntimeError as error:
            print(error)
        """
    )
    with artifact_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            cwd=os.getcwd(),
        )
    assert "current holder" in result.stdout
    assert f"pid={os.getpid()}" in result.stdout
    # The message must tell the operator how to resolve the conflict.
    assert "separate artifact path" in result.stdout


def test_the_lock_is_released_when_the_holder_dies(tmp_path) -> None:
    """A crashed process cannot leave an operating-system lock held forever.

    This is why file existence alone is not used as the lock signal.
    """
    script = textwrap.dedent(
        f"""
        import os
        from sion_translate.locking import artifact_lock
        with artifact_lock({str(tmp_path)!r}):
            os._exit(1)   # Exit immediately without running finally blocks.
        """
    )
    subprocess.run([sys.executable, "-c", script], capture_output=True, cwd=os.getcwd())
    # The lock file may remain, but its byte range must be unlocked.
    assert (tmp_path / LOCK_FILENAME).exists()
    with artifact_lock(tmp_path, timeout=0.0):
        pass


def test_a_timeout_waits_before_giving_up(tmp_path) -> None:
    with artifact_lock(tmp_path):
        script = textwrap.dedent(
            f"""
            import sys
            import time
            from sion_translate.locking import artifact_lock

            started = time.monotonic()
            try:
                with artifact_lock({str(tmp_path)!r}, timeout=0.2, poll_interval=0.05):
                    pass
            except RuntimeError as error:
                print(f"{{time.monotonic() - started:.9f}}")
                print(error, file=sys.stderr)
                sys.exit(3)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            cwd=os.getcwd(),
            timeout=5.0,
        )

    assert result.returncode == 3
    assert "artifact root is locked by another process" in result.stderr
    # Measure time inside artifact_lock, excluding child import and startup time.
    assert float(result.stdout.strip()) >= 0.15


def test_lock_rejects_a_symbolic_link_without_touching_its_target(tmp_path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("must remain unchanged", encoding="utf-8")
    lock_path = root / LOCK_FILENAME
    try:
        lock_path.symlink_to(sentinel)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable on this host: {error}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        with artifact_lock(root):
            raise AssertionError("a linked lock file must never be opened")

    assert sentinel.read_text(encoding="utf-8") == "must remain unchanged"
    assert lock_path.is_symlink()


def test_lock_does_not_create_directories_through_a_linked_parent(tmp_path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symbolic links are unavailable on this host: {error}")

    requested = linked_parent / "must-not-exist" / "artifacts"
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        with artifact_lock(requested):
            raise AssertionError("a lock root must not traverse a linked parent")

    assert not (real_parent / "must-not-exist").exists()


def test_lock_rejects_a_hard_link_without_overwriting_the_shared_file(tmp_path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("shared bytes", encoding="utf-8")
    lock_path = root / LOCK_FILENAME
    os.link(sentinel, lock_path)

    with pytest.raises(ValueError, match="multiple hard links"):
        with artifact_lock(root):
            raise AssertionError("a hard-linked lock file must never be opened")

    assert sentinel.read_text(encoding="utf-8") == "shared bytes"


def test_an_ordinary_stale_regular_lock_file_is_reusable(tmp_path) -> None:
    lock_path = tmp_path / LOCK_FILENAME
    lock_path.write_text("host=old pid=1 started=0\n", encoding="utf-8")

    with artifact_lock(tmp_path) as acquired:
        assert acquired == lock_path
        current = lock_path.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in current

    assert lock_path.is_file()
