"""artifact 루트의 프로세스 간 배타 락.

두 작업이 동시에 "토크나이저가 없으니 만들자"고 판단하면 실패가 아니라
**섞인 상태**가 남습니다 — 한 세대의 토크나이저와 다른 세대의 데이터셋.
지문 검사는 그 조합을 처음 보는 것으로만 인식하므로 아무도 이상을 눈치채지
못합니다.
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
    """순차 실행은 막지 않는다. 막는 것은 동시 실행뿐이다."""
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
    assert "학습 output_dir가 다른 프로세스에 잠겨" in result.stdout
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

    with pytest.raises(RuntimeError, match="rank 0.*run lock"):
        with train_module.coordinated_training_run_lock(tmp_path, context):
            raise AssertionError("a peer must not enter after rank 0 failed")

    assert not (tmp_path / TRAINING_RUN_LOCK_FILENAME).exists()


def test_a_second_process_is_refused_while_the_lock_is_held(tmp_path) -> None:
    """같은 프로세스 안에서는 OS 락이 재진입을 허용할 수 있으므로 별도
    프로세스로 확인합니다."""
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
    assert "현재 보유자" in result.stdout
    assert f"pid={os.getpid()}" in result.stdout
    # 무엇을 하면 되는지가 메시지 안에 있어야 한다.
    assert "artifact 경로" in result.stdout


def test_the_lock_is_released_when_the_holder_dies(tmp_path) -> None:
    """파일의 존재로 잠그면 크래시한 작업이 영영 풀리지 않는 락을 남긴다.

    OS 락은 프로세스가 어떻게 끝나든 커널이 놓아 줍니다.
    """
    script = textwrap.dedent(
        f"""
        import os
        from sion_translate.locking import artifact_lock
        with artifact_lock({str(tmp_path)!r}):
            os._exit(1)   # finally 를 건너뛰고 즉시 죽는다
        """
    )
    subprocess.run([sys.executable, "-c", script], capture_output=True, cwd=os.getcwd())
    # 락 파일은 남아 있지만 잠겨 있지는 않아야 한다.
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
    assert "artifact 루트가 다른 프로세스에 잠겨 있습니다" in result.stderr
    # 자식의 import/startup 시간이 아니라 artifact_lock 안에서 보낸 시간만 잽니다.
    assert float(result.stdout.strip()) >= 0.15
