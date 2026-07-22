"""인자 없이 실행하는 KJ-X 고성능 학습 진입점.

Linux에서 충분한 /dev/shm 공간이 있으면 원천 데이터와 전처리 산출물을 RAM
디스크에 배치합니다. tokenizer/dataset은 학습 전에 일반 디스크 artifacts/에도
원자적으로 동기화하고, checkpoints/exports는 항상 일반 디스크에 기록합니다.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PERSISTENT_ARTIFACTS = ROOT / "artifacts"
MIN_RAM_HEADROOM = 8 * 2**30
TMUX_SESSION = "kjx"


def _install_tmux() -> None:
    """Install tmux on the root-based Debian/Ubuntu images commonly used by Vast.ai."""

    if shutil.which("tmux"):
        return
    apt_get = shutil.which("apt-get")
    if apt_get is None:
        raise SystemExit("tmux가 없고 apt-get도 찾지 못했습니다. tmux를 설치한 뒤 다시 실행하세요.")
    command_prefix: list[str] = []
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if sudo is None:
            raise SystemExit(
                "tmux 설치에 root 권한이 필요하지만 sudo가 없습니다. "
                "관리자 권한으로 tmux를 설치하세요."
            )
        command_prefix = [sudo]
    print("[easy_run] tmux가 없어 자동으로 설치합니다.", flush=True)
    subprocess.run([*command_prefix, apt_get, "update"], check=True)
    subprocess.run([*command_prefix, apt_get, "install", "-y", "tmux"], check=True)
    if not shutil.which("tmux"):
        raise SystemExit("tmux 설치 명령은 끝났지만 실행 파일을 찾지 못했습니다.")


def _enter_tmux() -> None:
    """Re-exec easy_run inside a durable tmux session when launched interactively."""

    if os.name == "nt" or os.environ.get("TMUX") or os.environ.get("KJX_TMUX_ACTIVE") == "1":
        return
    _install_tmux()
    tmux = shutil.which("tmux")
    assert tmux is not None
    existing = (
        subprocess.run(
            [tmux, "has-session", "-t", TMUX_SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if existing:
        print(
            f"[easy_run] 기존 tmux 세션 '{TMUX_SESSION}'에 재접속합니다. "
            f"나중에는 tmux attach -t {TMUX_SESSION}",
            flush=True,
        )
        os.execv(tmux, [tmux, "attach-session", "-t", TMUX_SESSION])

    print(
        f"[easy_run] tmux 세션 '{TMUX_SESSION}'을 만들고 학습을 시작합니다. 분리: Ctrl+B, D",
        flush=True,
    )
    environment = os.environ.copy()
    environment["KJX_TMUX_ACTIVE"] = "1"
    os.execve(
        tmux,
        [
            tmux,
            "new-session",
            "-s",
            TMUX_SESSION,
            shlex.join([sys.executable, str(Path(__file__).resolve())]),
        ],
        environment,
    )


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _copy_files_parallel(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    files = [item for item in source.iterdir() if item.is_file()]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(files)))) as executor:
        list(executor.map(lambda item: shutil.copy2(item, destination / item.name), files))


def _atomic_sync_directory(source: Path, destination: Path) -> None:
    """Copy a RAM directory to persistent storage without exposing partial data."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.sync-{os.getpid()}")
    backup = destination.with_name(f".{destination.name}.previous-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(source, temporary)
    if destination.exists():
        destination.replace(backup)
    temporary.replace(destination)
    shutil.rmtree(backup, ignore_errors=True)


def _ram_workspace(required_bytes: int) -> Path | None:
    shm = Path("/dev/shm")
    if os.name == "nt" or not shm.is_dir():
        return None
    free = shutil.disk_usage(shm).free
    if free < required_bytes + MIN_RAM_HEADROOM:
        print(
            f"[easy_run] /dev/shm 여유 공간 부족: 필요 약 {required_bytes / 2**30:.1f} GiB "
            f"+ 여유 8 GiB, 사용 가능 {free / 2**30:.1f} GiB. 일반 디스크를 사용합니다."
        )
        return None
    identity = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:8]
    workspace = shm / f"kjx-{ROOT.name}-{identity}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _generated_config(raw_dir: Path, artifacts_dir: Path) -> Path:
    config_path = ROOT / "kjx.yaml"
    raw = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = raw.setdefault("data", {})
    data["raw_dir"] = str(raw_dir)
    # Source 규모 차이를 완화하되 사용자가 kjx.yaml에서 지정한 값은 보존합니다.
    data.setdefault("source_sampling_alpha", 0.9)
    data["tokenizer_model"] = str(artifacts_dir / "tokenizer" / "kjx.model")
    data["tokenizer_features"] = str(artifacts_dir / "tokenizer" / "token_features.npz")
    data["dataset_dir"] = str(artifacts_dir / "dataset")

    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", prefix="kjx-easy-", delete=False
    )
    with handle:
        yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)
    return Path(handle.name)


def _run(command: list[str], env: dict[str, str]) -> None:
    print("[easy_run] 실행:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    _enter_tmux()

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise SystemExit("CUDA GPU를 찾지 못했습니다. H100 CUDA 환경에서 실행하세요.")

    source_data = ROOT / "data"
    raw_files = list(source_data.glob("*.jsonl"))
    if not raw_files:
        raise SystemExit("data/*.jsonl 파일이 없습니다.")

    required = sum(path.stat().st_size for path in raw_files)
    required += max(_directory_size(PERSISTENT_ARTIFACTS), required * 3)
    ram = _ram_workspace(required)
    if ram is None:
        runtime_data = source_data
        runtime_artifacts = PERSISTENT_ARTIFACTS
        print("[easy_run] RAM 디스크 없이 일반 디스크에서 실행합니다.")
    else:
        runtime_data = ram / "data"
        runtime_artifacts = ram / "artifacts"
        print(f"[easy_run] RAM 디스크 사용: {ram}")
        print(f"[easy_run] 원천 데이터 {len(raw_files)}개를 RAM으로 복사합니다.")
        shutil.rmtree(runtime_data, ignore_errors=True)
        _copy_files_parallel(source_data, runtime_data)
        if PERSISTENT_ARTIFACTS.exists():
            shutil.rmtree(runtime_artifacts, ignore_errors=True)
            shutil.copytree(PERSISTENT_ARTIFACTS, runtime_artifacts)

    generated_config = _generated_config(runtime_data, runtime_artifacts)
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    try:
        # 전처리를 단일 rank에서 끝내야 torchrun worker가 장시간 barrier에서
        # 기다리거나 통신 timeout에 걸리지 않습니다.
        _run(
            [
                sys.executable,
                "-m",
                "kjx.cli.train",
                "--config",
                str(generated_config),
                "--prepare-only",
            ],
            env,
        )
        if ram is not None:
            print("[easy_run] tokenizer/dataset을 일반 디스크 artifacts/에 보존합니다.")
            _atomic_sync_directory(
                runtime_artifacts / "tokenizer", PERSISTENT_ARTIFACTS / "tokenizer"
            )
            _atomic_sync_directory(runtime_artifacts / "dataset", PERSISTENT_ARTIFACTS / "dataset")

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpu_count}",
            "-m",
            "kjx.cli.train",
            "--config",
            str(generated_config),
        ]
        print(f"[easy_run] H100 GPU {gpu_count}개로 학습을 시작합니다.")
        _run(command, env)
    finally:
        generated_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
