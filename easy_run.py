"""인자 없이 실행하는 sion_translate 고성능 학습 진입점.

Linux에서 충분한 /dev/shm 공간이 있으면 원천 데이터와 전처리 산출물을 RAM
디스크에 배치합니다. tokenizer/dataset은 학습 전에 버전이 고정된 일반 디스크
artifacts/sion-v6/에도 원자적으로 동기화하고, checkpoints/exports는 항상 일반
디스크에 기록합니다.
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
ARTIFACT_LAYOUT_VERSION = "sion-v6"
PERSISTENT_ARTIFACTS = ROOT / "artifacts" / ARTIFACT_LAYOUT_VERSION
EXPRESSIVE_CORPUS_NAME = "synthetic_expressive_cultural.jsonl"
MIN_RAM_HEADROOM = 8 * 2**30


def _tmux_session_name() -> str:
    """Return a stable session name that cannot collide with another checkout."""

    identity = hashlib.sha256(str(ROOT.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"sion-{identity}"


def _install_tmux() -> str | None:
    """Return an existing tmux binary without mutating the host system."""

    tmux = shutil.which("tmux")
    if tmux is None:
        print(
            "[easy_run] tmux가 없어 현재 프로세스에서 계속합니다. "
            "지속 세션이 필요하면 tmux를 설치하거나 nohup/Slurm을 사용하세요.",
            flush=True,
        )
    return tmux


def _enter_tmux() -> None:
    """Re-exec easy_run inside a durable tmux session when launched interactively."""

    if (
        os.name == "nt"
        or os.environ.get("TMUX")
        or os.environ.get("SION_TMUX_ACTIVE") == "1"
        or os.environ.get("SION_NO_TMUX") == "1"
        or not sys.stdin.isatty()
        or not sys.stdout.isatty()
    ):
        return
    tmux = _install_tmux()
    if tmux is None:
        return
    session = _tmux_session_name()
    existing = (
        subprocess.run(
            [tmux, "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if existing:
        print(
            f"[easy_run] 기존 tmux 세션 '{session}'에 재접속합니다. "
            f"나중에는 tmux attach -t {session}",
            flush=True,
        )
        os.execv(tmux, [tmux, "attach-session", "-t", session])

    print(
        f"[easy_run] tmux 세션 '{session}'을 만들고 학습을 시작합니다. 분리: Ctrl+B, D",
        flush=True,
    )
    environment = os.environ.copy()
    environment["SION_TMUX_ACTIVE"] = "1"
    os.execve(
        tmux,
        [
            tmux,
            "new-session",
            "-s",
            session,
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
    published = False
    try:
        shutil.copytree(source, temporary)
        if destination.exists():
            destination.replace(backup)
        try:
            temporary.replace(destination)
            published = True
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
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
    workspace = shm / f"sion-{ROOT.name}-{identity}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _runtime_artifact_directory(ram_workspace: Path | None) -> Path:
    """Resolve only the current artifact layout on disk or in shared memory."""

    if ram_workspace is None:
        return PERSISTENT_ARTIFACTS
    return ram_workspace / "artifacts" / ARTIFACT_LAYOUT_VERSION


def _generated_config(raw_dir: Path, artifacts_dir: Path) -> Path:
    config_path = ROOT / "sion_translate.yaml"
    raw = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = raw.setdefault("data", {})
    data["raw_dir"] = str(raw_dir)
    # Source 규모 차이를 완화하되 사용자가 sion_translate.yaml에서 지정한 값은 보존합니다.
    data.setdefault("source_sampling_alpha", 0.9)
    data["tokenizer_model"] = str(artifacts_dir / "tokenizer" / "sion.model")
    data["tokenizer_features"] = str(artifacts_dir / "tokenizer" / "token_features.npz")
    data["dataset_dir"] = str(artifacts_dir / "dataset")

    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", prefix="sion-easy-", delete=False
    )
    with handle:
        yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)
    return Path(handle.name)


def _run(command: list[str], env: dict[str, str]) -> None:
    print("[easy_run] 실행:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _build_expressive_cultural_corpus(data_dir: Path, env: dict[str, str]) -> Path:
    """Materialize the curated train split before corpus discovery.

    The builder owns the leakage boundary between its training pairs and the
    bidirectional challenge set. Running it on every launch is inexpensive and
    makes the generated training shard reproducible from the reviewed seed file.
    """

    builder = ROOT / "scripts" / "data" / "build_expressive_cultural_corpus.py"
    seed = ROOT / "examples" / "expressive_cultural_seed_pairs.jsonl"
    training_output = data_dir / EXPRESSIVE_CORPUS_NAME
    challenge_output = ROOT / "examples" / "expressive_cultural_cases.jsonl"
    if not builder.is_file() or not seed.is_file():
        raise SystemExit(
            f"[easy_run] 표현·문화 데이터 빌더 또는 시드 파일이 없습니다: {builder}, {seed}"
        )

    print("[easy_run] 표현·문화 학습 코퍼스를 재현 가능하게 빌드합니다.", flush=True)
    _run(
        [
            sys.executable,
            str(builder),
            "--seed",
            str(seed),
            "--training-output",
            str(training_output),
            "--challenge-output",
            str(challenge_output),
        ],
        env,
    )
    if not training_output.is_file():
        raise SystemExit(
            f"[easy_run] 표현·문화 코퍼스 빌더가 성공했지만 출력이 없습니다: {training_output}"
        )
    return training_output


def _discover_raw_files(data_dir: Path, env: dict[str, str]) -> list[Path]:
    """Build required generated shards, then return a stable corpus listing."""

    _build_expressive_cultural_corpus(data_dir, env)
    raw_files = sorted(data_dir.glob("*.jsonl"))
    if not raw_files:
        raise SystemExit("data/*.jsonl 파일이 없습니다.")
    return raw_files


def _validate_gpu_runtime(torch_module) -> tuple[int, tuple[str, ...]]:
    """Fail before preprocessing when the CUDA runtime cannot launch training."""

    gpu_count = int(torch_module.cuda.device_count())
    if not torch_module.cuda.is_available() or gpu_count < 1:
        raise SystemExit("CUDA GPU를 찾지 못했습니다. CUDA 지원 PyTorch 환경을 확인하세요.")
    if gpu_count > 1 and not torch_module.distributed.is_nccl_available():
        raise SystemExit(
            f"CUDA GPU {gpu_count}개를 찾았지만 PyTorch에 NCCL 지원이 없습니다. "
            "다중 GPU용 CUDA PyTorch 패키지를 설치하세요."
        )
    names = tuple(sorted({torch_module.cuda.get_device_name(index) for index in range(gpu_count)}))
    return gpu_count, names


def _check_shard_keys(env: dict[str, str]) -> None:
    """Refuse to start when a shard's keys match no configured language pair.

    Such a shard yields zero sentences and says nothing about it, so the loss is
    invisible until someone counts. One shard in this corpus shipped with the
    keys 한국어/일본어 and would have dropped 10,075 rows in silence.
    """

    checker = ROOT / "scripts" / "data" / "check_shard_keys.py"
    if not checker.exists():
        return
    print("[easy_run] shard 키 이름을 확인합니다 (설정된 언어쌍과 맞는지).", flush=True)
    result = subprocess.run(
        [sys.executable, str(checker)],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "[easy_run] 위 shard 는 설정된 언어쌍으로 읽히지 않아 학습에서 조용히 빠집니다.\n"
            "           JSONL 의 키 이름을 고치거나 sion_translate.yaml 의 "
            "data.language_pairs 를 맞춘 뒤 다시 실행하세요."
        )


def _verify_tokenizer(
    tokenizer_model: Path,
    data_dir: Path,
    *,
    sample_rows: int = 20_000,
    max_fallback_rate: float = 0.002,
) -> None:
    """Fail when the tokenizer splits too much of the corpus into raw bytes.

    A character that falls back to bytes costs three tokens where one would do
    and the model never sees it as a unit. The previously shipped tokenizer
    rendered the 한본어 fused syllable 넼 as ``<0xEB> <0x84> <0xBC>`` because that
    corpus did not exist when it was trained.

    The check samples the corpus rather than testing fixed strings: hardcoded
    probes would be wrong for any other language pair, and the corpus is what the
    model actually has to encode. Some fallback is expected and correct - genuinely
    rare characters are exactly what byte fallback is for - so the gate is a rate,
    and the offending characters are always printed so the rate can be judged.
    """

    try:
        import sentencepiece as spm
    except ImportError:
        print("[easy_run] sentencepiece 를 불러오지 못해 토크나이저 검증을 건너뜁니다.")
        return
    if not tokenizer_model.exists():
        print(f"[easy_run] 토크나이저를 찾지 못해 검증을 건너뜁니다: {tokenizer_model}")
        return

    import json
    from collections import Counter

    processor = spm.SentencePieceProcessor()
    processor.Load(str(tokenizer_model))

    shards = sorted(data_dir.glob("*.jsonl"))
    if not shards:
        print("[easy_run] 코퍼스를 찾지 못해 토크나이저 검증을 건너뜁니다.")
        return
    per_shard = max(1, sample_rows // len(shards))

    total_tokens = 0
    fallback_tokens = 0
    offenders: Counter[str] = Counter()
    for shard in shards:
        with shard.open("r", encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle):
                if index >= per_shard:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                for value in row.values():
                    if not isinstance(value, str) or not value:
                        continue
                    pieces = processor.EncodeAsPieces(value)
                    total_tokens += len(pieces)
                    hits = sum(1 for piece in pieces if piece.startswith("<0x"))
                    if not hits:
                        continue
                    fallback_tokens += hits
                    # Name the characters, not the byte pieces: a byte piece on
                    # its own says nothing about what to fix.
                    for character in value:
                        encoded = processor.EncodeAsPieces(character)
                        if any(piece.startswith("<0x") for piece in encoded):
                            offenders[character] += 1

    if total_tokens == 0:
        print("[easy_run] 표본에서 토큰을 얻지 못해 검증을 건너뜁니다.")
        return

    rate = fallback_tokens / total_tokens
    print(
        f"[easy_run] 토크나이저 검증: vocab {processor.vocab_size():,}, "
        f"표본 {total_tokens:,} 토큰 중 byte fallback {fallback_tokens:,} ({rate:.4%})",
        flush=True,
    )
    for character, count in offenders.most_common(10):
        print(f"           U+{ord(character):04X} {character!r}  {count:,}회")
    if rate > max_fallback_rate:
        raise SystemExit(
            f"[easy_run] byte fallback 비율 {rate:.4%} 이 상한 {max_fallback_rate:.4%} 을 넘습니다. "
            "학습을 중단합니다.\n"
            f"           artifacts/{ARTIFACT_LAYOUT_VERSION}/tokenizer 를 지우고 "
            "다시 실행하거나, "
            "sion-train-tokenizer 를 --required-character-min-occurrences 를 낮춰 직접 "
            "실행하세요."
        )
    print("[easy_run] byte fallback 비율이 허용 범위입니다. 계속합니다.", flush=True)


def main() -> None:
    _enter_tmux()

    import torch

    gpu_count, gpu_names = _validate_gpu_runtime(torch)

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    source_data = ROOT / "data"
    raw_files = _discover_raw_files(source_data, env)

    required = sum(path.stat().st_size for path in raw_files)
    required += max(_directory_size(PERSISTENT_ARTIFACTS), required * 3)
    ram = _ram_workspace(required)
    runtime_artifacts = _runtime_artifact_directory(ram)
    if ram is None:
        runtime_data = source_data
        print("[easy_run] RAM 디스크 없이 일반 디스크에서 실행합니다.")
    else:
        runtime_data = ram / "data"
        print(f"[easy_run] RAM 디스크 사용: {ram}")
        print(f"[easy_run] 원천 데이터 {len(raw_files)}개를 RAM으로 복사합니다.")
        shutil.rmtree(runtime_data, ignore_errors=True)
        _copy_files_parallel(source_data, runtime_data)
        # The /dev/shm workspace has a stable checkout-derived name and can
        # survive a crashed run. Never inherit an orphaned cache implicitly.
        shutil.rmtree(runtime_artifacts, ignore_errors=True)
        if PERSISTENT_ARTIFACTS.exists():
            shutil.copytree(PERSISTENT_ARTIFACTS, runtime_artifacts)

    generated_config = _generated_config(runtime_data, runtime_artifacts)
    # Before anything expensive: a shard the pipeline cannot read is worth
    # catching now rather than after hours of training.
    _check_shard_keys(env)
    try:
        # 전처리를 단일 rank에서 끝내야 torchrun worker가 장시간 barrier에서
        # 기다리거나 통신 timeout에 걸리지 않습니다.
        _run(
            [
                sys.executable,
                "-m",
                "sion_translate.cli.train",
                "--config",
                str(generated_config),
                "--prepare-only",
            ],
            env,
        )
        # The tokenizer exists now, whether it was reused or just trained. Check
        # it before spending GPU hours on a vocabulary that cannot represent the
        # corpus.
        _verify_tokenizer(runtime_artifacts / "tokenizer" / "sion.model", runtime_data)

        if ram is not None:
            print(
                "[easy_run] tokenizer/dataset을 일반 디스크 "
                f"artifacts/{ARTIFACT_LAYOUT_VERSION}/에 보존합니다."
            )
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
            "sion_translate.cli.train",
            "--config",
            str(generated_config),
        ]
        print(f"[easy_run] CUDA GPU {gpu_count}개({', '.join(gpu_names)})로 학습을 시작합니다.")
        _run(command, env)
    finally:
        generated_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
