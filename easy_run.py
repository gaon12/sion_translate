"""High-performance sion_translate training entry point.

On Linux, this script places source data and preprocessing outputs on a RAM disk
when ``/dev/shm`` has enough free space. Before training, it also synchronizes
the tokenizer and prepared datasets atomically to the persistent ``artifacts/``
directory.
Checkpoints and exports are always written to persistent storage.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from sion_translate.artifacts import DEFAULT_ARTIFACT_ROOT
from sion_translate.bundle_contract import (
    BundleContractError,
    EXPECTED_DEPENDENCY_TARGET,
    EXPECTED_RUNTIME_VERSIONS,
    load_embedded_training_contract,
    verify_embedded_bundle_payload,
)


ROOT = Path(__file__).resolve().parent
PERSISTENT_ARTIFACTS = ROOT / DEFAULT_ARTIFACT_ROOT
EXPRESSIVE_CORPUS_NAME = "synthetic_expressive_cultural.jsonl"
MIN_RAM_HEADROOM = 8 * 2**30
PREPARED_ARTIFACT_DIRECTORIES = ("tokenizer", "dataset", "foundation_dataset")
LOCAL_CHECKOUT_FLAG = "--allow-local-checkout"
TMUX_SUPPORTED = os.name != "nt"
CUDA_CANARY_TIMEOUT_SECONDS = 60
NCCL_CANARY_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class EmbeddedBundlePolicy:
    """Launch choices already authenticated by an extracted GPU bundle."""

    config_path: Path
    raw_parallel_data_included: bool
    monolingual_corpus_included: bool
    foundation_enabled: bool


def _tmux_session_name() -> str:
    """Return a stable session name that cannot collide with another checkout."""

    identity = hashlib.sha256(str(ROOT.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"sion-{identity}"


def _install_tmux() -> str | None:
    """Return an existing tmux binary without mutating the host system."""

    tmux = shutil.which("tmux")
    if tmux is None:
        print(
            "[easy_run] tmux is unavailable, so training will continue in the "
            "current process. Install tmux or use nohup/Slurm if you need a "
            "persistent session.",
            flush=True,
        )
    return tmux


def _parse_allow_local_checkout(arguments: Sequence[str] | None = None) -> bool:
    """Parse the launcher's one explicit development-trust option."""

    parser = argparse.ArgumentParser(
        description="Prepare, authenticate, and run the configured Sion training pipeline."
    )
    parser.add_argument(
        LOCAL_CHECKOUT_FLAG,
        action="store_true",
        help=(
            "trust this metadata-free Git development checkout; never use this option "
            "for an extracted GPU bundle"
        ),
    )
    return bool(parser.parse_args(arguments).allow_local_checkout)


def _tmux_training_command(*, allow_local_checkout: bool = False) -> list[str]:
    """Build the exact launcher command that a new tmux session must retain."""

    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *([LOCAL_CHECKOUT_FLAG] if allow_local_checkout else []),
    ]


def _enter_tmux(*, allow_local_checkout: bool = False) -> None:
    """Re-exec easy_run inside a durable tmux session when launched interactively."""

    if (
        not TMUX_SUPPORTED
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
            f"[easy_run] Reattaching to existing tmux session '{session}'. "
            f"To attach later, run: tmux attach -t {session}",
            flush=True,
        )
        os.execv(tmux, [tmux, "attach-session", "-t", session])

    print(
        f"[easy_run] Creating tmux session '{session}' and starting training. "
        "Detach with Ctrl+B, D.",
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
            shlex.join(_tmux_training_command(allow_local_checkout=allow_local_checkout)),
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
            f"[easy_run] Insufficient free space in /dev/shm: approximately "
            f"{required_bytes / 2**30:.1f} GiB plus an 8 GiB safety margin is "
            f"required, but only {free / 2**30:.1f} GiB is available. Using "
            "persistent storage instead."
        )
        return None
    identity = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:8]
    workspace = shm / f"sion-{ROOT.name}-{identity}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _runtime_artifact_directory(ram_workspace: Path | None) -> Path:
    """Resolve the canonical artifact root on disk or in shared memory."""

    if ram_workspace is None:
        return PERSISTENT_ARTIFACTS
    return ram_workspace / DEFAULT_ARTIFACT_ROOT


def _restore_active_artifacts(source: Path, destination: Path) -> None:
    """Restore only live prepared directories into a RAM workspace."""

    destination.mkdir(parents=True, exist_ok=True)
    for name in PREPARED_ARTIFACT_DIRECTORIES:
        active = source / name
        if active.exists():
            shutil.copytree(active, destination / name)


def _generated_config(raw_dir: Path, artifacts_dir: Path) -> Path:
    config_path = ROOT / "sion_translate.yaml"
    raw = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = raw.setdefault("data", {})
    data["raw_dir"] = str(raw_dir)
    # Reduce source-size imbalance while preserving any explicit user setting.
    data.setdefault("source_sampling_alpha", 0.9)
    data["tokenizer_model"] = str(artifacts_dir / "tokenizer" / "sion.model")
    data["tokenizer_features"] = str(artifacts_dir / "tokenizer" / "token_features.npz")
    data["dataset_dir"] = str(artifacts_dir / "dataset")
    foundation = raw.setdefault("foundation", {})
    foundation["dataset_dir"] = str(artifacts_dir / "foundation_dataset")

    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", prefix="sion-easy-", delete=False
    )
    with handle:
        yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)
    return Path(handle.name)


def _run(command: list[str], env: dict[str, str]) -> None:
    print("[easy_run] Running:", " ".join(command), flush=True)
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
            "[easy_run] The expressive/cultural data builder or seed file is "
            f"missing: {builder}, {seed}"
        )

    print(
        "[easy_run] Building the expressive/cultural training corpus reproducibly.",
        flush=True,
    )
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
            "[easy_run] The expressive/cultural corpus builder completed, but "
            f"its expected output is missing: {training_output}"
        )
    return training_output


def _discover_raw_files(data_dir: Path, env: dict[str, str]) -> list[Path]:
    """Build required generated shards, then return a stable corpus listing."""

    _build_expressive_cultural_corpus(data_dir, env)
    raw_files = sorted(data_dir.glob("*.jsonl"))
    if not raw_files:
        raise SystemExit("No data/*.jsonl files were found.")
    return raw_files


def _embedded_bundle_policy(
    root: Path | None = None,
    *,
    allow_local_checkout: bool = False,
) -> EmbeddedBundlePolicy | None:
    """Authenticate an extracted bundle before the launcher makes any choices."""

    bundle_root = (root or ROOT).resolve()
    try:
        contract = load_embedded_training_contract(
            bundle_root,
            require_project_identity=True,
            allow_local_checkout=allow_local_checkout,
        )
        if contract is None:
            return None
        verify_embedded_bundle_payload(contract)
    except BundleContractError as error:
        raise SystemExit(
            "[easy_run] The extracted GPU bundle failed runtime authentication: "
            f"{error}. Re-extract the archive and run verify-tree again."
        ) from error
    return EmbeddedBundlePolicy(
        config_path=contract.root / contract.config_path,
        raw_parallel_data_included=contract.raw_parallel_data_included,
        monolingual_corpus_included=bool(contract.records_for_origin("monolingual-corpus")),
        foundation_enabled=contract.foundation_enabled,
    )


def _existing_bundle_raw_files(data_dir: Path, policy: EmbeddedBundlePolicy) -> list[Path]:
    """List authenticated raw shards without rebuilding or modifying them."""

    if not policy.raw_parallel_data_included:
        return []
    raw_files = sorted(data_dir.glob("*.jsonl"))
    if not raw_files:
        raise SystemExit(
            "[easy_run] The verified bundle declares raw parallel data, but no "
            "data/*.jsonl payload is present. Re-extract and verify the archive."
        )
    return raw_files


def _report_embedded_bundle_policy(policy: EmbeddedBundlePolicy) -> None:
    if policy.raw_parallel_data_included:
        print(
            "[easy_run] The verified bundle includes raw parallel data. Existing "
            "payload files will be used without modifying them.",
            flush=True,
        )
    else:
        print(
            "[easy_run] The prepared-only bundle omits raw parallel data. "
            "sion-train will authenticate and reuse the prepared tokenizer and shards.",
            flush=True,
        )
    if not policy.foundation_enabled:
        print("[easy_run] Foundation training is disabled by the authenticated config.")
    elif policy.monolingual_corpus_included:
        print("[easy_run] The bundle includes the configured foundation source corpus.")
    else:
        print(
            "[easy_run] The foundation source corpus is omitted. sion-train will "
            "authenticate the prepared foundation generation before allocating the model."
        )


def _validate_gpu_runtime(torch_module) -> tuple[int, tuple[str, ...]]:
    """Fail before preprocessing when the CUDA runtime cannot launch training."""

    gpu_count = int(torch_module.cuda.device_count())
    if not torch_module.cuda.is_available() or gpu_count < 1:
        raise SystemExit(
            "No CUDA GPU was found. Check that PyTorch was installed with CUDA support."
        )
    if gpu_count > 1 and not torch_module.distributed.is_nccl_available():
        raise SystemExit(
            f"Found {gpu_count} CUDA GPUs, but this PyTorch build does not "
            "include NCCL support. Install a CUDA-enabled PyTorch package that "
            "supports multi-GPU training."
        )
    names = tuple(sorted({torch_module.cuda.get_device_name(index) for index in range(gpu_count)}))
    return gpu_count, names


def _validate_installed_dependency_runtime(
    torch_module,
    *,
    python_version: tuple[int, int] | None = None,
    python_implementation: str | None = None,
    operating_system: str | None = None,
    machine: str | None = None,
    libc_name: str | None = None,
    libc_version: str | None = None,
    distribution_version=None,
) -> dict[str, str]:
    """Require the live runtime target and versions to match the reviewed GPU lock."""

    resolved_python = python_version or (sys.version_info.major, sys.version_info.minor)
    resolved_implementation = (python_implementation or platform.python_implementation()).lower()
    resolved_system = (operating_system or platform.system()).lower()
    resolved_machine = (machine or platform.machine()).lower()
    detected_libc_name, detected_libc_version = platform.libc_ver()
    resolved_libc_name = (libc_name or detected_libc_name).lower()
    resolved_libc_version = libc_version or detected_libc_version
    version_reader = distribution_version or importlib_metadata.version
    actual_versions = {"torch": str(torch_module.__version__)}
    missing: list[str] = []
    for package in EXPECTED_RUNTIME_VERSIONS:
        if package == "torch":
            continue
        try:
            actual_versions[package] = str(version_reader(package))
        except importlib_metadata.PackageNotFoundError:
            missing.append(package)

    mismatches: list[str] = []
    expected_python = tuple(
        int(part) for part in str(EXPECTED_DEPENDENCY_TARGET["python_version"]).split(".")
    )
    if resolved_python != expected_python:
        mismatches.append(
            f"Python {resolved_python[0]}.{resolved_python[1]} is installed; "
            f"expected {expected_python[0]}.{expected_python[1]}"
        )
    if resolved_implementation != str(EXPECTED_DEPENDENCY_TARGET["python_implementation"]):
        mismatches.append(
            f"Python implementation {resolved_implementation!r} is installed; expected "
            f"{EXPECTED_DEPENDENCY_TARGET['python_implementation']!r}"
        )
    if resolved_system != str(EXPECTED_DEPENDENCY_TARGET["os"]):
        mismatches.append(
            f"operating system {resolved_system!r} is installed; "
            f"expected {EXPECTED_DEPENDENCY_TARGET['os']!r}"
        )
    normalized_machine = "x86_64" if resolved_machine == "amd64" else resolved_machine
    if normalized_machine != str(EXPECTED_DEPENDENCY_TARGET["machine"]):
        mismatches.append(
            f"machine {resolved_machine!r} is installed; "
            f"expected {EXPECTED_DEPENDENCY_TARGET['machine']!r}"
        )
    expected_manylinux = str(EXPECTED_DEPENDENCY_TARGET["manylinux"])
    try:
        required_glibc = tuple(int(part) for part in expected_manylinux.split("_", maxsplit=1))
        actual_glibc = tuple(int(part) for part in resolved_libc_version.split(".")[:2])
    except (TypeError, ValueError):
        actual_glibc = ()
        required_glibc = (2, 28)
    if resolved_libc_name != "glibc" or actual_glibc < required_glibc:
        mismatches.append(
            f"C library {resolved_libc_name or 'unknown'} {resolved_libc_version or 'unknown'} "
            f"is installed; expected glibc compatible with manylinux_{expected_manylinux} or newer"
        )
    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        actual = actual_versions.get(package)
        if actual is not None and actual != expected:
            mismatches.append(f"{package} {actual!r} is installed; expected {expected!r}")
    compiled_cuda = str(getattr(torch_module.version, "cuda", None))
    expected_cuda = str(EXPECTED_DEPENDENCY_TARGET["torch_backend"]).removeprefix("cu")
    expected_cuda = f"{int(expected_cuda) // 10}.{int(expected_cuda) % 10}"
    if compiled_cuda != expected_cuda:
        mismatches.append(
            f"PyTorch was compiled for CUDA {compiled_cuda!r}; expected {expected_cuda!r}"
        )
    if missing:
        mismatches.append("missing locked packages: " + ", ".join(sorted(missing)))
    if mismatches:
        details = "\n".join(f"  - {message}" for message in mismatches)
        raise SystemExit(
            "[easy_run] The live GPU runtime target or versions do not match the "
            "authenticated dependency contract:\n"
            f"{details}\n"
            "Recreate .venv with CPython 3.11 and run the authenticated uv pip sync "
            "command from the retraining runbook before allocating GPUs."
        )
    actual_versions["cuda"] = compiled_cuda
    return actual_versions


_CUDA_CANARY_REPORT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "device_index",
        "device_name",
        "compute_capability",
        "total_memory_bytes",
        "torch",
        "compiled_cuda",
        "bf16",
        "gqa_query_heads",
        "gqa_kv_heads",
        "head_dimension",
        "gradient_norm",
        "nccl_all_reduce",
        "peak_allocated_bytes",
        "elapsed_seconds",
    }
)


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_strict_json_report(line: str, *, role: str) -> dict[str, object]:
    """Decode one child report while rejecting JSON's optional NaN extensions."""

    try:
        value = json.loads(line, parse_constant=_reject_nonfinite_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"[easy_run] {role} returned no valid JSON report") from error
    if not isinstance(value, dict):
        raise SystemExit(f"[easy_run] {role} returned a non-object JSON report")
    return value


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_cuda_canary_report(
    report: dict[str, object],
    *,
    device_index: int,
) -> dict[str, object]:
    """Validate every field before a CUDA child is trusted as a success."""

    capability = report.get("compute_capability")
    device_name = report.get("device_name")
    total_memory = report.get("total_memory_bytes")
    gradient_norm = report.get("gradient_norm")
    peak_memory = report.get("peak_allocated_bytes")
    elapsed = report.get("elapsed_seconds")
    expected_cuda = str(EXPECTED_DEPENDENCY_TARGET["torch_backend"]).removeprefix("cu")
    expected_cuda = f"{int(expected_cuda) // 10}.{int(expected_cuda) % 10}"
    valid = (
        set(report) == _CUDA_CANARY_REPORT_FIELDS
        and report.get("schema") == "sion-cuda-canary-v1"
        and report.get("status") == "passed"
        and not isinstance(report.get("device_index"), bool)
        and report.get("device_index") == device_index
        and isinstance(device_name, str)
        and bool(device_name.strip())
        and len(device_name) <= 512
        and isinstance(capability, list)
        and len(capability) == 2
        and all(
            not isinstance(part, bool) and isinstance(part, int) and 0 <= part <= 99
            for part in capability
        )
        and capability[0] >= 8
        and not isinstance(total_memory, bool)
        and isinstance(total_memory, int)
        and total_memory > 0
        and report.get("torch") == EXPECTED_RUNTIME_VERSIONS["torch"]
        and report.get("compiled_cuda") == expected_cuda
        and report.get("bf16") is True
        and report.get("gqa_query_heads") == 12
        and report.get("gqa_kv_heads") == 6
        and report.get("head_dimension") == 72
        and _is_finite_number(gradient_norm)
        and float(gradient_norm) >= 0.0
        and isinstance(report.get("nccl_all_reduce"), bool)
        and not isinstance(peak_memory, bool)
        and isinstance(peak_memory, int)
        and peak_memory > 0
        and _is_finite_number(elapsed)
        and 0.0 < float(elapsed) <= CUDA_CANARY_TIMEOUT_SECONDS
    )
    if not valid:
        raise SystemExit(
            f"[easy_run] CUDA canary on device {device_index} returned an invalid success report"
        )
    return report


_DISTRIBUTED_NCCL_REPORT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "rank",
        "world_size",
        "device_index",
        "all_reduce_value",
        "torch",
        "compiled_cuda",
        "elapsed_seconds",
    }
)


def _validate_distributed_nccl_report(
    report: dict[str, object],
    *,
    gpu_count: int,
) -> dict[str, object]:
    """Authenticate one rank report from the all-GPU NCCL canary."""

    rank = report.get("rank")
    device_index = report.get("device_index")
    elapsed = report.get("elapsed_seconds")
    expected_sum = gpu_count * (gpu_count + 1) / 2
    expected_cuda = str(EXPECTED_DEPENDENCY_TARGET["torch_backend"]).removeprefix("cu")
    expected_cuda = f"{int(expected_cuda) // 10}.{int(expected_cuda) % 10}"
    valid = (
        set(report) == _DISTRIBUTED_NCCL_REPORT_FIELDS
        and report.get("schema") == "sion-distributed-nccl-canary-v1"
        and report.get("status") == "passed"
        and not isinstance(rank, bool)
        and isinstance(rank, int)
        and 0 <= rank < gpu_count
        and report.get("world_size") == gpu_count
        and not isinstance(device_index, bool)
        and isinstance(device_index, int)
        and device_index == rank
        and _is_finite_number(report.get("all_reduce_value"))
        and float(report["all_reduce_value"]) == expected_sum
        and report.get("torch") == EXPECTED_RUNTIME_VERSIONS["torch"]
        and report.get("compiled_cuda") == expected_cuda
        and _is_finite_number(elapsed)
        and 0.0 < float(elapsed) <= NCCL_CANARY_TIMEOUT_SECONDS
    )
    if not valid:
        raise SystemExit("[easy_run] The distributed NCCL canary returned an invalid report")
    return report


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Kill torchrun and every worker after a timeout or launcher interruption."""

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    if process.poll() is None:
        process.kill()


def _reap_killed_process(process: subprocess.Popen[str]) -> None:
    """Bound cleanup itself so a failed probe cannot pin the paid container."""

    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.kill()
        process.communicate()


def _run_multi_gpu_nccl_canary(
    gpu_count: int,
    env: dict[str, str],
) -> list[dict[str, object]]:
    """Run one bounded communicator across every visible GPU."""

    if gpu_count <= 1:
        return []
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--max-restarts=0",
        f"--nproc-per-node={gpu_count}",
        "-m",
        "sion_translate.gpu_runtime",
        "--distributed-nccl",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=NCCL_CANARY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        _kill_process_group(process)
        _reap_killed_process(process)
        raise SystemExit(
            f"[easy_run] The distributed NCCL canary timed out after "
            f"{NCCL_CANARY_TIMEOUT_SECONDS} seconds. Training was not started."
        ) from error
    except BaseException:
        _kill_process_group(process)
        _reap_killed_process(process)
        raise
    if process.returncode != 0:
        _kill_process_group(process)
        detail = (stderr or stdout or "no diagnostic output").strip()
        raise SystemExit(
            f"[easy_run] The distributed NCCL canary failed. Training was not started.\n{detail}"
        )

    reports: list[dict[str, object]] = []
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            report = _load_strict_json_report(candidate, role="distributed NCCL canary")
        except SystemExit:
            continue
        if report.get("schema") == "sion-distributed-nccl-canary-v1":
            reports.append(_validate_distributed_nccl_report(report, gpu_count=gpu_count))
    ranks = [report["rank"] for report in reports]
    if len(reports) != gpu_count or set(ranks) != set(range(gpu_count)):
        raise SystemExit(
            "[easy_run] The distributed NCCL canary did not return exactly one valid "
            "report from every GPU"
        )
    reports.sort(key=lambda report: int(report["rank"]))
    print(
        f"[easy_run] Distributed NCCL canary passed across {gpu_count} GPUs.",
        flush=True,
    )
    return reports


def _run_one_cuda_kernel_canary(
    device_index: int,
    env: dict[str, str],
) -> dict[str, object]:
    """Run and authenticate one isolated CUDA child result."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "sion_translate.gpu_runtime",
        "--device-index",
        str(device_index),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CUDA_CANARY_TIMEOUT_SECONDS,
            start_new_session=os.name != "nt",
        )
    except subprocess.TimeoutExpired as error:
        raise SystemExit(
            f"[easy_run] CUDA canary timed out on device {device_index} after "
            f"{CUDA_CANARY_TIMEOUT_SECONDS} seconds. Training was not started."
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise SystemExit(
            f"[easy_run] CUDA canary failed on device {device_index}. Training was not "
            f"started.\n{detail}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        report_line = lines[-1]
    except IndexError as error:
        raise SystemExit(
            f"[easy_run] CUDA canary on device {device_index} returned no valid JSON report"
        ) from error
    report = _load_strict_json_report(
        report_line,
        role=f"CUDA canary on device {device_index}",
    )
    return _validate_cuda_canary_report(report, device_index=device_index)


def _run_cuda_kernel_canaries(gpu_count: int, env: dict[str, str]) -> list[dict[str, object]]:
    """Probe every GPU concurrently so the whole server waits at most one timeout."""

    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        reports = list(
            executor.map(
                lambda device_index: _run_one_cuda_kernel_canary(device_index, env),
                range(gpu_count),
            )
        )
    for device_index, report in enumerate(reports):
        print(
            f"[easy_run] CUDA canary passed on device {device_index}: "
            f"{report.get('device_name', 'unknown GPU')}; "
            f"{float(report.get('elapsed_seconds', 0.0)):.2f}s; "
            f"peak {int(report.get('peak_allocated_bytes', 0)) / 2**20:.1f} MiB.",
            flush=True,
        )
    _run_multi_gpu_nccl_canary(gpu_count, env)
    return reports


def _report_foundation_corpus(config_path: Path) -> None:
    """Report which monolingual corpora will and will not enter training.

    Skipping foundation pretraining is a normal path because the stage runs only
    when its input directories are present. A silent skip could otherwise make a
    user wait for days while assuming that a multi-gigabyte corpus was included.
    This report therefore prints the reason and all excluded inputs first.

    Missing languages do not stop training because adding them later is a valid
    workflow. Set ``foundation.require_all_languages`` to require every language.
    """

    from sion_translate.config import load_config
    from sion_translate.foundation import plan_foundation_stage

    plan = plan_foundation_stage(load_config(config_path))
    print("[easy_run] Checking foundation (monolingual pretraining) corpora.", flush=True)
    for line in plan.report:
        print(f"[easy_run]   {line}", flush=True)
    if not plan.enabled:
        print(f"[easy_run] Skipping the foundation stage: {plan.reason}", flush=True)
        return
    for warning in plan.warnings:
        print(f"[easy_run] [warning] {warning}", flush=True)
    print(
        f"[easy_run] Running the foundation stage (languages: "
        f"{', '.join(plan.languages)}). Outputs are stored in runs/*/foundation/ "
        "separately from the translation model.",
        flush=True,
    )


def _check_shard_keys(env: dict[str, str]) -> None:
    """Refuse to start when a shard's keys match no configured language pair.

    Such a shard yields zero sentences and says nothing about it, so the loss is
    invisible until someone counts. One shard in this corpus shipped with the
    keys 한국어/일본어 and would have dropped 10,075 rows in silence.
    """

    checker = ROOT / "scripts" / "data" / "check_shard_keys.py"
    if not checker.exists():
        return
    print(
        "[easy_run] Checking whether shard keys match the configured language pairs.",
        flush=True,
    )
    result = subprocess.run(
        [sys.executable, str(checker)],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "[easy_run] The shard above cannot be read with the configured "
            "language pairs and would be silently excluded from training.\n"
            "           Correct the JSONL key names or update "
            "sion_translate.yaml data.language_pairs, then run again."
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
        print("[easy_run] sentencepiece is unavailable; skipping tokenizer validation.")
        return
    if not tokenizer_model.exists():
        print(f"[easy_run] Tokenizer not found; skipping validation: {tokenizer_model}")
        return

    import json
    from collections import Counter

    processor = spm.SentencePieceProcessor()
    processor.Load(str(tokenizer_model))

    shards = sorted(data_dir.glob("*.jsonl"))
    if not shards:
        print("[easy_run] No corpus was found; skipping tokenizer validation.")
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
        print("[easy_run] No tokens were produced from the sample; skipping validation.")
        return

    rate = fallback_tokens / total_tokens
    print(
        f"[easy_run] Tokenizer validation: vocab {processor.vocab_size():,}; "
        f"{fallback_tokens:,} byte-fallback tokens among {total_tokens:,} "
        f"sample tokens ({rate:.4%}).",
        flush=True,
    )
    for character, count in offenders.most_common(10):
        print(f"           U+{ord(character):04X} {character!r}  {count:,} occurrences")
    if rate > max_fallback_rate:
        raise SystemExit(
            f"[easy_run] The byte-fallback rate {rate:.4%} exceeds the allowed "
            f"maximum {max_fallback_rate:.4%}. Training is stopping.\n"
            f"           Remove {DEFAULT_ARTIFACT_ROOT}/tokenizer and run again, "
            "or run sion-train-tokenizer directly with a lower "
            "--required-character-min-occurrences value."
        )
    print("[easy_run] The byte-fallback rate is within the allowed limit. Continuing.", flush=True)


def main() -> None:
    # Hash the complete immutable package before importing the heavy GPU
    # runtime, creating a terminal session, or allowing the launcher to select
    # a data/artifact layout.
    allow_local_checkout = _parse_allow_local_checkout()
    bundle_policy = _embedded_bundle_policy(allow_local_checkout=allow_local_checkout)
    if bundle_policy is not None and allow_local_checkout:
        raise SystemExit(
            f"[easy_run] {LOCAL_CHECKOUT_FLAG} cannot be used with an authenticated GPU "
            "bundle. Run the verified bundle without the local-checkout option."
        )
    _enter_tmux(allow_local_checkout=allow_local_checkout)

    import torch

    gpu_count, gpu_names = _validate_gpu_runtime(torch)
    installed_versions = _validate_installed_dependency_runtime(torch)
    print(
        "[easy_run] Verified GPU runtime compatibility: "
        + ", ".join(f"{name}={version}" for name, version in sorted(installed_versions.items())),
        flush=True,
    )

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    _run_cuda_kernel_canaries(gpu_count, env)

    source_data = ROOT / "data"
    local_checkout_arguments = (
        [LOCAL_CHECKOUT_FLAG] if allow_local_checkout and bundle_policy is None else []
    )
    remove_generated_config = False
    if bundle_policy is not None:
        # A bundle's config hash and canonical paths are authenticated. Moving
        # its artifacts to RAM would require a new config, which the training
        # preflight correctly rejects. Keep this mode immutable and in place.
        raw_files = _existing_bundle_raw_files(source_data, bundle_policy)
        ram = None
        runtime_data = source_data
        runtime_artifacts = PERSISTENT_ARTIFACTS
        generated_config = bundle_policy.config_path
        print("[easy_run] Running the verified bundle in its authenticated layout.")
        _report_embedded_bundle_policy(bundle_policy)
    else:
        raw_files = _discover_raw_files(source_data, env)
        required = sum(path.stat().st_size for path in raw_files)
        active_artifact_size = sum(
            _directory_size(PERSISTENT_ARTIFACTS / name) for name in PREPARED_ARTIFACT_DIRECTORIES
        )
        required += max(active_artifact_size, required * 3)
        ram = _ram_workspace(required)
        runtime_artifacts = _runtime_artifact_directory(ram)
        if ram is None:
            runtime_data = source_data
            print("[easy_run] Running from persistent storage without a RAM disk.")
        else:
            runtime_data = ram / "data"
            print(f"[easy_run] Using RAM disk: {ram}")
            print(f"[easy_run] Copying {len(raw_files)} source data files to RAM.")
            shutil.rmtree(runtime_data, ignore_errors=True)
            _copy_files_parallel(source_data, runtime_data)
            # The /dev/shm workspace has a stable checkout-derived name and can
            # survive a crashed run. Never inherit an orphaned cache implicitly.
            shutil.rmtree(runtime_artifacts, ignore_errors=True)
            if PERSISTENT_ARTIFACTS.exists():
                _restore_active_artifacts(PERSISTENT_ARTIFACTS, runtime_artifacts)
        generated_config = _generated_config(runtime_data, runtime_artifacts)
        remove_generated_config = True

    # Before anything expensive: a shard the pipeline cannot read is worth
    # catching now rather than after hours of training.
    if bundle_policy is None:
        _check_shard_keys(env)
        _report_foundation_corpus(generated_config)
    try:
        # Complete preprocessing on one rank so torchrun workers do not wait at
        # a barrier for an extended period or hit a communication timeout.
        _run(
            [
                sys.executable,
                "-m",
                "sion_translate.cli.train",
                *local_checkout_arguments,
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
                "[easy_run] Preserving tokenizer and prepared datasets on persistent "
                f"storage at {DEFAULT_ARTIFACT_ROOT}/."
            )
            for name in PREPARED_ARTIFACT_DIRECTORIES:
                _atomic_sync_directory(
                    runtime_artifacts / name,
                    PERSISTENT_ARTIFACTS / name,
                )

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpu_count}",
            "-m",
            "sion_translate.cli.train",
            *local_checkout_arguments,
            "--config",
            str(generated_config),
        ]
        print(f"[easy_run] Starting training on {gpu_count} CUDA GPUs ({', '.join(gpu_names)}).")
        _run(command, env)
    finally:
        if remove_generated_config:
            generated_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
