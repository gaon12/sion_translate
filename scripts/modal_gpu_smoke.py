"""Run one short, budget-gated production GPU smoke test on Modal.

This script intentionally has no "all" mode. One invocation starts one selected
hardware target, and the caller must provide an authorization threshold that covers the
script's conservative two-attempt compute contingency. Modal does not expose an
account-level hard spending cap to this script.
"""

# Pyright cannot statically describe a dynamically supplied Torch module or
# Modal's heterogeneous decorator keyword mapping. Runtime contracts below
# validate both objects before any paid GPU work proceeds.
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportCallIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = PurePosixPath("/opt/sion")
LOCK_RELATIVE_PATH = Path("requirements/pylock.gpu-cp311-linux-x86_64-cu128.toml")
UV_BOOTSTRAP_RELATIVE_PATH = Path("requirements/modal-bootstrap.txt")
NATIVE_BUILD_RELATIVE_PATH = Path("scripts/build_sentencepiece_native.py")
NATIVE_REQUIREMENTS_RELATIVE_PATH = Path("requirements/sentencepiece-native-build.txt")
NATIVE_MANIFEST_PATH = REMOTE_ROOT / "native/manifest.json"
NATIVE_CORE_COMMIT = "31646a467d2051eb904e0b45de3a73e91fe1c1e3"
NATIVE_CORE_TREE = "a256eb7f5d3e634041fa11aa2cbb4b1de065359b"
LOCK_SHA256 = "0820c94d97a424e7c051cec1e01bba452a038904ae0df4730849fdabe50f350f"
FUNCTION_TIMEOUT_SECONDS = 300
STARTUP_TIMEOUT_SECONDS = 180
CHILD_TIMEOUT_SECONDS = 150
PARENT_CLEANUP_MARGIN_SECONDS = 20
MINIMUM_CHILD_RUNTIME_SECONDS = 30
SCALEDOWN_WINDOW_SECONDS = 2
PLATFORM_RECOVERY_ATTEMPTS = 2
CPU_CORES = 4.0
MEMORY_GIB = 32
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
EXPECTED_PARENT_PID_ENVIRONMENT = "SION_EXPECTED_GUARDIAN_PID"
INHERITED_SIGNAL_MASK_ENVIRONMENT = "SION_INHERITED_SIGNAL_MASK"
EXPECTED_MODAL_CLIENT_VERSION = "1.5.3"
APP_NAME = "sion-budget-gated-gpu-smoke"
RESULT_VOLUME_NAME = "sion-gpu-smoke-results"
RESULT_MOUNT = PurePosixPath("/sion-results")
JOURNAL_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^smoke-[a-z0-9][a-z0-9-]{7,79}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FUNCTION_CALL_ID_PATTERN = re.compile(r"^fc-[A-Za-z0-9_-]{8,128}$")
REMOTE_FUNCTION_NAMES = {
    "a100-40gb": "a100_40gb",
    "a100-80gb": "a100_80gb",
    "h100": "h100_exact",
    "a100-40gb-x2": "a100_40gb_x2",
}

# Modal's public per-second resource prices on 2026-08-31. The CLI uses these only
# as a conservative spending guard and labels every result as an estimate.
TARGETS: dict[str, dict[str, Any]] = {
    "a100-40gb": {
        "gpu": "A100-40GB",
        "gpu_count": 1,
        "usd_per_second": 0.000583,
    },
    "a100-80gb": {
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "usd_per_second": 0.000694,
    },
    "h100": {
        # The exclamation mark prevents Modal from substituting an H200.
        "gpu": "H100!",
        "gpu_count": 1,
        "usd_per_second": 0.001097,
    },
    "a100-40gb-x2": {
        "gpu": "A100-40GB:2",
        "gpu_count": 2,
        "usd_per_second": 0.000583,
    },
}
EXPECTED_RUNTIME_VERSIONS = {
    "numpy": "2.4.6",
    "nvidia-cudnn-cu12": "9.10.2.21",
    "nvidia-nccl-cu12": "2.27.5",
    "sentencepiece": "0.2.1",
    "torch": "2.10.0+cu128",
    "torchao": "0.17.0+cu128",
    "transformers": "5.16.1",
}
EXPECTED_CUDNN_VERSION = 91_002
EXPECTED_NCCL_VERSION = (2, 27, 5)
EXPECTED_REFINEMENT_GRADIENT_COUNT = 6
PRODUCTION_PARAMETER_COUNT = 287_127_073
DISTRIBUTED_CANARY_PARAMETER_COUNT = 222_553
OPTIMIZER_MOMENT_NAMES = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq"})
DISTRIBUTED_RANK_REPORT_KEYS = {
    "rank",
    "world_size",
    "backend",
    "norm",
    "loss",
    "parameter_count",
    "candidate_refinement_steps",
    "candidate_refinement_loss",
    "candidate_refinement_gain",
    "candidate_refinement_token_nll_gain",
    "candidate_refinement_gradient_count",
    "candidate_refinement_update",
    "optimizer_state_count",
    "checkpoint_step",
    "checkpoint_roundtrip",
    "ema_roundtrip",
    "optimizer_roundtrip",
    "scheduler_roundtrip",
    "rng_roundtrip",
    "device_name",
    "peak_allocated_gib",
    "peak_reserved_gib",
}


def _compute_usd_per_second(target: str) -> float:
    specification = TARGETS[target]
    return (
        float(specification["usd_per_second"]) * int(specification["gpu_count"])
        + CPU_CORES * CPU_USD_PER_CORE_SECOND
        + MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND
    )


def authorization_compute_charge(target: str) -> float:
    """Budget two complete startup/function compute windows for one target.

    Modal may recover infrastructure failures independently of application
    retries. This covers configured GPU, CPU, and memory only; it is an
    authorization threshold, not an account spending cap.
    """

    return (
        (STARTUP_TIMEOUT_SECONDS + FUNCTION_TIMEOUT_SECONDS + SCALEDOWN_WINDOW_SECONDS)
        * _compute_usd_per_second(target)
        * PLATFORM_RECOVERY_ATTEMPTS
    )


def estimated_function_compute_charge(target: str, elapsed_seconds: float) -> float:
    """Estimate one observed function body plus the configured idle window.

    This includes the provisioned GPU, CPU, and memory. Startup, image build,
    storage, network, and timeout overrun remain outside the observed estimate.
    The caller must enforce a workspace budget and inspect billing between
    targets.
    """

    if not _is_finite_number(elapsed_seconds) or float(elapsed_seconds) < 0.0:
        raise ValueError("observed elapsed time must be a finite non-negative number")
    billed_seconds = max(1, math.ceil(float(elapsed_seconds))) + SCALEDOWN_WINDOW_SECONDS
    return billed_seconds * _compute_usd_per_second(target)


def validate_cost_guard(target: str, max_dollars: float) -> float:
    """Require authorization for the configured two-attempt contingency."""

    if isinstance(max_dollars, bool) or not math.isfinite(max_dollars) or max_dollars <= 0:
        raise ValueError("--max-dollars must be a finite positive number")
    authorized = authorization_compute_charge(target)
    if authorized > max_dollars:
        raise ValueError(
            f"{target} requires a ${authorized:.4f} two-attempt compute contingency, "
            f"which exceeds the supplied ${max_dollars:.4f} authorization"
        )
    return authorized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one budget-gated Sion GPU smoke test on Modal."
    )
    parser.add_argument("--target", required=True, choices=tuple(TARGETS))
    parser.add_argument(
        "--max-dollars",
        required=True,
        type=float,
        help="Required compute-cost authorization threshold for this invocation.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_regular_file_sha256(path: Path) -> tuple[int, str]:
    """Hash one regular file and reject identity changes during the read."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"GPU smoke contract path is not a regular file: {path}")
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"GPU smoke contract path changed while hashing: {path}")
    return after.st_size, digest


def _verify_executed_entrypoint(executed_path: Path, reviewed_path: Path) -> None:
    """Bind Modal's mounted executable source to the reviewed image copy."""

    executed_size, executed_sha256 = _stable_regular_file_sha256(executed_path)
    reviewed_size, reviewed_sha256 = _stable_regular_file_sha256(reviewed_path)
    if (executed_size, executed_sha256) != (reviewed_size, reviewed_sha256):
        raise RuntimeError(
            "Modal's executed GPU smoke entrypoint differs from its reviewed image copy"
        )


def gpu_smoke_contract_sha256(root: Path) -> str:
    """Hash every reviewed byte copied into the paid GPU smoke image."""

    resolved_root = root.resolve()
    relative_paths = [
        LOCK_RELATIVE_PATH,
        UV_BOOTSTRAP_RELATIVE_PATH,
        NATIVE_BUILD_RELATIVE_PATH,
        NATIVE_REQUIREMENTS_RELATIVE_PATH,
        Path("scripts/modal_gpu_smoke.py"),
    ]
    source_root = resolved_root / "src"
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError(f"GPU smoke source root is not a regular directory: {source_root}")
    relative_paths.extend(
        sorted(
            (path.relative_to(resolved_root) for path in source_root.rglob("*.py")),
            key=lambda path: path.as_posix(),
        )
    )
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("GPU smoke contract contains duplicate paths")
    contract = hashlib.sha256()
    for relative_path in relative_paths:
        path = resolved_root / relative_path
        size, digest = _stable_regular_file_sha256(path)
        contract.update(f"{relative_path.as_posix()}\0{size}\0{digest}\n".encode("utf-8"))
    return contract.hexdigest()


def _validated_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("durable Modal run ID is invalid")
    return run_id


def _validated_contract_sha256(value: object) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("durable Modal contract SHA-256 is invalid")
    return value


def _validated_function_call_id(value: object) -> str:
    if not isinstance(value, str) or FUNCTION_CALL_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("durable Modal FunctionCall ID is invalid")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    """Publish one finite JSON document without exposing a partial file."""

    _validate_json_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                value,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class _DurableRunJournal:
    """Commit immutable progress events and a current status to one run directory."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        target: str,
        function_call_id: str,
        max_dollars: float,
        expected_contract_sha256: str,
        commit: Callable[[], None],
    ) -> None:
        self.run_id = _validated_run_id(run_id)
        if target not in TARGETS:
            raise ValueError(f"unsupported Modal GPU smoke target: {target}")
        self.target = target
        self.function_call_id = _validated_function_call_id(function_call_id)
        self.max_dollars = max_dollars
        self.expected_contract_sha256 = _validated_contract_sha256(expected_contract_sha256)
        self._commit = commit
        self._sequence = 0
        self._completed_phases: list[str] = []
        self._run_root = root / "runs" / self.run_id

    def _base_status(self, state: str) -> dict[str, Any]:
        return {
            "journal_version": JOURNAL_VERSION,
            "run_id": self.run_id,
            "target": self.target,
            "function_call_id": self.function_call_id,
            "state": state,
            "sequence": self._sequence,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "completed_phases": list(self._completed_phases),
            "max_dollars": self.max_dollars,
            "authorization_compute_charge_usd": authorization_compute_charge(self.target),
            "expected_contract_sha256": self.expected_contract_sha256,
        }

    def _publish(self, state: str, event: str, **details: object) -> None:
        self._sequence += 1
        status = self._base_status(state)
        status["event"] = event
        if details:
            status["details"] = details
        event_path = self._run_root / "events" / f"{self._sequence:04d}-{event}.json"
        _write_json_atomic(event_path, status)
        _write_json_atomic(self._run_root / "status.json", status)
        self._commit()

    def started(self) -> None:
        (self._run_root / "result.json").unlink(missing_ok=True)
        (self._run_root / "failure.json").unlink(missing_ok=True)
        self._publish("running", "started")

    def contract_verified(self, observed_contract_sha256: str) -> None:
        self._publish(
            "running",
            "contract-verified",
            observed_contract_sha256=_validated_contract_sha256(observed_contract_sha256),
        )

    def phase_completed(self, phase: str) -> None:
        if phase in self._completed_phases:
            raise RuntimeError(f"durable Modal phase was reported twice: {phase}")
        self._completed_phases.append(phase)
        self._publish("running", "phase-completed", phase=phase)

    def passed(self, result: dict[str, Any]) -> None:
        _write_json_atomic(self._run_root / "result.json", result)
        self._publish("passed", "finished", result_path="result.json")

    def failed(self, error: BaseException) -> None:
        (self._run_root / "result.json").unlink(missing_ok=True)
        failure = {
            "error_type": type(error).__name__,
            "message": str(error)[:4_000],
            "traceback_tail": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-16_000:],
        }
        _write_json_atomic(self._run_root / "failure.json", failure)
        self._publish("failed", "failed", failure_path="failure.json", **failure)


def _production_config(target: str):
    from sion_translate.config import ExperimentalConfig, ModelConfig

    return ModelConfig(
        vocab_size=48_000,
        d_model=864,
        encoder_layers=18,
        decoder_layers=9,
        num_heads=12,
        num_kv_heads=6,
        d_ff=2_304,
        max_seq_len=2_048,
        dropout=0.1,
        # Auto-configuration enables activation checkpointing below 70 GiB.
        gradient_checkpointing=target in {"a100-40gb", "a100-40gb-x2"},
        experimental=ExperimentalConfig(
            candidate_refinement_enabled=True,
            candidate_refinement_steps=1,
            candidate_refinement_temperature=1.0,
            candidate_refinement_loss_weight=0.25,
            candidate_refinement_vocab_chunk_size=2_048,
        ),
    )


def _distributed_smoke_training_state_template() -> dict[str, bool]:
    """Provide the key DCP needs in its caller-supplied load template."""

    return {"modal_distributed_smoke": False}


def _single_gpu_smoke_training_state_template() -> dict[str, bool]:
    """Provide the single-process progress key before checkpoint restore."""

    return {"smoke_complete": False}


def _assert_finite(torch_module, tensor, label: str) -> None:
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label} contains NaN or infinity")


def _mean_refinement_token_nll_gain(torch_module, token_gain, labels, label: str) -> float:
    """Validate the token-shaped refinement diagnostic and return its target mean."""

    if token_gain.shape != labels.shape:
        raise RuntimeError(
            f"{label} token NLL gain shape {tuple(token_gain.shape)} does not match "
            f"labels {tuple(labels.shape)}"
        )
    target_mask = labels.ne(-100)
    if not bool(target_mask.any().item()):
        raise RuntimeError(f"{label} has no target tokens for refinement evidence")
    detached_gain = token_gain.detach()
    target_gain = detached_gain[target_mask]
    _assert_finite(torch_module, target_gain, f"{label} target-token NLL gain")
    if bool(torch_module.count_nonzero(detached_gain[~target_mask]).item()):
        raise RuntimeError(f"{label} token NLL gain is nonzero outside target tokens")
    return float(target_gain.float().mean().item())


def _seed_smoke_rngs(torch_module, seed: int) -> None:
    """Seed every RNG family persisted by the production checkpoint."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def _rng_fingerprint(torch_module) -> str:
    """Return a stable digest of the current Python, NumPy, CPU, and CUDA RNGs."""

    import random

    import numpy as np

    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(numpy_state[2:]).encode("ascii"))
    digest.update(torch_module.get_rng_state().detach().cpu().numpy().tobytes())
    digest.update(torch_module.cuda.get_rng_state().detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _first_optimizer_state_tensor(torch_module, optimizer) -> tuple[str, Any]:
    """Return one materialized, non-scalar Adam moment as a detached local copy."""

    for state_index, state in enumerate(optimizer.state.values()):
        for state_name, value in state.items():
            if state_name not in OPTIMIZER_MOMENT_NAMES or not torch_module.is_tensor(value):
                continue
            to_local = getattr(value, "to_local", None)
            local_value = to_local() if callable(to_local) else value
            if local_value.numel() <= 1:
                continue
            return f"{state_index}:{state_name}", local_value.detach().clone()
    raise RuntimeError("optimizer created no non-scalar Adam moment to verify")


def _corrupt_first_optimizer_state_tensor(torch_module, optimizer) -> tuple[str, Any]:
    """Change one live non-scalar Adam moment so a no-op restore cannot pass."""

    for state_index, state in enumerate(optimizer.state.values()):
        for state_name, value in state.items():
            if state_name not in OPTIMIZER_MOMENT_NAMES or not torch_module.is_tensor(value):
                continue
            to_local = getattr(value, "to_local", None)
            local_value = to_local() if callable(to_local) else value
            if local_value.numel() <= 1:
                continue
            with torch_module.no_grad():
                local_value.add_(1)
            return f"{state_index}:{state_name}", local_value.detach().clone()
    raise RuntimeError("optimizer created no non-scalar Adam moment to corrupt")


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _native_sentencepiece_report() -> dict[str, Any]:
    """Verify the declared native overlay separately from the stock base lock."""
    helper = Path(str(REMOTE_ROOT / NATIVE_BUILD_RELATIVE_PATH))
    spec = importlib.util.spec_from_file_location("sion_native_binding_evidence", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError("the native SentencePiece verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _validate_native_sentencepiece_evidence(
        module.verify_installed(Path(str(NATIVE_MANIFEST_PATH)))
    )


def _validate_native_sentencepiece_evidence(value: object) -> dict[str, Any]:
    hash_fields = {
        "wrapper_sha256",
        "proxy_sha256",
        "wheel_sha256",
        "installed_extension_sha256",
        "manifest_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != hash_fields | {"source_commit", "source_tree", "swig_version"}
        or value.get("source_commit") != NATIVE_CORE_COMMIT
        or value.get("source_tree") != NATIVE_CORE_TREE
        or value.get("swig_version") != "4.4.0"
        or any(
            not isinstance(value.get(field), str) or SHA256_PATTERN.fullmatch(value[field]) is None
            for field in hash_fields
        )
    ):
        raise RuntimeError("Modal GPU smoke returned invalid native SentencePiece evidence")
    return value


def _runtime_report(torch_module) -> dict[str, Any]:
    observed_versions = {
        package: importlib.metadata.version(package) for package in EXPECTED_RUNTIME_VERSIONS
    }
    if observed_versions != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(
            f"installed versions differ from the authenticated GPU lock: {observed_versions}"
        )
    if str(torch_module.__version__) != EXPECTED_RUNTIME_VERSIONS["torch"]:
        raise RuntimeError("imported Torch version differs from installed package metadata")
    if torch_module.version.cuda != "12.8":
        raise RuntimeError(f"expected CUDA 12.8, got {torch_module.version.cuda!r}")
    cudnn_version = torch_module.backends.cudnn.version()
    if cudnn_version != EXPECTED_CUDNN_VERSION:
        raise RuntimeError(f"expected cuDNN {EXPECTED_CUDNN_VERSION}, got {cudnn_version!r}")
    if not torch_module.distributed.is_nccl_available():
        raise RuntimeError("the authenticated GPU environment requires NCCL")
    raw_nccl_version = torch_module.cuda.nccl.version()
    if isinstance(raw_nccl_version, int):
        nccl_version = (
            raw_nccl_version // 10_000,
            (raw_nccl_version % 10_000) // 100,
            raw_nccl_version % 100,
        )
    else:
        nccl_version = tuple(int(part) for part in raw_nccl_version)
    if nccl_version != EXPECTED_NCCL_VERSION:
        raise RuntimeError(f"expected NCCL {EXPECTED_NCCL_VERSION}, got {nccl_version!r}")
    if platform.python_implementation() != "CPython" or platform.python_version_tuple()[:2] != (
        "3",
        "11",
    ):
        raise RuntimeError("the authenticated GPU environment requires CPython 3.11")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("the authenticated GPU environment requires Linux x86-64")
    libc_name, libc_version = platform.libc_ver()
    try:
        glibc_version = tuple(int(part) for part in libc_version.split(".")[:2])
    except ValueError as error:
        raise RuntimeError(
            f"could not parse the Linux C library version: {libc_version!r}"
        ) from error
    if libc_name.lower() != "glibc" or glibc_version < (2, 28):
        raise RuntimeError(
            f"the authenticated GPU environment requires glibc 2.28 or newer; "
            f"got {libc_name} {libc_version}"
        )

    lock = Path(str(REMOTE_ROOT / LOCK_RELATIVE_PATH))
    if not lock.is_file() or lock.is_symlink() or _sha256(lock) != LOCK_SHA256:
        raise RuntimeError("the GPU dependency lock is missing or has changed")
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable")

    devices = []
    for index in range(torch_module.cuda.device_count()):
        properties = torch_module.cuda.get_device_properties(index)
        with torch_module.cuda.device(index):
            native_bf16 = torch_module.cuda.is_bf16_supported(including_emulation=False)
        if not native_bf16:
            raise RuntimeError(f"CUDA device {index} has no native BF16 support")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory_gib": properties.total_memory / 2**30,
                "bf16_native": True,
            }
        )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "libc": {"name": libc_name, "version": libc_version},
        "versions": observed_versions,
        "cuda_runtime": torch_module.version.cuda,
        "cudnn": cudnn_version,
        "nccl_available": True,
        "nccl_version": list(nccl_version),
        "devices": devices,
        "lock_sha256": LOCK_SHA256,
        "native_sentencepiece": _native_sentencepiece_report(),
    }


def _validate_target_hardware(target: str, runtime: dict[str, Any]) -> None:
    devices = runtime["devices"]
    expected_count = int(TARGETS[target]["gpu_count"])
    if len(devices) != expected_count:
        raise RuntimeError(f"expected {expected_count} visible GPU(s), got {len(devices)}")
    for device in devices:
        name = str(device["name"]).upper()
        capability = tuple(device["capability"])
        memory = float(device["total_memory_gib"])
        if target.startswith("a100"):
            if "A100" not in name or "H100" in name or "H200" in name or capability != (8, 0):
                raise RuntimeError(f"requested A100 but received {name} {capability}")
            minimum = 70.0 if target == "a100-80gb" else 35.0
            maximum = 50.0 if target != "a100-80gb" else 90.0
            if not minimum <= memory <= maximum:
                raise RuntimeError(f"unexpected A100 memory capacity: {memory:.1f} GiB")
        elif (
            "H100" not in name
            or "H200" in name
            or "A100" in name
            or capability != (9, 0)
            or not 70.0 <= memory <= 90.0
        ):
            raise RuntimeError(
                f"requested exact 80 GB H100 but received {name} {capability} with {memory:.1f} GiB"
            )


def _attention_optimizer_canary(torch_module, device) -> dict[str, Any]:
    from sion_translate.model.layers import GQAAttention, RotaryEmbedding
    from sion_translate.training.trainer import build_optimizer_param_groups

    started = time.perf_counter()
    torch_module.manual_seed(17)
    torch_module.cuda.manual_seed_all(17)
    left = torch_module.randn((64, 864), device=device, dtype=torch_module.bfloat16)
    right = torch_module.randn((864, 2_304), device=device, dtype=torch_module.bfloat16)
    product = left @ right
    _assert_finite(torch_module, product, "BF16 matrix multiplication")

    rope = RotaryEmbedding(72, 32).to(device)
    attention = GQAAttention(
        864,
        12,
        6,
        dropout=0.0,
        qk_norm=True,
        norm_eps=1e-6,
        rope=rope,
    ).to(device)
    attention.train()
    hidden = torch_module.randn((2, 12, 864), device=device, requires_grad=True)
    padding_mask = torch_module.tensor([[True] * 12, [True] * 9 + [False] * 3], device=device)
    with torch_module.autocast("cuda", dtype=torch_module.bfloat16):
        output = attention(hidden, key_padding_mask=padding_mask, is_causal=True)
        loss = output.float().square().mean()
    _assert_finite(torch_module, output, "masked GQA SDPA output")
    loss.backward()
    for name, parameter in attention.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"masked GQA parameter {name} has no gradient")
        _assert_finite(torch_module, parameter.grad, f"masked GQA gradient {name}")
    before = attention.q_proj.weight.detach().clone()
    optimizer = torch_module.optim.AdamW(
        build_optimizer_param_groups(attention, 0.01),
        lr=1e-4,
        weight_decay=0.0,
        fused=True,
    )
    optimizer.step()
    if torch_module.equal(before, attention.q_proj.weight):
        raise RuntimeError("fused AdamW did not update the attention parameters")
    for state in optimizer.state.values():
        for value in state.values():
            if torch_module.is_tensor(value):
                _assert_finite(torch_module, value, "fused AdamW state")
    torch_module.cuda.synchronize(device)
    return {"elapsed_seconds": time.perf_counter() - started, "loss": float(loss.item())}


def _production_model_canary(torch_module, device, target: str) -> dict[str, Any]:
    from sion_translate.config import TrainingConfig
    from sion_translate.model import SionForConditionalGeneration
    from sion_translate.training.checkpoint import load_checkpoint, save_checkpoint
    from sion_translate.training.distributed import DistributedContext
    from sion_translate.training.ema import EMAWeights
    from sion_translate.training.trainer import build_optimizer_param_groups, cosine_scheduler

    started = time.perf_counter()
    torch_module.manual_seed(20260710)
    torch_module.cuda.manual_seed_all(20260710)
    with torch_module.device("meta"):
        model = SionForConditionalGeneration(_production_config(target), pad_id=0)
    parameter_count = model.parameter_count()
    if parameter_count != PRODUCTION_PARAMETER_COUNT:
        raise RuntimeError(f"production parameter count changed: {parameter_count:,}")
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    training = TrainingConfig()
    optimizer = torch_module.optim.AdamW(
        build_optimizer_param_groups(model, training.weight_decay),
        lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_eps,
        weight_decay=0.0,
        fused=True,
    )
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps=1,
        max_steps=2,
        min_ratio=training.min_learning_rate_ratio,
    )
    ema = EMAWeights(model, training.ema_decay)
    batch = {
        "input_ids": torch_module.tensor([[4, 5, 6, 3]], device=device),
        "attention_mask": torch_module.tensor([[True, True, True, True]], device=device),
        "decoder_input_ids": torch_module.tensor([[2, 7, 8, 9]], device=device),
        "labels": torch_module.tensor([[7, 8, 9, 3]], device=device),
    }
    with torch_module.autocast("cuda", dtype=torch_module.bfloat16):
        output = model(**batch, reasoning_level=9)
    if output.loss is None or output.candidate_refinement_steps is None:
        raise RuntimeError("production forward omitted loss or refinement diagnostics")
    if int(output.candidate_refinement_steps.item()) != 1:
        raise RuntimeError("production training did not execute T1-to-T2 refinement")
    _assert_finite(torch_module, output.loss, "production loss")
    refinement_diagnostics = {
        "loss": output.candidate_refinement_loss,
        "gain": output.candidate_refinement_gain,
        "token_nll_gain": output.candidate_refinement_token_nll_gain,
    }
    for name, value in refinement_diagnostics.items():
        if value is None:
            raise RuntimeError(f"production training omitted candidate-refinement {name}")
        _assert_finite(torch_module, value, f"candidate-refinement {name}")
    token_nll_gain_mean = _mean_refinement_token_nll_gain(
        torch_module,
        output.candidate_refinement_token_nll_gain,
        batch["labels"],
        "production candidate refinement",
    )
    if not math.isclose(
        token_nll_gain_mean,
        float(output.candidate_refinement_gain.item()),
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("production aggregate and token-level refinement gains disagree")
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("production model produced no gradients")
    for gradient in gradients:
        _assert_finite(torch_module, gradient, "production gradient")
    if model.candidate_refinement is None:
        raise RuntimeError("production model omitted the candidate-refinement module")
    refinement_gradients = {
        name: parameter.grad for name, parameter in model.candidate_refinement.named_parameters()
    }
    if not refinement_gradients or any(
        gradient is None for gradient in refinement_gradients.values()
    ):
        raise RuntimeError("candidate-refinement parameters did not all receive gradients")
    for name, gradient in refinement_gradients.items():
        assert gradient is not None
        _assert_finite(torch_module, gradient, f"candidate-refinement gradient {name}")
    scale_gradient = model.candidate_refinement.refinement_scale.grad
    assert scale_gradient is not None
    if int(torch_module.count_nonzero(scale_gradient).item()) == 0:
        raise RuntimeError("candidate-refinement scale received a zero training gradient")
    grad_norm = torch_module.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
    _assert_finite(torch_module, grad_norm, "production gradient norm")
    first_ema_name, first_ema_before = next(iter(ema.shadow.items()))
    first_ema_before = first_ema_before.detach().clone()
    refinement_scale_before = model.candidate_refinement.refinement_scale.detach().clone()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    optimizer.zero_grad(set_to_none=True)
    if torch_module.equal(first_ema_before, ema.shadow[first_ema_name]):
        raise RuntimeError("production EMA did not follow the optimizer update")
    refinement_scale = float(model.candidate_refinement.refinement_scale.item())
    refinement_update = not torch_module.equal(
        refinement_scale_before,
        model.candidate_refinement.refinement_scale,
    )
    if not math.isfinite(refinement_scale) or not refinement_update:
        raise RuntimeError("fused AdamW did not update the candidate-refinement scale")

    model.eval()
    inference_batch = {name: value for name, value in batch.items() if name != "labels"}
    cached_refinement_calls = 0
    assert model.candidate_refinement is not None

    def count_refinement_call(_module, _inputs, _output) -> None:
        nonlocal cached_refinement_calls
        cached_refinement_calls += 1

    refinement_hook = model.candidate_refinement.register_forward_hook(count_refinement_call)
    try:
        with torch_module.no_grad(), torch_module.autocast("cuda", dtype=torch_module.bfloat16):
            disabled = model(**inference_batch, reasoning_level=0)
            enabled = model(**inference_batch, reasoning_level=9)
            calls_before_generation = cached_refinement_calls
            generated = model.generate(
                batch["input_ids"],
                batch["attention_mask"],
                bos_id=2,
                eos_id=3,
                max_new_tokens=3,
                reasoning_level=9,
            )
    finally:
        refinement_hook.remove()
    if int(disabled.candidate_refinement_steps.item()) != 0:
        raise RuntimeError("reasoning level zero did not disable refinement")
    if int(enabled.candidate_refinement_steps.item()) != 1:
        raise RuntimeError("inference did not execute T1-to-T2 refinement")
    if cached_refinement_calls <= calls_before_generation:
        raise RuntimeError("cached generation did not execute T1-to-T2 refinement")
    _assert_finite(torch_module, enabled.logits, "refined inference logits")
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise RuntimeError("cached generation returned an invalid shape")

    context = DistributedContext(0, 0, 1, device, False)
    with tempfile.TemporaryDirectory(prefix="sion-gpu-smoke-") as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        _seed_smoke_rngs(torch_module, 20260711)
        save_started = time.perf_counter()
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            1,
            context,
            training_state={"smoke_complete": True},
            ema=ema,
        )
        save_elapsed = time.perf_counter() - save_started
        saved_rng_fingerprint = _rng_fingerprint(torch_module)
        saved_parameter = next(model.parameters()).detach().clone()
        saved_ema = ema.shadow[first_ema_name].detach().clone()
        saved_optimizer_name, saved_optimizer_tensor = _first_optimizer_state_tensor(
            torch_module, optimizer
        )
        saved_scheduler_state = deepcopy(scheduler.state_dict())
        with torch_module.no_grad():
            next(model.parameters()).zero_()
            ema.shadow[first_ema_name].zero_()
        if torch_module.equal(saved_parameter, next(model.parameters())):
            raise RuntimeError("model corruption probe did not change live weights")
        if torch_module.equal(saved_ema, ema.shadow[first_ema_name]):
            raise RuntimeError("EMA corruption probe did not change live weights")
        corrupted_optimizer_name, corrupted_optimizer_tensor = (
            _corrupt_first_optimizer_state_tensor(torch_module, optimizer)
        )
        if corrupted_optimizer_name != saved_optimizer_name or torch_module.equal(
            corrupted_optimizer_tensor, saved_optimizer_tensor
        ):
            raise RuntimeError("optimizer corruption probe did not change live state")
        corrupted_scheduler_state = deepcopy(saved_scheduler_state)
        corrupted_scheduler_state["last_epoch"] = int(corrupted_scheduler_state["last_epoch"]) + 17
        scheduler.load_state_dict(corrupted_scheduler_state)
        if scheduler.state_dict() == saved_scheduler_state:
            raise RuntimeError("scheduler corruption probe did not change live state")
        _seed_smoke_rngs(torch_module, 20260712)
        if _rng_fingerprint(torch_module) == saved_rng_fingerprint:
            raise RuntimeError("RNG corruption probe did not change live state")
        restored_state: dict[str, Any] = _single_gpu_smoke_training_state_template()
        load_started = time.perf_counter()
        restored_step = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            context,
            training_state=restored_state,
            ema=ema,
        )
        load_elapsed = time.perf_counter() - load_started
        if restored_step != 1 or restored_state.get("smoke_complete") is not True:
            raise RuntimeError("checkpoint progress did not round-trip")
        if not torch_module.equal(saved_parameter, next(model.parameters())):
            raise RuntimeError("checkpoint weights did not round-trip")
        if not torch_module.equal(saved_ema, ema.shadow[first_ema_name]):
            raise RuntimeError("checkpoint EMA weights did not round-trip")
        restored_optimizer_name, restored_optimizer_tensor = _first_optimizer_state_tensor(
            torch_module, optimizer
        )
        if restored_optimizer_name != saved_optimizer_name or not torch_module.equal(
            saved_optimizer_tensor, restored_optimizer_tensor
        ):
            raise RuntimeError("checkpoint optimizer state did not round-trip")
        if scheduler.state_dict() != saved_scheduler_state:
            raise RuntimeError("checkpoint scheduler state did not round-trip")
        if _rng_fingerprint(torch_module) != saved_rng_fingerprint:
            raise RuntimeError("checkpoint RNG state did not round-trip")
        checkpoint_bytes = sum(
            path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
        )
    torch_module.cuda.synchronize(device)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count,
        "dropout": model.config.dropout,
        "gradient_checkpointing": model.config.gradient_checkpointing,
        "loss": float(output.loss.item()),
        "gradient_norm": float(grad_norm.item()),
        "candidate_refinement_loss": float(refinement_diagnostics["loss"].item()),
        "candidate_refinement_steps": int(output.candidate_refinement_steps.item()),
        "candidate_refinement_gain": float(refinement_diagnostics["gain"].item()),
        "candidate_refinement_token_nll_gain": token_nll_gain_mean,
        "candidate_refinement_scale_gradient": float(scale_gradient.item()),
        "candidate_refinement_scale": refinement_scale,
        "candidate_refinement_gradient_count": len(refinement_gradients),
        "candidate_refinement_update": refinement_update,
        "cached_refinement_calls": cached_refinement_calls - calls_before_generation,
        "generated_shape": list(generated.shape),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_save_seconds": save_elapsed,
        "checkpoint_load_seconds": load_elapsed,
        "ema_decay": ema.decay,
        "ema_roundtrip": True,
        "optimizer_roundtrip": True,
        "scheduler_roundtrip": True,
        "rng_roundtrip": True,
    }


def _distributed_child() -> int:
    import torch

    from sion_translate.config import ExperimentalConfig, ModelConfig, TrainingConfig
    from sion_translate.model import SionForConditionalGeneration
    from sion_translate.training.checkpoint import load_checkpoint, save_checkpoint
    from sion_translate.training.distributed import (
        cleanup_distributed,
        initialize_distributed,
        parallelize_model,
    )
    from sion_translate.training.ema import EMAWeights
    from sion_translate.training.trainer import (
        _normalize_and_clip_finite_gradients,
        _preflight_optimizer_step_inputs,
        build_optimizer_param_groups,
        cosine_scheduler,
    )

    context = initialize_distributed()
    try:
        if context.world_size != 2 or context.backend != "nccl":
            raise RuntimeError("distributed smoke requires exactly two NCCL ranks")
        torch.manual_seed(20260710)
        torch.cuda.manual_seed_all(20260710)
        with torch.device("meta"):
            model = SionForConditionalGeneration(
                ModelConfig(
                    vocab_size=128,
                    d_model=72,
                    encoder_layers=2,
                    decoder_layers=1,
                    num_heads=6,
                    num_kv_heads=3,
                    d_ff=192,
                    max_seq_len=32,
                    dropout=0.1,
                    gradient_checkpointing=True,
                    experimental=ExperimentalConfig(candidate_refinement_enabled=True),
                )
            )
        parameter_count = model.parameter_count()
        if parameter_count != DISTRIBUTED_CANARY_PARAMETER_COUNT:
            raise RuntimeError(f"distributed canary parameter count changed: {parameter_count:,}")
        model = parallelize_model(
            model,
            context,
            strategy="fsdp2",
            precision="bf16",
            reduce_dtype="auto",
            reshard_after_forward=True,
            materialize_meta=True,
        )
        torch.manual_seed(20260710 + context.rank)
        torch.cuda.manual_seed_all(20260710 + context.rank)
        training = TrainingConfig()
        optimizer = torch.optim.AdamW(
            build_optimizer_param_groups(model, training.weight_decay),
            lr=training.learning_rate,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_eps,
            weight_decay=0.0,
            fused=True,
        )
        scheduler = cosine_scheduler(
            optimizer,
            warmup_steps=1,
            max_steps=2,
            min_ratio=training.min_learning_rate_ratio,
        )
        ema = EMAWeights(model, training.ema_decay)
        input_ids = torch.tensor([[4, 5, 3]], device=context.device)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        decoder = torch.tensor([[2, 6, 7]], device=context.device)
        labels = torch.tensor([[6, 7, 3]], device=context.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, mask, decoder, labels, reasoning_level=9)
        if output.loss is None or output.candidate_refinement_steps is None:
            raise RuntimeError("FSDP2 smoke omitted loss or refinement diagnostics")
        if int(output.candidate_refinement_steps.item()) != 1:
            raise RuntimeError("FSDP2 training did not execute T1-to-T2 refinement")
        refinement_diagnostics = {
            "loss": output.candidate_refinement_loss,
            "gain": output.candidate_refinement_gain,
            "token_nll_gain": output.candidate_refinement_token_nll_gain,
        }
        for name, value in refinement_diagnostics.items():
            if value is None:
                raise RuntimeError(f"FSDP2 training omitted candidate-refinement {name}")
            _assert_finite(torch, value, f"FSDP2 candidate-refinement {name}")
        token_nll_gain_mean = _mean_refinement_token_nll_gain(
            torch,
            output.candidate_refinement_token_nll_gain,
            labels,
            "FSDP2 candidate refinement",
        )
        if not math.isclose(
            token_nll_gain_mean,
            float(output.candidate_refinement_gain.item()),
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise RuntimeError("FSDP2 aggregate and token-level refinement gains disagree")
        output.loss.backward()
        refinement_gradients = {
            name: parameter.grad
            for name, parameter in model.candidate_refinement.named_parameters()
        }
        if len(refinement_gradients) != EXPECTED_REFINEMENT_GRADIENT_COUNT or any(
            gradient is None for gradient in refinement_gradients.values()
        ):
            raise RuntimeError(
                "FSDP2 candidate-refinement parameters did not all receive gradients"
            )
        for name, gradient in refinement_gradients.items():
            assert gradient is not None
            to_local = getattr(gradient, "to_local", None)
            local_gradient = to_local() if callable(to_local) else gradient
            _assert_finite(torch, local_gradient, f"FSDP2 candidate-refinement gradient {name}")
        scale_parameter = model.candidate_refinement.refinement_scale
        scale_gradient = scale_parameter.grad
        assert scale_gradient is not None
        to_local = getattr(scale_gradient, "to_local", None)
        local_scale_gradient = to_local() if callable(to_local) else scale_gradient
        nonzero_scale_gradients = torch.count_nonzero(local_scale_gradient).to(
            device=context.device,
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(nonzero_scale_gradients, op=torch.distributed.ReduceOp.SUM)
        if int(nonzero_scale_gradients.item()) == 0:
            raise RuntimeError("FSDP2 candidate-refinement scale received a zero gradient")
        to_local = getattr(scale_parameter, "to_local", None)
        scale_before_update = (
            to_local().detach().clone() if callable(to_local) else scale_parameter.detach().clone()
        )
        parameters, normalizer = _preflight_optimizer_step_inputs(
            model.parameters(),
            accumulated_loss=output.loss,
            accumulated_local_normalizer=torch.ones((), device=context.device, dtype=torch.float64),
            context=context,
            stage_name="modal-fsdp2-smoke",
            next_step=1,
        )
        norm = _normalize_and_clip_finite_gradients(
            parameters,
            global_normalizer=normalizer,
            max_norm=training.grad_clip,
            context=context,
            stage_name="modal-fsdp2-smoke",
            next_step=1,
        )
        tracked_parameter = next(model.parameters())
        first_ema_name, first_ema_before = next(iter(ema.shadow.items()))
        first_ema_before = first_ema_before.detach().clone()
        local_tensor = getattr(tracked_parameter, "to_local", None)
        before_update = (
            local_tensor().detach().clone()
            if callable(local_tensor)
            else tracked_parameter.detach().clone()
        )
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        to_local = getattr(scale_parameter, "to_local", None)
        scale_after_update = to_local().detach() if callable(to_local) else scale_parameter.detach()
        refinement_update = torch.tensor(
            int(not torch.equal(scale_before_update, scale_after_update)),
            device=context.device,
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(refinement_update, op=torch.distributed.ReduceOp.SUM)
        if int(refinement_update.item()) == 0:
            raise RuntimeError("FSDP2 optimizer did not update candidate refinement")
        local_tensor = getattr(tracked_parameter, "to_local", None)
        after_update = (
            local_tensor().detach().clone()
            if callable(local_tensor)
            else tracked_parameter.detach().clone()
        )
        if torch.equal(before_update, after_update):
            raise RuntimeError("distributed fused AdamW did not update the model")
        first_ema_after = ema.shadow[first_ema_name]
        if torch.equal(first_ema_before, first_ema_after):
            raise RuntimeError("distributed EMA did not follow the optimizer update")
        if not optimizer.state:
            raise RuntimeError("distributed fused AdamW created no optimizer state")
        for state in optimizer.state.values():
            for value in state.values():
                if not torch.is_tensor(value):
                    continue
                to_local = getattr(value, "to_local", None)
                local_value = to_local() if callable(to_local) else value
                if not bool(torch.isfinite(local_value).all().item()):
                    raise RuntimeError("distributed fused AdamW state is non-finite")

        checkpoint_root = os.environ.get("SION_MODAL_CHECKPOINT_DIR")
        if not checkpoint_root:
            raise RuntimeError("distributed checkpoint directory was not provided")
        checkpoint = Path(checkpoint_root) / "checkpoint"
        _seed_smoke_rngs(torch, 20260711 + context.rank)
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            1,
            context,
            training_state={"modal_distributed_smoke": True},
            ema=ema,
        )
        saved_rng_fingerprint = _rng_fingerprint(torch)
        local_tensor = getattr(tracked_parameter, "to_local", None)
        saved_parameter = (
            local_tensor().detach().clone()
            if callable(local_tensor)
            else tracked_parameter.detach().clone()
        )
        saved_ema = ema.shadow[first_ema_name].detach().clone()
        saved_optimizer_name, saved_optimizer_tensor = _first_optimizer_state_tensor(
            torch, optimizer
        )
        saved_scheduler_state = deepcopy(scheduler.state_dict())
        with torch.no_grad():
            tracked_parameter.zero_()
            ema.shadow[first_ema_name].zero_()
        local_tensor = getattr(tracked_parameter, "to_local", None)
        corrupted_parameter = (
            local_tensor().detach() if callable(local_tensor) else tracked_parameter.detach()
        )
        if torch.equal(saved_parameter, corrupted_parameter):
            raise RuntimeError("distributed model corruption probe did not change weights")
        if torch.equal(saved_ema, ema.shadow[first_ema_name]):
            raise RuntimeError("distributed EMA corruption probe did not change weights")
        corrupted_optimizer_name, corrupted_optimizer_tensor = (
            _corrupt_first_optimizer_state_tensor(torch, optimizer)
        )
        if corrupted_optimizer_name != saved_optimizer_name or torch.equal(
            corrupted_optimizer_tensor, saved_optimizer_tensor
        ):
            raise RuntimeError("distributed optimizer corruption probe did not change state")
        corrupted_scheduler_state = deepcopy(saved_scheduler_state)
        corrupted_scheduler_state["last_epoch"] = int(corrupted_scheduler_state["last_epoch"]) + 17
        scheduler.load_state_dict(corrupted_scheduler_state)
        if scheduler.state_dict() == saved_scheduler_state:
            raise RuntimeError("distributed scheduler corruption probe did not change state")
        _seed_smoke_rngs(torch, 20260721 + context.rank)
        if _rng_fingerprint(torch) == saved_rng_fingerprint:
            raise RuntimeError("distributed RNG corruption probe did not change state")
        restored_state: dict[str, Any] = _distributed_smoke_training_state_template()
        restored_step = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            context,
            training_state=restored_state,
            ema=ema,
        )
        local_tensor = getattr(tracked_parameter, "to_local", None)
        restored_parameter = (
            local_tensor().detach() if callable(local_tensor) else tracked_parameter.detach()
        )
        if restored_step != 1 or restored_state.get("modal_distributed_smoke") is not True:
            raise RuntimeError("distributed checkpoint progress did not round-trip")
        if not torch.equal(saved_parameter, restored_parameter):
            raise RuntimeError("distributed checkpoint weights did not round-trip")
        if not torch.equal(saved_ema, ema.shadow[first_ema_name]):
            raise RuntimeError("distributed checkpoint EMA weights did not round-trip")
        restored_optimizer_name, restored_optimizer_tensor = _first_optimizer_state_tensor(
            torch, optimizer
        )
        if restored_optimizer_name != saved_optimizer_name or not torch.equal(
            saved_optimizer_tensor, restored_optimizer_tensor
        ):
            raise RuntimeError("distributed checkpoint optimizer state did not round-trip")
        if scheduler.state_dict() != saved_scheduler_state:
            raise RuntimeError("distributed checkpoint scheduler state did not round-trip")
        if _rng_fingerprint(torch) != saved_rng_fingerprint:
            raise RuntimeError("distributed checkpoint RNG state did not round-trip")
        torch.cuda.synchronize(context.device)
        properties = torch.cuda.get_device_properties(context.device)
        report = {
            "rank": context.rank,
            "world_size": context.world_size,
            "backend": context.backend,
            "norm": float(norm.item()),
            "loss": float(output.loss.item()),
            "parameter_count": parameter_count,
            "candidate_refinement_steps": int(output.candidate_refinement_steps.item()),
            "candidate_refinement_loss": float(refinement_diagnostics["loss"].item()),
            "candidate_refinement_gain": float(refinement_diagnostics["gain"].item()),
            "candidate_refinement_token_nll_gain": token_nll_gain_mean,
            "candidate_refinement_gradient_count": len(refinement_gradients),
            "candidate_refinement_update": True,
            "optimizer_state_count": len(optimizer.state),
            "checkpoint_step": restored_step,
            "checkpoint_roundtrip": True,
            "ema_roundtrip": True,
            "optimizer_roundtrip": True,
            "scheduler_roundtrip": True,
            "rng_roundtrip": True,
            "device_name": properties.name,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / 2**30,
        }
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        return 0
    finally:
        cleanup_distributed(context)


def _terminate_child_group(process: subprocess.Popen[str]) -> None:
    """Kill and finitely reap torchrun plus every NCCL worker it created."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as final_error:
            raise RuntimeError(
                "could not reap the terminated distributed smoke workers"
            ) from final_error
        raise RuntimeError(
            "distributed smoke workers required a second forced termination"
        ) from error


def _block_guarded_child_signals() -> set[signal.Signals] | None:
    """Block shutdown signals across spawn until the Linux child guard is armed."""

    if sys.platform != "linux" or not hasattr(signal, "pthread_sigmask"):
        return None
    termination_signals = {
        candidate
        for name in ("SIGHUP", "SIGQUIT", "SIGINT", "SIGTERM")
        if isinstance((candidate := getattr(signal, name, None)), signal.Signals)
    }
    return set(signal.pthread_sigmask(signal.SIG_BLOCK, termination_signals))


def _restore_guarded_child_signals(previous: set[signal.Signals] | None) -> None:
    if previous is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _validated_distributed_reports(value: object) -> dict[str, Any]:
    """Require exactly two complete rank reports and derive their aggregate."""

    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("distributed smoke did not emit one success report from each rank")
    reports: list[dict[str, Any]] = []
    for candidate in value:
        if not isinstance(candidate, dict) or set(candidate) != DISTRIBUTED_RANK_REPORT_KEYS:
            raise RuntimeError("distributed smoke emitted an unexpected rank-report schema")
        reports.append(candidate)
    if any(type(report.get("rank")) is not int for report in reports) or {
        int(report["rank"]) for report in reports
    } != {0, 1}:
        raise RuntimeError("distributed smoke did not emit one success report from each rank")
    for report in reports:
        finite_nonnegative = (
            "norm",
            "loss",
            "candidate_refinement_loss",
            "peak_allocated_gib",
            "peak_reserved_gib",
        )
        finite_signed = (
            "candidate_refinement_gain",
            "candidate_refinement_token_nll_gain",
        )
        if any(
            not _is_finite_number(report.get(name))
            for name in (*finite_nonnegative, *finite_signed)
        ):
            raise RuntimeError(f"distributed smoke emitted non-finite metrics: {report!r}")
        if (
            report.get("world_size") != 2
            or type(report.get("world_size")) is not int
            or report.get("backend") != "nccl"
            or any(float(report[name]) < 0.0 for name in finite_nonnegative)
            or float(report["norm"]) == 0.0
            or float(report["loss"]) == 0.0
            or float(report["candidate_refinement_loss"]) == 0.0
            or float(report["peak_allocated_gib"]) == 0.0
            or float(report["peak_reserved_gib"]) == 0.0
            or float(report["peak_allocated_gib"])
            < DISTRIBUTED_CANARY_PARAMETER_COUNT * 4 / 2 / 2**30
            or type(report.get("parameter_count")) is not int
            or report.get("parameter_count") != DISTRIBUTED_CANARY_PARAMETER_COUNT
            or type(report.get("candidate_refinement_steps")) is not int
            or report.get("candidate_refinement_steps") != 1
            or type(report.get("candidate_refinement_gradient_count")) is not int
            or report.get("candidate_refinement_gradient_count")
            != EXPECTED_REFINEMENT_GRADIENT_COUNT
            or report.get("candidate_refinement_update") is not True
            or not math.isclose(
                float(report["candidate_refinement_gain"]),
                float(report["candidate_refinement_token_nll_gain"]),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            or float(report["peak_reserved_gib"]) < float(report["peak_allocated_gib"])
            or type(report.get("optimizer_state_count")) is not int
            or int(report["optimizer_state_count"]) <= 0
            or type(report.get("checkpoint_step")) is not int
            or report.get("checkpoint_step") != 1
            or report.get("checkpoint_roundtrip") is not True
            or report.get("ema_roundtrip") is not True
            or report.get("optimizer_roundtrip") is not True
            or report.get("scheduler_roundtrip") is not True
            or report.get("rng_roundtrip") is not True
            or not isinstance(report.get("device_name"), str)
            or not str(report["device_name"]).strip()
            or "A100" not in str(report["device_name"]).upper()
        ):
            raise RuntimeError(f"distributed smoke emitted an invalid success report: {report!r}")
    ordered = sorted(reports, key=lambda report: int(report["rank"]))
    return {
        "ranks": ordered,
        "max_peak_allocated_gib": max(float(report["peak_allocated_gib"]) for report in ordered),
        "max_peak_reserved_gib": max(float(report["peak_reserved_gib"]) for report in ordered),
    }


def _validated_distributed_report(stdout: str) -> dict[str, Any]:
    """Parse torchrun output and require complete evidence from both ranks."""

    reports: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(
                candidate,
                parse_constant=_reject_nonfinite_json,
            )
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict) and set(value) == DISTRIBUTED_RANK_REPORT_KEYS:
            reports.append(value)
    return _validated_distributed_reports(reports)


def _two_gpu_canary(timeout_seconds: float) -> dict[str, Any]:
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds < MINIMUM_CHILD_RUNTIME_SECONDS
        or timeout_seconds > CHILD_TIMEOUT_SECONDS
    ):
        raise ValueError("distributed child timeout is outside the configured safe range")
    started = time.perf_counter()
    command = [
        sys.executable,
        "-u",
        "-m",
        "sion_translate.process_guard",
        "launcher",
        "torch.distributed.run",
        "--",
        "--standalone",
        "--max-restarts=0",
        "--nproc-per-node=2",
        "--module",
        "sion_translate.process_guard",
        "worker",
        "scripts.modal_gpu_smoke",
        "--",
        "--distributed-child",
    ]
    with tempfile.TemporaryDirectory(prefix="sion-modal-distributed-") as temporary:
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT))),
            "SION_DISTRIBUTED_TIMEOUT_SECONDS": "60",
            "SION_MODAL_CHECKPOINT_DIR": temporary,
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ENABLE_MONITORING": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "60",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
            EXPECTED_PARENT_PID_ENVIRONMENT: str(os.getpid()),
        }
        process: subprocess.Popen[str] | None = None
        try:
            inherited_signal_mask = _block_guarded_child_signals()
            try:
                if inherited_signal_mask is None:
                    environment.pop(INHERITED_SIGNAL_MASK_ENVIRONMENT, None)
                else:
                    environment[INHERITED_SIGNAL_MASK_ENVIRONMENT] = ",".join(
                        str(int(signum)) for signum in sorted(inherited_signal_mask, key=int)
                    )
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                    env=environment,
                )
            finally:
                _restore_guarded_child_signals(inherited_signal_mask)
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    f"distributed NCCL/FSDP2 smoke exceeded {timeout_seconds:g} seconds"
                ) from error
            if process.returncode != 0:
                detail = (stderr or stdout or "no child diagnostic").strip()[-4_000:]
                raise RuntimeError(f"distributed NCCL/FSDP2 smoke failed:\n{detail}")
        except BaseException:
            if process is not None:
                _terminate_child_group(process)
            raise
        return {
            "elapsed_seconds": time.perf_counter() - started,
            "report": _validated_distributed_report(stdout),
            "stdout_tail": stdout[-4_000:],
            "stderr_tail": stderr[-4_000:],
        }


def _remaining_child_timeout(elapsed_seconds: float) -> float:
    """Reserve enough parent time to kill and reap every torchrun worker."""

    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ValueError("parent elapsed time must be finite and non-negative")
    remaining = FUNCTION_TIMEOUT_SECONDS - elapsed_seconds - PARENT_CLEANUP_MARGIN_SECONDS
    timeout_seconds = min(float(CHILD_TIMEOUT_SECONDS), remaining)
    if timeout_seconds < MINIMUM_CHILD_RUNTIME_SECONDS:
        raise TimeoutError(
            "not enough parent execution time remains for the distributed smoke and cleanup"
        )
    return timeout_seconds


def _validate_json_value(value: Any, *, path: str = "result") -> None:
    """Reject values that Modal could serialize ambiguously or non-portably."""

    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{path} is not finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            if not isinstance(name, str):
                raise RuntimeError(f"{path} contains a non-string key")
            _validate_json_value(item, path=f"{path}.{name}")
        return
    raise RuntimeError(f"{path} contains unsupported value type {type(value).__name__}")


def _version_tuple(value: object, *, components: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise RuntimeError(f"Modal GPU smoke returned an invalid {label}")
    parts = value.split(".")
    if len(parts) != components or any(not part.isdigit() for part in parts):
        raise RuntimeError(f"Modal GPU smoke returned an invalid {label}")
    return tuple(int(part) for part in parts)


def _validate_runtime_evidence(target: str, value: object) -> dict[str, Any]:
    required = {
        "python",
        "python_implementation",
        "libc",
        "versions",
        "cuda_runtime",
        "cudnn",
        "nccl_available",
        "nccl_version",
        "devices",
        "lock_sha256",
        "native_sentencepiece",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("Modal GPU smoke returned an unexpected runtime schema")
    runtime = value
    _validate_native_sentencepiece_evidence(runtime.get("native_sentencepiece"))
    if (
        _version_tuple(runtime.get("python"), components=3, label="Python version")[:2] != (3, 11)
        or runtime.get("python_implementation") != "CPython"
        or runtime.get("versions") != EXPECTED_RUNTIME_VERSIONS
        or runtime.get("cuda_runtime") != "12.8"
        or type(runtime.get("cudnn")) is not int
        or runtime.get("cudnn") != EXPECTED_CUDNN_VERSION
        or runtime.get("nccl_available") is not True
        or runtime.get("nccl_version") != list(EXPECTED_NCCL_VERSION)
        or runtime.get("lock_sha256") != LOCK_SHA256
    ):
        raise RuntimeError("Modal GPU smoke returned an unauthenticated runtime envelope")
    libc = runtime.get("libc")
    if (
        not isinstance(libc, dict)
        or set(libc) != {"name", "version"}
        or libc.get("name") != "glibc"
        or _version_tuple(libc.get("version"), components=2, label="glibc version") < (2, 28)
    ):
        raise RuntimeError("Modal GPU smoke returned an unsupported C library")
    devices = runtime.get("devices")
    expected_count = int(TARGETS[target]["gpu_count"])
    if not isinstance(devices, list) or len(devices) != expected_count:
        raise RuntimeError("Modal GPU smoke returned an invalid visible-device inventory")
    required_device_keys = {
        "index",
        "name",
        "capability",
        "total_memory_gib",
        "bf16_native",
    }
    for expected_index, device in enumerate(devices):
        if not isinstance(device, dict) or set(device) != required_device_keys:
            raise RuntimeError("Modal GPU smoke returned an unexpected device schema")
        capability = device.get("capability")
        if (
            type(device.get("index")) is not int
            or device.get("index") != expected_index
            or not isinstance(device.get("name"), str)
            or not str(device["name"]).strip()
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(type(part) is not int for part in capability)
            or not _is_finite_number(device.get("total_memory_gib"))
            or float(device["total_memory_gib"]) <= 0.0
            or device.get("bf16_native") is not True
        ):
            raise RuntimeError("Modal GPU smoke returned invalid device evidence")
    _validate_target_hardware(target, runtime)
    return runtime


def _validate_attention_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"elapsed_seconds", "loss"}:
        raise RuntimeError("Modal GPU smoke returned invalid attention evidence")
    if (
        not _is_finite_number(value.get("elapsed_seconds"))
        or float(value["elapsed_seconds"]) <= 0.0
        or not _is_finite_number(value.get("loss"))
        or float(value["loss"]) < 0.0
    ):
        raise RuntimeError("Modal GPU smoke returned invalid attention evidence")
    return value


def _validate_production_evidence(target: str, value: object) -> dict[str, Any]:
    required = {
        "elapsed_seconds",
        "parameter_count",
        "dropout",
        "gradient_checkpointing",
        "loss",
        "gradient_norm",
        "candidate_refinement_loss",
        "candidate_refinement_steps",
        "candidate_refinement_gain",
        "candidate_refinement_token_nll_gain",
        "candidate_refinement_scale_gradient",
        "candidate_refinement_scale",
        "candidate_refinement_gradient_count",
        "candidate_refinement_update",
        "cached_refinement_calls",
        "generated_shape",
        "checkpoint_bytes",
        "checkpoint_save_seconds",
        "checkpoint_load_seconds",
        "ema_decay",
        "ema_roundtrip",
        "optimizer_roundtrip",
        "scheduler_roundtrip",
        "rng_roundtrip",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("Modal GPU smoke returned invalid production-model evidence")
    finite_positive = (
        "loss",
        "candidate_refinement_loss",
        "checkpoint_save_seconds",
        "checkpoint_load_seconds",
    )
    finite_signed = (
        "candidate_refinement_gain",
        "candidate_refinement_token_nll_gain",
        "candidate_refinement_scale_gradient",
        "candidate_refinement_scale",
    )
    if (
        not _is_finite_number(value.get("elapsed_seconds"))
        or float(value["elapsed_seconds"]) <= 0.0
        or not _is_finite_number(value.get("gradient_norm"))
        or float(value["gradient_norm"]) <= 0.0
        or any(not _is_finite_number(value.get(name)) for name in finite_positive)
        or any(float(value[name]) <= 0.0 for name in finite_positive)
        or any(not _is_finite_number(value.get(name)) for name in finite_signed)
        or float(value["candidate_refinement_scale_gradient"]) == 0.0
        or float(value["candidate_refinement_scale"]) == 0.0
        or type(value.get("parameter_count")) is not int
        or value.get("parameter_count") != PRODUCTION_PARAMETER_COUNT
        or value.get("dropout") != 0.1
        or value.get("gradient_checkpointing") is not (target in {"a100-40gb", "a100-40gb-x2"})
        or type(value.get("cached_refinement_calls")) is not int
        or int(value["cached_refinement_calls"]) <= 0
        or type(value.get("candidate_refinement_gradient_count")) is not int
        or value.get("candidate_refinement_gradient_count") != EXPECTED_REFINEMENT_GRADIENT_COUNT
        or type(value.get("candidate_refinement_steps")) is not int
        or value.get("candidate_refinement_steps") != 1
        or value.get("candidate_refinement_update") is not True
        or not math.isclose(
            float(value["candidate_refinement_gain"]),
            float(value["candidate_refinement_token_nll_gain"]),
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
        or type(value.get("checkpoint_bytes")) is not int
        or int(value["checkpoint_bytes"]) < PRODUCTION_PARAMETER_COUNT * 4
        or not isinstance(value.get("generated_shape"), list)
        or len(value["generated_shape"]) != 2
        or any(type(part) is not int for part in value["generated_shape"])
        or value["generated_shape"][0] != 1
        or not 2 <= value["generated_shape"][1] <= 4
        or value.get("ema_decay") != 0.999
        or value.get("ema_roundtrip") is not True
        or value.get("optimizer_roundtrip") is not True
        or value.get("scheduler_roundtrip") is not True
        or value.get("rng_roundtrip") is not True
    ):
        raise RuntimeError("Modal GPU smoke returned invalid production-model evidence")
    return value


def _validate_distributed_phase(
    value: object,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"elapsed_seconds", "report", "stdout_tail", "stderr_tail"}
        or not _is_finite_number(value.get("elapsed_seconds"))
        or float(value["elapsed_seconds"]) <= 0.0
        or not isinstance(value.get("stdout_tail"), str)
        or len(value["stdout_tail"]) > 4_000
        or not isinstance(value.get("stderr_tail"), str)
        or len(value["stderr_tail"]) > 4_000
    ):
        raise RuntimeError("Modal GPU smoke returned invalid distributed phase evidence")
    report = value.get("report")
    if not isinstance(report, dict) or set(report) != {
        "ranks",
        "max_peak_allocated_gib",
        "max_peak_reserved_gib",
    }:
        raise RuntimeError("Modal GPU smoke returned invalid distributed phase evidence")
    validated_report = _validated_distributed_reports(report.get("ranks"))
    if report != validated_report:
        raise RuntimeError("Modal GPU smoke returned inconsistent distributed aggregates")
    devices = runtime["devices"]
    for rank_report in validated_report["ranks"]:
        device = devices[int(rank_report["rank"])]
        if rank_report["device_name"] != device["name"] or float(
            rank_report["peak_reserved_gib"]
        ) > float(device["total_memory_gib"]):
            raise RuntimeError(
                "Modal GPU smoke returned distributed evidence that does not match "
                "the authenticated runtime devices"
            )
    return value


def _validated_remote_result(target: str, value: object) -> dict[str, Any]:
    """Authenticate every field in the remote success envelope before reporting a pass."""

    if target not in TARGETS:
        raise RuntimeError(f"unknown Modal GPU smoke target: {target!r}")
    if not isinstance(value, dict):
        raise RuntimeError("Modal GPU smoke returned a non-object result")
    result = dict(value)
    required = {
        "status",
        "target",
        "runtime",
        "phases",
        "elapsed_seconds",
        "estimated_function_compute_charge_usd",
        "authorization_compute_charge_usd",
        "peak_allocated_gib",
        "peak_reserved_gib",
    }
    if set(result) != required:
        raise RuntimeError("Modal GPU smoke returned an unexpected result schema")
    _validate_json_value(result)
    if result.get("status") != "passed" or result.get("target") != target:
        raise RuntimeError("Modal GPU smoke returned a mismatched success result")
    runtime = _validate_runtime_evidence(target, result.get("runtime"))
    phases = result.get("phases")
    expected_phases = {"attention_optimizer", "production_model"}
    if int(TARGETS[target]["gpu_count"]) == 2:
        expected_phases.add("distributed_fsdp2")
    if not isinstance(phases, dict) or set(phases) != expected_phases:
        raise RuntimeError("Modal GPU smoke returned incomplete phase evidence")
    attention = _validate_attention_evidence(phases.get("attention_optimizer"))
    production = _validate_production_evidence(target, phases.get("production_model"))
    distributed: dict[str, Any] | None = None
    if "distributed_fsdp2" in expected_phases:
        distributed = _validate_distributed_phase(phases.get("distributed_fsdp2"), runtime)
    positive_metrics = (
        "elapsed_seconds",
        "estimated_function_compute_charge_usd",
        "authorization_compute_charge_usd",
    )
    positive_memory_metrics = ("peak_allocated_gib", "peak_reserved_gib")
    if any(
        not _is_finite_number(result.get(name)) or float(result[name]) <= 0.0
        for name in positive_metrics
    ) or any(
        not _is_finite_number(result.get(name)) or float(result[name]) <= 0.0
        for name in positive_memory_metrics
    ):
        raise RuntimeError("Modal GPU smoke returned invalid top-level metrics")
    elapsed = float(result["elapsed_seconds"])
    phase_elapsed = float(attention["elapsed_seconds"]) + float(production["elapsed_seconds"])
    if distributed is not None:
        phase_elapsed += float(distributed["elapsed_seconds"])
    if (
        phase_elapsed > elapsed + max(1e-6, elapsed * 1e-9)
        or float(production["checkpoint_save_seconds"])
        + float(production["checkpoint_load_seconds"])
        > float(production["elapsed_seconds"]) + 1e-6
        or float(result["peak_reserved_gib"]) < float(result["peak_allocated_gib"])
        or float(result["peak_allocated_gib"]) < PRODUCTION_PARAMETER_COUNT * 4 / 2**30
        or float(result["peak_reserved_gib"]) > float(runtime["devices"][0]["total_memory_gib"])
    ):
        raise RuntimeError("Modal GPU smoke returned inconsistent timing or memory evidence")
    if not math.isclose(
        float(result["authorization_compute_charge_usd"]),
        authorization_compute_charge(target),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Modal GPU smoke returned a mismatched authorization estimate")
    if not math.isclose(
        float(result["estimated_function_compute_charge_usd"]),
        estimated_function_compute_charge(target, elapsed),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Modal GPU smoke returned a mismatched function-body estimate")
    return result


def _run_remote(
    target: str,
    phase_completed: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    import torch

    runtime = _runtime_report(torch)
    _validate_target_hardware(target, runtime)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    phases: dict[str, Any] = {}
    phases["attention_optimizer"] = _attention_optimizer_canary(torch, device)
    if phase_completed is not None:
        phase_completed("attention_optimizer")
    phases["production_model"] = _production_model_canary(torch, device, target)
    if phase_completed is not None:
        phase_completed("production_model")
    if int(TARGETS[target]["gpu_count"]) == 2:
        # The full-model probe has returned and owns no live tensors, but the
        # parent's caching allocator may still reserve GPU 0 memory needed by
        # the two child ranks.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        child_timeout = _remaining_child_timeout(time.perf_counter() - started)
        phases["distributed_fsdp2"] = _two_gpu_canary(child_timeout)
        if phase_completed is not None:
            phase_completed("distributed_fsdp2")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return _validated_remote_result(
        target,
        {
            "status": "passed",
            "target": target,
            "runtime": runtime,
            "phases": phases,
            "elapsed_seconds": elapsed,
            "estimated_function_compute_charge_usd": estimated_function_compute_charge(
                target, elapsed
            ),
            "authorization_compute_charge_usd": authorization_compute_charge(target),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        },
    )


def _run_durable_remote(
    target: str,
    run_id: str,
    function_call_id: str,
    max_dollars: float,
    expected_contract_sha256: str,
    *,
    journal_root: Path,
    commit: Callable[[], None],
) -> dict[str, Any]:
    """Run one smoke probe while persisting enough state to recover without its client."""

    journal = _DurableRunJournal(
        journal_root,
        run_id=run_id,
        target=target,
        function_call_id=function_call_id,
        max_dollars=max_dollars,
        expected_contract_sha256=expected_contract_sha256,
        commit=commit,
    )
    try:
        journal.started()
        validate_cost_guard(target, max_dollars)
        observed_contract_sha256 = gpu_smoke_contract_sha256(Path(REMOTE_ROOT))
        if observed_contract_sha256 != expected_contract_sha256:
            raise RuntimeError(
                "deployed Modal GPU smoke bytes differ from the submitted local contract"
            )
        _verify_executed_entrypoint(
            Path(__file__),
            Path(REMOTE_ROOT) / "scripts" / "modal_gpu_smoke.py",
        )
        journal.contract_verified(observed_contract_sha256)
        result = _run_remote(target, journal.phase_completed)
        journal.passed(result)
        return result
    except BaseException as error:
        try:
            journal.failed(error)
        except BaseException as journal_error:
            print(
                "Failed to persist the durable Modal failure journal: "
                f"{type(journal_error).__name__}: {journal_error}",
                file=sys.stderr,
                flush=True,
            )
        raise RuntimeError(
            f"durable Modal GPU smoke {run_id} failed; inspect its Volume journal"
        ) from error


def _dispatch_remote(
    target: str,
    run_id: str = "",
    function_call_id: str = "",
    max_dollars: float = 0.0,
    expected_contract_sha256: str = "",
    *,
    journal_root: Path,
    commit: Callable[[], None],
) -> dict[str, Any]:
    durable_arguments = (
        bool(run_id),
        bool(function_call_id),
        max_dollars != 0.0,
        bool(expected_contract_sha256),
    )
    if not any(durable_arguments):
        return _run_remote(target)
    if not all(durable_arguments):
        raise ValueError("durable Modal invocation arguments must be supplied together")
    return _run_durable_remote(
        target,
        run_id,
        function_call_id,
        max_dollars,
        expected_contract_sha256,
        journal_root=journal_root,
        commit=commit,
    )


if __name__ == "__main__" and sys.argv[1:] == ["--distributed-child"]:
    # The guarded torchrun workers need only the locked training environment.
    # Exit through the child path before importing or constructing Modal SDK
    # objects, so worker correctness does not depend on an injected client.
    raise SystemExit(_distributed_child())


try:
    import modal
except ModuleNotFoundError:  # Unit tests can validate the CLI without Modal.
    modal = None


def _validate_modal_client_version() -> None:
    """Reject an unreviewed local client before it can create a paid function."""

    observed = importlib.metadata.version("modal")
    if observed != EXPECTED_MODAL_CLIENT_VERSION:
        raise RuntimeError(
            "the Modal GPU smoke requires local Modal client "
            f"{EXPECTED_MODAL_CLIENT_VERSION}, got {observed}"
        )


def _dispatch_modal_function(
    target: str,
    run_id: str,
    max_dollars: float,
    expected_contract_sha256: str,
    *,
    journal_root: Path,
    commit: Callable[[], None],
) -> dict[str, Any]:
    if modal is None:
        raise RuntimeError("Modal is unavailable inside the remote function")
    function_call_id = ""
    if run_id:
        observed_call_id = modal.current_function_call_id()
        if observed_call_id is None:
            raise RuntimeError("Modal did not expose the current durable FunctionCall ID")
        function_call_id = observed_call_id
    return _dispatch_remote(
        target,
        run_id,
        function_call_id,
        max_dollars,
        expected_contract_sha256,
        journal_root=journal_root,
        commit=commit,
    )


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential", "git")
        .add_local_file(
            str(REPOSITORY_ROOT / NATIVE_BUILD_RELATIVE_PATH),
            str(REMOTE_ROOT / NATIVE_BUILD_RELATIVE_PATH),
            copy=True,
        )
        .add_local_file(
            str(REPOSITORY_ROOT / NATIVE_REQUIREMENTS_RELATIVE_PATH),
            str(REMOTE_ROOT / NATIVE_REQUIREMENTS_RELATIVE_PATH),
            copy=True,
        )
        .add_local_file(
            str(REPOSITORY_ROOT / UV_BOOTSTRAP_RELATIVE_PATH),
            str(REMOTE_ROOT / UV_BOOTSTRAP_RELATIVE_PATH),
            copy=True,
        )
        .add_local_file(
            str(REPOSITORY_ROOT / LOCK_RELATIVE_PATH),
            str(REMOTE_ROOT / LOCK_RELATIVE_PATH),
            copy=True,
        )
        .add_local_dir(
            str(REPOSITORY_ROOT / "src"),
            str(REMOTE_ROOT / "src"),
            copy=True,
            ignore=("**/__pycache__/**", "**/*.pyc", "**/*.pyo"),
        )
        .add_local_file(
            str(REPOSITORY_ROOT / "scripts/modal_gpu_smoke.py"),
            str(REMOTE_ROOT / "scripts/modal_gpu_smoke.py"),
            copy=True,
        )
        .run_commands(
            "python -m pip install --disable-pip-version-check --no-cache-dir "
            "--only-binary :all: --require-hashes "
            f"-r {REMOTE_ROOT / UV_BOOTSTRAP_RELATIVE_PATH}",
            "uv pip sync --system --no-config "
            f"{REMOTE_ROOT / LOCK_RELATIVE_PATH} "
            "--require-hashes --strict --only-binary :all:",
            "python -m venv /opt/sion-native-tools",
            "/opt/sion-native-tools/bin/python -m pip install "
            f"-r {REMOTE_ROOT / NATIVE_REQUIREMENTS_RELATIVE_PATH}",
            "git clone --depth 1 --branch v0.2.1 "
            "https://github.com/google/sentencepiece.git /opt/sion-native-source",
            "PATH=/opt/sion-native-tools/bin:$PATH /opt/sion-native-tools/bin/python "
            f"{REMOTE_ROOT / NATIVE_BUILD_RELATIVE_PATH} "
            "--source /opt/sion-native-source --output /opt/sion/native --jobs 2",
            "uv pip install --system --no-deps --reinstall /opt/sion/native/wheels/*.whl",
            f"python {REMOTE_ROOT / NATIVE_BUILD_RELATIVE_PATH} verify-installed "
            f"--manifest {NATIVE_MANIFEST_PATH}",
            f"python {REMOTE_ROOT / NATIVE_BUILD_RELATIVE_PATH} verify "
            f"--manifest {NATIVE_MANIFEST_PATH} --output /opt/sion/native-verification",
        )
        .env(
            {
                "PYTHONPATH": str(REMOTE_ROOT / "src"),
                "PYTHONUNBUFFERED": "1",
                "PYTHONWARNINGS": "error",
            }
        )
    )
    app = modal.App(APP_NAME, image=image, include_source=False)
    result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
    _common_options = {
        "retries": 0,
        "timeout": FUNCTION_TIMEOUT_SECONDS,
        "startup_timeout": STARTUP_TIMEOUT_SECONDS,
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "scaledown_window": SCALEDOWN_WINDOW_SECONDS,
        "single_use_containers": True,
        "cpu": CPU_CORES,
        "memory": MEMORY_GIB * 1024,
        "volumes": {str(RESULT_MOUNT): result_volume},
        # Modal must mount this entrypoint so the remote worker can import its
        # decorated functions. The same reviewed file is also copied to
        # /opt/sion for the timeout-configured torchrun child.
        "include_source": True,
    }

    @app.function(gpu="A100-40GB", name="a100_40gb", **_common_options)
    def smoke_a100_40gb(
        run_id: str = "",
        max_dollars: float = 0.0,
        expected_contract_sha256: str = "",
    ) -> dict[str, Any]:
        return _dispatch_modal_function(
            "a100-40gb",
            run_id,
            max_dollars,
            expected_contract_sha256,
            journal_root=Path(RESULT_MOUNT),
            commit=result_volume.commit,
        )

    @app.function(gpu="A100-80GB", name="a100_80gb", **_common_options)
    def smoke_a100_80gb(
        run_id: str = "",
        max_dollars: float = 0.0,
        expected_contract_sha256: str = "",
    ) -> dict[str, Any]:
        return _dispatch_modal_function(
            "a100-80gb",
            run_id,
            max_dollars,
            expected_contract_sha256,
            journal_root=Path(RESULT_MOUNT),
            commit=result_volume.commit,
        )

    @app.function(gpu="H100!", name="h100_exact", **_common_options)
    def smoke_h100(
        run_id: str = "",
        max_dollars: float = 0.0,
        expected_contract_sha256: str = "",
    ) -> dict[str, Any]:
        return _dispatch_modal_function(
            "h100",
            run_id,
            max_dollars,
            expected_contract_sha256,
            journal_root=Path(RESULT_MOUNT),
            commit=result_volume.commit,
        )

    @app.function(gpu="A100-40GB:2", name="a100_40gb_x2", **_common_options)
    def smoke_a100_40gb_x2(
        run_id: str = "",
        max_dollars: float = 0.0,
        expected_contract_sha256: str = "",
    ) -> dict[str, Any]:
        return _dispatch_modal_function(
            "a100-40gb-x2",
            run_id,
            max_dollars,
            expected_contract_sha256,
            journal_root=Path(RESULT_MOUNT),
            commit=result_volume.commit,
        )

    smoke_functions = {
        "a100-40gb": smoke_a100_40gb,
        "a100-80gb": smoke_a100_80gb,
        "h100": smoke_h100,
        "a100-40gb-x2": smoke_a100_40gb_x2,
    }

    @app.local_entrypoint()
    def main(target: str, max_dollars: float) -> None:
        _validate_modal_client_version()
        authorized = validate_cost_guard(target, max_dollars)
        print(
            "Authorized two-attempt compute contingency: "
            f"${authorized:.4f}. This is not an account-level hard cap.",
            flush=True,
        )
        invocation_started = time.perf_counter()
        result = _validated_remote_result(target, smoke_functions[target].remote())
        result["client_roundtrip_seconds"] = time.perf_counter() - invocation_started
        _validate_json_value(result)
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

else:
    app = None
    smoke_functions: dict[str, Any] = {}


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    authorized = validate_cost_guard(arguments.target, arguments.max_dollars)
    raise SystemExit(
        "Use `python -m modal run scripts/modal_gpu_smoke.py "
        f"--target {arguments.target} --max-dollars {authorized:.4f}` to launch the selected target."
    )
