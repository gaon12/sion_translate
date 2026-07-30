"""학습 재개용 체크포인트 저장/복원.

여기서 저장하는 체크포인트는 '학습을 이어서 하기 위한' 것으로,
모델 가중치 외에 optimizer / scheduler / (fp16이면) scaler / 진행 상태
(best loss, early-stopping 카운터, epoch)까지 전부 포함합니다.

번역(추론)에만 쓸 가벼운 저장본은 ``sion_translate.training.export`` 가 따로 만듭니다.

저장 형식은 학습 방식에 따라 둘로 나뉩니다.
- 단일 프로세스: ``checkpoint.pt`` 파일 하나에 torch.save 로 저장
- 분산 학습(FSDP2/DDP): torch.distributed.checkpoint(DCP) 형식의 디렉터리.
  가중치가 rank 별로 조각나 있어도 그대로 저장/복원할 수 있습니다.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .distributed import DistributedContext, barrier

CHECKPOINT_SCHEMA = "sion-training-checkpoint-v2"
CHECKPOINT_IDENTITY_SCHEMA = "sion-checkpoint-identity-v1"
DCP_COMPLETION_FILENAME = ".sion_checkpoint_complete.json"


def _json_compatible(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation for identity metadata."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"checkpoint identity contains unsupported value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"filename": path.name}
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        identity["status"] = "missing"
        return identity
    except OSError as error:
        identity.update(status="unreadable", error_type=type(error).__name__)
        return identity
    if not path.is_file():
        identity["status"] = "not-a-file"
        return identity
    try:
        sha256 = _sha256_file(path)
    except OSError as error:
        identity.update(status="unreadable", error_type=type(error).__name__)
        return identity
    identity.update(status="ok", size=size, sha256=sha256)
    return identity


def build_checkpoint_identity(
    *,
    model_config: Any,
    tokenizer_path: str | Path,
    token_features_path: str | Path | None,
    dataset_dir: str | Path,
    data_config: Any | None = None,
    sampling_seed: int | None = None,
    stage_name: str | None = None,
    loader_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a portable identity for the model, tokenizer, and prepared data.

    File identities use portable names and content hashes. The effective
    model/data configuration is retained verbatim, including any explicitly
    configured paths, so a changed run layout must be acknowledged as a new run
    instead of silently resuming with different artifacts.
    """

    model_payload = _json_compatible(model_config)
    data_payload = _json_compatible(data_config) if data_config is not None else None
    dataset_path = Path(dataset_dir)
    tokenizer_model_path = Path(tokenizer_path)
    data_identity: dict[str, Any] = {
        "directory_name": dataset_path.name,
        "manifest": _file_identity(dataset_path / "manifest.json"),
        "raw_fingerprint": _file_identity(dataset_path / "raw_fingerprint.json"),
    }
    if data_payload is not None:
        data_identity["config"] = data_payload
        data_identity["config_sha256"] = hashlib.sha256(
            _canonical_json(data_payload).encode("utf-8")
        ).hexdigest()
    if sampling_seed is not None:
        data_identity["sampling_seed"] = int(sampling_seed)
    if loader_config is not None:
        data_identity["loader"] = _json_compatible(loader_config)
    return {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "stage": stage_name,
        "model": {
            "config": model_payload,
            "config_sha256": hashlib.sha256(
                _canonical_json(model_payload).encode("utf-8")
            ).hexdigest(),
        },
        "tokenizer": {
            "model": _file_identity(tokenizer_model_path),
            "metadata": _file_identity(tokenizer_model_path.parent / "tokenizer_metadata.json"),
            "token_features": (
                _file_identity(Path(token_features_path))
                if token_features_path is not None
                else {"status": "not-configured"}
            ),
        },
        "data": data_identity,
    }


def _identity_differences(expected: Any, actual: Any, path: str = "identity") -> list[str]:
    """Return concise paths that differ, capped to keep resume errors readable."""

    if len(path) > 512:
        return [path]
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            child = f"{path}.{key}"
            if key not in expected or key not in actual:
                differences.append(child)
            else:
                differences.extend(_identity_differences(expected[key], actual[key], child))
            if len(differences) >= 8:
                return differences[:8]
        return differences
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        differences = []
        if len(expected) != len(actual):
            differences.append(f"{path}.length")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=False)):
            differences.extend(
                _identity_differences(expected_item, actual_item, f"{path}[{index}]")
            )
            if len(differences) >= 8:
                return differences[:8]
        return differences
    return [] if expected == actual else [path]


def _validate_identity(
    state: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None,
) -> None:
    if expected_identity is None:
        return
    stored_identity = state.get("identity")
    if stored_identity is None:
        warnings.warn(
            "이전 버전 체크포인트에는 model/tokenizer/data identity가 없습니다. "
            "이번 재개에서는 안전한 동일성 검사를 건너뜁니다. 다음 저장부터는 identity가 기록됩니다.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    if not isinstance(stored_identity, Mapping):
        raise ValueError("checkpoint identity must be an object")
    expected = _json_compatible(expected_identity)
    actual = _json_compatible(stored_identity)
    if expected != actual:
        differences = _identity_differences(expected, actual)
        detail = ", ".join(differences) if differences else "unknown fields"
        raise ValueError(
            "checkpoint identity does not match the current model/tokenizer/data "
            f"({detail}). Refusing to resume with incompatible artifacts."
        )


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    if not isinstance(python_state, tuple):
        raise ValueError("checkpoint Python RNG state is invalid")
    if not isinstance(numpy_state, Mapping) or not isinstance(
        numpy_state.get("keys"), torch.Tensor
    ):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise ValueError("checkpoint torch RNG state is invalid")

    random.setstate(python_state)
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy_state["keys"].detach().cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu_state.detach().cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if not isinstance(cuda_states, list) or not all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            raise ValueError("checkpoint CUDA RNG state is invalid")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG state count does not match the current CUDA device count"
            )
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Durably replace one checkpoint file without exposing a partial write."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        if os.name != "nt":
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _dcp_sibling(path: Path, suffix: str) -> Path:
    candidate = path.with_name(f".{path.name}.{suffix}")
    if candidate.parent.resolve() != path.parent.resolve():
        raise ValueError("distributed checkpoint sibling escaped its parent directory")
    return candidate


def _write_dcp_completion(
    directory: Path,
    *,
    step: int,
    world_size: int,
) -> None:
    marker = directory / DCP_COMPLETION_FILENAME
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "world_size": int(world_size),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{DCP_COMPLETION_FILENAME}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_dcp_staging(staging: Path, destination: Path) -> None:
    """Publish a complete DCP directory while retaining one recoverable version."""

    previous = _dcp_sibling(destination, "previous")
    _remove_path(previous)
    moved_previous = False
    if destination.exists() or destination.is_symlink():
        os.replace(destination, previous)
        moved_previous = True
    try:
        os.replace(staging, destination)
    except BaseException:
        if moved_previous and not destination.exists():
            os.replace(previous, destination)
        raise


def _valid_dcp_completion(path: Path, *, world_size: int) -> bool:
    marker = path / DCP_COMPLETION_FILENAME
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, Mapping)
        and payload.get("schema") == CHECKPOINT_SCHEMA
        and isinstance(payload.get("step"), int)
        and payload.get("world_size") == world_size
        and (path / ".metadata").is_file()
    )


def _resolve_dcp_checkpoint(path: Path, *, world_size: int) -> Path:
    """Resolve a complete current DCP checkpoint, or its retained predecessor."""

    if _valid_dcp_completion(path, world_size=world_size):
        return path
    previous = _dcp_sibling(path, "previous")
    if _valid_dcp_completion(previous, world_size=world_size):
        warnings.warn(
            f"{path} is incomplete; resuming from retained checkpoint {previous}",
            RuntimeWarning,
            stacklevel=3,
        )
        return previous
    # Accept pre-v2 DCP directories once for backward compatibility. New saves
    # always carry a completion marker and never overwrite this directory in place.
    if (path / ".metadata").is_file():
        warnings.warn(
            "legacy distributed checkpoint has no Sion completion marker; "
            "loading it without atomic-publication verification",
            RuntimeWarning,
            stacklevel=3,
        )
        return path
    raise FileNotFoundError(f"no complete distributed checkpoint found at {path}")


def checkpoint_path_exists(path: str | Path) -> bool:
    """Return whether a local or distributed checkpoint can be resumed."""

    path = Path(path)
    previous = _dcp_sibling(path, "previous")
    return any(
        (
            (path / "checkpoint.pt").is_file(),
            (path / ".metadata").is_file(),
            (previous / ".metadata").is_file(),
        )
    )


def _validate_loaded_state(state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint payload must be an object")
    schema = state.get("schema")
    if schema is not None and schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {schema!r}")
    missing = [
        key
        for key in ("model", "optimizer", "scheduler", "step", "training_state")
        if key not in state
    ]
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")
    if isinstance(state["step"], bool) or not isinstance(state["step"], int):
        raise ValueError("checkpoint step must be an integer")
    if not isinstance(state["training_state"], Mapping):
        raise ValueError("checkpoint training_state must be an object")
    return state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    context: DistributedContext,
    *,
    scaler: Any | None = None,
    training_state: dict[str, Any] | None = None,
    ema: Any | None = None,
    identity: Mapping[str, Any] | None = None,
) -> None:
    """현재 학습 상태 전체를 ``path`` 디렉터리에 저장합니다.

    분산 학습에서는 모든 rank 가 함께 호출해야 합니다(집단 통신 발생).
    """
    path = Path(path)
    if context.is_main:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not context.distributed:
            path.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "scheduler": scheduler.state_dict(),
        "step": step,
        "training_state": dict(training_state or {}),
    }
    if identity is not None:
        state["identity"] = _json_compatible(identity)
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        # EMA shadow 가중치도 함께 저장해 재개 시 평균 이력이 끊기지 않게 합니다.
        state["ema"] = ema.state_dict()

    if context.distributed:
        # 분산 학습: DCP 가 rank 별 조각(shard)을 병렬로 저장합니다.
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        staging = _dcp_sibling(path, "staging")
        if context.is_main:
            _remove_path(staging)
            staging.mkdir(parents=True)
        barrier(context)
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state["model"] = model_state
        state["optimizer"] = optimizer_state
        dcp.save(state, checkpoint_id=staging)
        _atomic_torch_save(
            {"schema": CHECKPOINT_SCHEMA, "rng_state": _capture_rng_state()},
            staging / f"rng-rank-{context.rank:05d}.pt",
        )
        barrier(context)
        if context.is_main:
            _write_dcp_completion(staging, step=step, world_size=context.world_size)
            _publish_dcp_staging(staging, path)
    elif context.is_main:
        # 단일 프로세스: 파일 하나로 충분합니다.
        state["model"] = model.state_dict()
        state["optimizer"] = optimizer.state_dict()
        # RNG state를 함께 저장하면 같은 데이터 위치에서 재개할 때 Python,
        # NumPy, torch의 확률적 연산도 중단 전 상태에서 이어집니다.
        state["rng_state"] = _capture_rng_state()
        _atomic_torch_save(state, path / "checkpoint.pt")
    # 저장이 완전히 끝난 뒤에만 모든 rank 가 다음 단계로 진행합니다.
    barrier(context)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    *,
    scaler: Any | None = None,
    training_state: dict[str, Any] | None = None,
    ema: Any | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> int:
    """``path`` 의 체크포인트를 읽어 학습 상태를 복원하고, 재개할 step 을 반환합니다.

    ``training_state`` dict 를 넘기면 best loss / early-stopping 카운터 /
    epoch 같은 진행 상태가 그 안에 채워집니다.
    """
    path = Path(path)
    if context.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        path = _resolve_dcp_checkpoint(path, world_size=context.world_size)
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state: dict[str, Any] = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
            "step": 0,
            "training_state": dict(training_state or {}),
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if ema is not None:
            # DCP 는 여기 넣어 둔 텐서 '안으로' 값을 읽어들이므로,
            # 현재 shadow 텐서를 로드 대상으로 미리 등록합니다.
            state["ema"] = ema.state_dict()
        if expected_identity is not None:
            state["identity"] = _json_compatible(expected_identity)
        state["schema"] = CHECKPOINT_SCHEMA
        # 체크포인트가 불완전하거나 구조가 맞지 않으면 여기서 바로 실패합니다.
        # 일부 파라미터가 초기값인 채로 조용히 재개되는 것이 훨씬 위험하기 때문입니다.
        dcp.load(state, checkpoint_id=path)
        _validate_loaded_state(state)
        _validate_identity(state, expected_identity)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema"):
            ema.load_state_dict(state["ema"])
        if training_state is not None:
            training_state.update(state.get("training_state", {}))
        rng_file = path / f"rng-rank-{context.rank:05d}.pt"
        if rng_file.is_file():
            rng_payload = torch.load(
                rng_file,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if not isinstance(rng_payload, Mapping) or not isinstance(
                rng_payload.get("rng_state"), Mapping
            ):
                raise ValueError("distributed checkpoint RNG payload is invalid")
            _restore_rng_state(rng_payload["rng_state"])
        return int(state["step"])

    # 단일 프로세스: 임의 코드 실행이 가능한 일반 pickle 로드는 사용하지 않습니다.
    try:
        loaded = torch.load(
            path / "checkpoint.pt",
            map_location=context.device,
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise RuntimeError(
            "checkpoint could not be loaded with PyTorch's safe weights-only loader; "
            "refusing to fall back to executable pickle"
        ) from error
    state = _validate_loaded_state(loaded)
    # 모델/optimizer를 변경하기 전에 현재 실행과 체크포인트의 정체성을 비교합니다.
    _validate_identity(state, expected_identity)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema"):
        ema.load_state_dict(state["ema"])
    if training_state is not None:
        training_state.update(state.get("training_state", {}))
    rng_state = state.get("rng_state")
    if rng_state is not None:
        if not isinstance(rng_state, Mapping):
            raise ValueError("checkpoint RNG state must be an object")
        _restore_rng_state(rng_state)
    return int(state["step"])
