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
import stat
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .distributed import DistributedContext, broadcast_text

CHECKPOINT_SCHEMA = "sion-training-checkpoint-v2"
CHECKPOINT_IDENTITY_SCHEMA = "sion-checkpoint-identity-v1"
DCP_COMPLETION_FILENAME = ".sion_checkpoint_complete.json"
DCP_COMPLETION_SCHEMA = "sion-dcp-completion-v2"
_CHECKPOINT_IO_HEARTBEAT_SECONDS = 1.0


@dataclass(frozen=True)
class _VerifiedDcpPublication:
    source: str
    world_size: int
    marker_sha256: str


@dataclass(frozen=True)
class _VerifiedCheckpointLease:
    source: str
    world_size: int
    marker_sha256: str


@dataclass(frozen=True)
class VerifiedCheckpointGeneration:
    """Exact source selected by an authenticated checkpoint-generation lease."""

    source: Path
    step: int
    artifact_sha256: str


@dataclass(frozen=True)
class CheckpointGenerationBinding:
    """Discovery hint that binds one logical generation to its artifact digest.

    This object is not load authority. Pass its digest back to
    ``verified_checkpoint_generation_lease`` so that a semantic failure in the
    current generation can be followed by an exact attempt of retained previous.
    """

    source: Path
    artifact_sha256: str


class _VerifiedCheckpointLeaseState(threading.local):
    active: _VerifiedCheckpointLease | None

    def __init__(self) -> None:
        self.active = None


_VERIFIED_CHECKPOINT_LEASE = _VerifiedCheckpointLeaseState()
RUNTIME_DATA_PATH_FIELDS = frozenset(
    {
        "raw_dir",
        "tokenizer_model",
        "tokenizer_features",
        "dataset_dir",
    }
)


def _json_compatible(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation for identity metadata."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in cast(list[Any] | tuple[Any, ...], value)]
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_checkpoint_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


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


def _unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the model below torch.compile without unwrapping DDP or FSDP."""

    unwrapped = model
    while True:
        original = getattr(unwrapped, "_orig_mod", None)
        if not isinstance(original, nn.Module):
            return unwrapped
        unwrapped = original


def _stage_transfer_ema_template(model: nn.Module) -> dict[str, torch.Tensor]:
    """Allocate the exact tensor mapping DCP needs to restore EMA parameters."""

    return {
        name: parameter.detach().clone()
        for name, parameter in _unwrap_compiled_model(model).named_parameters()
        if parameter.requires_grad
    }


def _canonical_stage_transfer_key(raw_name: str) -> str:
    name = raw_name
    while name.startswith("_orig_mod."):
        name = name.removeprefix("_orig_mod.")
    return name


def _validate_stage_transfer_model_state(
    model: nn.Module,
    model_state: object,
) -> dict[str, torch.Tensor]:
    """Preflight every raw model tensor before ``load_state_dict`` can mutate it."""

    if not isinstance(model_state, Mapping):
        raise ValueError("stage-transfer checkpoint model state must be an object")
    expected = _unwrap_compiled_model(model).state_dict()
    normalized: dict[str, torch.Tensor] = {}
    for raw_name, raw_tensor in cast(Mapping[object, object], model_state).items():
        if not isinstance(raw_name, str) or not isinstance(raw_tensor, torch.Tensor):
            raise ValueError("stage-transfer checkpoint model state is malformed")
        name = _canonical_stage_transfer_key(raw_name)
        if name in normalized:
            raise ValueError(f"stage-transfer checkpoint model contains duplicate key: {name}")
        normalized[name] = raw_tensor
    missing = sorted(set(expected) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "stage-transfer checkpoint model does not match the architecture: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, target in expected.items():
        source = normalized[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                "stage-transfer checkpoint model tensor metadata mismatch for "
                f"{name}: checkpoint=(shape={tuple(source.shape)}, dtype={source.dtype}), "
                f"model=(shape={tuple(target.shape)}, dtype={target.dtype})"
            )
    return normalized


def _validate_dcp_tensor_group_metadata(
    metadata: Any,
    *,
    prefix: str,
    expected: Mapping[str, torch.Tensor],
) -> bool:
    """Validate DCP keys/shapes/dtypes before its loader can cast or ignore them."""

    prefix_with_dot = f"{prefix}."
    actual: dict[str, Any] = {}
    for raw_key, tensor_metadata in metadata.state_dict_metadata.items():
        flat_key = str(raw_key)
        if not flat_key.startswith(prefix_with_dot):
            continue
        name = _canonical_stage_transfer_key(flat_key.removeprefix(prefix_with_dot))
        if name in actual:
            raise ValueError(f"distributed checkpoint {prefix} contains duplicate key: {name}")
        actual[name] = tensor_metadata
    if not actual:
        return False
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"distributed checkpoint {prefix} does not match the model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, target in expected.items():
        stored = actual[name]
        size = getattr(stored, "size", None)
        properties = getattr(stored, "properties", None)
        dtype = getattr(properties, "dtype", None)
        if tuple(size or ()) != tuple(target.shape) or dtype != target.dtype:
            raise ValueError(
                f"distributed checkpoint {prefix} tensor metadata mismatch for {name}: "
                f"checkpoint=(shape={tuple(size or ())}, dtype={dtype}), "
                f"model=(shape={tuple(target.shape)}, dtype={target.dtype})"
            )
    return True


def _validate_stage_transfer_ema(
    model: nn.Module,
    ema_state: object,
) -> dict[str, torch.Tensor] | None:
    """Validate a complete EMA parameter set without mutating the live model."""

    if ema_state is None:
        return None
    if not isinstance(ema_state, Mapping):
        raise ValueError("stage-transfer checkpoint EMA state must be an object")
    expected = {
        name: parameter
        for name, parameter in _unwrap_compiled_model(model).named_parameters()
        if parameter.requires_grad
    }
    normalized: dict[str, torch.Tensor] = {}
    for raw_name, raw_tensor in cast(Mapping[object, object], ema_state).items():
        if not isinstance(raw_name, str) or not isinstance(raw_tensor, torch.Tensor):
            raise ValueError("stage-transfer checkpoint EMA state is malformed")
        name = _canonical_stage_transfer_key(raw_name)
        if name in normalized:
            raise ValueError(f"stage-transfer checkpoint EMA contains duplicate key: {name}")
        normalized[name] = raw_tensor
    missing = sorted(set(expected) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "stage-transfer checkpoint EMA does not match the model parameters: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    mismatched = [
        name
        for name, parameter in expected.items()
        if normalized[name].shape != parameter.shape or normalized[name].dtype != parameter.dtype
    ]
    if mismatched:
        name = mismatched[0]
        tensor = normalized[name]
        parameter = expected[name]
        raise ValueError(
            "stage-transfer checkpoint EMA tensor metadata mismatch for "
            f"{name}: checkpoint=(shape={tuple(tensor.shape)}, dtype={tensor.dtype}), "
            f"model=(shape={tuple(parameter.shape)}, dtype={parameter.dtype})"
        )
    return normalized


@torch.no_grad()
def _copy_validated_stage_transfer_ema(
    model: nn.Module,
    ema_state: Mapping[str, torch.Tensor] | None,
) -> bool:
    """Select a preflighted EMA parameter set for the next stage."""

    if ema_state is None:
        return False
    expected = {
        name: parameter
        for name, parameter in _unwrap_compiled_model(model).named_parameters()
        if parameter.requires_grad
    }
    for name, parameter in expected.items():
        parameter.copy_(ema_state[name])
    return True


def _source_identity_ema_policy(state: Mapping[str, Any]) -> bool | None:
    """Return a hash-authenticated EMA policy, or None for a legacy identity."""

    identity = state.get("identity")
    if not isinstance(identity, Mapping):
        return None
    identity_mapping = cast(Mapping[object, object], identity)
    objective = identity_mapping.get("objective")
    if not isinstance(objective, Mapping):
        return None
    objective_mapping = cast(Mapping[object, object], objective)
    stored_sha256 = objective_mapping.get("sha256")
    if stored_sha256 is None:
        return None
    if not isinstance(stored_sha256, str):
        raise ValueError("checkpoint objective identity SHA256 is invalid")
    unhashed_objective = {
        str(key): value for key, value in objective_mapping.items() if key != "sha256"
    }
    calculated_sha256 = hashlib.sha256(
        _canonical_json(unhashed_objective).encode("utf-8")
    ).hexdigest()
    if stored_sha256 != calculated_sha256:
        raise ValueError("checkpoint objective identity SHA256 does not authenticate its fields")
    supervised = objective_mapping.get("supervised")
    if not isinstance(supervised, Mapping) or "ema_decay" not in supervised:
        return None
    supervised_mapping = cast(Mapping[object, object], supervised)
    decay = supervised_mapping.get("ema_decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)):
        raise ValueError("checkpoint objective ema_decay is invalid")
    numeric_decay = float(decay)
    if not np.isfinite(numeric_decay) or not 0.0 <= numeric_decay < 1.0:
        raise ValueError("checkpoint objective ema_decay is invalid")
    return numeric_decay > 0.0


def _portable_data_config(data_config: Any) -> Any:
    """Remove storage locations while retaining every semantic data setting."""

    payload = _json_compatible(data_config)
    if not isinstance(payload, Mapping):
        return payload
    mapping = cast(Mapping[object, object], payload)
    return {
        key: value for key, value in mapping.items() if str(key) not in RUNTIME_DATA_PATH_FIELDS
    }


# 재개 시 같아야 하는 최적화·목적함수 설정. 여기 빠진 값을 바꾸고 재개하면
# optimizer state 를 그대로 이어받으면서 다른 목적을 최적화하게 되고, 과거
# best 지표와 새 지표를 같은 축에 놓고 비교하게 됩니다.
_SUPERVISED_OBJECTIVE_FIELDS = (
    "learning_rate",
    "min_learning_rate_ratio",
    "warmup_steps",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "grad_clip",
    "precision",
    "ema_decay",
    "sft_selection_metric",
)

# MRT 는 reward 정의 자체가 선택 지표입니다. 가중치 하나만 바꿔도
# validation_reward 는 다른 축의 수치가 되므로 반드시 identity 에 들어갑니다.
_POSTTRAINING_OBJECTIVE_FIELDS = (
    "method",
    "learning_rate",
    "warmup_steps",
    "samples_per_source",
    "sampling_temperature",
    "top_k",
    "max_new_tokens",
    "risk_weight",
    "mrt_alpha",
    "preference_weight",
    "preference_min_gap",
    "preference_temperature",
    "reward_chrf_weight",
    "reward_token_f1_weight",
    "reward_number_weight",
    "reward_structured_weight",
    "reward_slot_weight",
    "reward_language_weight",
    "reward_length_weight",
    "reward_repetition_penalty",
    "reward_copy_penalty",
    "reward_number_corruption_penalty",
    "roundtrip_enabled",
    "roundtrip_reward_weight",
    "roundtrip_failure_penalty",
    "roundtrip_min_score",
    "roundtrip_num_beams",
    "roundtrip_max_new_tokens",
    "validation_num_beams",
    "validation_length_penalty",
    "decode_min_new_tokens",
    "decode_no_repeat_ngram_size",
    "decode_max_output_length_ratio",
    "decode_max_output_length_margin",
    "selection_metric",
)


def build_objective_identity(
    training_config: Any,
    posttraining_config: Any = None,
    *,
    include_posttraining: bool = False,
) -> dict[str, Any]:
    """무엇을 최적화하고 무엇으로 best 를 고르는지의 정체성.

    모델·토크나이저·데이터가 같아도 목적이 다르면 재개는 안전하지 않습니다.
    학습률 스케줄이나 Adam 계수를 바꾸고 optimizer state 를 이어받으면 momentum
    이 다른 곡률을 가리키고, MRT 의 reward 가중치를 바꾸면 ``validation_reward``
    가 다른 축의 수치가 되는데 early stopping 은 과거 best 와 비교합니다.
    """

    payload: dict[str, Any] = {
        "supervised": {
            field: _json_compatible(getattr(training_config, field))
            for field in _SUPERVISED_OBJECTIVE_FIELDS
        }
    }
    if include_posttraining and posttraining_config is not None:
        payload["posttraining"] = {
            field: _json_compatible(getattr(posttraining_config, field))
            for field in _POSTTRAINING_OBJECTIVE_FIELDS
        }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


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
    objective_identity: Mapping[str, Any] | None = None,
    pipeline_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a portable identity for the model, tokenizer, and prepared data.

    File identities use portable names and content hashes. The effective
    model/data configuration retains semantic preprocessing and sampling fields,
    while runtime-only artifact locations are excluded so an identical run can
    move between ``/dev/shm`` and persistent storage.
    """

    model_payload = _json_compatible(model_config)
    data_payload = _portable_data_config(data_config) if data_config is not None else None
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
    identity: dict[str, Any] = {
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
    if objective_identity is not None:
        identity["objective"] = _json_compatible(objective_identity)
    if pipeline_identity is not None:
        identity["pipeline"] = _json_compatible(pipeline_identity)
    return identity


def _normalize_identity_for_comparison(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy path-bearing identities to the portable representation."""

    payload = _json_compatible(identity)
    payload = cast(dict[str, Any], payload)
    model_identity = payload.get("model")
    if isinstance(model_identity, dict):
        typed_model_identity = cast(dict[str, Any], model_identity)
        model_config = typed_model_identity.get("config")
        if isinstance(model_config, dict):
            typed_model_config = cast(dict[str, Any], model_config)
            experimental = typed_model_config.get("experimental")
            if isinstance(experimental, dict):
                typed_experimental = cast(dict[str, Any], experimental)
                # Before candidate feedback existed, absence meant the exact
                # architecture represented by today's disabled defaults. Add
                # only those defaults; enabled=true must still mismatch. The
                # old hash must authenticate the pre-injection config first so
                # normalization cannot hide a corrupted identity hash.
                candidate_defaults = (
                    ("candidate_refinement_enabled", False),
                    ("candidate_refinement_steps", 1),
                    ("candidate_refinement_temperature", 1.0),
                    ("candidate_refinement_loss_weight", 0.25),
                    ("candidate_refinement_vocab_chunk_size", 2048),
                )
                missing_candidate_fields = [
                    name for name, _ in candidate_defaults if name not in typed_experimental
                ]
                old_config_sha256 = hashlib.sha256(
                    _canonical_json(typed_model_config).encode("utf-8")
                ).hexdigest()
                if len(missing_candidate_fields) == len(candidate_defaults) and (
                    typed_model_identity.get("config_sha256") == old_config_sha256
                ):
                    for name, value in candidate_defaults:
                        typed_experimental[name] = value
                    typed_model_identity["config_sha256"] = hashlib.sha256(
                        _canonical_json(typed_model_config).encode("utf-8")
                    ).hexdigest()
    pipeline_identity = payload.get("pipeline")
    if pipeline_identity == {
        "schema": "sion-translation-pipeline-v1",
        "branch": "translation-only",
    }:
        payload["pipeline"] = {
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
        }
    data_identity = payload.get("data")
    if not isinstance(data_identity, dict):
        return payload
    typed_data_identity = cast(dict[str, Any], data_identity)
    data_config = typed_data_identity.get("config")
    if data_config is None:
        return payload
    portable_config = _portable_data_config(data_config)
    typed_data_identity["config"] = portable_config
    typed_data_identity["config_sha256"] = hashlib.sha256(
        _canonical_json(portable_config).encode("utf-8")
    ).hexdigest()
    return payload


def _identity_differences(expected: Any, actual: Any, path: str = "identity") -> list[str]:
    """Return concise paths that differ, capped to keep resume errors readable."""

    if len(path) > 512:
        return [path]
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_mapping = cast(Mapping[object, Any], expected)
        actual_mapping = cast(Mapping[object, Any], actual)
        differences: list[str] = []
        keys = sorted(set(expected_mapping) | set(actual_mapping), key=str)
        for key in keys:
            child = f"{path}.{key}"
            if key not in expected_mapping or key not in actual_mapping:
                differences.append(child)
            else:
                differences.extend(
                    _identity_differences(expected_mapping[key], actual_mapping[key], child)
                )
            if len(differences) >= 8:
                return differences[:8]
        return differences
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        expected_sequence = cast(list[Any] | tuple[Any, ...], expected)
        actual_sequence = cast(list[Any] | tuple[Any, ...], actual)
        differences = []
        if len(expected_sequence) != len(actual_sequence):
            differences.append(f"{path}.length")
        for index, (expected_item, actual_item) in enumerate(
            zip(expected_sequence, actual_sequence, strict=False)
        ):
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
        if "pipeline" in expected_identity:
            raise ValueError(
                "checkpoint has no recorded pipeline identity. Refusing to resume because "
                "its foundation-vs-translation-only ancestry cannot be verified. Start a "
                "new training.output_dir or resume from a checkpoint created by this pipeline."
            )
        warnings.warn(
            "이전 버전 체크포인트에는 model/tokenizer/data identity가 없습니다. "
            "이번 재개에서는 안전한 동일성 검사를 건너뜁니다. 다음 저장부터는 identity가 기록됩니다.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    if not isinstance(stored_identity, Mapping):
        raise ValueError("checkpoint identity must be an object")
    expected = _normalize_identity_for_comparison(expected_identity)
    actual = _normalize_identity_for_comparison(cast(Mapping[str, Any], stored_identity))
    if "pipeline" in expected and "pipeline" not in actual:
        raise ValueError(
            "checkpoint has no recorded pipeline identity. Refusing to resume because "
            "its foundation-vs-translation-only ancestry cannot be verified. Start a new "
            "training.output_dir or resume from a checkpoint created by this pipeline."
        )
    if "objective" in expected and "objective" not in actual:
        # 목적함수 identity 가 없던 시절의 체크포인트입니다. 재개 자체를 막지는
        # 않되, 무엇을 검사하지 못했는지 밝힙니다 — 그 사이에 학습률이나 reward
        # 가중치가 바뀌었다면 이 재개는 과거 best 와 비교 불가능한 숫자를
        # 이어받습니다.
        warnings.warn(
            "이 체크포인트에는 목적함수/최적화 identity 가 없습니다(구버전). "
            "학습률·Adam 계수·EMA·MRT reward 가중치가 그대로인지 확인할 수 없으므로 "
            "그 부분의 동일성 검사는 건너뜁니다.",
            RuntimeWarning,
            stacklevel=3,
        )
        expected = {key: value for key, value in expected.items() if key != "objective"}
    if expected != actual:
        differences = _identity_differences(expected, actual)
        detail = ", ".join(differences) if differences else "unknown fields"
        raise ValueError(
            "checkpoint identity does not match the current model/tokenizer/data "
            f"({detail}). Refusing to resume with incompatible artifacts."
        )


def _dcp_mapping_probe(metadata: Any, root: str) -> dict[str, Any]:
    """Build a DCP target for one nested mapping from keys actually stored."""

    from torch.distributed.checkpoint._nested_dict import (  # pyright: ignore[reportPrivateImportUsage, reportUnknownVariableType]
        unflatten_state_dict,  # pyright: ignore[reportUnknownVariableType]
    )

    planner_data_raw = getattr(metadata, "planner_data", None)
    planner_data: Mapping[object, object] = (
        cast(Mapping[object, object], planner_data_raw)
        if isinstance(planner_data_raw, Mapping)
        else cast(Mapping[object, object], {})
    )
    flattened: dict[str, Any] = {}
    paths: dict[str, tuple[str | int, ...]] = {}
    for raw_key in metadata.state_dict_metadata:
        flat_key = str(raw_key)
        if flat_key != root and not flat_key.startswith(f"{root}."):
            continue
        raw_path = planner_data.get(raw_key)
        if raw_path is None:
            raw_path = planner_data.get(flat_key)
        if isinstance(raw_path, list):
            raw_parts = tuple(cast(list[object], raw_path))
        elif isinstance(raw_path, tuple):
            raw_parts = cast(tuple[object, ...], raw_path)
        else:
            raw_parts = ()
        if raw_path is not None:
            if not raw_parts or not all(type(part) in (str, int) for part in raw_parts):
                raise ValueError(
                    f"distributed checkpoint has an invalid {root} metadata path: {flat_key}"
                )
            path = tuple(cast(str | int, part) for part in raw_parts)
        else:
            path = tuple(flat_key.split("."))
        if not path or path[0] != root:
            raise ValueError(
                f"distributed checkpoint has an invalid {root} metadata path: {flat_key}"
            )
        flattened[flat_key] = None
        paths[flat_key] = path
    try:
        return cast(dict[str, Any], unflatten_state_dict(flattened, paths))
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"distributed checkpoint {root} metadata is malformed") from error


def _dcp_identity_probe(metadata: Any) -> dict[str, Any]:
    """Build an identity-only DCP target from keys actually in the checkpoint."""

    return _dcp_mapping_probe(metadata, "identity")


def _restore_expected_empty_mappings(
    actual: dict[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Restore empty mappings, which DCP intentionally omits from metadata."""

    for key, expected_value in expected.items():
        if key not in actual:
            if isinstance(expected_value, Mapping) and not expected_value:
                actual[key] = {}
            continue
        actual_value = actual[key]
        if isinstance(actual_value, dict) and isinstance(expected_value, Mapping):
            _restore_expected_empty_mappings(
                cast(dict[str, Any], actual_value),
                cast(Mapping[str, Any], expected_value),
            )


def _preflight_dcp_identity(
    path: Path,
    expected_identity: Mapping[str, Any] | None,
) -> int:
    """Validate DCP identity and step without touching live training tensors."""

    import torch.distributed.checkpoint as dcp

    try:
        metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
            path
        ).read_metadata()
    except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
        raise ValueError(
            f"distributed checkpoint metadata could not be preflighted: {path}"
        ) from error
    probe = _dcp_identity_probe(metadata)
    if "step" not in metadata.state_dict_metadata:
        raise ValueError("distributed checkpoint is missing step metadata")
    probe["step"] = 0
    try:
        dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            probe,
            checkpoint_id=path,
            no_dist=True,
        )
    except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
        raise ValueError(
            f"distributed checkpoint identity/step could not be preflighted: {path}"
        ) from error
    stored_identity = probe.get("identity")
    if isinstance(stored_identity, dict) and expected_identity is not None:
        _restore_expected_empty_mappings(
            cast(dict[str, Any], stored_identity),
            cast(Mapping[str, Any], _json_compatible(expected_identity)),
        )
    _validate_identity(probe, expected_identity)
    step = probe.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("distributed checkpoint step must be an integer")
    return step


def _preflight_dcp_stage_transfer(
    path: Path,
    expected_identity: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Validate ancestry and step before DCP touches the live model."""

    import torch.distributed.checkpoint as dcp

    try:
        metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
            path
        ).read_metadata()
    except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
        raise ValueError(
            f"distributed stage-transfer metadata could not be preflighted: {path}"
        ) from error
    probe = _dcp_identity_probe(metadata)
    if "step" not in metadata.state_dict_metadata:
        raise ValueError("distributed stage-transfer checkpoint is missing step metadata")
    probe["step"] = 0
    try:
        dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            probe,
            checkpoint_id=path,
            no_dist=True,
        )
    except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
        raise ValueError(
            f"distributed stage-transfer identity/step could not be preflighted: {path}"
        ) from error
    stored_identity = probe.get("identity")
    if isinstance(stored_identity, dict) and expected_identity is not None:
        _restore_expected_empty_mappings(
            cast(dict[str, Any], stored_identity),
            cast(Mapping[str, Any], _json_compatible(expected_identity)),
        )
    _validate_stage_transfer(probe, expected_identity, source=path)
    step = probe.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("distributed stage-transfer checkpoint step must be an integer")
    return probe


def preflight_checkpoint_identity(
    path: str | Path,
    context: DistributedContext,
    expected_identity: Mapping[str, Any] | None,
) -> int | None:
    """Validate resume provenance and step before mutating training state."""

    path = resolve_checkpoint_source(path, context)
    if context.distributed:
        return _preflight_dcp_identity(path, expected_identity)
    try:
        loaded = torch.load(
            path / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise RuntimeError(
            "checkpoint identity could not be loaded with PyTorch's safe weights-only loader"
        ) from error
    loaded_state = _validate_loaded_state(loaded)
    _validate_identity(loaded_state, expected_identity)
    return int(loaded_state["step"])


def inspect_checkpoint_identity(
    path: str | Path,
    context: DistributedContext,
) -> dict[str, Any]:
    """Read the authenticated recorded identity without mutating live state.

    Distributed callers should invoke this inside a verified generation lease;
    ``resolve_checkpoint_source`` then performs only the lease's small marker
    check instead of hashing every shard again.
    """

    source = resolve_checkpoint_source(path, context)
    if context.distributed:
        import torch.distributed.checkpoint as dcp

        try:
            metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
                source
            ).read_metadata()
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint metadata could not be inspected: {source}"
            ) from error
        probe = _dcp_identity_probe(metadata)
        try:
            dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
                probe,
                checkpoint_id=source,
                no_dist=True,
            )
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint identity could not be inspected: {source}"
            ) from error
        raw_identity = probe.get("identity")
    else:
        try:
            loaded = torch.load(
                source / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as error:
            raise RuntimeError(
                "checkpoint identity could not be inspected with PyTorch's safe loader"
            ) from error
        loaded_state = _validate_loaded_state(loaded)
        raw_identity = loaded_state.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("checkpoint has no recorded identity object")
    normalized = _json_compatible(cast(Mapping[str, Any], raw_identity))
    if not isinstance(normalized, dict):
        raise ValueError("checkpoint identity must normalize to an object")
    return cast(dict[str, Any], normalized)


def inspect_checkpoint_training_state(
    path: str | Path,
    context: DistributedContext,
) -> dict[str, Any]:
    """Read authenticated scalar progress metadata without loading live tensors.

    Distributed callers should keep this inspection inside the same verified
    generation lease that will later authorize the full resume load.
    """

    source = resolve_checkpoint_source(path, context)
    if context.distributed:
        import torch.distributed.checkpoint as dcp

        try:
            metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
                source
            ).read_metadata()
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint metadata could not be inspected: {source}"
            ) from error
        probe = _dcp_mapping_probe(metadata, "training_state")
        if not probe:
            return {}
        try:
            dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
                probe,
                checkpoint_id=source,
                no_dist=True,
            )
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint training state could not be inspected: {source}"
            ) from error
        raw_training_state = probe.get("training_state")
    else:
        try:
            loaded = torch.load(
                source / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as error:
            raise RuntimeError(
                "checkpoint training state could not be inspected with PyTorch's safe loader"
            ) from error
        loaded_state = _validate_loaded_state(loaded)
        raw_training_state = loaded_state.get("training_state")
    if not isinstance(raw_training_state, Mapping):
        raise ValueError("checkpoint has no recorded training_state object")
    return {
        str(key): value for key, value in cast(Mapping[object, Any], raw_training_state).items()
    }


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "keys": torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                numpy_state[1].copy()
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        # Each distributed rank owns one current CUDA device. Capturing every
        # visible device here creates CUDA contexts on peer GPUs and couples a
        # rank-local RNG file to the machine's total device count.
        state["torch_cuda"] = torch.cuda.get_rng_state().cpu()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    _validate_rng_state(state)
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    assert isinstance(python_state, tuple)
    assert isinstance(numpy_state, Mapping)
    typed_numpy_state = cast(Mapping[str, Any], numpy_state)
    assert isinstance(typed_numpy_state.get("keys"), torch.Tensor)
    assert isinstance(torch_cpu_state, torch.Tensor)

    random.setstate(cast(Any, python_state))
    numpy_keys = cast(torch.Tensor, typed_numpy_state["keys"])
    np.random.set_state(
        (
            str(typed_numpy_state["algorithm"]),
            numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
            int(typed_numpy_state["position"]),
            int(typed_numpy_state["has_gauss"]),
            float(typed_numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu_state.detach().cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if isinstance(cuda_state, torch.Tensor):
            selected_cuda_state = cuda_state
        elif isinstance(cuda_state, list) and all(
            isinstance(item, torch.Tensor) for item in cast(list[object], cuda_state)
        ):
            # Backward compatibility for checkpoints that stored every visible
            # device. Restore only this rank's current device instead of
            # initializing or overwriting RNG streams owned by other ranks.
            current_device = torch.cuda.current_device()
            typed_cuda_state = cast(list[torch.Tensor], cuda_state)
            if current_device >= len(typed_cuda_state):
                raise ValueError(
                    "legacy checkpoint CUDA RNG state has no entry for the current device"
                )
            selected_cuda_state = typed_cuda_state[current_device]
        else:
            raise ValueError("checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state(selected_cuda_state.detach().cpu())


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate a complete RNG snapshot without changing process-global RNGs."""

    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    if not isinstance(python_state, tuple):
        raise ValueError("checkpoint Python RNG state is invalid")
    try:
        random.Random().setstate(cast(Any, python_state))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    if not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    typed_numpy_state = cast(Mapping[str, Any], numpy_state)
    numpy_keys = typed_numpy_state.get("keys")
    if not isinstance(numpy_keys, torch.Tensor):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    try:
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(
            (
                str(typed_numpy_state["algorithm"]),
                numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
                int(typed_numpy_state["position"]),
                int(typed_numpy_state["has_gauss"]),
                float(typed_numpy_state["cached_gaussian"]),
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint NumPy RNG state is invalid") from error
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise ValueError("checkpoint torch RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu_state.detach().cpu())
    except (RuntimeError, TypeError) as error:
        raise ValueError("checkpoint torch RNG state is invalid") from error
    cuda_state = state.get("torch_cuda")
    if cuda_state is None:
        return
    cuda_states = (
        [cuda_state]
        if isinstance(cuda_state, torch.Tensor)
        else cast(list[object], cuda_state)
        if isinstance(cuda_state, list)
        else []
    )
    if not cuda_states or not all(
        isinstance(item, torch.Tensor) and item.ndim == 1 and item.dtype == torch.uint8
        for item in cuda_states
    ):
        raise ValueError("checkpoint CUDA RNG state is invalid")


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
        _fsync_directory(destination.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    """Flush a closed regular file before publishing a completion marker."""

    # Windows' ``os.fsync`` delegates to ``_commit`` and rejects a descriptor
    # opened read-only with EBADF. DCP artifacts are files owned by this
    # publication transaction, so open without truncation but with write access
    # on every platform before issuing the durable flush.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist namespace mutations where directory fsync is supported."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [root] + [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_dir() and not candidate.is_symlink()
    ]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _run_checkpoint_io_action(
    action: Callable[[], None],
    context: DistributedContext,
    *,
    operation: str,
    rank_zero_only: bool,
) -> None:
    """Run slow checkpoint I/O without parking peers in one long collective.

    The I/O runs in worker threads while every rank's main thread exchanges a
    tiny completion heartbeat. No individual collective includes the long I/O,
    so a filesystem scan may safely exceed the process-group operation timeout.
    Once every participating worker finishes, the lowest failing rank broadcasts
    a bounded diagnostic before any rank raises.
    """

    should_run = not rank_zero_only or context.is_main
    if not context.distributed or context.world_size == 1:
        if should_run:
            action()
        return

    import torch.distributed as dist

    completed = threading.Event()
    abort_requested = threading.Event()
    errors: list[BaseException] = []

    def run_action() -> None:
        try:
            if not abort_requested.is_set():
                action()
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    worker: threading.Thread | None = None
    if should_run:
        worker = threading.Thread(
            target=run_action,
            name=f"sion-checkpoint-{operation}",
            daemon=False,
        )
        worker.start()
    else:
        completed.set()

    try:
        all_completed = torch.zeros((), dtype=torch.int32, device=context.device)
        while True:
            all_completed.fill_(int(completed.is_set()))
            dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
                all_completed,
                op=dist.ReduceOp.MIN,
            )
            if bool(all_completed.item()):
                break
            if should_run:
                completed.wait(_CHECKPOINT_IO_HEARTBEAT_SECONDS)
            else:
                time.sleep(_CHECKPOINT_IO_HEARTBEAT_SECONDS)
    except BaseException:
        abort_requested.set()
        raise
    finally:
        if worker is not None:
            # A failed collective must not let an orphaned worker keep hashing,
            # renaming, or publishing checkpoint bytes after this call returns.
            # Python cannot safely cancel arbitrary filesystem I/O, so wait for
            # ownership to end before propagating the collective failure.
            worker.join()

    failure_source = torch.tensor(
        context.rank if errors else context.world_size,
        dtype=torch.int64,
        device=context.device,
    )
    dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
        failure_source,
        op=dist.ReduceOp.MIN,
    )
    source_rank = int(failure_source.item())
    if source_rank == context.world_size:
        return
    local_error = errors[0] if errors else None
    local_detail = None
    if context.rank == source_rank and local_error is not None:
        rendered = " ".join(str(local_error).splitlines())
        local_detail = f"{type(local_error).__name__}: {rendered}"[:2048]
    detail = broadcast_text(
        local_detail,
        context,
        source=source_rank,
    )
    if context.rank == source_rank and local_error is not None:
        raise local_error
    raise RuntimeError(f"{operation} failed on distributed rank {source_rank}: {detail}")


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
) -> _VerifiedDcpPublication:
    if type(step) is not int or step < 0:
        raise ValueError("distributed checkpoint step must be a non-negative integer")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("distributed checkpoint world size must be a positive integer")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"distributed checkpoint is not a regular directory: {directory}")
    marker = directory / DCP_COMPLETION_FILENAME
    if marker.is_symlink():
        raise ValueError(f"distributed checkpoint completion marker is a symlink: {marker}")
    inventory: list[dict[str, Any]] = []
    for artifact in sorted(directory.rglob("*")):
        if artifact == marker:
            continue
        artifact_stat = artifact.lstat()
        if stat.S_ISLNK(artifact_stat.st_mode):
            raise ValueError(f"distributed checkpoint contains a symlink: {artifact}")
        if stat.S_ISDIR(artifact_stat.st_mode):
            continue
        if not stat.S_ISREG(artifact_stat.st_mode):
            raise ValueError(f"distributed checkpoint contains a non-regular file: {artifact}")
        artifact_sha256 = _sha256_file(artifact)
        _fsync_file(artifact)
        final_stat = artifact.lstat()
        initial_identity = (
            artifact_stat.st_mode,
            artifact_stat.st_size,
            artifact_stat.st_mtime_ns,
            artifact_stat.st_ctime_ns,
            artifact_stat.st_dev,
            artifact_stat.st_ino,
        )
        final_identity = (
            final_stat.st_mode,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
            final_stat.st_dev,
            final_stat.st_ino,
        )
        if initial_identity != final_identity:
            raise ValueError(f"distributed checkpoint artifact changed while hashing: {artifact}")
        inventory.append(
            {
                "path": artifact.relative_to(directory).as_posix(),
                "size": final_stat.st_size,
                "sha256": artifact_sha256,
            }
        )
    _fsync_directory_tree(directory)
    payload = {
        "schema": DCP_COMPLETION_SCHEMA,
        "step": int(step),
        "world_size": int(world_size),
        "files": inventory,
    }
    _validated_dcp_v2_inventory(payload, world_size=world_size)
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
        _fsync_directory(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    stored_payload, marker_sha256 = _read_dcp_completion_payload(marker)
    if stored_payload != payload:
        raise ValueError("distributed checkpoint completion marker changed while being written")
    _validated_dcp_v2_inventory(stored_payload, world_size=world_size)
    return _VerifiedDcpPublication(
        source=_canonical_checkpoint_path(directory),
        world_size=world_size,
        marker_sha256=marker_sha256,
    )


def _publish_dcp_staging(
    staging: Path,
    destination: Path,
    *,
    world_size: int,
    verified: object,
) -> None:
    """Publish a complete DCP directory while retaining one recoverable version.

    ``verified`` is the path-bound capability returned by the immediately
    preceding ``_write_dcp_completion`` call. The private staging namespace is
    protected by the run-wide training lock; an uncooperative external writer
    remains outside the guarantees of PyTorch's path-based DCP API.
    """

    previous = _dcp_sibling(destination, "previous")
    if not isinstance(verified, _VerifiedDcpPublication):
        raise TypeError("distributed checkpoint publication requires a verified capability")
    if verified.source != _canonical_checkpoint_path(staging):
        raise ValueError("distributed checkpoint publication capability source does not match")
    if verified.world_size != world_size:
        raise ValueError("distributed checkpoint publication capability world size does not match")
    marker = staging / DCP_COMPLETION_FILENAME
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("distributed checkpoint staging marker is not a regular file")
    marker_payload, marker_sha256 = _read_dcp_completion_payload(marker)
    if marker_sha256 != verified.marker_sha256:
        raise ValueError("distributed checkpoint staging marker changed before publication")
    _validated_dcp_v2_inventory(marker_payload, world_size=world_size)
    destination_status = _dcp_completion_status(destination, world_size=world_size)
    previous_status = _dcp_completion_status(previous, world_size=world_size)
    destination_recoverable = destination_status == "valid"
    previous_recoverable = previous_status == "valid"
    moved_previous = False
    if destination_recoverable:
        _remove_path(previous)
        os.replace(destination, previous)
        _fsync_directory(destination.parent)
        moved_previous = True
    else:
        # A corrupt current publication must never replace the last valid
        # predecessor. Remove only the corrupt current; keep a recoverable
        # predecessor until the new staging directory is installed.
        _remove_path(destination)
        if not previous_recoverable:
            _remove_path(previous)
    try:
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if moved_previous and not destination.exists():
            os.replace(previous, destination)
        _fsync_directory(destination.parent)
        raise


def _read_dcp_completion_payload(marker: Path) -> tuple[dict[str, Any], str]:
    try:
        encoded = marker.read_bytes()
        raw_payload: object = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"distributed checkpoint completion marker is invalid: {marker}"
        ) from error
    if not isinstance(raw_payload, dict):
        raise ValueError("distributed checkpoint completion marker must be an object")
    return cast(dict[str, Any], raw_payload), _sha256_bytes(encoded)


def _validated_dcp_v2_inventory(
    marker_payload: Mapping[str, Any],
    *,
    world_size: int,
) -> list[dict[str, Any]]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("distributed checkpoint world size must be a positive integer")
    if marker_payload.get("schema") != DCP_COMPLETION_SCHEMA:
        raise ValueError("distributed checkpoint completion marker is not v2")
    if set(marker_payload) != {"schema", "step", "world_size", "files"}:
        raise ValueError("distributed checkpoint completion marker fields are invalid")
    if isinstance(marker_payload.get("step"), bool) or not isinstance(
        marker_payload.get("step"), int
    ):
        raise ValueError("distributed checkpoint completion step must be an integer")
    marker_world_size = marker_payload.get("world_size")
    if (
        isinstance(marker_world_size, bool)
        or not isinstance(marker_world_size, int)
        or marker_world_size != world_size
    ):
        raise ValueError(
            "distributed checkpoint completion world size does not match "
            f"({marker_world_size!r} != {world_size})"
        )
    raw_inventory = marker_payload.get("files")
    if not isinstance(raw_inventory, list):
        raise ValueError("distributed checkpoint completion inventory must be a list")
    inventory: list[dict[str, Any]] = []
    inventory_paths: set[str] = set()
    for raw_entry in cast(list[object], raw_inventory):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("distributed checkpoint completion inventory entry is invalid")
        entry_mapping = cast(Mapping[object, object], raw_entry)
        if set(entry_mapping) != {"path", "size", "sha256"}:
            raise ValueError("distributed checkpoint completion inventory entry is invalid")
        raw_path = entry_mapping.get("path")
        raw_size = entry_mapping.get("size")
        raw_sha256 = entry_mapping.get("sha256")
        if not isinstance(raw_path, str):
            raise ValueError("distributed checkpoint completion inventory path is invalid")
        normalized_path = PurePosixPath(raw_path)
        if (
            not raw_path
            or not normalized_path.parts
            or "\\" in raw_path
            or normalized_path.is_absolute()
            or normalized_path.as_posix() != raw_path
            or ".." in normalized_path.parts
            or raw_path == DCP_COMPLETION_FILENAME
            or raw_path in inventory_paths
        ):
            raise ValueError(
                f"distributed checkpoint completion inventory path is unsafe: {raw_path!r}"
            )
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ValueError(
                f"distributed checkpoint completion inventory size is invalid: {raw_path!r}"
            )
        if (
            not isinstance(raw_sha256, str)
            or len(raw_sha256) != 64
            or raw_sha256 != raw_sha256.lower()
            or any(character not in "0123456789abcdef" for character in raw_sha256)
        ):
            raise ValueError(
                f"distributed checkpoint completion inventory digest is invalid: {raw_path!r}"
            )
        entry = {"path": raw_path, "size": raw_size, "sha256": raw_sha256}
        inventory.append(entry)
        inventory_paths.add(raw_path)
    required_paths = {".metadata"} | {f"rng-rank-{rank:05d}.pt" for rank in range(world_size)}
    missing = sorted(required_paths - inventory_paths)
    if missing:
        raise ValueError(
            "distributed checkpoint completion inventory is missing required files: "
            + ", ".join(missing)
        )
    expected_rng_paths = required_paths - {".metadata"}
    stored_rng_paths = {
        artifact_path
        for artifact_path in inventory_paths
        if artifact_path.startswith("rng-rank-") and artifact_path.endswith(".pt")
    }
    if stored_rng_paths != expected_rng_paths:
        raise ValueError("distributed checkpoint completion inventory has unexpected RNG files")
    return inventory


def _dcp_completion_status(
    path: Path,
    *,
    world_size: int,
    expected_marker_sha256: str | None = None,
) -> str:
    marker = path / DCP_COMPLETION_FILENAME
    if not marker.exists() and not marker.is_symlink():
        return "absent"
    if marker.is_symlink() or not marker.is_file():
        return "invalid"
    try:
        marker_payload, marker_sha256 = _read_dcp_completion_payload(marker)
    except ValueError:
        return "invalid"
    if expected_marker_sha256 is not None and marker_sha256 != expected_marker_sha256:
        return "invalid"
    if isinstance(marker_payload.get("step"), bool) or not isinstance(
        marker_payload.get("step"), int
    ):
        return "invalid"
    marker_world_size = marker_payload.get("world_size")
    if (
        isinstance(marker_world_size, bool)
        or not isinstance(marker_world_size, int)
        or marker_world_size != world_size
    ):
        return "invalid"
    schema = marker_payload.get("schema")
    if schema == CHECKPOINT_SCHEMA:
        if set(marker_payload) != {"schema", "step", "world_size"}:
            return "invalid"
        required = [path / ".metadata"] + [
            path / f"rng-rank-{rank:05d}.pt" for rank in range(world_size)
        ]
        return (
            "legacy"
            if all(item.is_file() and not item.is_symlink() for item in required)
            else "invalid"
        )
    try:
        stored_inventory = _validated_dcp_v2_inventory(
            marker_payload,
            world_size=world_size,
        )
    except ValueError:
        return "invalid"
    try:
        current_inventory: list[dict[str, Any]] = []
        for artifact in sorted(path.rglob("*")):
            if artifact == marker:
                continue
            before = artifact.lstat()
            if stat.S_ISLNK(before.st_mode):
                return "invalid"
            if stat.S_ISDIR(before.st_mode):
                continue
            if not stat.S_ISREG(before.st_mode):
                return "invalid"
            artifact_sha256 = _sha256_file(artifact)
            after = artifact.lstat()
            before_identity = (
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_dev,
                before.st_ino,
            )
            after_identity = (
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_dev,
                after.st_ino,
            )
            if before_identity != after_identity:
                return "invalid"
            current_inventory.append(
                {
                    "path": artifact.relative_to(path).as_posix(),
                    "size": after.st_size,
                    "sha256": artifact_sha256,
                }
            )
    except OSError:
        return "invalid"
    if stored_inventory != current_inventory:
        return "invalid"
    try:
        _, final_marker_sha256 = _read_dcp_completion_payload(marker)
    except ValueError:
        return "invalid"
    return "valid" if final_marker_sha256 == marker_sha256 else "invalid"


def _resolve_dcp_checkpoint(path: Path, *, world_size: int) -> Path:
    """Resolve a complete current DCP checkpoint, or its retained predecessor."""

    current_status = _dcp_completion_status(path, world_size=world_size)
    if current_status == "valid":
        return path
    previous = _dcp_sibling(path, "previous")
    previous_status = _dcp_completion_status(previous, world_size=world_size)
    if previous_status == "valid":
        warnings.warn(
            f"{path} is incomplete; resuming from retained checkpoint {previous}",
            RuntimeWarning,
            stacklevel=3,
        )
        return previous
    raise FileNotFoundError(
        f"no authenticated v2 distributed checkpoint found at {path} "
        f"(current={current_status}, previous={previous_status})"
    )


def _logical_checkpoint_current(path: Path) -> Path:
    previous_suffix = ".previous"
    if path.name.startswith(".") and path.name.endswith(previous_suffix):
        current_name = path.name[1 : -len(previous_suffix)]
        if current_name:
            return path.with_name(current_name)
    return path


def _is_lightweight_dcp_candidate(path: Path, *, world_size: int) -> bool:
    """Check only cheap v2 structure; this does not authenticate shard bytes."""

    marker = path / DCP_COMPLETION_FILENAME
    try:
        if path.is_symlink() or marker.is_symlink() or not marker.is_file():
            return False
        marker_payload, _ = _read_dcp_completion_payload(marker)
        _validated_dcp_v2_inventory(
            marker_payload,
            world_size=world_size,
        )
        return all(
            required.is_file() and not required.is_symlink()
            for required in _required_dcp_files(path, world_size=world_size)
        )
    except (OSError, ValueError):
        return False


def checkpoint_generation_candidates(
    path: str | Path,
    context: DistributedContext,
) -> tuple[Path, ...]:
    """Return current/previous v2 DCP candidates without hashing large shards.

    This is an ordering helper, not an authentication result. Rank 0 may use it
    to enumerate current then retained previous, broadcast each exact path and
    small marker digest, and have every rank attempt
    ``verified_checkpoint_source_lease``. A candidate must never be loaded merely
    because it appears in this tuple.
    """

    if not context.distributed:
        raise ValueError("distributed checkpoint candidates require DCP topology")
    if context.world_size <= 0:
        raise ValueError("distributed checkpoint world size must be positive")
    current = _logical_checkpoint_current(Path(path))
    _reject_mixed_checkpoint_formats(current)
    previous = _dcp_sibling(current, "previous")
    return tuple(
        candidate
        for candidate in (current, previous)
        if _is_lightweight_dcp_candidate(candidate, world_size=context.world_size)
    )


def checkpoint_generation_candidate_metadata(
    path: str | Path,
    context: DistributedContext,
) -> tuple[str, int]:
    """Return a candidate's small marker digest and declared step.

    This helper validates only the v2 marker structure and required-file
    presence. The returned values are rank-zero control-plane hints, not proof
    that shard contents are authentic. Broadcast them and pass the digest to
    ``verified_checkpoint_source_lease`` before preflight or load.
    """

    candidate = Path(path)
    if not context.distributed:
        raise ValueError("distributed checkpoint candidate metadata requires DCP topology")
    if context.world_size <= 0:
        raise ValueError("distributed checkpoint world size must be positive")
    if not _is_lightweight_dcp_candidate(candidate, world_size=context.world_size):
        raise ValueError(f"checkpoint is not a structurally valid v2 DCP candidate: {candidate}")
    marker_payload, marker_sha256 = _read_dcp_completion_payload(
        candidate / DCP_COMPLETION_FILENAME
    )
    _validated_dcp_v2_inventory(marker_payload, world_size=context.world_size)
    marker_step = marker_payload.get("step")
    if isinstance(marker_step, bool) or not isinstance(marker_step, int) or marker_step < 0:
        raise ValueError("distributed checkpoint completion marker step is invalid")
    return marker_sha256, marker_step


def checkpoint_path_exists(path: str | Path) -> bool:
    """Return whether checkpoint-like artifacts exist, without authenticating them.

    This is discovery only. Callers deciding whether to resume or preserve best
    state must use ``verified_checkpoint_generation_lease`` (DCP) or an exact
    local preflight before trusting the candidate.
    """

    path = _logical_checkpoint_current(Path(path))
    previous = _dcp_sibling(path, "previous")
    return any(
        (
            (path / "checkpoint.pt").is_file(),
            (previous / "checkpoint.pt").is_file(),
            (path / ".metadata").is_file(),
            (previous / ".metadata").is_file(),
        )
    )


def resolve_checkpoint_source(
    path: str | Path,
    context: DistributedContext,
) -> Path:
    """Resolve the exact authenticated checkpoint generation a load will use."""

    path = Path(path)
    _reject_mixed_checkpoint_formats(path)
    if context.distributed:
        leased = _resolve_active_checkpoint_lease(path, context)
        if leased is not None:
            return leased
        resolved = _resolve_dcp_checkpoint(path, world_size=context.world_size)
        _reject_mixed_checkpoint_formats(resolved)
        return resolved
    if (path / "checkpoint.pt").is_file():
        return path
    raise FileNotFoundError(f"no local checkpoint payload found at {path}")


def _reject_mixed_checkpoint_formats(path: Path) -> None:
    """Reject a logical current/previous pair with topology-dependent formats."""

    current = _logical_checkpoint_current(path)
    previous = _dcp_sibling(current, "previous")
    generations = (current, previous)
    has_local_payload = any((generation / "checkpoint.pt").is_file() for generation in generations)
    has_distributed_payload = any(
        (generation / filename).is_file()
        for generation in generations
        for filename in (".metadata", DCP_COMPLETION_FILENAME)
    )
    if has_local_payload and has_distributed_payload:
        raise ValueError(
            "checkpoint generation pair mixes local checkpoint.pt and distributed "
            f"checkpoint artifacts: current={current}, previous={previous}"
        )


def _validate_sha256_digest(value: str, *, label: str) -> None:
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _required_dcp_files(path: Path, *, world_size: int) -> list[Path]:
    return [path / ".metadata"] + [path / f"rng-rank-{rank:05d}.pt" for rank in range(world_size)]


def register_verified_checkpoint_source(
    path: str | Path,
    context: DistributedContext,
    expected_artifact_sha256: str,
) -> Path:
    """Authenticate one rank-consistent v2 DCP source against rank 0's marker.

    Every rank hashes its complete visible inventory. A marker digest alone, or
    mutable file-stat metadata, cannot authenticate a non-shared filesystem.
    In distributed execution every rank must call this function together; short
    heartbeats keep the full hashes outside any single collective's timeout.
    """

    path = Path(path)
    if context.world_size <= 0 or context.rank < 0 or context.rank >= context.world_size:
        raise ValueError("distributed checkpoint context rank/world size is invalid")
    _validate_sha256_digest(
        expected_artifact_sha256,
        label="expected distributed checkpoint artifact digest",
    )

    def authenticate_visible_inventory() -> None:
        _reject_mixed_checkpoint_formats(path)
        marker = path / DCP_COMPLETION_FILENAME
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("verified checkpoint registration requires a regular v2 marker")
        marker_payload, marker_sha256 = _read_dcp_completion_payload(marker)
        if marker_sha256 != expected_artifact_sha256:
            raise ValueError(
                "distributed checkpoint completion marker digest does not match "
                f"rank 0 ({marker_sha256} != {expected_artifact_sha256})"
            )
        _validated_dcp_v2_inventory(
            marker_payload,
            world_size=context.world_size,
        )
        if (
            _dcp_completion_status(
                path,
                world_size=context.world_size,
                expected_marker_sha256=expected_artifact_sha256,
            )
            != "valid"
        ):
            raise ValueError(f"distributed checkpoint inventory verification failed: {path}")

    _run_checkpoint_io_action(
        authenticate_visible_inventory,
        context,
        operation="distributed checkpoint inventory authentication",
        rank_zero_only=False,
    )
    return path


def _active_checkpoint_lease() -> _VerifiedCheckpointLease | None:
    return _VERIFIED_CHECKPOINT_LEASE.active


def _revoke_checkpoint_lease() -> None:
    _VERIFIED_CHECKPOINT_LEASE.active = None


def _resolve_active_checkpoint_lease(
    path: Path,
    context: DistributedContext,
) -> Path | None:
    lease = _active_checkpoint_lease()
    if lease is None:
        return None
    if not context.distributed:
        _revoke_checkpoint_lease()
        raise ValueError("distributed checkpoint verification lease used with local topology")
    if lease.world_size != context.world_size:
        _revoke_checkpoint_lease()
        raise ValueError("distributed checkpoint verification lease world size does not match")
    if lease.source != _canonical_checkpoint_path(path):
        _revoke_checkpoint_lease()
        raise ValueError("distributed checkpoint verification lease source does not match")
    marker = path / DCP_COMPLETION_FILENAME
    try:
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("leased distributed checkpoint marker is not a regular file")
        marker_payload, marker_sha256 = _read_dcp_completion_payload(marker)
        if marker_sha256 != lease.marker_sha256:
            raise ValueError("leased distributed checkpoint marker digest changed")
        _validated_dcp_v2_inventory(
            marker_payload,
            world_size=context.world_size,
        )
    except ValueError:
        _revoke_checkpoint_lease()
        raise
    return path


@contextmanager
def verified_checkpoint_source_lease(
    path: str | Path,
    context: DistributedContext,
    expected_artifact_sha256: str,
) -> Generator[Path, None, None]:
    """Fully authenticate DCP once for one locked resume call-chain.

    The caller must hold the run-wide ``training_run_lock`` for the entire
    context so cooperating code cannot mutate the checkpoint namespace. Within
    that contract, existing path APIs recheck the exact canonical source, world
    size, v2 marker structure, and marker digest but avoid repeated shard hashes.
    PyTorch DCP exposes paths rather than immutable file handles, so this lease
    cannot prevent an uncooperative external writer from creating a TOCTOU race.

    In distributed execution every rank must enter and leave this context in the
    same control-flow order. The lease is process- and thread-local, opaque to
    callers, non-nestable, and unconditionally discarded on exit or mismatch.
    """

    if not context.distributed:
        raise ValueError("checkpoint verification leases require distributed DCP topology")
    if _active_checkpoint_lease() is not None:
        raise RuntimeError("checkpoint verification leases cannot be nested")
    source = register_verified_checkpoint_source(
        path,
        context,
        expected_artifact_sha256,
    )
    lease = _VerifiedCheckpointLease(
        source=_canonical_checkpoint_path(source),
        world_size=context.world_size,
        marker_sha256=expected_artifact_sha256,
    )
    _VERIFIED_CHECKPOINT_LEASE.active = lease
    try:
        yield source
    finally:
        if _active_checkpoint_lease() == lease:
            _revoke_checkpoint_lease()


def _coordinated_checkpoint_generation_bindings(
    path: Path,
    context: DistributedContext,
    *,
    expected_artifact_sha256: str | None,
) -> tuple[tuple[Path, str, int], ...]:
    """Rank 0 orders cheap candidate hints and broadcasts the exact bindings."""

    prepared: list[dict[str, object]] = []

    def prepare_candidates() -> None:
        for candidate in checkpoint_generation_candidates(path, context):
            artifact_sha256, marker_step = checkpoint_generation_candidate_metadata(
                candidate,
                context,
            )
            if expected_artifact_sha256 is not None and artifact_sha256 != expected_artifact_sha256:
                continue
            prepared.append(
                {
                    "source": str(candidate.resolve()),
                    "artifact_sha256": artifact_sha256,
                    "step": marker_step,
                }
            )

    _run_checkpoint_io_action(
        prepare_candidates,
        context,
        operation="distributed checkpoint candidate discovery",
        rank_zero_only=True,
    )
    encoded_here = json.dumps(prepared, ensure_ascii=True, separators=(",", ":"))
    encoded = (
        encoded_here
        if context.world_size == 1
        else broadcast_text(encoded_here if context.is_main else None, context)
    )
    try:
        raw_bindings: object = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "distributed checkpoint candidate broadcast is not valid JSON"
        ) from error
    if not isinstance(raw_bindings, list):
        raise RuntimeError("distributed checkpoint candidate broadcast is malformed")
    typed_bindings = cast(list[object], raw_bindings)
    if len(typed_bindings) > 2:
        raise RuntimeError("distributed checkpoint candidate broadcast is malformed")

    current = _logical_checkpoint_current(path)
    allowed_sources = {
        _canonical_checkpoint_path(current),
        _canonical_checkpoint_path(_dcp_sibling(current, "previous")),
    }
    bindings: list[tuple[Path, str, int]] = []
    seen_sources: set[str] = set()
    for raw_binding in typed_bindings:
        if not isinstance(raw_binding, dict):
            raise RuntimeError("distributed checkpoint candidate binding is malformed")
        typed_binding = cast(dict[str, object], raw_binding)
        if set(typed_binding) != {
            "source",
            "artifact_sha256",
            "step",
        }:
            raise RuntimeError("distributed checkpoint candidate binding is malformed")
        source_value = typed_binding.get("source")
        artifact_sha256 = typed_binding.get("artifact_sha256")
        marker_step = typed_binding.get("step")
        if not isinstance(source_value, str) or not source_value:
            raise RuntimeError("distributed checkpoint candidate source is malformed")
        if not isinstance(artifact_sha256, str):
            raise RuntimeError("distributed checkpoint candidate digest is malformed")
        _validate_sha256_digest(
            artifact_sha256,
            label="distributed checkpoint candidate marker digest",
        )
        if isinstance(marker_step, bool) or not isinstance(marker_step, int) or marker_step < 0:
            raise RuntimeError("distributed checkpoint candidate step is malformed")
        source = Path(source_value)
        canonical_source = _canonical_checkpoint_path(source)
        if canonical_source not in allowed_sources or canonical_source in seen_sources:
            raise RuntimeError(
                "distributed checkpoint candidate source is outside the logical pair"
            )
        if expected_artifact_sha256 is not None and artifact_sha256 != expected_artifact_sha256:
            raise RuntimeError("distributed checkpoint candidate does not match the bound marker")
        seen_sources.add(canonical_source)
        bindings.append((source, artifact_sha256, marker_step))
    return tuple(bindings)


def checkpoint_generation_bindings(
    path: str | Path,
    context: DistributedContext,
) -> tuple[CheckpointGenerationBinding, ...]:
    """Return ordered current/previous digest bindings for semantic attempts.

    Discovery never grants permission to load. Distributed bindings authenticate
    only the small v2 marker contract; local bindings hash the payload file. Each
    returned digest must be supplied to ``verified_checkpoint_generation_lease``,
    which performs the full inventory and identity checks inside a one-shot lease.
    """

    logical_path = Path(path)
    if context.distributed:
        return tuple(
            CheckpointGenerationBinding(source, artifact_sha256)
            for source, artifact_sha256, _step in _coordinated_checkpoint_generation_bindings(
                logical_path,
                context,
                expected_artifact_sha256=None,
            )
        )

    current = _logical_checkpoint_current(logical_path)
    _reject_mixed_checkpoint_formats(current)
    bindings: list[CheckpointGenerationBinding] = []
    for candidate in (current, _dcp_sibling(current, "previous")):
        payload = candidate / "checkpoint.pt"
        try:
            if payload.is_symlink() or not payload.is_file():
                continue
            artifact_sha256 = _sha256_file(payload)
        except OSError:
            continue
        bindings.append(CheckpointGenerationBinding(candidate, artifact_sha256))
    return tuple(bindings)


def _preflight_verified_checkpoint_candidate(
    source: Path,
    context: DistributedContext,
    expected_identity: Mapping[str, Any] | None,
    marker_step: int,
    expected_step: int | None,
    observed_steps: list[int],
    lease: _VerifiedCheckpointLease,
) -> None:
    worker_lease = _active_checkpoint_lease()
    if worker_lease is not None and worker_lease != lease:
        raise RuntimeError("checkpoint preflight worker already has an active lease")
    installed_here = worker_lease is None
    if installed_here:
        _VERIFIED_CHECKPOINT_LEASE.active = lease
    try:
        actual_step = preflight_checkpoint_identity(source, context, expected_identity)
        if actual_step is None:
            raise ValueError("distributed checkpoint has no recorded resume step")
        if actual_step != marker_step:
            raise ValueError(
                "distributed checkpoint payload step does not match its completion marker "
                f"({actual_step} != {marker_step})"
            )
        if expected_step is not None and actual_step != expected_step:
            raise ValueError(
                "distributed checkpoint step does not match the bound step "
                f"({actual_step} != {expected_step})"
            )
        observed_steps.append(actual_step)
    finally:
        if installed_here and _active_checkpoint_lease() == lease:
            _revoke_checkpoint_lease()


@contextmanager
def verified_checkpoint_generation_lease(
    path: str | Path,
    context: DistributedContext,
    expected_identity: Mapping[str, Any] | None = None,
    *,
    expected_artifact_sha256: str | None = None,
    expected_step: int | None = None,
) -> Generator[VerifiedCheckpointGeneration, None, None]:
    """Select, authenticate, and preflight one exact checkpoint generation.

    For DCP, rank 0 broadcasts ordered current/previous marker bindings. Every
    rank then authenticates the full inventory and preflights identity plus the
    marker-declared step inside one verification lease. A corrupt or incompatible
    current generation is rejected consistently before trying retained previous.
    Markerless/legacy DCP is never a candidate. If ``expected_artifact_sha256``
    is supplied, only the generation bound to that marker digest may be selected.

    Keep any load inside this context. A successful or failed full checkpoint
    load consumes the opaque lease immediately; lexical context exit remains safe.
    Callers must hold ``training_run_lock`` for this complete operation.
    """

    logical_path = Path(path)
    if expected_step is not None and (type(expected_step) is not int or expected_step < 0):
        raise ValueError("expected checkpoint step must be a non-negative integer")
    if expected_artifact_sha256 is not None:
        _validate_sha256_digest(
            expected_artifact_sha256,
            label="expected checkpoint artifact digest",
        )

    if not context.distributed:
        current = _logical_checkpoint_current(logical_path)
        _reject_mixed_checkpoint_formats(current)
        last_local_error: Exception | None = None
        for candidate in (current, _dcp_sibling(current, "previous")):
            try:
                source = resolve_checkpoint_source(candidate, context)
                payload = source / "checkpoint.pt"
                if payload.is_symlink() or not payload.is_file():
                    raise ValueError("local checkpoint payload is not a regular file")
                actual_digest = _sha256_file(payload)
                if (
                    expected_artifact_sha256 is not None
                    and actual_digest != expected_artifact_sha256
                ):
                    raise ValueError(
                        "local checkpoint payload digest does not match the bound artifact "
                        f"({actual_digest} != {expected_artifact_sha256})"
                    )
                actual_step = preflight_checkpoint_identity(source, context, expected_identity)
                if actual_step is None:
                    raise ValueError("checkpoint has no recorded resume step")
                if expected_step is not None and actual_step != expected_step:
                    raise ValueError(
                        "checkpoint step does not match the bound step "
                        f"({actual_step} != {expected_step})"
                    )
            except Exception as error:
                last_local_error = error
                continue
            yield VerifiedCheckpointGeneration(source, actual_step, actual_digest)
            return
        failure = FileNotFoundError(f"no local checkpoint generation matched {logical_path}")
        if last_local_error is not None:
            raise failure from last_local_error
        raise failure

    bindings = _coordinated_checkpoint_generation_bindings(
        logical_path,
        context,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    last_error: Exception | None = None
    selected_stack: ExitStack | None = None
    selected_source: Path | None = None
    selected_step: int | None = None
    for candidate, artifact_sha256, marker_step in bindings:
        candidate_stack = ExitStack()
        try:
            source = candidate_stack.enter_context(
                verified_checkpoint_source_lease(
                    candidate,
                    context,
                    artifact_sha256,
                )
            )
            observed_steps: list[int] = []
            active_lease = _active_checkpoint_lease()
            if active_lease is None:
                raise RuntimeError("distributed checkpoint candidate lease was not activated")
            _run_checkpoint_io_action(
                partial(
                    _preflight_verified_checkpoint_candidate,
                    source,
                    context,
                    expected_identity,
                    marker_step,
                    expected_step,
                    observed_steps,
                    active_lease,
                ),
                context,
                operation="distributed checkpoint generation preflight",
                rank_zero_only=False,
            )
            if len(observed_steps) != 1:
                raise RuntimeError("distributed checkpoint preflight did not return one local step")
        except Exception as error:
            last_error = error
            candidate_stack.close()
            continue
        selected_stack = candidate_stack
        selected_source = source
        selected_step = observed_steps[0]
        break

    if selected_stack is None or selected_source is None or selected_step is None:
        failure = FileNotFoundError(
            f"no rank-consistent authenticated v2 checkpoint generation matched {logical_path}"
        )
        if last_error is not None:
            raise failure from last_error
        raise failure
    try:
        selected_binding = next(binding for binding in bindings if binding[0] == selected_source)
        yield VerifiedCheckpointGeneration(
            selected_source,
            selected_step,
            selected_binding[1],
        )
    finally:
        selected_stack.close()


def _validate_legacy_dcp_layout(path: Path, *, world_size: int) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"legacy distributed checkpoint is not a directory: {path}")
    expected_rng_names = {f"rng-rank-{rank:05d}.pt" for rank in range(world_size)}
    actual_rng_names = {candidate.name for candidate in path.glob("rng-rank-*.pt")}
    if actual_rng_names != expected_rng_names:
        raise ValueError(
            "legacy distributed checkpoint RNG files do not match world size "
            f"(expected={sorted(expected_rng_names)}, actual={sorted(actual_rng_names)})"
        )
    missing_or_unsafe = [
        required.name
        for required in _required_dcp_files(path, world_size=world_size)
        if not required.is_file() or required.is_symlink()
    ]
    if missing_or_unsafe:
        raise ValueError(
            "legacy distributed checkpoint is missing required regular files: "
            + ", ".join(missing_or_unsafe)
        )


def upgrade_legacy_dcp_completion(
    path: str | Path,
    world_size: int,
    step: int,
) -> str:
    """Atomically upgrade one pre-v2 DCP marker and return its v2 marker digest.

    This is an explicit rank-zero recovery operation, not an automatic resume
    path. It seals the bytes currently present; it cannot prove that a legacy,
    digest-less checkpoint was not corrupted before this call. It validates the
    DCP step and exact per-rank RNG layout before hashing the complete directory.
    Existing valid v2 markers are verified and left byte-for-byte unchanged.
    """

    path = Path(path)
    _reject_mixed_checkpoint_formats(path)
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("distributed checkpoint world size must be a positive integer")
    if type(step) is not int or step < 0:
        raise ValueError("distributed checkpoint step must be a non-negative integer")
    marker = path / DCP_COMPLETION_FILENAME
    marker_payload: dict[str, Any] | None = None
    marker_sha256: str | None = None
    if marker.exists() or marker.is_symlink():
        if not marker.is_file() or marker.is_symlink():
            raise ValueError(
                f"distributed checkpoint completion marker is not a regular file: {marker}"
            )
        marker_payload, marker_sha256 = _read_dcp_completion_payload(marker)
        schema = marker_payload.get("schema")
        if schema == DCP_COMPLETION_SCHEMA:
            _validated_dcp_v2_inventory(marker_payload, world_size=world_size)
            if marker_payload.get("step") != step:
                raise ValueError(
                    "distributed checkpoint completion step does not match requested step "
                    f"({marker_payload.get('step')!r} != {step})"
                )
            if _dcp_completion_status(path, world_size=world_size) != "valid":
                raise ValueError(f"distributed checkpoint inventory is invalid: {path}")
            actual_step = _preflight_dcp_identity(path, None)
            if actual_step != step:
                raise ValueError(
                    "distributed checkpoint payload step does not match requested step "
                    f"({actual_step} != {step})"
                )
            assert marker_sha256 is not None
            return marker_sha256
        if schema != CHECKPOINT_SCHEMA or set(marker_payload) != {
            "schema",
            "step",
            "world_size",
        }:
            raise ValueError("distributed checkpoint has an invalid legacy completion marker")
        if isinstance(marker_payload.get("step"), bool) or marker_payload.get("step") != step:
            raise ValueError(
                "legacy distributed checkpoint completion step does not match requested step"
            )
        if (
            isinstance(marker_payload.get("world_size"), bool)
            or marker_payload.get("world_size") != world_size
        ):
            raise ValueError("legacy distributed checkpoint completion world size does not match")

    _validate_legacy_dcp_layout(path, world_size=world_size)
    actual_step = _preflight_dcp_identity(path, None)
    if actual_step != step:
        raise ValueError(
            "legacy distributed checkpoint payload step does not match requested step "
            f"({actual_step} != {step})"
        )
    _write_dcp_completion(path, step=step, world_size=world_size)
    upgraded_payload, upgraded_sha256 = _read_dcp_completion_payload(marker)
    _validated_dcp_v2_inventory(
        upgraded_payload,
        world_size=world_size,
    )
    return upgraded_sha256


def _validate_loaded_state(state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint payload must be an object")
    typed_state = cast(Mapping[str, Any], state)
    schema = typed_state.get("schema")
    if schema is not None and schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {schema!r}")
    missing = [
        key
        for key in ("model", "optimizer", "scheduler", "step", "training_state")
        if key not in typed_state
    ]
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")
    if isinstance(typed_state["step"], bool) or not isinstance(typed_state["step"], int):
        raise ValueError("checkpoint step must be an integer")
    if not isinstance(typed_state["training_state"], Mapping):
        raise ValueError("checkpoint training_state must be an object")
    return typed_state


def _validate_local_optimizer_structure(model: nn.Module, state: object) -> int:
    """Validate AdamW's serialized topology without constructing live optimizer state."""

    if not isinstance(state, Mapping):
        raise ValueError("checkpoint optimizer state must be an object")
    typed_state = cast(Mapping[object, object], state)
    raw_slots = typed_state.get("state")
    raw_groups = typed_state.get("param_groups")
    if not isinstance(raw_slots, Mapping) or not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("checkpoint optimizer state is missing state/param_groups")
    typed_groups = cast(list[object], raw_groups)

    parameter_ids: list[int] = []
    for raw_group in typed_groups:
        if not isinstance(raw_group, Mapping):
            raise ValueError("checkpoint optimizer parameter group is malformed")
        raw_parameters = cast(Mapping[object, object], raw_group).get("params")
        if not isinstance(raw_parameters, list):
            raise ValueError("checkpoint optimizer parameter group has no parameter list")
        for raw_parameter_id in cast(list[object], raw_parameters):
            if (
                isinstance(raw_parameter_id, bool)
                or not isinstance(raw_parameter_id, int)
                or raw_parameter_id < 0
            ):
                raise ValueError("checkpoint optimizer parameter id is invalid")
            parameter_ids.append(raw_parameter_id)

    expected_parameter_count = sum(
        1 for parameter in _unwrap_compiled_model(model).parameters() if parameter.requires_grad
    )
    if len(parameter_ids) != expected_parameter_count or len(set(parameter_ids)) != len(
        parameter_ids
    ):
        raise ValueError(
            "checkpoint optimizer parameter topology does not match the model "
            f"({len(parameter_ids)} serialized != {expected_parameter_count} trainable)"
        )
    known_parameters = set(parameter_ids)
    for raw_parameter_id, raw_slot in cast(Mapping[object, object], raw_slots).items():
        if (
            isinstance(raw_parameter_id, bool)
            or not isinstance(raw_parameter_id, int)
            or raw_parameter_id not in known_parameters
            or not isinstance(raw_slot, Mapping)
        ):
            raise ValueError("checkpoint optimizer slot state is malformed")
    return len(typed_groups)


def _validate_local_scheduler_structure(state: object, *, optimizer_group_count: int) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint scheduler state must be an object")
    typed_state = cast(Mapping[object, object], state)
    last_epoch = typed_state.get("last_epoch")
    if isinstance(last_epoch, bool) or not isinstance(last_epoch, int):
        raise ValueError("checkpoint scheduler last_epoch is invalid")
    for key in ("base_lrs", "_last_lr"):
        values = typed_state.get(key)
        if not isinstance(values, list):
            raise ValueError(
                f"checkpoint scheduler {key} does not match optimizer parameter groups"
            )
        typed_values = cast(list[object], values)
        if len(typed_values) != optimizer_group_count:
            raise ValueError(
                f"checkpoint scheduler {key} does not match optimizer parameter groups"
            )


def _validate_dcp_resume_metadata(
    metadata: Any,
    model: nn.Module,
    *,
    require_scaler: bool,
    require_ema: bool,
) -> None:
    expected_model = _unwrap_compiled_model(model).state_dict()
    if not _validate_dcp_tensor_group_metadata(
        metadata,
        prefix="model",
        expected=expected_model,
    ):
        raise ValueError("distributed checkpoint model state is missing")
    flat_keys = {str(key) for key in metadata.state_dict_metadata}
    required_exact = {
        "schema",
        "step",
        "scheduler.base_lrs",
        "scheduler.last_epoch",
        "scheduler._last_lr",
    }
    missing_exact = sorted(required_exact - flat_keys)
    if missing_exact:
        raise ValueError(
            "distributed checkpoint resume metadata is incomplete: " + ", ".join(missing_exact)
        )
    if not any(
        key.startswith("optimizer.param_groups.") and key.endswith(".params") for key in flat_keys
    ):
        raise ValueError("distributed checkpoint optimizer parameter groups are missing")
    if require_scaler and not any(key.startswith("scaler.") for key in flat_keys):
        raise ValueError(
            "checkpoint resume has a scaler enabled but the checkpoint scaler state is missing"
        )
    expected_ema = {
        name: parameter.detach()
        for name, parameter in _unwrap_compiled_model(model).named_parameters()
        if parameter.requires_grad
    }
    if require_ema and not _validate_dcp_tensor_group_metadata(
        metadata,
        prefix="ema",
        expected=expected_ema,
    ):
        raise ValueError(
            "checkpoint resume has EMA enabled but the checkpoint EMA state is missing"
        )


def preflight_checkpoint_load_structure(
    path: str | Path,
    model: nn.Module,
    context: DistributedContext,
    *,
    require_scaler: bool = False,
    require_ema: bool = False,
) -> int:
    """Prove that a generation is structurally loadable without mutating live state.

    Call this while holding the exact generation lease that will later authorize
    ``load_checkpoint``. It closes the gap between identity-only preflight and
    PyTorch's state restoration, where malformed model tensors or optimizer
    containers would otherwise make a bad current generation mask recoverable
    retained previous state.
    """

    source = resolve_checkpoint_source(path, context)
    if context.distributed:
        import torch.distributed.checkpoint as dcp

        try:
            metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
                source
            ).read_metadata()
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint metadata could not be loaded: {source}"
            ) from error
        _validate_dcp_resume_metadata(
            metadata,
            model,
            require_scaler=require_scaler,
            require_ema=require_ema,
        )
        rng_file = source / f"rng-rank-{context.rank:05d}.pt"
        if rng_file.is_symlink() or not rng_file.is_file():
            raise ValueError(f"distributed checkpoint RNG payload is missing: {rng_file}")
        try:
            rng_payload = torch.load(
                rng_file,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as error:
            raise ValueError(
                f"distributed checkpoint RNG payload could not be loaded: {rng_file}"
            ) from error
        if not isinstance(rng_payload, Mapping):
            raise ValueError("distributed checkpoint RNG payload is invalid")
        raw_rng_state = cast(Mapping[str, Any], rng_payload).get("rng_state")
        if not isinstance(raw_rng_state, Mapping):
            raise ValueError("distributed checkpoint RNG payload is invalid")
        _validate_rng_state(cast(Mapping[str, Any], raw_rng_state))
        return _preflight_dcp_identity(source, None)

    try:
        loaded = torch.load(
            source / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise RuntimeError(
            "checkpoint structure could not be loaded with PyTorch's safe weights-only loader"
        ) from error
    loaded_state = _validate_loaded_state(loaded)
    _validate_stage_transfer_model_state(model, loaded_state["model"])
    optimizer_group_count = _validate_local_optimizer_structure(
        model,
        loaded_state["optimizer"],
    )
    _validate_local_scheduler_structure(
        loaded_state["scheduler"],
        optimizer_group_count=optimizer_group_count,
    )
    current_schema = loaded_state.get("schema") == CHECKPOINT_SCHEMA
    scaler_state = loaded_state.get("scaler")
    if require_scaler and scaler_state is None and current_schema:
        raise ValueError(
            "checkpoint resume has a scaler enabled but the checkpoint scaler state is missing"
        )
    if scaler_state is not None and not isinstance(scaler_state, Mapping):
        raise ValueError("checkpoint scaler state must be an object")
    ema_state = loaded_state.get("ema")
    if require_ema:
        if ema_state is None:
            raise ValueError(
                "checkpoint resume has EMA enabled but the checkpoint EMA state is missing"
            )
        _validate_stage_transfer_ema(model, ema_state)
    rng_state = loaded_state.get("rng_state")
    if rng_state is None and current_schema:
        raise ValueError("checkpoint RNG state is missing")
    if rng_state is not None:
        if not isinstance(rng_state, Mapping):
            raise ValueError("checkpoint RNG state must be an object")
        _validate_rng_state(cast(Mapping[str, Any], rng_state))
    return int(loaded_state["step"])


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
    if context.is_main and not context.distributed:
        path.parent.mkdir(parents=True, exist_ok=True)
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

        def prepare_staging() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            _remove_path(staging)
            staging.mkdir(parents=True)

        _run_checkpoint_io_action(
            prepare_staging,
            context,
            operation="distributed checkpoint staging preparation",
            rank_zero_only=True,
        )
        checkpoint_model = _unwrap_compiled_model(model)
        model_state, optimizer_state = get_state_dict(checkpoint_model, optimizer)
        state["model"] = model_state
        state["optimizer"] = optimizer_state
        dcp.save(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            state, checkpoint_id=staging
        )
        rng_payload = {
            "schema": CHECKPOINT_SCHEMA,
            "rng_state": _capture_rng_state(),
        }

        def write_rank_rng() -> None:
            _atomic_torch_save(
                rng_payload,
                staging / f"rng-rank-{context.rank:05d}.pt",
            )

        _run_checkpoint_io_action(
            write_rank_rng,
            context,
            operation="distributed checkpoint RNG persistence",
            rank_zero_only=False,
        )

        def publish_checkpoint() -> None:
            verified = _write_dcp_completion(
                staging,
                step=step,
                world_size=context.world_size,
            )
            _publish_dcp_staging(
                staging,
                path,
                world_size=context.world_size,
                verified=verified,
            )

        _run_checkpoint_io_action(
            publish_checkpoint,
            context,
            operation="distributed checkpoint completion publication",
            rank_zero_only=True,
        )
    elif context.is_main:
        # 단일 프로세스: 파일 하나로 충분합니다.
        state["model"] = _unwrap_compiled_model(model).state_dict()
        state["optimizer"] = optimizer.state_dict()
        # RNG state를 함께 저장하면 같은 데이터 위치에서 재개할 때 Python,
        # NumPy, torch의 확률적 연산도 중단 전 상태에서 이어집니다.
        state["rng_state"] = _capture_rng_state()
        _atomic_torch_save(state, path / "checkpoint.pt")


def _initialize_model_from_checkpoint_impl(
    path: str | Path,
    model: nn.Module,
    context: DistributedContext,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """다른 **단계**의 체크포인트에서 가중치만 가져온다 (재개가 아님).

    foundation → SFT 처럼 단계가 바뀔 때 쓰는 경로입니다. ``load_checkpoint``
    와 의도적으로 다릅니다.

    - optimizer/scheduler/scaler/RNG/step 을 **복원하지 않습니다.** 새
      단계는 새 목적함수와 새 LR schedule 을 가지므로, 이전 단계의 Adam
      moment 와 step 카운터를 이어받으면 warmup 이 건너뛰어지고 momentum 이
      다른 loss 표면의 것을 가리킵니다.
    - 단, 체크포인트에 EMA shadow가 있으면 raw model 대신 그 파라미터를 다음
      단계의 초기 가중치로 선택합니다. 이는 fresh foundation 종료 직후 trainer가
      선택하는 가중치와 완료-marker 재사용 경로를 동일하게 만듭니다. EMA의
      decay/history 자체를 새 단계로 이어받는 것은 아닙니다.
    - dataset identity 를 비교하지 않습니다. 두 단계는 서로 다른 데이터셋을
      쓰는 것이 정상입니다(단일어 대 병렬).
    - 대신 **tokenizer 와 model config 는 반드시 같아야 합니다.** 토크나이저가
      다르면 임베딩 행이 가리키는 것이 달라져 가중치 인계가 조용히 무의미해지고,
      model config 가 다르면 애초에 모양이 맞지 않습니다.

    반환값은 출처 정보(step, stage)로, 호출자가 provenance 에 기록합니다.
    """

    path = resolve_checkpoint_source(path, context)
    if context.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            set_model_state_dict,
        )

        resolved = path
        source_identity = _preflight_dcp_stage_transfer(resolved, expected_identity)
        checkpoint_model = _unwrap_compiled_model(model)
        # DCP 는 여기 넣어 둔 값 '안으로' 읽어들이므로, 가져올 것만 등록합니다.
        # optimizer/scheduler는 등록하지 않고, EMA는 초기 파라미터 선택용으로만
        # 읽습니다. 새 단계가 EMA history 자체를 이어받는 것은 아닙니다.
        state: dict[str, Any] = {"model": get_model_state_dict(checkpoint_model), "step": 0}
        try:
            metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
                resolved
            ).read_metadata()
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed stage-transfer metadata could not be loaded: {resolved}"
            ) from error
        _validate_dcp_tensor_group_metadata(
            metadata,
            prefix="model",
            expected=cast(Mapping[str, torch.Tensor], state["model"]),
        )
        ema_template = _stage_transfer_ema_template(model)
        has_ema = _validate_dcp_tensor_group_metadata(
            metadata,
            prefix="ema",
            expected=ema_template,
        )
        ema_policy = _source_identity_ema_policy(source_identity)
        if ema_policy is True and not has_ema:
            raise ValueError(
                "stage-transfer checkpoint was trained with EMA enabled but its EMA state "
                "is missing"
            )
        if ema_policy is False and has_ema:
            raise ValueError(
                "stage-transfer checkpoint records EMA as disabled but contains EMA state"
            )
        if has_ema:
            state["ema"] = ema_template
        if expected_identity is not None:
            state["identity"] = _json_compatible(expected_identity)
        try:
            dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
                state, checkpoint_id=resolved
            )
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed stage-transfer payload could not be loaded: {resolved}"
            ) from error
        _validate_stage_transfer(state, expected_identity, source=resolved)
        selected_ema_state = _validate_stage_transfer_ema(model, state.get("ema"))
        set_model_state_dict(checkpoint_model, state["model"])
        selected_ema = _copy_validated_stage_transfer_ema(model, selected_ema_state)
        identity = state.get("identity")
        identity_mapping: Mapping[object, object] = (
            cast(Mapping[object, object], identity) if isinstance(identity, Mapping) else {}
        )
        return {
            "source": str(resolved),
            "step": int(state.get("step") or 0),
            "stage": identity_mapping.get("stage"),
            "weights": "ema" if selected_ema else "raw",
        }

    try:
        loaded = torch.load(
            path / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise RuntimeError(
            "stage-transfer checkpoint could not be loaded with PyTorch's safe "
            "weights-only loader; refusing to fall back to executable pickle"
        ) from error
    loaded_state = _validate_loaded_state(loaded)
    _validate_stage_transfer(loaded_state, expected_identity, source=path)
    ema_policy = _source_identity_ema_policy(loaded_state)
    has_ema = loaded_state.get("ema") is not None
    if ema_policy is True and not has_ema:
        raise ValueError(
            "stage-transfer checkpoint was trained with EMA enabled but its EMA state is missing"
        )
    if ema_policy is False and has_ema:
        raise ValueError("stage-transfer checkpoint records EMA as disabled but contains EMA state")
    validated_model_state = _validate_stage_transfer_model_state(
        model,
        loaded_state["model"],
    )
    selected_ema_state = _validate_stage_transfer_ema(model, loaded_state.get("ema"))
    _unwrap_compiled_model(model).load_state_dict(validated_model_state)
    selected_ema = _copy_validated_stage_transfer_ema(model, selected_ema_state)
    identity = loaded_state.get("identity")
    identity_mapping: Mapping[object, object] = (
        cast(Mapping[object, object], identity) if isinstance(identity, Mapping) else {}
    )
    return {
        "source": str(path),
        "step": int(loaded_state.get("step") or 0),
        "stage": identity_mapping.get("stage"),
        "weights": "ema" if selected_ema else "raw",
    }


def initialize_model_from_checkpoint(
    path: str | Path,
    model: nn.Module,
    context: DistributedContext,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load stage-transfer weights and consume any active verification lease."""

    lease = _active_checkpoint_lease()
    try:
        return _initialize_model_from_checkpoint_impl(
            path,
            model,
            context,
            expected_identity=expected_identity,
        )
    finally:
        if lease is not None and _active_checkpoint_lease() == lease:
            _revoke_checkpoint_lease()


def _validate_stage_transfer(
    state: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None,
    *,
    source: Path,
) -> None:
    """단계 인계에서 반드시 같아야 하는 것만 비교한다.

    데이터셋은 달라야 정상이므로 비교하지 않습니다. tokenizer 와 model config
    가 다르면 인계 자체가 뜻을 잃으므로 여기서 막습니다.
    """

    if expected_identity is None:
        return
    recorded = state.get("identity")
    if not isinstance(recorded, Mapping):
        raise ValueError(
            f"stage-transfer checkpoint has no recorded identity: {source}. "
            "Refusing to inherit weights whose tokenizer cannot be verified."
        )
    recorded = _normalize_identity_for_comparison(cast(Mapping[str, Any], recorded))
    expected = _normalize_identity_for_comparison(expected_identity)
    differences: list[str] = []
    for section in ("tokenizer", "model"):
        differences.extend(
            _identity_differences(
                expected.get(section),
                recorded.get(section),
                path=f"identity.{section}",
            )
        )
    if differences:
        rendered = "\n  - ".join(differences)
        raise ValueError(
            "stage-transfer checkpoint does not match this run's tokenizer/model "
            f"identity ({source}):\n  - {rendered}\n"
            "Weight transfer across stages is only meaningful when both stages "
            "share a tokenizer and a model architecture."
        )


def _load_checkpoint_impl(
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
    path = resolve_checkpoint_source(path, context)
    if context.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        _preflight_dcp_identity(path, expected_identity)
        checkpoint_model = _unwrap_compiled_model(model)
        model_state, optimizer_state = get_state_dict(checkpoint_model, optimizer)
        state: dict[str, Any] = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
            "step": 0,
            "training_state": dict(training_state or {}),
        }
        try:
            metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
                path
            ).read_metadata()
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint metadata could not be loaded: {path}"
            ) from error
        _validate_dcp_resume_metadata(
            metadata,
            model,
            require_scaler=scaler is not None,
            require_ema=ema is not None,
        )
        # DCP only restores keys present in the caller-supplied template. These
        # fields were added after the original progress schema, so discover
        # them from authenticated metadata before loading. Do not add absent
        # keys unconditionally: legacy DCP generations legitimately lack them
        # and strict loading would otherwise reject those checkpoints.
        progress_template = cast(dict[str, Any], state["training_state"])
        stored_flat_keys = {str(key) for key in metadata.state_dict_metadata}
        for optional_key in (
            "configured_selection_metric",
            "best_selection_metric",
            "best_checkpoint_artifact_sha256",
        ):
            if (
                optional_key not in progress_template
                and f"training_state.{optional_key}" in stored_flat_keys
            ):
                progress_template[optional_key] = None
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if ema is not None:
            ema_template = ema.state_dict()
            if not _validate_dcp_tensor_group_metadata(
                metadata,
                prefix="ema",
                expected=ema_template,
            ):
                raise ValueError(
                    "checkpoint resume has EMA enabled but the checkpoint EMA state is missing"
                )
            # DCP loads into supplied tensors. Use detached clones so a failed
            # load cannot partially overwrite the live EMA shadow.
            state["ema"] = {name: tensor.detach().clone() for name, tensor in ema_template.items()}
        if expected_identity is not None:
            state["identity"] = _json_compatible(expected_identity)
        state["schema"] = CHECKPOINT_SCHEMA
        rng_file = path / f"rng-rank-{context.rank:05d}.pt"
        if rng_file.is_symlink() or not rng_file.is_file():
            raise ValueError(f"distributed checkpoint RNG payload is missing: {rng_file}")
        try:
            rng_payload = torch.load(
                rng_file,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as error:
            raise ValueError(
                f"distributed checkpoint RNG payload could not be loaded: {rng_file}"
            ) from error
        if not isinstance(rng_payload, Mapping):
            raise ValueError("distributed checkpoint RNG payload is invalid")
        typed_rng_payload = cast(Mapping[str, Any], rng_payload)
        if not isinstance(typed_rng_payload.get("rng_state"), Mapping):
            raise ValueError("distributed checkpoint RNG payload is invalid")
        validated_rng_state = cast(Mapping[str, Any], typed_rng_payload["rng_state"])
        _validate_rng_state(validated_rng_state)
        # 체크포인트가 불완전하거나 구조가 맞지 않으면 여기서 바로 실패합니다.
        # 일부 파라미터가 초기값인 채로 조용히 재개되는 것이 훨씬 위험하기 때문입니다.
        try:
            dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
                state, checkpoint_id=path
            )
        except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
            raise ValueError(
                f"distributed checkpoint payload could not be loaded: {path}"
            ) from error
        _validate_loaded_state(state)
        _validate_identity(state, expected_identity)
        if ema is not None:
            ema.validate_state_dict(state.get("ema"))
        set_state_dict(
            checkpoint_model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        if training_state is not None:
            training_state.update(state.get("training_state", {}))
        _restore_rng_state(validated_rng_state)
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
    loaded_state = _validate_loaded_state(loaded)
    # 모델/optimizer를 변경하기 전에 현재 실행과 체크포인트의 정체성을 비교합니다.
    _validate_identity(loaded_state, expected_identity)
    _validate_stage_transfer_model_state(model, loaded_state["model"])
    optimizer_group_count = _validate_local_optimizer_structure(
        model,
        loaded_state["optimizer"],
    )
    _validate_local_scheduler_structure(
        loaded_state["scheduler"],
        optimizer_group_count=optimizer_group_count,
    )
    current_schema = loaded_state.get("schema") == CHECKPOINT_SCHEMA
    scaler_state = loaded_state.get("scaler")
    if scaler is not None:
        if scaler_state is None and current_schema:
            raise ValueError(
                "checkpoint resume has a scaler enabled but the checkpoint scaler state is missing"
            )
        if scaler_state is not None and not isinstance(scaler_state, Mapping):
            raise ValueError("checkpoint scaler state must be an object")
    if ema is not None:
        ema_state = loaded_state.get("ema")
        if ema_state is None:
            raise ValueError(
                "checkpoint resume has EMA enabled but the checkpoint EMA state is missing"
            )
        ema.validate_state_dict(ema_state)
    rng_state = loaded_state.get("rng_state")
    if rng_state is None and current_schema:
        raise ValueError("checkpoint RNG state is missing")
    if rng_state is not None and not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint RNG state must be an object")
    if isinstance(rng_state, Mapping):
        _validate_rng_state(cast(Mapping[str, Any], rng_state))
    _unwrap_compiled_model(model).load_state_dict(loaded_state["model"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    scheduler.load_state_dict(loaded_state["scheduler"])
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if ema is not None:
        ema.load_state_dict(loaded_state["ema"])
    if training_state is not None:
        training_state.update(loaded_state.get("training_state", {}))
    if rng_state is not None:
        _restore_rng_state(cast(Mapping[str, Any], rng_state))
    return int(loaded_state["step"])


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
    """Restore a full training checkpoint and consume its verification lease."""

    lease = _active_checkpoint_lease()
    try:
        return _load_checkpoint_impl(
            path,
            model,
            optimizer,
            scheduler,
            context,
            scaler=scaler,
            training_state=training_state,
            ema=ema,
            expected_identity=expected_identity,
        )
    finally:
        if lease is not None and _active_checkpoint_lease() == lease:
            _revoke_checkpoint_lease()
