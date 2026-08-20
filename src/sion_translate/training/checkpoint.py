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
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .distributed import DistributedContext, barrier

CHECKPOINT_SCHEMA = "sion-training-checkpoint-v2"
CHECKPOINT_IDENTITY_SCHEMA = "sion-checkpoint-identity-v1"
DCP_COMPLETION_FILENAME = ".sion_checkpoint_complete.json"
DCP_COMPLETION_SCHEMA = "sion-dcp-completion-v2"
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
    "validation_num_beams",
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


def _dcp_identity_probe(metadata: Any) -> dict[str, Any]:
    """Build an identity-only DCP target from keys actually in the checkpoint."""

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
        if flat_key != "identity" and not flat_key.startswith("identity."):
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
                    f"distributed checkpoint has an invalid identity metadata path: {flat_key}"
                )
            path = tuple(cast(str | int, part) for part in raw_parts)
        else:
            path = tuple(flat_key.split("."))
        if not path or path[0] != "identity":
            raise ValueError(
                f"distributed checkpoint has an invalid identity metadata path: {flat_key}"
            )
        flattened[flat_key] = None
        paths[flat_key] = path
    try:
        return cast(dict[str, Any], unflatten_state_dict(flattened, paths))
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("distributed checkpoint identity metadata is malformed") from error


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
) -> None:
    """Validate DCP identity without binding checkpoint data to live tensors."""

    if expected_identity is None:
        return
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
        path
    ).read_metadata()
    probe = _dcp_identity_probe(metadata)
    if "identity" not in probe:
        _validate_identity(probe, expected_identity)
        return
    try:
        dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            probe,
            checkpoint_id=path,
        )
    except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
        raise ValueError(
            f"distributed checkpoint identity could not be preflighted: {path}"
        ) from error
    stored_identity = probe.get("identity")
    if isinstance(stored_identity, dict):
        _restore_expected_empty_mappings(
            cast(dict[str, Any], stored_identity),
            cast(Mapping[str, Any], _json_compatible(expected_identity)),
        )
    _validate_identity(probe, expected_identity)


def _preflight_dcp_stage_transfer(
    path: Path,
    expected_identity: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Validate model/tokenizer ancestry before DCP touches the live model."""

    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
        path
    ).read_metadata()
    probe = _dcp_identity_probe(metadata)
    if "identity" not in probe:
        _validate_stage_transfer(probe, expected_identity, source=path)
        return probe
    try:
        dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            probe,
            checkpoint_id=path,
        )
    except dcp.CheckpointException as error:  # pyright: ignore[reportPrivateImportUsage]
        raise ValueError(
            f"distributed stage-transfer identity could not be preflighted: {path}"
        ) from error
    stored_identity = probe.get("identity")
    if isinstance(stored_identity, dict) and expected_identity is not None:
        _restore_expected_empty_mappings(
            cast(dict[str, Any], stored_identity),
            cast(Mapping[str, Any], _json_compatible(expected_identity)),
        )
    _validate_stage_transfer(probe, expected_identity, source=path)
    return probe


def preflight_checkpoint_identity(
    path: str | Path,
    context: DistributedContext,
    expected_identity: Mapping[str, Any] | None,
) -> None:
    """Fail on incompatible resume provenance before mutating training state."""

    if expected_identity is None:
        return
    path = Path(path)
    if context.distributed:
        resolved = _resolve_dcp_checkpoint(path, world_size=context.world_size)
        _preflight_dcp_identity(resolved, expected_identity)
        return
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
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    if not isinstance(python_state, tuple):
        raise ValueError("checkpoint Python RNG state is invalid")
    if not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    typed_numpy_state = cast(Mapping[str, Any], numpy_state)
    if not isinstance(typed_numpy_state.get("keys"), torch.Tensor):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise ValueError("checkpoint torch RNG state is invalid")

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
    inventory: list[dict[str, Any]] = []
    for artifact in sorted(directory.rglob("*")):
        if artifact == marker:
            continue
        if artifact.is_symlink():
            raise ValueError(f"distributed checkpoint contains a symlink: {artifact}")
        if not artifact.is_file():
            continue
        inventory.append(
            {
                "path": artifact.relative_to(directory).as_posix(),
                "size": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        )
    payload = {
        "schema": DCP_COMPLETION_SCHEMA,
        "step": int(step),
        "world_size": int(world_size),
        "files": inventory,
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


def _publish_dcp_staging(
    staging: Path,
    destination: Path,
    *,
    world_size: int,
    staging_verified: bool = False,
) -> None:
    """Publish a complete DCP directory while retaining one recoverable version.

    ``staging_verified`` is reserved for the immediate caller that just built
    the hashed completion marker; it avoids reading every large shard a second
    time before the atomic rename.
    """

    previous = _dcp_sibling(destination, "previous")
    staging_is_valid = (
        (staging / DCP_COMPLETION_FILENAME).is_file()
        if staging_verified
        else _dcp_completion_status(staging, world_size=world_size) == "valid"
    )
    if not staging_is_valid:
        raise ValueError(f"refusing to publish an incomplete DCP staging directory: {staging}")
    destination_status = _dcp_completion_status(destination, world_size=world_size)
    previous_status = _dcp_completion_status(previous, world_size=world_size)
    destination_recoverable = destination_status in {"valid", "legacy"} or (
        destination_status == "absent" and (destination / ".metadata").is_file()
    )
    previous_recoverable = previous_status in {"valid", "legacy"} or (
        previous_status == "absent" and (previous / ".metadata").is_file()
    )
    moved_previous = False
    if destination_recoverable:
        _remove_path(previous)
        os.replace(destination, previous)
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
    except BaseException:
        if moved_previous and not destination.exists():
            os.replace(previous, destination)
        raise


def _dcp_completion_status(path: Path, *, world_size: int) -> str:
    marker = path / DCP_COMPLETION_FILENAME
    if not marker.is_file():
        return "absent"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    if not isinstance(payload, Mapping):
        return "invalid"
    marker_payload = cast(Mapping[object, object], payload)
    if type(marker_payload.get("step")) is not int:
        return "invalid"
    if marker_payload.get("world_size") != world_size:
        return "invalid"
    schema = marker_payload.get("schema")
    if schema == CHECKPOINT_SCHEMA:
        required = [path / ".metadata"] + [
            path / f"rng-rank-{rank:05d}.pt" for rank in range(world_size)
        ]
        return "legacy" if all(item.is_file() for item in required) else "invalid"
    if schema != DCP_COMPLETION_SCHEMA:
        return "invalid"
    stored_inventory = marker_payload.get("files")
    if not isinstance(stored_inventory, list):
        return "invalid"
    try:
        current_inventory: list[dict[str, Any]] = []
        for artifact in sorted(path.rglob("*")):
            if artifact == marker:
                continue
            if artifact.is_symlink():
                return "invalid"
            if not artifact.is_file():
                continue
            current_inventory.append(
                {
                    "path": artifact.relative_to(path).as_posix(),
                    "size": artifact.stat().st_size,
                    "sha256": _sha256_file(artifact),
                }
            )
    except OSError:
        return "invalid"
    if stored_inventory != current_inventory:
        return "invalid"
    inventory_paths = {item.get("path") for item in current_inventory}
    required_paths = {".metadata"} | {f"rng-rank-{rank:05d}.pt" for rank in range(world_size)}
    return "valid" if required_paths <= inventory_paths else "invalid"


def _resolve_dcp_checkpoint(path: Path, *, world_size: int) -> Path:
    """Resolve a complete current DCP checkpoint, or its retained predecessor."""

    current_status = _dcp_completion_status(path, world_size=world_size)
    if current_status in {"valid", "legacy"}:
        if current_status == "legacy":
            warnings.warn(
                f"{path} uses a legacy checkpoint completion marker without file hashes",
                RuntimeWarning,
                stacklevel=3,
            )
        return path
    previous = _dcp_sibling(path, "previous")
    previous_status = _dcp_completion_status(previous, world_size=world_size)
    if previous_status in {"valid", "legacy"}:
        warnings.warn(
            f"{path} is incomplete; resuming from retained checkpoint {previous}",
            RuntimeWarning,
            stacklevel=3,
        )
        return previous
    # Accept pre-v2 DCP directories once for backward compatibility. New saves
    # always carry a completion marker and never overwrite this directory in place.
    if current_status == "absent" and (path / ".metadata").is_file():
        warnings.warn(
            "legacy distributed checkpoint has no Sion completion marker; "
            "loading it without atomic-publication verification",
            RuntimeWarning,
            stacklevel=3,
        )
        return path
    raise FileNotFoundError(
        f"no complete distributed checkpoint found at {path} "
        f"(current={current_status}, previous={previous_status})"
    )


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
        checkpoint_model = _unwrap_compiled_model(model)
        model_state, optimizer_state = get_state_dict(checkpoint_model, optimizer)
        state["model"] = model_state
        state["optimizer"] = optimizer_state
        dcp.save(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            state, checkpoint_id=staging
        )
        _atomic_torch_save(
            {"schema": CHECKPOINT_SCHEMA, "rng_state": _capture_rng_state()},
            staging / f"rng-rank-{context.rank:05d}.pt",
        )
        barrier(context)
        if context.is_main:
            _write_dcp_completion(staging, step=step, world_size=context.world_size)
            _publish_dcp_staging(
                staging,
                path,
                world_size=context.world_size,
                staging_verified=True,
            )
    elif context.is_main:
        # 단일 프로세스: 파일 하나로 충분합니다.
        state["model"] = _unwrap_compiled_model(model).state_dict()
        state["optimizer"] = optimizer.state_dict()
        # RNG state를 함께 저장하면 같은 데이터 위치에서 재개할 때 Python,
        # NumPy, torch의 확률적 연산도 중단 전 상태에서 이어집니다.
        state["rng_state"] = _capture_rng_state()
        _atomic_torch_save(state, path / "checkpoint.pt")
    # 저장이 완전히 끝난 뒤에만 모든 rank 가 다음 단계로 진행합니다.
    barrier(context)


def initialize_model_from_checkpoint(
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

    path = Path(path)
    if context.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            set_model_state_dict,
        )

        resolved = _resolve_dcp_checkpoint(path, world_size=context.world_size)
        source_identity = _preflight_dcp_stage_transfer(resolved, expected_identity)
        checkpoint_model = _unwrap_compiled_model(model)
        # DCP 는 여기 넣어 둔 값 '안으로' 읽어들이므로, 가져올 것만 등록합니다.
        # optimizer/scheduler는 등록하지 않고, EMA는 초기 파라미터 선택용으로만
        # 읽습니다. 새 단계가 EMA history 자체를 이어받는 것은 아닙니다.
        state: dict[str, Any] = {"model": get_model_state_dict(checkpoint_model), "step": 0}
        metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
            resolved
        ).read_metadata()
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
        dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            state, checkpoint_id=resolved
        )
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
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if ema is not None:
            metadata = dcp.FileSystemReader(  # pyright: ignore[reportPrivateImportUsage]
                path
            ).read_metadata()
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
        # 체크포인트가 불완전하거나 구조가 맞지 않으면 여기서 바로 실패합니다.
        # 일부 파라미터가 초기값인 채로 조용히 재개되는 것이 훨씬 위험하기 때문입니다.
        dcp.load(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
            state, checkpoint_id=path
        )
        _validate_loaded_state(state)
        _validate_identity(state, expected_identity)
        if ema is not None:
            ema.validate_state_dict(state.get("ema"))
        rng_file = path / f"rng-rank-{context.rank:05d}.pt"
        if not rng_file.is_file():
            raise ValueError(f"distributed checkpoint RNG payload is missing: {rng_file}")
        rng_payload = torch.load(
            rng_file,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(rng_payload, Mapping):
            raise ValueError("distributed checkpoint RNG payload is invalid")
        typed_rng_payload = cast(Mapping[str, Any], rng_payload)
        if not isinstance(typed_rng_payload.get("rng_state"), Mapping):
            raise ValueError("distributed checkpoint RNG payload is invalid")
        validated_rng_state = cast(Mapping[str, Any], typed_rng_payload["rng_state"])
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
