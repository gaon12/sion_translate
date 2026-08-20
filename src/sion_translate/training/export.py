"""Inference export, conversion, integrity metadata, and portable quantization.

Training checkpoints contain optimizer and scheduler state.  Files produced here
contain only what inference needs and are deliberately self-describing so that a
model/tokenizer mismatch fails loudly instead of producing plausible bad output.

The native ``.pt`` formats contain state dictionaries accepted by PyTorch's
weights-only loader. Legacy module pickles require an explicit unsafe migration
opt-in and are never loaded by the normal inference path.
The custom GGUF file is a real mixed Q4_K_M container, but llama.cpp does not
implement the Sion encoder-decoder architecture; it is therefore an honest
storage/interchange artifact rather than a falsely advertised llama.cpp model.
"""

# Optional TorchAO/GGUF/remote-code APIs do not publish complete typing metadata.
# Keep strict checking for known types while containing Unknown values at this
# integration boundary.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import copy
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import types
import uuid
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from sion_translate.artifacts import (
    FOUNDATION_RELEASE_NAME,
    MODEL_RELEASE_VERSION,
    TRANSLATION_RELEASE_NAME,
)
from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.fp8 import DEFAULT_BLOCK, FORWARD_DTYPE, Fp8Policy, scale_for
from sion_translate.fp8_runtime import apply_fp8_weights
from sion_translate.locking import artifact_lock
from sion_translate.model import SionForConditionalGeneration
from sion_translate.model.layers import RotaryEmbedding, SwiGLU

from .distributed import DistributedContext

EXPORT_SCHEMA = "sion-inference-export-v2"
MANIFEST_SCHEMA = "sion-export-manifest-v2"
SUPPORTED_FORMATS = (
    "fp32",
    "fp16",
    "bf16",
    "int8",
    "int4",
    "fp8",
    "gguf_q4_k_m",
    "transformers",
)
DEFAULT_TRAINING_FORMATS = ("fp32",)
DEFAULT_CONVERSION_FORMATS = SUPPORTED_FORMATS

_FORMAT_FILENAMES = {
    "fp32": "model.pt",
    "fp16": "model_fp16.pt",
    "bf16": "model_bf16.pt",
    "int8": "model_int8.pt",
    "int4": "model_int4.pt",
    "fp8": "model_fp8.pt",
    "gguf_q4_k_m": "model-q4_k_m.gguf",
    "transformers": "transformers",
}
_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_INT4_GROUP_SIZE = 128
_EXPORT_LOCK_TIMEOUT_SECONDS = 60.0


@contextmanager
def _legacy_kjx_pickle_aliases():
    """Temporarily map the project's pre-rename pickle module paths.

    Early exports stored a quantized ``nn.Module`` under ``kjx.*`` rather
    than a portable state dictionary. PyTorch pickle resolves those module
    names before the payload schema can be inspected, so aliases must exist
    around ``torch.load`` itself.
    """
    import sion_translate
    import sion_translate.config as config_module
    import sion_translate.model as model_module
    import sion_translate.model.experimental as experimental_module
    import sion_translate.model.layers as layers_module

    legacy_model_module = types.ModuleType("kjx.model.kjx")
    legacy_model_module.__dict__["KJXForConditionalGeneration"] = SionForConditionalGeneration
    aliases = {
        "kjx": sion_translate,
        "kjx.config": config_module,
        "kjx.model": model_module,
        "kjx.model.experimental": experimental_module,
        "kjx.model.layers": layers_module,
        "kjx.model.kjx": legacy_model_module,
    }
    added: list[str] = []
    for name, module in aliases.items():
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    try:
        yield
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)


def _hydrate_legacy_module_attributes(model: nn.Module, config: ModelConfig) -> None:
    """Restore non-parameter runtime attributes absent from early pickles."""

    if isinstance(model, SionForConditionalGeneration):
        model._synchronize_generation_across_ranks = getattr(  # pyright: ignore[reportPrivateUsage]
            model,
            "_synchronize_generation_across_ranks",
            False,
        )
        model.recurrent_block_layers = getattr(
            model,
            "recurrent_block_layers",
            min(config.experimental.recurrent_block_layers, config.encoder_layers),
        )
        model.recurrent_steps = getattr(
            model,
            "recurrent_steps",
            max(1, config.experimental.recurrent_steps),
        )
    for module in model.modules():
        if isinstance(module, SwiGLU):
            module.gate_beta = getattr(module, "gate_beta", None)
            module.up_beta = getattr(module, "up_beta", None)
        elif isinstance(module, RotaryEmbedding):
            module.head_dim = getattr(module, "head_dim", config.d_model // config.num_heads)
            module.max_seq_len = getattr(module, "max_seq_len", config.max_seq_len)
            module.base = getattr(module, "base", 10000.0)
            module._cache_device = getattr(  # pyright: ignore[reportPrivateUsage]
                module, "_cache_device", str(module.cos.device)
            )


def _model_config_from_dict(raw: Mapping[str, Any]) -> ModelConfig:
    values = dict(raw)
    experimental = values.pop("experimental", {})
    if isinstance(experimental, ExperimentalConfig):
        experimental_config = experimental
    else:
        experimental_config = ExperimentalConfig(**dict(experimental))
    return ModelConfig(**values, experimental=experimental_config)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


_MODEL_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?")


def _validated_release_identity(
    release_name: object,
    release_version: object,
    *,
    translation_capable: object,
) -> tuple[str, str, bool]:
    if not isinstance(release_name, str) or not release_name.strip():
        raise ValueError("release_name must be a non-empty string")
    if not isinstance(release_version, str) or not _MODEL_VERSION_PATTERN.fullmatch(
        release_version.strip()
    ):
        raise ValueError("release_version must use a numeric major.minor[.patch] value")
    if not isinstance(translation_capable, bool):
        raise ValueError("translation_capable must be a boolean")
    normalized_name = release_name.strip()
    normalized_version = release_version.strip()
    if normalized_name == FOUNDATION_RELEASE_NAME and translation_capable:
        raise ValueError("the sion foundation release cannot be translation-capable")
    if normalized_name == TRANSLATION_RELEASE_NAME and not translation_capable:
        raise ValueError("the sion_translate release must be translation-capable")
    return normalized_name, normalized_version, translation_capable


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "filename": resolved.name,
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _directory_entry(path: Path) -> dict[str, Any]:
    """Describe a directory with a path-independent deterministic tree hash."""

    if not path.is_dir():
        raise FileNotFoundError(path)
    files: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if not child.is_file():
            continue
        files.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size": child.stat().st_size,
                "sha256": _sha256_file(child),
            }
        )
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "file": path.name,
        "artifact_type": "directory",
        "size": sum(int(item["size"]) for item in files),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    cpu = tensor.detach().to("cpu").contiguous()
    return memoryview(cpu.view(torch.uint8).numpy().tobytes())


def _state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes and bytes in a deterministic order."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _artifact_set_id(
    state_sha256: str,
    model_config: ModelConfig,
    pad_id: int,
) -> str:
    material = {
        "state_sha256": state_sha256,
        "model_config": asdict(model_config),
        "pad_id": int(pad_id),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_language_pairs(
    *,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
) -> list[list[str]]:
    if language_pair is not None and language_pairs is not None:
        raise ValueError("use either language_pair or language_pairs, not both")
    raw_pairs: Sequence[Sequence[str]]
    if language_pairs is not None:
        raw_pairs = language_pairs
    elif language_pair is not None:
        raw_pairs = [language_pair]
    else:
        return []
    if not raw_pairs:
        return []
    normalized: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_pair in raw_pairs:
        pair = [str(language).strip() for language in raw_pair]
        if len(pair) != 2 or not all(pair) or pair[0] == pair[1]:
            raise ValueError("each language pair must contain two distinct non-empty languages")
        key = (pair[0], pair[1])
        if key not in seen:
            seen.add(key)
            normalized.append(pair)
    return normalized


def _metadata_language_pairs(metadata: Mapping[str, Any]) -> list[list[str]]:
    if metadata.get("language_pairs") is not None:
        value = metadata["language_pairs"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("metadata.language_pairs must be a sequence")
        return _normalize_language_pairs(language_pairs=value)
    if metadata.get("language_pair") is not None:
        value = metadata["language_pair"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("metadata.language_pair must be a sequence")
        return _normalize_language_pairs(language_pair=value)
    return []


def _metadata_languages(metadata: Mapping[str, Any]) -> list[str] | None:
    raw_languages = metadata.get("languages")
    if raw_languages is None:
        pairs = _metadata_language_pairs(metadata)
        return _languages_from_pairs(pairs) if pairs else None
    if not isinstance(raw_languages, Sequence) or isinstance(raw_languages, (str, bytes)):
        raise ValueError("metadata.languages must be a sequence")
    languages = list(dict.fromkeys(str(language).strip() for language in raw_languages))
    if any(not language for language in languages):
        raise ValueError("metadata.languages must contain only non-empty languages")
    return languages


def _normalize_translation_directions(
    language_pairs: Sequence[Sequence[str]],
    *,
    translation_directions: Sequence[Sequence[str]] | None = None,
    bidirectional: bool = True,
) -> list[list[str]]:
    pairs = _normalize_language_pairs(language_pairs=language_pairs)
    if translation_directions is None:
        directions: list[list[str]] = []
        for source, target in pairs:
            directions.append([source, target])
            if bidirectional:
                directions.append([target, source])
        return directions
    directions = _normalize_language_pairs(language_pairs=translation_directions)
    if pairs and not directions:
        raise ValueError(
            "translation_directions cannot be empty when language pairs are configured"
        )
    allowed_edges = {frozenset(pair) for pair in pairs}
    disconnected = [
        direction for direction in directions if frozenset(direction) not in allowed_edges
    ]
    if disconnected:
        raise ValueError(
            f"translation directions must belong to configured language pairs: {disconnected!r}"
        )
    return directions


def _metadata_translation_directions(metadata: Mapping[str, Any]) -> list[list[str]]:
    pairs = _metadata_language_pairs(metadata)
    raw_directions = metadata.get("translation_directions")
    if raw_directions is None:
        # Manifests predating explicit direction metadata represented
        # bidirectional checkpoints, so preserve their historical contract.
        return _normalize_translation_directions(pairs, bidirectional=True)
    if not isinstance(raw_directions, Sequence) or isinstance(
        raw_directions,
        (str, bytes),
    ):
        raise ValueError("metadata.translation_directions must be a sequence")
    return _normalize_translation_directions(
        pairs,
        translation_directions=raw_directions,
    )


def _metadata_translation_capable(metadata: Mapping[str, Any]) -> bool:
    value = metadata.get("translation_capable", True)
    if not isinstance(value, bool):
        raise ValueError("metadata.translation_capable must be a boolean")
    return value


def _default_reasoning_level(model_config: ModelConfig) -> int:
    experimental = model_config.experimental
    return (
        9
        if experimental.evidence_repair_enabled or experimental.candidate_refinement_enabled
        else 0
    )


def _validate_generation_defaults(
    metadata: Mapping[str, Any],
    model_config: ModelConfig,
    *,
    required: bool,
) -> int:
    expected = _default_reasoning_level(model_config)
    raw_defaults = metadata.get("generation_defaults")
    if raw_defaults is None and not required:
        return expected
    if not isinstance(raw_defaults, Mapping):
        raise ValueError("generation_defaults must be an object")
    stored_level = raw_defaults.get("reasoning_level")
    if isinstance(stored_level, bool) or not isinstance(stored_level, int):
        raise ValueError("generation_defaults.reasoning_level must be an integer")
    if stored_level != expected:
        raise ValueError(
            "generation_defaults.reasoning_level does not match model features: "
            f"stored={stored_level}, expected={expected}"
        )
    return expected


def _metadata_revision_capability(metadata: Mapping[str, Any]) -> bool | None:
    capabilities = metadata.get("capabilities")
    if capabilities is None:
        return None
    if not isinstance(capabilities, Mapping):
        raise ValueError("metadata.capabilities must be an object")
    value = capabilities.get("revision_trained")
    if value is not None and not isinstance(value, bool):
        raise ValueError("metadata.capabilities.revision_trained must be a boolean")
    return value


def _languages_from_pairs(language_pairs: Sequence[Sequence[str]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(language) for language_pair in language_pairs for language in language_pair
        )
    )


def _metadata_compatibility_id(metadata: Mapping[str, Any]) -> str:
    """Hash non-volatile metadata that must agree across one artifact set."""

    material = {
        str(key): copy.deepcopy(value)
        for key, value in metadata.items()
        if key
        not in {
            "created_unix",
            "source",
            "step",
            "quantization",
            "language_pair",
            "language_pairs",
            "languages",
            "translation_directions",
        }
    }
    pairs = _metadata_language_pairs(metadata)
    if pairs:
        material["language_pairs"] = pairs
        material["languages"] = _languages_from_pairs(pairs)
        material["translation_directions"] = _metadata_translation_directions(metadata)
    elif metadata.get("languages"):
        material["languages"] = [str(value) for value in metadata["languages"]]
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_export_metadata(
    model_config: ModelConfig,
    *,
    tokenizer_path: str | Path | None = None,
    token_features_path: str | Path | None = None,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    languages: Sequence[str] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    bidirectional: bool = True,
    revision_trained: bool | None = None,
    step: int | None = None,
    source: str | Path | None = None,
    release_name: str = TRANSLATION_RELEASE_NAME,
    release_version: str = MODEL_RELEASE_VERSION,
    translation_capable: bool = True,
) -> dict[str, Any]:
    """Build provenance and compatibility metadata shared by every format.

    ``translation_capable`` 은 이름표가 아니라 계약입니다. foundation 단계의
    산출물은 번역쌍을 한 번도 보지 않았지만 구조가 번역 모델과 완전히 같아서,
    그대로 실으면 방향 태그를 받아들이고 그럴듯한 쓰레기를 냅니다.
    """

    release_name, release_version, translation_capable = _validated_release_identity(
        release_name,
        release_version,
        translation_capable=translation_capable,
    )

    experimental = model_config.experimental
    metadata: dict[str, Any] = {
        "created_unix": time.time(),
        "release_name": release_name,
        "release_version": release_version,
        "translation_capable": bool(translation_capable),
    }
    if source is not None:
        # Export metadata may be published. Preserve a content-addressed source
        # identity without leaking a workstation or mounted-volume path.
        metadata["source"] = _file_identity(source)
    if step is not None:
        metadata["step"] = int(step)
    if tokenizer_path is not None:
        metadata["tokenizer"] = _file_identity(tokenizer_path)
    if token_features_path is not None:
        metadata["token_features"] = _file_identity(token_features_path)
    pairs = _normalize_language_pairs(
        language_pair=language_pair,
        language_pairs=language_pairs,
    )
    explicit_languages = (
        list(dict.fromkeys(str(language).strip() for language in languages))
        if languages is not None
        else None
    )
    if explicit_languages is not None and (
        not explicit_languages or any(not language for language in explicit_languages)
    ):
        raise ValueError("languages must contain at least one non-empty language")
    if pairs and not translation_capable:
        # 파운데이션 모델에는 언어쌍이 없습니다. 단일어 복원만 학습했으므로
        # 다룰 줄 아는 것은 **언어**이고, 쌍도 방향도 존재하지 않습니다.
        # 쌍을 적어 두면 아래 검증이 방향을 요구하고, 그 방향은 거짓입니다.
        metadata["languages"] = explicit_languages or _languages_from_pairs(pairs)
    elif pairs:
        metadata["language_pairs"] = pairs
        metadata["languages"] = _languages_from_pairs(pairs)
        metadata["translation_directions"] = _normalize_translation_directions(
            pairs,
            translation_directions=translation_directions,
            bidirectional=bidirectional,
        )
        if len(pairs) == 1:
            metadata["language_pair"] = pairs[0]
    elif explicit_languages is not None:
        metadata["languages"] = explicit_languages
    elif translation_directions is not None:
        raise ValueError("translation_directions require at least one language pair")
    metadata["feature_flags"] = {
        "bats": bool(experimental.bats_enabled),
        "core": bool(experimental.core_enabled),
        "tetm": bool(experimental.tetm_enabled),
        "morphoscript": bool(experimental.morphoscript_enabled),
        "evidence_repair": bool(experimental.evidence_repair_enabled),
        "candidate_refinement": bool(experimental.candidate_refinement_enabled),
        "semantic_parity": bool(experimental.semantic_parity_enabled),
        "situglu": bool(experimental.situglu_enabled),
        "recurrent_block": bool(
            getattr(
                model_config,
                "recurrent_block_layers",
                getattr(experimental, "recurrent_block_layers", 0),
            )
        ),
    }
    metadata["generation_defaults"] = {
        "reasoning_level": _default_reasoning_level(model_config),
    }
    if revision_trained is not None:
        metadata["capabilities"] = {"revision_trained": bool(revision_trained)}
    return metadata


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}{path.suffix}")


def _remove_artifact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _remove_staging_artifact(path: Path) -> None:
    """Remove staging, tolerating only a locked and already-empty shell."""

    try:
        _remove_artifact(path)
    except OSError:
        if path.is_dir() and not path.is_symlink():
            try:
                if next(path.iterdir(), None) is None:
                    return
            except OSError:
                pass
        raise


def _export_publish_lock_root(destination: Path) -> Path:
    """Return a stable sibling lock root for one canonical export destination."""

    lock_name = f".{destination.name}.sion-export-publish-lock"
    return destination.parent / lock_name


def _install_directory(temporary: Path, destination: Path) -> None:
    """staging 디렉터리를 목적지 자리에 설치한다.

    보통은 rename 한 번이면 끝나고, 그것이 원자적이라 선호합니다.

    Windows 에서는 그 rename 이 실패할 수 있습니다. Transformers export 를
    검증할 때 원격 코드를 ``transformers_modules.*`` 로 import 하는데, 그
    과정에서 staging 디렉터리 **자체**에 핸들이 남습니다. 실측으로 확인한
    상태는 이렇습니다.

    - staging 디렉터리를 어떤 이름으로도 rename 할 수 없다
    - 그 안의 **모든 자식**(파일과 하위 디렉터리)은 개별적으로 rename 된다
    - 목적지 자리에는 새로 mkdir 할 수 있다
    - ``gc.collect()`` 로는 풀리지 않는다 (import 시스템이 잡고 있으므로)

    그래서 자식을 하나씩 옮기는 것으로 물러섭니다. Windows 에는 애초에
    디렉터리의 원자적 교체가 없으므로 이것이 잃는 보장은 없습니다. 호출자가
    이전 판을 backup 으로 들고 있어 실패 시 되돌릴 수 있다는 점도 그대로입니다.
    """

    try:
        os.replace(temporary, destination)
        return
    except OSError:
        if not temporary.is_dir():
            raise

    # Never expose a partially populated canonical destination. Move children
    # into a fresh, unlocked handoff shell and rename that shell only after all
    # child moves succeed.
    handoff = destination.with_name(f".{destination.name}.handoff-{uuid.uuid4().hex}")
    try:
        handoff.mkdir(parents=False, exist_ok=False)
        for child in list(temporary.iterdir()):
            os.replace(child, handoff / child.name)
        try:
            temporary.rmdir()
        except OSError:
            # 내용은 전부 옮겼습니다. 잠긴 빈 껍데기 때문에 성공한 export 를
            # 실패로 만들지 않습니다.
            pass
        os.replace(handoff, destination)
    except BaseException:
        try:
            _remove_artifact(handoff)
        except OSError:
            # A partial handoff has a randomized non-canonical name. Preserve
            # the original installation error; recovery never reads handoffs.
            pass
        raise


def _atomic_replace_directory_unlocked(temporary: Path, destination: Path) -> None:
    """Install a fully written directory while retaining the prior one on failure."""

    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved_existing = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        _install_directory(temporary, destination)
    except BaseException as install_error:
        rollback_errors: list[BaseException] = []
        if destination.exists():
            try:
                _remove_artifact(destination)
            except BaseException as error:
                rollback_errors.append(error)
        if moved_existing and backup.exists() and not destination.exists():
            try:
                os.replace(backup, destination)
                moved_existing = False
            except BaseException as error:
                rollback_errors.append(error)
        if rollback_errors:
            failure = RuntimeError(
                "failed to restore the previous export directory after installation failed; "
                f"recoverable backup: {backup}"
            )
            for error in rollback_errors:
                failure.add_note(f"rollback error: {type(error).__name__}: {error}")
            raise failure from install_error
        raise
    else:
        if moved_existing:
            moved_existing = False
            try:
                _remove_artifact(backup)
            except OSError:
                # The new artifact is already installed atomically. A locked
                # stale backup must not turn a valid export into a failed one.
                pass


def _atomic_replace_directory(temporary: Path, destination: Path) -> None:
    """Serialize publication per destination and atomically replace one directory."""

    with artifact_lock(
        _export_publish_lock_root(destination),
        timeout=_EXPORT_LOCK_TIMEOUT_SECONDS,
        poll_interval=0.05,
    ):
        _atomic_replace_directory_unlocked(temporary, destination)


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Write and CRC-check a PyTorch zip before replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        torch.save(payload, temporary)
        with zipfile.ZipFile(temporary, "r") as archive:
            broken = archive.testzip()
            if broken is not None:
                raise RuntimeError(f"corrupt tensor archive member: {broken}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy_file(source: str | Path, destination: Path) -> None:
    source = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    temporary = _temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        if _sha256_file(temporary) != _sha256_file(source):
            raise RuntimeError(f"copied sidecar failed SHA256 verification: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loaded = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError("manifest did not round-trip as an object")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inspect_transformers_checkpoint(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for Transformers validation") from error

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as error:
        raise RuntimeError("transformers is required for checkpoint validation") from error

    config_payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    auto_map = config_payload.get("auto_map")
    if not isinstance(auto_map, Mapping):
        raise RuntimeError("Transformers checkpoint is missing its remote-code auto_map")

    def load_remote_class(auto_class: str, mapping: Mapping[str, Any]) -> type:
        reference = mapping.get(auto_class)
        if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
            reference = next((item for item in reference if isinstance(item, str)), None)
        if not isinstance(reference, str):
            raise RuntimeError(f"Transformers checkpoint is missing auto_map.{auto_class}")
        return get_class_from_dynamic_module(
            reference,
            path,
            local_files_only=True,
        )

    remote_config_class = load_remote_class("AutoConfig", auto_map)
    config = remote_config_class.from_pretrained(path)
    config_release_name = getattr(config, "release_name", None)
    config_release_version = getattr(config, "release_version", None)
    if not isinstance(config_release_name, str) or not config_release_name.strip():
        raise RuntimeError("Transformers config release_name must be a non-empty string")
    if not isinstance(config_release_version, str) or not config_release_version.strip():
        raise RuntimeError("Transformers config release_version must be a non-empty string")
    export_payload = json.loads((path / "sion_export.json").read_text(encoding="utf-8"))
    if not isinstance(export_payload, Mapping):
        raise RuntimeError("Transformers sion_export.json must contain an object")
    if export_payload.get("release_name") != config_release_name:
        raise RuntimeError("Transformers config and sion_export.json disagree about release_name")
    if export_payload.get("release_version") != config_release_version:
        raise RuntimeError(
            "Transformers config and sion_export.json disagree about release_version"
        )
    try:
        transformers_reasoning_level = _validate_generation_defaults(
            export_payload,
            config.to_model_config(),
            required=True,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    generation_payload = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))
    if not isinstance(generation_payload, Mapping):
        raise RuntimeError("Transformers generation_config.json must contain an object")
    generation_reasoning_level = generation_payload.get("reasoning_level")
    if isinstance(generation_reasoning_level, bool) or not isinstance(
        generation_reasoning_level, int
    ):
        raise RuntimeError("Transformers generation_config reasoning_level must be an integer")
    if not 0 <= generation_reasoning_level <= 9:
        raise RuntimeError("Transformers generation_config reasoning_level must be between 0 and 9")
    if generation_reasoning_level != transformers_reasoning_level:
        raise RuntimeError(
            "Transformers generation_config.json and sion_export.json disagree "
            "about reasoning_level"
        )
    config_translation_capable = getattr(config, "translation_capable", True)
    if not isinstance(config_translation_capable, bool):
        raise RuntimeError("Transformers config translation_capable must be a boolean")
    weight_files = sorted(path.glob("model*.safetensors"))
    if not weight_files:
        raise RuntimeError("Transformers checkpoint has no model*.safetensors weights")
    tensor_count = 0
    tensor_names: set[str] = set()
    dtypes: set[str] = set()
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            tensor_count += len(names)
            tensor_names.update(names)
            for name in names:
                tensor_slice = handle.get_slice(name)
                serialized_dtype = str(tensor_slice.get_dtype())
                dtypes.add(
                    {
                        "F32": "torch.float32",
                        "F16": "torch.float16",
                        "BF16": "torch.bfloat16",
                    }.get(serialized_dtype, serialized_dtype)
                )
                tensor_shapes[name] = tuple(tensor_slice.get_shape())
    if "model.token_embedding.weight" not in tensor_names:
        raise RuntimeError("Transformers checkpoint is missing model.token_embedding.weight")
    if (path / "tokenizer.model").is_file():
        tokenizer_payload = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer_auto_map = tokenizer_payload.get("auto_map")
        if not isinstance(tokenizer_auto_map, Mapping):
            raise RuntimeError("Transformers tokenizer is missing its remote-code auto_map")
        remote_tokenizer_class = load_remote_class("AutoTokenizer", tokenizer_auto_map)
        remote_tokenizer = remote_tokenizer_class.from_pretrained(path)
        if not remote_tokenizer.__class__.__module__.startswith("transformers_modules."):
            raise RuntimeError(
                "Transformers checkpoint imported an installed tokenizer instead of bundled code"
            )
        tokenizer_translation_capable = getattr(
            remote_tokenizer,
            "translation_capable",
            True,
        )
        if not isinstance(tokenizer_translation_capable, bool):
            raise RuntimeError("Transformers tokenizer translation_capable must be a boolean")
        if tokenizer_translation_capable is not config_translation_capable:
            raise RuntimeError(
                "Transformers config/tokenizer disagree about translation capability: "
                f"config={config_translation_capable!r}, "
                f"tokenizer={tokenizer_translation_capable!r}"
            )
        del remote_tokenizer

    # Import the exact bundled remote code and materialize its architecture on
    # the meta device. This catches syntax/import/API drift and verifies every
    # safetensors key and shape without allocating a second 8B/32B CPU model.
    remote_model_class = load_remote_class("AutoModelForSeq2SeqLM", auto_map)
    with torch.device("meta"):
        remote_model = remote_model_class(config)
    expected_state = remote_model.state_dict()
    expected_names = set(expected_state)
    if tensor_names != expected_names:
        missing = sorted(expected_names - tensor_names)
        unexpected = sorted(tensor_names - expected_names)
        raise RuntimeError(
            "Transformers state dictionary does not match bundled model code: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    mismatched_shapes = {
        name: (tensor_shapes[name], tuple(expected_state[name].shape))
        for name in tensor_names
        if tensor_shapes[name] != tuple(expected_state[name].shape)
    }
    if mismatched_shapes:
        raise RuntimeError(
            "Transformers tensor shapes do not match bundled model code: "
            f"{dict(list(sorted(mismatched_shapes.items()))[:8])}"
        )
    runtime_model_class = (
        f"{remote_model.model.__class__.__module__}.{remote_model.model.__class__.__qualname__}"
    )
    if not remote_model.model.__class__.__module__.startswith("transformers_modules."):
        raise RuntimeError(
            "Transformers checkpoint imported an installed Sion runtime instead of bundled code"
        )
    del remote_model, expected_state
    gc.collect()
    return {
        "tensor_count": tensor_count,
        "weight_files": len(weight_files),
        "dtypes": sorted(dtypes),
        "model_type": config.model_type,
        "runtime_model_class": runtime_model_class,
        "release_name": config_release_name,
        "release_version": config_release_version,
        "reasoning_level": generation_reasoning_level,
        "languages": list(config.languages),
        "language_pairs": [list(pair) for pair in config.language_pairs],
        "translation_directions": [list(direction) for direction in config.translation_directions],
        "translation_capable": config_translation_capable,
        "revision_trained": config.revision_trained,
    }


def _write_transformers_checkpoint(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    pad_id: int,
    *,
    tokenizer_path: str | Path | None,
    token_features_path: str | Path | None,
    languages: Sequence[str] | None,
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]],
    translation_capable: bool,
    revision_trained: bool | None,
    release_name: str,
    release_version: str,
) -> dict[str, Any]:
    from sion_translate.hf.conversion import save_transformers_checkpoint

    temporary = _temporary_path(path)
    try:
        save_transformers_checkpoint(
            temporary,
            dict(state_dict),
            model_config,
            pad_id=pad_id,
            tokenizer_path=tokenizer_path,
            token_features_path=token_features_path,
            languages=languages,
            language_pairs=language_pairs,
            translation_directions=translation_directions,
            translation_capable=translation_capable,
            revision_trained=revision_trained,
            release_name=release_name,
            release_version=release_version,
            allow_language_subset=not bool(language_pairs),
        )
        inspection = _inspect_transformers_checkpoint(temporary)
        if inspection["release_name"] != release_name:
            raise RuntimeError("Transformers checkpoint release_name changed during export")
        if inspection["release_version"] != release_version:
            raise RuntimeError("Transformers checkpoint release_version changed during export")
        # The public export transaction owns the destination lock. Reacquiring
        # it here would deadlock on platforms with non-reentrant file locks.
        _atomic_replace_directory_unlocked(temporary, path)
        return inspection
    finally:
        if temporary.exists():
            _remove_staging_artifact(temporary)


def _precision_state(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        value = tensor.detach().to(device="cpu")
        if value.is_floating_point() and value.dtype != dtype:
            value = value.to(dtype=dtype)
        converted[name] = value
    return converted


def _cpu_model(
    model_config: ModelConfig,
    state_dict: Mapping[str, torch.Tensor],
    pad_id: int,
) -> SionForConditionalGeneration:
    config = copy.deepcopy(model_config)
    config.gradient_checkpointing = False
    with torch.device("meta"):
        model = SionForConditionalGeneration(config, pad_id=pad_id)
    # Bind the stable CPU snapshot without first allocating an equally large
    # initialized model. TorchAO replaces eligible Parameters after this point.
    model.load_state_dict(dict(state_dict), assign=True)
    model.eval()
    return model


def _torchao_quantized_state(
    model_config: ModelConfig,
    state_dict: Mapping[str, torch.Tensor],
    pad_id: int,
    *,
    bits: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )
    except ImportError as error:
        raise RuntimeError("torchao is required for this quantization backend") from error

    model = _cpu_model(model_config, state_dict, pad_id)
    if bits == 8:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        quantization = {
            "backend": "torchao",
            "algorithm": "int8-dynamic-activation-int8-weight",
            "runtime_device": "cpu",
        }
    elif bits == 4:
        quantize_(model, Int4WeightOnlyConfig(group_size=_INT4_GROUP_SIZE))
        quantization = {
            "backend": "torchao",
            "algorithm": "int4-weight-only",
            "group_size": _INT4_GROUP_SIZE,
            "runtime_device": "cpu",
        }
    else:
        raise ValueError("bits must be 4 or 8")
    return dict(model.state_dict()), quantization


def _pack_int4_state(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Portable symmetric groupwise INT4 with two offset nibbles per byte."""

    packed_state: dict[str, dict[str, Any]] = {}
    for name, original in state_dict.items():
        tensor = original.detach().to("cpu")
        if tensor.is_floating_point() and tensor.ndim == 2:
            flat = tensor.float().contiguous().view(-1)
            numel = flat.numel()
            padded_numel = math.ceil(numel / _INT4_GROUP_SIZE) * _INT4_GROUP_SIZE
            if padded_numel != numel:
                flat = torch.nn.functional.pad(flat, (0, padded_numel - numel))
            groups = flat.view(-1, _INT4_GROUP_SIZE)
            scales = groups.abs().amax(dim=1).div(7.0)
            scales = torch.where(scales > 0, scales, torch.ones_like(scales))
            values = torch.round(groups / scales[:, None]).clamp(-7, 7).to(torch.int16)
            offset = (values + 8).to(torch.uint8).reshape(-1)
            packed = offset[0::2] | (offset[1::2] << 4)
            packed_state[name] = {
                "kind": "packed_int4",
                "shape": list(tensor.shape),
                "numel": numel,
                "group_size": _INT4_GROUP_SIZE,
                "scales": scales.to(torch.float16),
                "packed": packed,
            }
        else:
            value = (
                tensor.to(torch.float16, copy=True)
                if tensor.is_floating_point()
                else tensor.clone()
            )
            packed_state[name] = {"kind": "tensor", "value": value}
    quantization = {
        "backend": "sion-packed",
        "algorithm": "symmetric-groupwise-int4-nibbles",
        "group_size": _INT4_GROUP_SIZE,
        "runtime_device": "any",
        "runtime_dequantizes_to": "fp32",
        "tensor_policy": "2d-weights-int4; norms-biases-and-other-tensors-fp16",
    }
    return packed_state, quantization


def _pack_fp8_state(
    state_dict: Mapping[str, torch.Tensor],
    policy: Fp8Policy,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """가중치 전용 FP8. 정책이 허용한 projection 만 내리고 나머지는 그대로 둡니다.

    활성값은 건드리지 않습니다. 가중치만 내리면 저장·상주 바이트를 줄이면서
    실측 출력 오차도 더 작습니다 — 2.57% 대 3.63%. 기본 runtime 은 FP8
    가중치를 매 forward 에서 BF16(BF16 미지원 CUDA에서는 FP16)으로
    역양자화한 뒤 dense GEMM 을 사용하므로 실행 대역폭·연산량 이득을
    보장하지 않습니다.

    무엇을 내리지 않는지가 더 중요합니다. ``Fp8Policy`` 의 기본 범위는 FFN
    뿐이고, 어휘 projection 은 어떤 범위에서도 제외됩니다. 자세한 실측 근거는
    ``sion_translate.fp8`` 모듈 문서에 있습니다.
    """

    policy.validate()
    packed_state: dict[str, dict[str, Any]] = {}
    quantized_parameters = 0
    preserved_parameters = 0
    for name, original in state_dict.items():
        tensor = original.detach().to("cpu")
        eligible = (
            policy.allows(name)
            and tensor.is_floating_point()
            and tensor.ndim == 2
            and tensor.shape[-1] % policy.block == 0
        )
        if not eligible:
            preserved_parameters += tensor.numel()
            packed_state[name] = {"kind": "tensor", "value": tensor}
            continue
        work = tensor.float()
        scales = scale_for(work, dtype=FORWARD_DTYPE, block=policy.block)
        grouped = work.reshape(*work.shape[:-1], -1, policy.block)
        packed_state[name] = {
            "kind": "block_fp8",
            "shape": list(tensor.shape),
            "block": policy.block,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            # 스케일은 fp32 로 둡니다. 개수가 값의 1/block 이라 용량이 무의미하고,
            # fp16 으로 낮추면 스케일 자체가 양자화 오차를 더합니다.
            "scales": scales.squeeze(-1).contiguous(),
            "values": (grouped / scales).to(FORWARD_DTYPE).reshape(tensor.shape).contiguous(),
        }
        quantized_parameters += tensor.numel()
    total = quantized_parameters + preserved_parameters
    quantization = {
        "algorithm": "weight-only-fp8-e4m3-blockwise",
        "format": "fp8",
        "activation_dtype": "bfloat16",
        "activation_fallback_dtype": "float16",
        "activation_dtype_policy": "bf16-if-supported-else-fp16",
        "weight_dtype": "float8_e4m3fn",
        "block": policy.block,
        "scope": policy.scope,
        "quantized_parameters": quantized_parameters,
        "preserved_parameters": preserved_parameters,
        "quantized_fraction": (quantized_parameters / total) if total else 0.0,
        "runtime_device": "any",
    }
    return packed_state, quantization


def _unpack_fp8_state(
    packed_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, entry in packed_state.items():
        kind = entry.get("kind")
        if kind == "tensor":
            value = entry.get("value")
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name}: invalid packed tensor entry")
            state[name] = value
            continue
        if kind != "block_fp8":
            raise ValueError(f"{name}: unknown packed FP8 entry {kind!r}")
        values = entry.get("values")
        scales = entry.get("scales")
        if not isinstance(values, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise ValueError(f"{name}: invalid packed FP8 tensors")
        block = int(entry.get("block", DEFAULT_BLOCK))
        shape = tuple(map(int, entry["shape"]))
        grouped = values.float().reshape(*shape[:-1], -1, block)
        restored = (grouped * scales.float().unsqueeze(-1)).reshape(shape)
        state[name] = restored.to(getattr(torch, str(entry.get("dtype", "float32"))))
    return state


def _unpack_int4_state(
    packed_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, entry in packed_state.items():
        kind = entry.get("kind")
        if kind == "tensor":
            value = entry.get("value")
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name}: invalid packed tensor entry")
            state[name] = value
            continue
        if kind != "packed_int4":
            raise ValueError(f"{name}: unknown packed INT4 entry {kind!r}")
        packed = entry.get("packed")
        scales = entry.get("scales")
        if not isinstance(packed, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise ValueError(f"{name}: invalid packed INT4 tensors")
        offset = torch.empty(packed.numel() * 2, dtype=torch.int16)
        offset[0::2] = (packed.to(torch.int16) & 0x0F) - 8
        offset[1::2] = ((packed.to(torch.int16) >> 4) & 0x0F) - 8
        group_size = int(entry.get("group_size", _INT4_GROUP_SIZE))
        restored = offset.float().view(-1, group_size)
        restored.mul_(scales.float().view(-1, 1))
        numel = int(entry["numel"])
        shape = tuple(map(int, entry["shape"]))
        state[name] = restored.reshape(-1)[:numel].reshape(shape)
    return state


def _pack_scale_min(scales: np.ndarray, minima: np.ndarray) -> np.ndarray:
    """Pack eight unsigned 6-bit scales and minima into the K-quant 12 bytes."""

    scales = scales.astype(np.uint8, copy=False)
    minima = minima.astype(np.uint8, copy=False)
    packed = np.empty((scales.shape[0], 12), dtype=np.uint8)
    packed[:, 0:4] = (scales[:, 0:4] & 0x3F) | ((scales[:, 4:8] & 0x30) << 2)
    packed[:, 4:8] = (minima[:, 0:4] & 0x3F) | ((minima[:, 4:8] & 0x30) << 2)
    packed[:, 8:12] = (scales[:, 4:8] & 0x0F) | ((minima[:, 4:8] & 0x0F) << 4)
    return packed


def _quantize_affine_k(values: np.ndarray, *, bits: int) -> np.ndarray:
    """NumPy Q4_K/Q5_K encoder matching llama.cpp's 256-value block layout.

    gguf-python currently implements K-quant dequantization but raises
    ``NotImplementedError`` for quantization.  This affine encoder supplies the
    missing direction and intentionally includes zero in every local range,
    which keeps all-positive/all-negative groups well behaved.
    """

    if bits not in {4, 5}:
        raise ValueError("bits must be 4 or 5")
    source = np.asarray(values, dtype=np.float32)
    if source.ndim == 0 or source.shape[-1] % 256:
        raise ValueError("K-quant requires the last dimension to be divisible by 256")
    original_shape = source.shape
    blocks = np.nan_to_num(source, copy=True).reshape((-1, 8, 32))
    levels = (1 << bits) - 1

    local_min = np.minimum(blocks.min(axis=-1), 0.0)
    local_max = np.maximum(blocks.max(axis=-1), 0.0)
    local_scale = (local_max - local_min) / levels
    minimum_magnitude = -local_min

    global_scale = local_scale.max(axis=1) / 63.0
    global_min = minimum_magnitude.max(axis=1) / 63.0
    global_scale = global_scale.astype(np.float16).astype(np.float32)
    global_min = global_min.astype(np.float16).astype(np.float32)

    safe_scale = np.where(global_scale > 0, global_scale, 1.0)
    safe_min = np.where(global_min > 0, global_min, 1.0)
    scales = np.rint(local_scale / safe_scale[:, None]).clip(0, 63).astype(np.uint8)
    minima = np.rint(minimum_magnitude / safe_min[:, None]).clip(0, 63).astype(np.uint8)
    scales = np.where(global_scale[:, None] > 0, scales, 0).astype(np.uint8)
    minima = np.where(global_min[:, None] > 0, minima, 0).astype(np.uint8)

    effective_scale = global_scale[:, None] * scales
    effective_min = global_min[:, None] * minima
    divisor = np.where(effective_scale > 0, effective_scale, 1.0)
    quantized = np.rint((blocks + effective_min[:, :, None]) / divisor[:, :, None]).clip(0, levels)
    quantized = np.where(
        effective_scale[:, :, None] > 0,
        quantized,
        0,
    ).astype(np.uint8)

    block_count = blocks.shape[0]
    scale_bytes = _pack_scale_min(scales, minima)
    low = np.empty((block_count, 128), dtype=np.uint8)
    for pair in range(4):
        low[:, pair * 32 : (pair + 1) * 32] = (quantized[:, pair * 2] & 0x0F) | (
            (quantized[:, pair * 2 + 1] & 0x0F) << 4
        )

    parts = [
        global_scale.astype(np.float16).view(np.uint8).reshape(block_count, 2),
        global_min.astype(np.float16).view(np.uint8).reshape(block_count, 2),
        scale_bytes,
    ]
    if bits == 5:
        high = np.zeros((block_count, 32), dtype=np.uint8)
        for group in range(8):
            high |= ((quantized[:, group] >> 4) & 1) << group
        parts.append(high)
    parts.append(low)
    encoded = np.concatenate(parts, axis=1)
    bytes_per_block = 144 if bits == 4 else 176
    output_shape = (*original_shape[:-1], original_shape[-1] // 256 * bytes_per_block)
    return np.ascontiguousarray(encoded.reshape(output_shape))


def _write_sion_gguf(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    pad_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, int]:
    try:
        import gguf
    except ImportError as error:
        raise RuntimeError("gguf-python is required for GGUF export") from error

    temporary = _temporary_path(path)
    counts = {"q4_k": 0, "q5_k": 0, "f16": 0}
    writer = None
    try:
        writer = gguf.GGUFWriter(temporary, "sion")
        writer.add_name("sion_translate")
        writer.add_description(
            "Custom Sion encoder-decoder Transformer; storage/interchange only. "
            "No llama.cpp runtime implementation is provided."
        )
        writer.add_string("sion.runtime_support", "unsupported-by-llama.cpp")
        writer.add_string("sion.export_schema", EXPORT_SCHEMA)
        writer.add_uint32("sion.vocab_size", int(model_config.vocab_size))
        writer.add_uint32("sion.embedding_length", int(model_config.d_model))
        writer.add_uint32("sion.encoder.block_count", int(model_config.encoder_layers))
        writer.add_uint32("sion.decoder.block_count", int(model_config.decoder_layers))
        writer.add_uint32("sion.context_length", int(model_config.max_seq_len))
        writer.add_uint32("sion.pad_token_id", int(pad_id))
        language_pairs = _metadata_language_pairs(metadata)
        if language_pairs:
            writer.add_string(
                "sion.language_pairs",
                json.dumps(language_pairs, ensure_ascii=False, separators=(",", ":")),
            )
            if len(language_pairs) == 1:
                writer.add_array("sion.language_pair", language_pairs[0])
        tokenizer = metadata.get("tokenizer")
        if isinstance(tokenizer, Mapping) and tokenizer.get("sha256"):
            writer.add_string("sion.tokenizer.sha256", str(tokenizer["sha256"]))
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q4_K_M))
        writer.add_string(
            "sion.quantization.recipe",
            "Q4_K matrices; Q5_K embeddings/output/attention-out; incompatible tensors F16",
        )

        for name, tensor in state_dict.items():
            array = tensor.detach().float().cpu().contiguous().numpy()
            if array.ndim == 2 and array.shape[-1] % 256 == 0:
                use_q5 = name == "token_embedding.weight" or name.endswith(".out_proj.weight")
                qtype = gguf.GGMLQuantizationType.Q5_K if use_q5 else gguf.GGMLQuantizationType.Q4_K
                encoded = _quantize_affine_k(array, bits=5 if use_q5 else 4)
                writer.add_tensor(name, encoded, raw_dtype=qtype)
                counts["q5_k" if use_q5 else "q4_k"] += 1
            else:
                writer.add_tensor(name, array.astype(np.float16))
                counts["f16"] += 1
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        writer = None

        reader = gguf.GGUFReader(temporary)
        try:
            if len(reader.tensors) != len(state_dict):
                raise RuntimeError(
                    f"GGUF tensor count mismatch: {len(reader.tensors)} != {len(state_dict)}"
                )
            for tensor in reader.tensors:
                expected = state_dict.get(tensor.name)
                if expected is None:
                    raise RuntimeError(f"unexpected GGUF tensor: {tensor.name}")
                if tuple(map(int, tensor.shape)) != tuple(reversed(expected.shape)):
                    raise RuntimeError(f"GGUF shape mismatch for {tensor.name}")
        finally:
            memory_map = getattr(reader.data, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
            del reader
        gc.collect()
        os.replace(temporary, path)
        return counts
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _inspect_sion_gguf(path: Path) -> dict[str, Any]:
    try:
        import gguf
    except ImportError as error:
        raise RuntimeError("gguf-python is required for GGUF validation") from error

    reader = gguf.GGUFReader(path)
    try:
        counts = {"q4_k": 0, "q5_k": 0, "f16": 0}
        for tensor in reader.tensors:
            name = tensor.tensor_type.name.lower()
            if name in counts:
                counts[name] += 1
        return {
            "tensor_count": len(reader.tensors),
            "tensor_counts": counts,
        }
    finally:
        memory_map = getattr(reader.data, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
        del reader
        gc.collect()


def _base_payload(
    model_config: ModelConfig,
    pad_id: int,
    step: int,
    metadata: Mapping[str, Any],
    *,
    format_name: str,
) -> dict[str, Any]:
    return {
        "schema": EXPORT_SCHEMA,
        "format": format_name,
        "step": int(step),
        "model_config": asdict(model_config),
        "pad_id": int(pad_id),
        "metadata": copy.deepcopy(dict(metadata)),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict) or loaded.get("schema") != MANIFEST_SCHEMA:
        return None
    return loaded


def _existing_entry_is_valid(directory: Path, entry: Mapping[str, Any]) -> bool:
    if entry.get("status") != "ok" or not isinstance(entry.get("file"), str):
        return False
    if entry.get("size") is None or not _is_sha256(entry.get("sha256")):
        return False
    path = directory / str(entry["file"])
    try:
        if path.is_dir():
            actual = _directory_entry(path)
            if entry.get("artifact_type") not in {None, "directory"}:
                return False
        elif path.is_file():
            actual = _file_entry(path)
        else:
            return False
    except OSError:
        return False
    if actual["size"] != int(entry["size"]):
        return False
    if entry.get("file_count") is not None and actual.get("file_count") != int(entry["file_count"]):
        return False
    return actual["sha256"] == entry["sha256"]


def resolve_manifest_artifact(
    directory: str | Path,
    format_names: Sequence[str],
) -> Path | None:
    """Resolve one declared artifact after validating its manifest identity.

    ``None`` means that no manifest exists, or that none of the requested
    formats is marked successful. An existing but malformed/tampered manifest
    raises instead of allowing callers to fall back to an unverified stale file.
    """

    directory = Path(directory)
    manifest_path = directory / "export_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"missing or invalid export manifest: {manifest_path}")
    for field in ("state_sha256", "artifact_set_id"):
        if not _is_sha256(manifest.get(field)):
            raise ValueError(f"manifest.{field} must be a SHA256 digest")
    formats = manifest.get("formats")
    if not isinstance(formats, Mapping):
        raise ValueError("manifest.formats must be an object")

    root = directory.resolve()
    for format_name in format_names:
        raw_entry = formats.get(format_name)
        if raw_entry is None:
            continue
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"manifest format {format_name!r} must be an object")
        if raw_entry.get("status") != "ok":
            continue
        if raw_entry.get("artifact_set_id") != manifest["artifact_set_id"]:
            raise ValueError(f"manifest format {format_name!r} has a mismatched artifact_set_id")
        size = raw_entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"manifest format {format_name!r} has an invalid size")
        if not _is_sha256(raw_entry.get("sha256")):
            raise ValueError(f"manifest format {format_name!r} has an invalid SHA256")
        filename = raw_entry.get("file")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"manifest format {format_name!r} has no artifact path")
        artifact = (directory / filename).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ValueError("artifact path escapes export directory") from error
        if not artifact.is_file():
            raise ValueError(f"manifest format {format_name!r} does not point to a regular file")
        if not _existing_entry_is_valid(directory, raw_entry):
            raise ValueError(
                f"manifest format {format_name!r} size or SHA256 does not match the artifact"
            )
        return artifact
    return None


def _failure_entry(
    error: Exception,
    artifact_set_id: str,
    *,
    attempts: int,
) -> dict[str, Any]:
    return {
        "status": "error",
        "artifact_set_id": artifact_set_id,
        "error_type": type(error).__name__,
        "message": str(error),
        "failed_export_attempts": attempts,
        "created_unix": time.time(),
    }


def _normalize_formats(formats: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in formats))
    if not normalized or any(not value for value in normalized):
        raise ValueError("at least one non-empty export format is required")
    unknown = sorted(set(normalized) - set(SUPPORTED_FORMATS))
    if unknown:
        raise ValueError(f"unsupported export formats: {unknown}; supported: {SUPPORTED_FORMATS}")
    return normalized


def _export_state_dict_formats_unlocked(
    directory: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    pad_id: int,
    *,
    step: int = 0,
    formats: Sequence[str] = DEFAULT_CONVERSION_FORMATS,
    metadata: Mapping[str, Any] | None = None,
    tokenizer_path: str | Path | None = None,
    token_features_path: str | Path | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    int4_backend: str = "auto",
    fp8_policy: Fp8Policy | None = None,
    release_name: str = TRANSLATION_RELEASE_NAME,
    translation_capable: bool = True,
    llama_quantize: str | Path | None = None,
    _filename_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Export one state dict to requested formats and atomically merge a manifest.

    A same-weight subset conversion retains existing valid entries.  Different
    weights start a new artifact set.  If a re-export fails, a previously valid
    artifact remains referenced and receives ``last_error`` diagnostics.
    """

    del llama_quantize  # Custom Sion K-quant is deterministic and needs no subprocess.
    requested = _normalize_formats(formats)
    if int4_backend not in {"auto", "torchao", "packed"}:
        raise ValueError("int4_backend must be one of: auto, torchao, packed")
    # FP8 export 는 그 자체가 "FP8 로 내보내라"는 요청이므로 여기서는 켠 정책이
    # 기본입니다. 무엇을 내릴지는 여전히 정책이 정합니다 (기본 범위 = FFN).
    fp8_policy = fp8_policy or Fp8Policy(enabled=True)
    fp8_policy.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Callers already hand us a stable snapshot. Avoid another full host copy:
    # a 32B FP32 state is about 128 GiB before any per-format conversion.
    cpu_state = {name: tensor.detach().to("cpu") for name, tensor in state_dict.items()}
    state_hash = _state_sha256(cpu_state)
    artifact_id = _artifact_set_id(state_hash, model_config, pad_id)
    manifest_path = directory / "export_manifest.json"
    previous_manifest = _read_manifest(manifest_path)
    same_weights = (
        previous_manifest is not None and previous_manifest.get("artifact_set_id") == artifact_id
    )
    if metadata is None and previous_manifest is not None and same_weights:
        export_metadata = copy.deepcopy(previous_manifest.get("metadata") or {})
    else:
        export_metadata = copy.deepcopy(
            dict(metadata)
            if metadata is not None
            else build_export_metadata(
                model_config,
                tokenizer_path=tokenizer_path,
                token_features_path=token_features_path,
                language_pairs=language_pairs,
                step=step,
                release_name=release_name,
                translation_capable=translation_capable,
            )
        )
    validated_name, validated_version, validated_capability = _validated_release_identity(
        export_metadata.get("release_name"),
        export_metadata.get("release_version"),
        translation_capable=export_metadata.get("translation_capable"),
    )
    export_metadata["release_name"] = validated_name
    export_metadata["release_version"] = validated_version
    export_metadata["translation_capable"] = validated_capability
    if "generation_defaults" not in export_metadata:
        export_metadata["generation_defaults"] = {
            "reasoning_level": _default_reasoning_level(model_config),
        }
    _validate_generation_defaults(export_metadata, model_config, required=True)
    export_metadata.setdefault("step", int(step))
    if tokenizer_path is not None:
        export_metadata["tokenizer"] = _file_identity(tokenizer_path)
    if token_features_path is not None:
        export_metadata["token_features"] = _file_identity(token_features_path)
    embedded_sidecars: list[str] = []
    for sidecar_path, metadata_name in (
        (tokenizer_path, "tokenizer"),
        (token_features_path, "token_features"),
    ):
        identity = export_metadata.get(metadata_name)
        if sidecar_path is not None and isinstance(identity, Mapping):
            _atomic_copy_file(
                sidecar_path,
                directory / str(identity["filename"]),
            )
            embedded_sidecars.append(metadata_name)
    if embedded_sidecars:
        export_metadata["embedded_sidecars"] = embedded_sidecars
    resolved_language_pairs = (
        _normalize_language_pairs(language_pairs=language_pairs)
        if language_pairs is not None
        else _metadata_language_pairs(export_metadata)
    )
    if language_pairs is not None:
        if resolved_language_pairs:
            export_metadata["language_pairs"] = resolved_language_pairs
            export_metadata["languages"] = _languages_from_pairs(resolved_language_pairs)
            if len(resolved_language_pairs) == 1:
                export_metadata["language_pair"] = resolved_language_pairs[0]
            else:
                export_metadata.pop("language_pair", None)
        else:
            export_metadata.pop("language_pair", None)
            export_metadata.pop("language_pairs", None)
            # A foundation export has languages but deliberately has no
            # translation pairs. Dropping both would erase the only truthful
            # statement of what its denoising weights were trained on.
            if export_metadata.get("translation_capable") is not False:
                export_metadata.pop("languages", None)
            export_metadata.pop("translation_directions", None)
    # 번역 불가 산출물(foundation)에는 방향을 유도해 넣지 않습니다. 이 함수는
    # 방향이 비어 있으면 language_pairs 에서 만들어 채우는데, 그 가중치는
    # 어떤 방향으로도 번역할 수 없습니다.
    resolved_translation_capable = _metadata_translation_capable(export_metadata)
    resolved_translation_directions: list[list[str]] = []
    if resolved_translation_capable:
        resolved_translation_directions = _metadata_translation_directions(export_metadata)
        if resolved_translation_directions:
            export_metadata["translation_directions"] = resolved_translation_directions
    resolved_languages = _metadata_languages(export_metadata)
    metadata_compatibility_id = _metadata_compatibility_id(export_metadata)
    previous_metadata_compatibility_id = None
    if same_weights and previous_manifest is not None:
        previous_metadata_compatibility_id = previous_manifest.get("metadata_compatibility_id")
        if previous_metadata_compatibility_id is None:
            previous_metadata = previous_manifest.get("metadata")
            if isinstance(previous_metadata, Mapping):
                previous_metadata_compatibility_id = _metadata_compatibility_id(previous_metadata)
    same_artifact = same_weights and previous_metadata_compatibility_id == metadata_compatibility_id
    previous_formats = (
        dict(previous_manifest.get("formats") or {})
        if same_artifact and previous_manifest is not None
        else {}
    )
    filenames = dict(_FORMAT_FILENAMES)
    filenames.update(dict(_filename_overrides or {}))
    format_entries = dict(previous_formats)

    for format_name in requested:
        filename = filenames[format_name]
        path = directory / filename
        previous = previous_formats.get(format_name)
        try:
            payload = _base_payload(
                model_config,
                pad_id,
                step,
                export_metadata,
                format_name=format_name,
            )
            details: dict[str, Any]
            if format_name in _PRECISION_DTYPES:
                dtype = _PRECISION_DTYPES[format_name]
                payload["model"] = _precision_state(cpu_state, dtype)
                _atomic_torch_save(payload, path)
                details = {"dtype": ("float32" if dtype == torch.float32 else str(dtype))}
            elif format_name == "int8":
                quantized, quantization = _torchao_quantized_state(
                    model_config,
                    cpu_state,
                    pad_id,
                    bits=8,
                )
                payload["model"] = quantized
                payload["quantization"] = quantization
                _atomic_torch_save(payload, path)
                details = {"quantization": quantization}
            elif format_name == "int4":
                quantized: Mapping[str, Any]
                if int4_backend in {"auto", "torchao"}:
                    try:
                        quantized, quantization = _torchao_quantized_state(
                            model_config,
                            cpu_state,
                            pad_id,
                            bits=4,
                        )
                    except Exception:
                        if int4_backend == "torchao":
                            raise
                        quantized, quantization = _pack_int4_state(cpu_state)
                else:
                    quantized, quantization = _pack_int4_state(cpu_state)
                payload["model"] = dict(quantized)
                payload["quantization"] = quantization
                _atomic_torch_save(payload, path)
                details = {"quantization": quantization}
            elif format_name == "fp8":
                quantized, quantization = _pack_fp8_state(cpu_state, fp8_policy)
                payload["model"] = dict(quantized)
                payload["quantization"] = quantization
                _atomic_torch_save(payload, path)
                details = {"quantization": quantization}
            elif format_name == "gguf_q4_k_m":
                tensor_counts = _write_sion_gguf(
                    path,
                    cpu_state,
                    model_config,
                    pad_id,
                    export_metadata,
                )
                details = {
                    "quantization": "Q4_K_M",
                    "architecture": "sion",
                    "runtime_supported_by_llama_cpp": False,
                    "purpose": "storage/interchange",
                    "backend": "gguf-python",
                    "tensor_counts": tensor_counts,
                }
            else:
                if model_config.experimental.morphoscript_enabled and token_features_path is None:
                    raise ValueError(
                        "MorphoScript-enabled Transformers export requires token_features_path"
                    )
                if tokenizer_path is None and isinstance(
                    export_metadata.get("tokenizer"),
                    Mapping,
                ):
                    raise ValueError(
                        "Transformers export metadata references a tokenizer, "
                        "but tokenizer_path was not provided or resolvable"
                    )
                if token_features_path is None and isinstance(
                    export_metadata.get("token_features"),
                    Mapping,
                ):
                    raise ValueError(
                        "Transformers export metadata references token features, "
                        "but token_features_path was not provided or resolvable"
                    )
                inspection = _write_transformers_checkpoint(
                    path,
                    cpu_state,
                    model_config,
                    pad_id,
                    tokenizer_path=tokenizer_path,
                    token_features_path=token_features_path,
                    languages=resolved_languages,
                    language_pairs=resolved_language_pairs,
                    translation_directions=resolved_translation_directions,
                    translation_capable=resolved_translation_capable,
                    revision_trained=_metadata_revision_capability(export_metadata),
                    release_name=str(export_metadata["release_name"]),
                    release_version=str(export_metadata["release_version"]),
                )
                details = {
                    "dtype": (
                        inspection["dtypes"][0].removeprefix("torch.")
                        if len(inspection["dtypes"]) == 1
                        else inspection["dtypes"]
                    ),
                    "format": "transformers-safetensors-v1",
                    "backend": "transformers",
                    **inspection,
                }
            artifact_entry = (
                _directory_entry(path) if format_name == "transformers" else _file_entry(path)
            )
            entry = {
                "status": "ok",
                **artifact_entry,
                "artifact_set_id": artifact_id,
                **details,
            }
            format_entries[format_name] = entry
        except Exception as error:
            prior_attempts = (
                int(previous.get("failed_export_attempts", 0))
                if isinstance(previous, Mapping)
                else 0
            )
            failure = _failure_entry(
                error,
                artifact_id,
                attempts=prior_attempts + 1,
            )
            if isinstance(previous, Mapping) and _existing_entry_is_valid(directory, previous):
                retained = dict(previous)
                retained["last_error"] = failure
                retained["failed_export_attempts"] = prior_attempts + 1
                format_entries[format_name] = retained
            else:
                format_entries[format_name] = failure

    requested_union = [
        format_name for format_name in SUPPORTED_FORMATS if format_name in format_entries
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_unix": (
            previous_manifest.get("created_unix", time.time())
            if same_artifact and previous_manifest is not None
            else time.time()
        ),
        "state_sha256": state_hash,
        "artifact_set_id": artifact_id,
        "metadata_compatibility_id": metadata_compatibility_id,
        "requested_formats": requested_union,
        "last_requested_formats": list(requested),
        "formats": format_entries,
        "metadata": export_metadata,
    }
    _atomic_json_dump(manifest, manifest_path)
    return manifest


def export_state_dict_formats(
    directory: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    pad_id: int,
    *,
    step: int = 0,
    formats: Sequence[str] = DEFAULT_CONVERSION_FORMATS,
    metadata: Mapping[str, Any] | None = None,
    tokenizer_path: str | Path | None = None,
    token_features_path: str | Path | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    int4_backend: str = "auto",
    fp8_policy: Fp8Policy | None = None,
    release_name: str = TRANSLATION_RELEASE_NAME,
    translation_capable: bool = True,
    llama_quantize: str | Path | None = None,
    _filename_overrides: Mapping[str, str] | None = None,
    _acquire_publish_lock: bool = True,
) -> dict[str, Any]:
    """Export one complete file/manifest generation under a destination lock."""

    if not _acquire_publish_lock:
        return _export_state_dict_formats_unlocked(
            directory,
            state_dict,
            model_config,
            pad_id,
            step=step,
            formats=formats,
            metadata=metadata,
            tokenizer_path=tokenizer_path,
            token_features_path=token_features_path,
            language_pairs=language_pairs,
            int4_backend=int4_backend,
            fp8_policy=fp8_policy,
            release_name=release_name,
            translation_capable=translation_capable,
            llama_quantize=llama_quantize,
            _filename_overrides=_filename_overrides,
        )
    destination = Path(directory)
    with artifact_lock(
        _export_publish_lock_root(destination),
        timeout=_EXPORT_LOCK_TIMEOUT_SECONDS,
        poll_interval=0.05,
    ):
        return _export_state_dict_formats_unlocked(
            destination,
            state_dict,
            model_config,
            pad_id,
            step=step,
            formats=formats,
            metadata=metadata,
            tokenizer_path=tokenizer_path,
            token_features_path=token_features_path,
            language_pairs=language_pairs,
            int4_backend=int4_backend,
            fp8_policy=fp8_policy,
            release_name=release_name,
            translation_capable=translation_capable,
            llama_quantize=llama_quantize,
            _filename_overrides=_filename_overrides,
        )


def convert_export(
    source: str | Path,
    directory: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_CONVERSION_FORMATS,
    tokenizer_path: str | Path | None = None,
    token_features_path: str | Path | None = None,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    bidirectional: bool | None = None,
    revision_trained: bool | None = None,
    int4_backend: str = "auto",
    fp8_policy: Fp8Policy | None = None,
    release_name: str | None = None,
    release_version: str | None = None,
    translation_capable: bool | None = None,
    llama_quantize: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a stable state-dict export without mutating the source artifact."""

    source = Path(source)
    requested = _normalize_formats(formats)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        raise ValueError(f"{source} cannot be read as a safe weights-only Sion export") from error
    if not isinstance(payload, dict) or "model_config" not in payload:
        raise ValueError(f"{source} is not a Sion inference export")
    stored = payload.get("model")
    if not isinstance(stored, Mapping) or stored.get("kind") == "packed_int4":
        raise ValueError("conversion source must contain a stable state dict")
    if any(not isinstance(value, torch.Tensor) for value in stored.values()):
        raise ValueError("conversion source contains a quantized/non-tensor state dict")
    quantization = payload.get("quantization")
    if quantization:
        raise ValueError("conversion source must be FP32/FP16/BF16, not quantized")
    config = _model_config_from_dict(payload["model_config"])
    step = int(payload.get("step", 0))
    pad_id = int(payload.get("pad_id", 0))
    inherited = copy.deepcopy(payload.get("metadata") or {})

    def resolve_release_identity(explicit: str | None, metadata_name: str) -> str:
        inherited_value: object = inherited.get(metadata_name)
        if (
            explicit is not None
            and isinstance(inherited_value, str)
            and inherited_value.strip()
            and explicit.strip() != inherited_value.strip()
        ):
            raise ValueError(
                f"explicit {metadata_name} {explicit!r} conflicts with source metadata "
                f"{inherited_value!r}; conversion cannot relabel weights"
            )
        value: object = explicit if explicit is not None else inherited_value
        if not isinstance(value, str) or not value.strip():
            option = metadata_name.replace("_", "-")
            raise ValueError(
                f"source export has no trustworthy {metadata_name}; pass --{option} explicitly"
            )
        return value.strip()

    resolved_release_name = resolve_release_identity(release_name, "release_name")
    resolved_release_version = resolve_release_identity(release_version, "release_version")
    inherited_translation_capable = inherited.get("translation_capable")
    if (
        translation_capable is not None
        and isinstance(inherited_translation_capable, bool)
        and translation_capable is not inherited_translation_capable
    ):
        raise ValueError(
            "explicit translation_capable conflicts with source metadata; "
            "conversion cannot relabel weights"
        )
    if translation_capable is None and not isinstance(inherited_translation_capable, bool):
        raise ValueError(
            "source export has no trustworthy translation_capable; pass "
            "--translation-capable or --foundation-only explicitly"
        )
    resolved_translation_capable = (
        translation_capable
        if translation_capable is not None
        else cast(bool, inherited_translation_capable)
    )

    def resolve_inherited_sidecar(
        explicit: str | Path | None,
        metadata_name: str,
    ) -> str | Path | None:
        if explicit is not None or "transformers" not in requested:
            return explicit
        identity = inherited.get(metadata_name)
        if not isinstance(identity, Mapping):
            return None
        filename = identity.get("filename")
        expected_hash = identity.get("sha256")
        if not isinstance(filename, str) or not _is_sha256(expected_hash):
            return None
        candidate = source.parent / Path(filename).name
        if candidate.is_file() and _sha256_file(candidate) == expected_hash:
            return candidate
        return None

    tokenizer_path = resolve_inherited_sidecar(tokenizer_path, "tokenizer")
    token_features_path = resolve_inherited_sidecar(
        token_features_path,
        "token_features",
    )
    metadata = build_export_metadata(
        config,
        tokenizer_path=tokenizer_path,
        token_features_path=token_features_path,
        language_pair=language_pair,
        language_pairs=language_pairs,
        bidirectional=True if bidirectional is None else bidirectional,
        revision_trained=revision_trained,
        step=step,
        source=source,
        release_name=resolved_release_name,
        release_version=resolved_release_version,
        translation_capable=resolved_translation_capable,
    )
    if tokenizer_path is None and inherited.get("tokenizer"):
        metadata["tokenizer"] = inherited["tokenizer"]
    if token_features_path is None and inherited.get("token_features"):
        metadata["token_features"] = inherited["token_features"]
    if language_pair is None and language_pairs is None:
        inherited_pairs = _metadata_language_pairs(inherited)
        if inherited_pairs:
            metadata["language_pairs"] = inherited_pairs
            metadata["languages"] = _languages_from_pairs(inherited_pairs)
            if len(inherited_pairs) == 1:
                metadata["language_pair"] = inherited_pairs[0]
            metadata["translation_directions"] = _metadata_translation_directions(inherited)
        elif inherited.get("languages"):
            metadata["languages"] = _metadata_languages(inherited)
    if revision_trained is None and inherited.get("capabilities"):
        metadata["capabilities"] = inherited["capabilities"]
    return export_state_dict_formats(
        directory,
        dict(stored),
        config,
        pad_id,
        step=step,
        formats=requested,
        metadata=metadata,
        tokenizer_path=tokenizer_path,
        token_features_path=token_features_path,
        language_pairs=_metadata_language_pairs(metadata),
        int4_backend=int4_backend,
        fp8_policy=fp8_policy,
        release_name=resolved_release_name,
        translation_capable=resolved_translation_capable,
        llama_quantize=llama_quantize,
    )


def load_exported_model(
    path: str | Path,
    *,
    return_metadata: bool = False,
    allow_legacy: bool = True,
    unsafe_allow_pickle: bool = False,
) -> tuple[nn.Module, ModelConfig, int] | tuple[nn.Module, ModelConfig, int, dict[str, Any]]:
    """Load native precision, TorchAO INT8/INT4, packed INT4, or legacy exports.

    The default loader accepts only PyTorch's weights-only representation.
    ``unsafe_allow_pickle`` exists solely for an explicit, one-time migration of
    a trusted legacy bundle that pickled an ``nn.Module``.
    """

    path = Path(path)
    if unsafe_allow_pickle:
        warnings.warn(
            "unsafe_allow_pickle=True can execute code embedded in the model file. "
            "Use it only to migrate a file whose origin and contents you trust.",
            RuntimeWarning,
            stacklevel=2,
        )
        with _legacy_kjx_pickle_aliases():
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception as error:
            raise ValueError(
                f"{path} cannot be read with the safe weights-only loader. "
                "Refusing executable pickle; trusted legacy migrations must explicitly "
                "pass unsafe_allow_pickle=True."
            ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain an export payload")
    schema = payload.get("schema")
    if schema is None and not allow_legacy:
        raise ValueError(f"{path} is a legacy export without {EXPORT_SCHEMA} metadata")
    if schema is not None and schema != EXPORT_SCHEMA:
        raise ValueError(f"unsupported export schema: {schema}")
    config = _model_config_from_dict(payload["model_config"])
    pad_id = int(payload["pad_id"])
    stored = payload["model"]
    quantization = payload.get("quantization")

    if isinstance(stored, nn.Module):
        model: nn.Module = stored
        _hydrate_legacy_module_attributes(model, config)
    else:
        fp8_packed = (
            isinstance(quantization, Mapping)
            and quantization.get("format") == "fp8"
            and isinstance(stored, Mapping)
        )
        if fp8_packed:
            # 먼저 고정밀도로 실은 뒤 아래에서 FP8 모듈로 바꿔 끼웁니다. 이렇게
            # 하는 이유는 모듈 구조와 키를 그대로 검증받기 위해서이고, 교체가
            # 끝나면 상주 가중치는 FP8 입니다.
            state = _unpack_fp8_state(stored)
        elif isinstance(quantization, Mapping) and quantization.get("format") == "fp8":
            raise ValueError("FP8 export has no packed state dictionary")
        elif isinstance(quantization, Mapping) and quantization.get("backend") == "sion-packed":
            if not isinstance(stored, Mapping):
                raise ValueError("packed INT4 export has no packed state dictionary")
            state = _unpack_int4_state(stored)
        elif isinstance(stored, Mapping):
            state = dict(stored)
        else:
            raise ValueError("export model must be a module or state dictionary")
        runtime_config = copy.deepcopy(config)
        runtime_config.gradient_checkpointing = False
        with torch.random.fork_rng(devices=[]):
            with torch.device("meta"):
                model = SionForConditionalGeneration(runtime_config, pad_id=pad_id)
        # Bind mmap-backed tensors directly instead of allocating and initializing
        # another full model before copying.  This keeps strict validation of a
        # 32B FP32 artifact near one model's host-memory footprint rather than two.
        model.load_state_dict(state, assign=True)
        if fp8_packed:
            # 되돌린 가중치를 버리고 FP8 을 상주시킵니다. 현재 계산은 forward
            # 마다 BF16(미지원 CUDA는 FP16)으로 역양자화하지만, 모델의 상주
            # 메모리는 FP8 로 줄입니다.
            replaced = apply_fp8_weights(model, dict(stored))
            if not replaced:
                raise ValueError("FP8 export contains no quantized weights")
    model.eval()
    metadata = copy.deepcopy(payload.get("metadata") or {})
    _validate_generation_defaults(metadata, config, required=False)
    if isinstance(quantization, Mapping):
        metadata.setdefault("quantization", copy.deepcopy(dict(quantization)))
    if return_metadata:
        return model, config, pad_id, metadata
    return model, config, pad_id


def validate_export_directory(
    directory: str | Path,
    *,
    expected_release_name: str | None = None,
    expected_release_version: str | None = None,
    expected_translation_capable: bool | None = None,
) -> dict[str, Any]:
    """Verify manifest integrity and minimally load every successful artifact."""

    if expected_translation_capable is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        expected_translation_capable, bool
    ):
        raise TypeError("expected_translation_capable must be a boolean or None")
    if expected_release_name is not None and (
        not isinstance(expected_release_name, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        or not expected_release_name.strip()
    ):
        raise ValueError("expected_release_name must be a non-empty string or None")
    if expected_release_version is not None and (
        not isinstance(expected_release_version, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        or _MODEL_VERSION_PATTERN.fullmatch(expected_release_version.strip()) is None
    ):
        raise ValueError(
            "expected_release_version must use a numeric major.minor[.patch] value or None"
        )
    if expected_release_name == FOUNDATION_RELEASE_NAME and expected_translation_capable is True:
        raise ValueError("expected sion foundation identity cannot be translation-capable")
    if expected_release_name == TRANSLATION_RELEASE_NAME and expected_translation_capable is False:
        raise ValueError("expected sion_translate identity must be translation-capable")

    directory = Path(directory)
    manifest_path = directory / "export_manifest.json"
    manifest = _read_manifest(manifest_path)
    report: dict[str, Any] = {
        "valid": False,
        "directory": str(directory),
        "schema": None,
        "artifact_set_id": None,
        "formats": {},
        "errors": [],
    }
    if manifest is None:
        report["errors"].append(
            {
                "error_type": "InvalidManifest",
                "message": f"missing or invalid manifest: {manifest_path}",
            }
        )
        return report
    report["schema"] = manifest.get("schema")
    report["artifact_set_id"] = manifest.get("artifact_set_id")
    for field in ("state_sha256", "artifact_set_id"):
        if not _is_sha256(manifest.get(field)):
            report["errors"].append(
                {
                    "error_type": "InvalidManifest",
                    "message": f"manifest.{field} must be a SHA256 digest",
                }
            )
    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, Mapping):
        report["errors"].append(
            {
                "error_type": "InvalidManifest",
                "message": "manifest.metadata must be an object",
            }
        )
        return report
    try:
        validated_name, validated_version, validated_capability = _validated_release_identity(
            manifest_metadata.get("release_name"),
            manifest_metadata.get("release_version"),
            translation_capable=manifest_metadata.get("translation_capable"),
        )
        if (
            manifest_metadata.get("release_name") != validated_name
            or manifest_metadata.get("release_version") != validated_version
            or manifest_metadata.get("translation_capable") is not validated_capability
        ):
            raise ValueError("manifest release identity is not normalized")
    except (TypeError, ValueError) as error:
        report["errors"].append(
            {
                "error_type": "InvalidReleaseIdentity",
                "message": str(error),
            }
        )
    for metadata_name, expected in (
        ("release_name", expected_release_name),
        ("release_version", expected_release_version),
        ("translation_capable", expected_translation_capable),
    ):
        if expected is not None and manifest_metadata.get(metadata_name) != expected:
            report["errors"].append(
                {
                    "error_type": "UnexpectedIdentity",
                    "message": (
                        f"manifest.metadata.{metadata_name} is "
                        f"{manifest_metadata.get(metadata_name)!r}, expected {expected!r}"
                    ),
                }
            )
    metadata_compatibility_id = _metadata_compatibility_id(manifest_metadata)
    report["metadata_compatibility_id"] = metadata_compatibility_id
    stored_metadata_compatibility_id = manifest.get("metadata_compatibility_id")
    if stored_metadata_compatibility_id is not None and (
        not _is_sha256(stored_metadata_compatibility_id)
        or stored_metadata_compatibility_id != metadata_compatibility_id
    ):
        report["errors"].append(
            {
                "error_type": "MetadataMismatch",
                "message": "manifest metadata compatibility hash does not match",
            }
        )
    embedded_sidecars = manifest_metadata.get("embedded_sidecars") or []
    if not isinstance(embedded_sidecars, list):
        report["errors"].append(
            {
                "error_type": "InvalidManifest",
                "message": "manifest.metadata.embedded_sidecars must be a list",
            }
        )
    else:
        for metadata_name in embedded_sidecars:
            identity = manifest_metadata.get(metadata_name)
            if not isinstance(metadata_name, str) or not isinstance(identity, Mapping):
                report["errors"].append(
                    {
                        "error_type": "InvalidManifest",
                        "message": f"invalid embedded sidecar identity: {metadata_name!r}",
                    }
                )
                continue
            filename = identity.get("filename")
            safe_filename = (
                filename if isinstance(filename, str) and Path(filename).name == filename else None
            )
            sidecar = directory / str(safe_filename)
            if (
                safe_filename is None
                or not sidecar.is_file()
                or identity.get("size") != sidecar.stat().st_size
                or not _is_sha256(identity.get("sha256"))
                or _sha256_file(sidecar) != identity["sha256"]
            ):
                report["errors"].append(
                    {
                        "error_type": "SidecarMismatch",
                        "message": f"embedded {metadata_name} does not match metadata",
                    }
                )
    formats = manifest.get("formats")
    if not isinstance(formats, Mapping):
        report["errors"].append(
            {
                "error_type": "InvalidManifest",
                "message": "manifest.formats must be an object",
            }
        )
        return report
    if "transformers" in formats and stored_metadata_compatibility_id is None:
        report["errors"].append(
            {
                "error_type": "InvalidManifest",
                "message": (
                    "manifest.metadata_compatibility_id is required for Transformers exports"
                ),
            }
        )

    root = directory.resolve()
    for format_name, raw_entry in formats.items():
        validation: dict[str, Any] = {"valid": False}
        try:
            if format_name not in SUPPORTED_FORMATS:
                raise ValueError(f"unsupported manifest format: {format_name}")
            if not isinstance(raw_entry, Mapping):
                raise ValueError("format manifest entry must be an object")
            if raw_entry.get("status") != "ok":
                raise RuntimeError(f"artifact status is {raw_entry.get('status', 'missing')!r}")
            if raw_entry.get("artifact_set_id") != manifest.get("artifact_set_id"):
                raise RuntimeError("artifact_set_id does not match the manifest")
            if raw_entry.get("size") is None or not _is_sha256(raw_entry.get("sha256")):
                raise ValueError("artifact entry requires size and SHA256")
            if format_name == "transformers" and (
                raw_entry.get("artifact_type") != "directory"
                or raw_entry.get("file_count") is None
                or not isinstance(raw_entry.get("files"), list)
            ):
                raise ValueError("Transformers entry requires deterministic directory metadata")
            filename = raw_entry.get("file")
            if not isinstance(filename, str) or not filename:
                raise ValueError("artifact entry has no file")
            artifact = (directory / filename).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as error:
                raise ValueError("artifact path escapes export directory") from error
            if not _existing_entry_is_valid(directory, raw_entry):
                raise RuntimeError("artifact size or SHA256 does not match the manifest")
            if format_name == "transformers":
                actual_directory = _directory_entry(artifact)
                if raw_entry.get("files") != actual_directory["files"]:
                    raise RuntimeError(
                        "Transformers file list does not match deterministic directory metadata"
                    )

            if format_name in {*_PRECISION_DTYPES, "int8", "int4"}:
                model, config, pad_id, artifact_metadata = cast(
                    tuple[nn.Module, ModelConfig, int, dict[str, Any]],
                    load_exported_model(
                        artifact,
                        return_metadata=True,
                    ),
                )
                expected_artifact_set_id = _artifact_set_id(
                    str(manifest["state_sha256"]),
                    config,
                    pad_id,
                )
                if expected_artifact_set_id != manifest.get("artifact_set_id"):
                    raise RuntimeError(
                        "artifact_set_id does not match state hash, model config, and pad ID"
                    )
                if _metadata_compatibility_id(artifact_metadata) != metadata_compatibility_id:
                    raise RuntimeError("native payload metadata does not match the manifest")
                validation["inspection"] = {
                    "loader": "load_exported_model",
                    "model_class": type(model).__name__,
                    "vocab_size": config.vocab_size,
                    "pad_id": pad_id,
                }
                del model, config, artifact_metadata
                gc.collect()
            elif format_name == "gguf_q4_k_m":
                inspection = _inspect_sion_gguf(artifact)
                expected_counts = raw_entry.get("tensor_counts")
                if isinstance(expected_counts, Mapping) and dict(inspection["tensor_counts"]) != {
                    str(name): int(count) for name, count in expected_counts.items()
                }:
                    raise RuntimeError("GGUF tensor counts do not match the manifest")
                validation["inspection"] = inspection
            else:
                for metadata_name, default_filename in (
                    ("tokenizer", "tokenizer.model"),
                    ("token_features", "token_features.npz"),
                ):
                    sidecar_metadata = manifest_metadata.get(metadata_name)
                    if not isinstance(sidecar_metadata, Mapping):
                        continue
                    sidecar_file = artifact / str(
                        sidecar_metadata.get("filename", default_filename)
                    )
                    if (
                        not sidecar_file.is_file()
                        or sidecar_metadata.get("size") != sidecar_file.stat().st_size
                        or not _is_sha256(sidecar_metadata.get("sha256"))
                        or _sha256_file(sidecar_file) != sidecar_metadata["sha256"]
                    ):
                        raise RuntimeError(
                            f"Transformers {metadata_name} does not match manifest metadata"
                        )
                inspection = _inspect_transformers_checkpoint(artifact)
                expected_languages = _metadata_languages(manifest_metadata)
                if expected_languages is not None and inspection["languages"] != expected_languages:
                    raise RuntimeError("Transformers languages do not match the manifest")
                expected_pairs = _metadata_language_pairs(manifest_metadata)
                if expected_pairs and inspection["language_pairs"] != expected_pairs:
                    raise RuntimeError("Transformers language pairs do not match the manifest")
                expected_directions = _metadata_translation_directions(manifest_metadata)
                if inspection["translation_directions"] != expected_directions:
                    raise RuntimeError(
                        "Transformers translation directions do not match the manifest"
                    )
                expected_translation_capable = _metadata_translation_capable(manifest_metadata)
                if inspection["translation_capable"] is not expected_translation_capable:
                    raise RuntimeError(
                        "Transformers translation capability does not match the manifest"
                    )
                if inspection["release_name"] != manifest_metadata.get("release_name"):
                    raise RuntimeError("Transformers release_name does not match the manifest")
                if inspection["release_version"] != manifest_metadata.get("release_version"):
                    raise RuntimeError("Transformers release_version does not match the manifest")
                manifest_defaults = manifest_metadata.get("generation_defaults")
                manifest_reasoning_level = (
                    manifest_defaults.get("reasoning_level")
                    if isinstance(manifest_defaults, Mapping)
                    else None
                )
                if inspection["reasoning_level"] != manifest_reasoning_level:
                    raise RuntimeError(
                        "Transformers reasoning endpoint does not match the manifest"
                    )
                expected_revision = _metadata_revision_capability(manifest_metadata)
                if inspection["revision_trained"] is not expected_revision:
                    raise RuntimeError(
                        "Transformers revision capability does not match the manifest"
                    )
                validation["inspection"] = inspection
            validation["valid"] = True
        except Exception as error:
            validation["error_type"] = type(error).__name__
            validation["message"] = str(error)
            report["errors"].append(
                {
                    "format": str(format_name),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        report["formats"][str(format_name)] = validation

    report["valid"] = not report["errors"] and bool(report["formats"])
    return report


def unwrap_model(model: nn.Module) -> nn.Module:
    """Remove torch.compile and DDP wrappers without touching FSDP2."""

    unwrapped = model
    while True:
        original = getattr(unwrapped, "_orig_mod", None)
        if isinstance(original, nn.Module):
            unwrapped = original
            continue
        module = getattr(unwrapped, "module", None)
        if isinstance(module, nn.Module):
            unwrapped = module
            continue
        return unwrapped


def gather_full_state_dict(
    model: nn.Module,
    context: DistributedContext,
) -> dict[str, torch.Tensor]:
    """Gather a complete CPU state dict from local, DDP, or FSDP2 training."""

    if context.distributed:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return cast(
            dict[str, torch.Tensor],
            get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            ),
        )
    return {
        # ``copy=True`` is required when the source already lives on CPU.
        # Otherwise an EMA swap snapshot aliases the live parameter storage and
        # silently turns back into raw weights when the swap context exits.
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in unwrap_model(model).state_dict().items()
    }


def _synchronize_rank0_exception(
    context: DistributedContext,
    error: Exception | None,
) -> None:
    """Broadcast a rank-0 export failure so peers never wait at a blind barrier."""

    serialized = (
        {
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if context.is_main and error is not None
        else None
    )
    payload: list[dict[str, str] | None] = [serialized]
    if context.distributed:
        dist.broadcast_object_list(
            payload,
            src=0,
            device=context.device,
        )
    if payload[0] is None:
        return
    if error is not None:
        raise error
    raise RuntimeError(
        f"rank 0 model export failed: {payload[0]['error_type']}: {payload[0]['message']}"
    )


def export_inference_models(
    directory: str | Path,
    model: nn.Module,
    model_config: ModelConfig,
    context: DistributedContext,
    step: int,
    *,
    ema: Any | None = None,
    formats: Sequence[str] = DEFAULT_TRAINING_FORMATS,
    tokenizer_path: str | Path | None = None,
    token_features_path: str | Path | None = None,
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    languages: Sequence[str] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    bidirectional: bool = True,
    revision_trained: bool | None = None,
    int4_backend: str = "auto",
    fp8_policy: Fp8Policy | None = None,
    release_name: str = TRANSLATION_RELEASE_NAME,
    translation_capable: bool = True,
    strict: bool = False,
) -> dict[str, Any] | None:
    """Gather selected weights and export them without rank skew or blind barriers."""

    requested = _normalize_formats(formats)
    directory = Path(directory)
    working_directory = _temporary_path(directory) if context.is_main and strict else directory
    manifest: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    pad_id = 0
    setup_error: Exception | None = None
    if context.is_main:
        try:
            pad_id = int(getattr(unwrap_model(model), "pad_id", 0))
            metadata = build_export_metadata(
                model_config,
                tokenizer_path=tokenizer_path,
                token_features_path=token_features_path,
                language_pair=language_pair,
                language_pairs=language_pairs,
                languages=languages,
                translation_directions=translation_directions,
                bidirectional=bidirectional,
                revision_trained=revision_trained,
                step=step,
                release_name=release_name,
                translation_capable=translation_capable,
            )
        except Exception as error:
            setup_error = error
    _synchronize_rank0_exception(context, setup_error)

    # With EMA enabled, checkpoints already retain the raw training weights.
    # Gather only the selected EMA deployment state here; exporting both full
    # copies would double collectives, host RAM, and filesystem traffic.
    if ema is not None:
        with ema.swap(model):
            deployment_state = gather_full_state_dict(model, context)
    else:
        deployment_state = gather_full_state_dict(model, context)

    # The final seven-format CPU conversion can take longer than the process
    # group's collective timeout for 8B/32B models. The full-state gather is
    # complete, so non-main ranks can release their process groups while rank 0
    # finishes the transactional export. torchrun still propagates rank-0 failure.
    if strict and not context.is_main:
        del deployment_state
        gc.collect()
        return None

    export_error: Exception | None = None
    if context.is_main:
        assert metadata is not None
        try:
            filename_overrides = {"fp32": "model_ema.pt"} if ema is not None else None
            manifest = export_state_dict_formats(
                working_directory,
                deployment_state,
                model_config,
                pad_id,
                step=step,
                formats=requested,
                metadata=metadata,
                tokenizer_path=tokenizer_path,
                token_features_path=token_features_path,
                language_pairs=_metadata_language_pairs(metadata),
                int4_backend=int4_backend,
                fp8_policy=fp8_policy,
                release_name=release_name,
                translation_capable=translation_capable,
                _filename_overrides=filename_overrides,
                _acquire_publish_lock=not strict,
            )
            if strict:
                # Conversion is complete. Release the full deployment snapshot
                # before strict validation reloads native artifacts one at a
                # time, otherwise a 32B FP32 state overlaps every validator.
                deployment_state.clear()
                gc.collect()
                failures = {
                    name: entry
                    for name in requested
                    if (entry := manifest["formats"].get(name, {})).get("status") != "ok"
                }
                if failures:
                    summaries = ", ".join(
                        f"{name}={entry.get('error_type', entry.get('status', 'missing'))}: "
                        f"{entry.get('message', 'unknown error')}"
                        for name, entry in failures.items()
                    )
                    raise RuntimeError(f"required model exports failed: {summaries}")
                validation = validate_export_directory(working_directory)
                if not validation["valid"]:
                    raise RuntimeError(
                        "final export validation failed: "
                        + json.dumps(validation["errors"], ensure_ascii=False)
                    )
                # 검증은 방금 쓴 산출물을 mmap 으로 다시 읽습니다. 그 핸들이
                # 살아 있으면 Windows 는 그 파일을 품은 staging 디렉터리의
                # rename 을 거부합니다(POSIX 는 허용하므로 지금까지 드러나지
                # 않았습니다). 위쪽 gc.collect() 는 검증 *이전* 것이라 검증이
                # 새로 만든 핸들은 잡지 못합니다.
                gc.collect()
                _atomic_replace_directory(working_directory, directory)
        except Exception as error:
            export_error = error
            if strict and working_directory.exists():
                _remove_artifact(working_directory)
    if strict:
        if export_error is not None:
            raise export_error
    else:
        _synchronize_rank0_exception(context, export_error)
    del deployment_state
    gc.collect()
    return manifest
