"""Conversion helpers for standard Transformers checkpoints."""

# Safetensors and Transformers remote-code APIs have incomplete type metadata.
# pyright: reportArgumentType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
import torch

from sion_translate.config import ModelConfig
from sion_translate.tokenizer import (
    OPTIONAL_CONTROL_SYMBOLS,
    SHARED_CONTROL_SYMBOLS,
    SionTokenizer as NativeSionTokenizer,
)

from .configuration_sion import SionConfig
from .modeling_sion import SionForConditionalGeneration
from .tokenization_sion import SionTokenizer

_SUPPORTED_EXPORT_DTYPES = {
    torch.float32,
    torch.float16,
    torch.bfloat16,
}
_TOKEN_FEATURE_NAMES = ("script", "onset", "vowel", "coda")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_feature_metadata(
    path: Path,
    *,
    vocab_size: int,
    script_classes: int,
) -> tuple[str, dict[str, list[int]]]:
    maximum_ids = {
        "script": script_classes,
        "onset": 20,
        "vowel": 22,
        "coda": 29,
    }
    shapes: dict[str, list[int]] = {}
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(_TOKEN_FEATURE_NAMES):
            raise ValueError(
                "token feature file must contain exactly "
                f"{', '.join(_TOKEN_FEATURE_NAMES)}; got {sorted(loaded.files)}"
            )
        for name in _TOKEN_FEATURE_NAMES:
            values = np.asarray(loaded[name])
            if values.shape != (vocab_size,):
                raise ValueError(
                    f"token feature {name} has shape {values.shape}; expected ({vocab_size},)"
                )
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"token feature {name} must use an integer dtype")
            if values.size and (int(values.min()) < 0 or int(values.max()) >= maximum_ids[name]):
                raise ValueError(
                    f"token feature {name} contains IDs outside [0, {maximum_ids[name]})"
                )
            shapes[name] = list(values.shape)
    return _file_sha256(path), shapes


def _copy_self_contained_runtime(output_dir: Path) -> list[str]:
    """Copy the native torch runtime under relative names for Hub remote code."""

    package_dir = Path(__file__).parents[1]
    model_dir = package_dir / "model"
    config_source = (package_dir / "config.py").read_text(encoding="utf-8")
    config_start = config_source.index("@dataclass\nclass ExperimentalConfig")
    config_end = config_source.index("\n\n@dataclass\nclass DataConfig")
    native_config = (
        "from __future__ import annotations\n\n"
        "import warnings\n"
        "from dataclasses import dataclass, field\n\n"
        f"{config_source[config_start:config_end]}\n"
    )
    runtime_sources = {
        "sion_native_config.py": native_config,
        "sion_native_layers.py": (model_dir / "layers.py").read_text(encoding="utf-8"),
        "sion_native_experimental.py": (
            (model_dir / "experimental.py")
            .read_text(encoding="utf-8")
            .replace(
                "from .layers import GQAAttention, RMSNorm",
                "from .sion_native_layers import GQAAttention, RMSNorm",
            )
        ),
        "sion_native_transformer.py": (
            (model_dir / "transformer.py")
            .read_text(encoding="utf-8")
            .replace(
                "from sion_translate.config import ModelConfig",
                "from .sion_native_config import ModelConfig",
            )
            .replace("from .experimental import (", "from .sion_native_experimental import (")
            .replace("from .layers import ", "from .sion_native_layers import ")
        ),
    }
    for filename, source in runtime_sources.items():
        (output_dir / filename).write_text(source, encoding="utf-8")
    return sorted(runtime_sources)


def _validate_language_contract(
    tokenizer: NativeSionTokenizer,
    *,
    model_config: ModelConfig,
    pad_id: int,
    languages: Sequence[str] | None,
    language_pairs: Sequence[Sequence[str]] | None,
    allow_language_subset: bool = False,
) -> tuple[list[str], list[list[str]]]:
    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            "tokenizer vocabulary size does not match model config: "
            f"{len(tokenizer)} != {model_config.vocab_size}"
        )
    if tokenizer.pad_id != pad_id:
        raise ValueError(
            f"tokenizer pad ID does not match export pad_id: {tokenizer.pad_id} != {pad_id}"
        )
    tokenizer_languages = list(
        tokenizer.languages if language_pairs else tokenizer.denoise_languages
    )
    exported_languages = (
        tokenizer_languages if languages is None else list(dict.fromkeys(map(str, languages)))
    )
    if allow_language_subset:
        unknown_languages = sorted(set(exported_languages) - set(tokenizer_languages))
        if unknown_languages:
            raise ValueError(
                "configured languages are not represented by tokenizer language tags: "
                f"{unknown_languages} not in {sorted(tokenizer_languages)}"
            )
    elif set(exported_languages) != set(tokenizer_languages):
        raise ValueError(
            "configured languages do not match tokenizer language tags: "
            f"{sorted(exported_languages)} != {sorted(tokenizer_languages)}"
        )
    pairs: list[list[str]] = []
    for raw_pair in language_pairs or ():
        pair = list(map(str, raw_pair))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"invalid language pair: {raw_pair!r}")
        if any(language not in tokenizer.language_tags for language in pair):
            raise ValueError(
                f"language pair {pair!r} is not represented by tokenizer tags "
                f"{sorted(tokenizer.language_tags)}"
            )
        pairs.append(pair)
    return exported_languages, pairs


def _state_dict_float_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype:
    dtypes = {tensor.dtype for tensor in state_dict.values() if tensor.is_floating_point()}
    if not dtypes:
        raise ValueError("state_dict must contain floating-point model weights")
    if len(dtypes) != 1:
        raise ValueError(f"state_dict must use one floating dtype, got {sorted(map(str, dtypes))}")
    dtype = next(iter(dtypes))
    if dtype not in _SUPPORTED_EXPORT_DTYPES:
        raise ValueError(
            f"Transformers export supports FP32, FP16, or BF16 weights; received {dtype}"
        )
    return dtype


def _translation_directions(
    language_pairs: Sequence[Sequence[str]],
    configured: Sequence[Sequence[str]] | None,
) -> list[list[str]]:
    pairs = [list(map(str, pair)) for pair in language_pairs]
    allowed_edges = {frozenset(pair) for pair in pairs}
    raw_directions = (
        configured
        if configured is not None
        else [direction for pair in pairs for direction in (pair, list(reversed(pair)))]
    )
    directions: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    if pairs and not raw_directions:
        raise ValueError(
            "translation_directions cannot be empty when language pairs are configured"
        )
    for raw_direction in raw_directions:
        direction = list(map(str, raw_direction))
        key = tuple(direction)
        if (
            len(direction) != 2
            or direction[0] == direction[1]
            or frozenset(direction) not in allowed_edges
        ):
            raise ValueError(f"invalid translation direction: {raw_direction!r}")
        if key not in seen:
            seen.add(key)
            directions.append(direction)
    return directions


def _generation_suppress_tokens(
    tokenizer: NativeSionTokenizer | None,
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    """Return source-only control IDs that must not enter decoder output."""

    suppressed = {int(pad_id), int(bos_id)}
    if tokenizer is not None:
        suppressed.add(int(tokenizer.unk_id))
        suppressed.update(map(int, tokenizer.language_tags.values()))
        suppressed.update(map(int, tokenizer.denoise_tags.values()))
        suppressed.update(map(int, tokenizer.reasoning_tags.values()))
        suppressed.update(map(int, tokenizer.reasoning_trace_ids.values()))
        for symbol in (*SHARED_CONTROL_SYMBOLS, *OPTIONAL_CONTROL_SYMBOLS):
            token_id = tokenizer.piece_id(symbol)
            if token_id >= 0 and tokenizer.processor.id_to_piece(token_id) == symbol:
                suppressed.add(token_id)
    # EOS remains the normal completion path, and protected slot tokens must
    # remain generatable so structured values and glossary entries can survive.
    suppressed.discard(int(eos_id))
    if tokenizer is not None:
        suppressed.difference_update(map(int, tokenizer.slot_ids))
    return sorted(suppressed)


def save_transformers_checkpoint(
    output_dir: str | Path,
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
    *,
    pad_id: int = 0,
    tokenizer_path: str | Path | None = None,
    token_features_path: str | Path | None = None,
    languages: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    translation_capable: bool = True,
    revision_trained: bool | None = None,
    allow_language_subset: bool = False,
    max_shard_size: str = "5GB",
) -> Path:
    """Save native Sion weights as a safe, AutoClass-compatible directory."""

    export_dtype = _state_dict_float_dtype(state_dict)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bos_id = 2
    eos_id = 3
    tokenizer: NativeSionTokenizer | None = None
    tokenizer_sha256: str | None = None
    slot_token_ids: list[int] = []
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)
        tokenizer = NativeSionTokenizer(tokenizer_path)
        languages, pairs = _validate_language_contract(
            tokenizer,
            model_config=model_config,
            pad_id=pad_id,
            languages=languages,
            language_pairs=language_pairs,
            allow_language_subset=allow_language_subset,
        )
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        tokenizer_sha256 = _file_sha256(tokenizer_path)
        slot_token_ids = list(tokenizer.slot_ids)
    else:
        pairs = [list(map(str, pair)) for pair in (language_pairs or [])]
    directions = _translation_directions(pairs, translation_directions)

    if token_features_path is None and tokenizer_path is not None:
        sibling_features = tokenizer_path.parent / "token_features.npz"
        if sibling_features.is_file():
            token_features_path = sibling_features
    resolved_features = Path(token_features_path) if token_features_path is not None else None
    token_features_sha256: str | None = None
    token_features_shapes: dict[str, list[int]] = {}
    if resolved_features is not None:
        if not resolved_features.is_file():
            raise FileNotFoundError(f"token feature file does not exist: {resolved_features}")
        token_features_sha256, token_features_shapes = _token_feature_metadata(
            resolved_features,
            vocab_size=model_config.vocab_size,
            script_classes=model_config.experimental.script_classes,
        )
    elif model_config.experimental.morphoscript_enabled:
        raise FileNotFoundError(
            "MorphoScript is enabled, but no token_features.npz was provided or found "
            "next to tokenizer.model"
        )

    SionConfig.register_for_auto_class()
    SionForConditionalGeneration.register_for_auto_class("AutoModelForSeq2SeqLM")
    SionTokenizer.register_for_auto_class()
    config = SionConfig.from_model_config(
        model_config,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        languages=list(languages or []),
        language_pairs=pairs,
        translation_directions=directions,
        translation_capable=translation_capable,
        revision_trained=revision_trained,
        slot_token_ids=slot_token_ids,
        tokenizer_sha256=tokenizer_sha256,
        token_features_sha256=token_features_sha256,
        token_features_shapes=token_features_shapes,
    )
    # Build only metadata tensors, then bind the caller's stable CPU snapshot
    # directly. A normal 32B wrapper would allocate another ~128 GiB of FP32
    # parameters before loading the same state.
    with torch.device("meta"):
        model = SionForConditionalGeneration(config)
    model.model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    model.generation_config.num_beams = 4
    model.generation_config.length_penalty = 1.0
    model.generation_config.max_new_tokens = min(256, model_config.max_seq_len)
    model.generation_config.no_repeat_ngram_size = 4
    model.generation_config.suppress_tokens = _generation_suppress_tokens(
        tokenizer,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    if resolved_features is not None:
        features_destination = output_dir / "token_features.npz"
        if resolved_features.resolve() != features_destination.resolve():
            shutil.copyfile(resolved_features, features_destination)
        original_features_destination = output_dir / resolved_features.name
        if original_features_destination.resolve() != features_destination.resolve():
            shutil.copyfile(resolved_features, original_features_destination)
    if tokenizer_path is not None:
        tokenizer_destination = output_dir / "tokenizer.model"
        if tokenizer_path.resolve() != tokenizer_destination.resolve():
            shutil.copyfile(tokenizer_path, tokenizer_destination)
        # The outer multi-format manifest records the source sidecar filename.
        # Preserve that identity as an alias while keeping the Transformers
        # standard ``tokenizer.model`` filename used by AutoTokenizer.
        original_name_destination = output_dir / tokenizer_path.name
        if original_name_destination.resolve() != tokenizer_destination.resolve():
            shutil.copyfile(tokenizer_path, original_name_destination)
        hf_tokenizer = SionTokenizer(
            str(tokenizer_destination),
            model_max_length=model_config.max_seq_len,
            token_features_file=("token_features.npz" if resolved_features is not None else None),
            token_features_sha256=token_features_sha256,
            tokenizer_sha256=tokenizer_sha256,
            slot_token_ids=slot_token_ids,
            language_pairs=pairs,
            translation_directions=directions,
            translation_capable=translation_capable,
            script_classes=model_config.experimental.script_classes,
            tetm_type_id=min(8, model_config.experimental.tetm_types - 1),
            tetm_mode_id=min(4, model_config.experimental.tetm_modes - 1),
        )
        hf_tokenizer.save_pretrained(output_dir)
        tokenizer_metadata = tokenizer_path.parent / "tokenizer_metadata.json"
        if tokenizer_metadata.is_file():
            metadata_destination = output_dir / tokenizer_metadata.name
            if tokenizer_metadata.resolve() != metadata_destination.resolve():
                shutil.copyfile(tokenizer_metadata, metadata_destination)

    module_dir = Path(__file__).parent
    for filename in (
        "configuration_sion.py",
        "modeling_sion.py",
        "tokenization_sion.py",
    ):
        source = (module_dir / filename).read_text(encoding="utf-8")
        if filename == "configuration_sion.py":
            installed_fallback = (
                "try:\n"
                "    from sion_translate.config import ExperimentalConfig, ModelConfig\n"
                "except ImportError:\n"
                "    # ``save_transformers_checkpoint`` writes this small runtime module next to\n"
                "    # the remote-code files.  Keeping the fallback in a relative import lets a\n"
                "    # Hub checkpoint load without installing the Sion source package.\n"
                "    from importlib import import_module\n\n"
                '    _native_config = import_module(f"{__package__}.sion_native_config")\n'
                "    ExperimentalConfig = _native_config.ExperimentalConfig\n"
                "    ModelConfig = _native_config.ModelConfig\n"
            )
            if installed_fallback not in source:
                raise RuntimeError("could not rewrite configuration_sion.py for remote loading")
            source = source.replace(
                installed_fallback,
                "from .sion_native_config import ExperimentalConfig, ModelConfig\n",
            )
        elif filename == "modeling_sion.py":
            installed_fallback = (
                "try:\n"
                "    from sion_translate.model import (\n"
                "        SionForConditionalGeneration as NativeSionForConditionalGeneration,\n"
                "    )\n"
                "except ImportError:\n"
                "    # Remote checkpoints contain the native runtime under relative module names,\n"
                "    # so only torch/transformers are needed when sion-translate is not installed.\n"
                "    from importlib import import_module\n\n"
                "    NativeSionForConditionalGeneration = import_module(\n"
                '        f"{__package__}.sion_native_transformer"\n'
                "    ).SionForConditionalGeneration\n"
            )
            if installed_fallback not in source:
                raise RuntimeError("could not rewrite modeling_sion.py for remote loading")
            source = source.replace(
                installed_fallback,
                "from .sion_native_transformer import (\n"
                "    SionForConditionalGeneration as NativeSionForConditionalGeneration,\n"
                ")\n",
            )
        (output_dir / filename).write_text(source, encoding="utf-8")
    runtime_files = _copy_self_contained_runtime(output_dir)
    metadata = {
        "format": "transformers-safetensors-v1",
        "dtype": str(export_dtype).removeprefix("torch."),
        "languages": list(languages or []),
        "language_pairs": pairs,
        "translation_directions": directions,
        "translation_capable": translation_capable,
        "capabilities": (
            {"revision_trained": revision_trained} if revision_trained is not None else {}
        ),
        "native_state_dict_prefix": "model.",
        "self_contained_remote_code": True,
        "runtime_files": runtime_files,
        "tokenizer_sha256": tokenizer_sha256,
        "slot_token_ids": slot_token_ids,
        "token_features": (
            {
                "file": "token_features.npz",
                "sha256": token_features_sha256,
                "shapes": token_features_shapes,
            }
            if resolved_features is not None
            else None
        ),
        "transformers_auto_classes": [
            "AutoConfig",
            "AutoModelForSeq2SeqLM",
            "AutoTokenizer",
        ],
        "requires": [
            "numpy>=2.0",
            "safetensors>=0.5",
            "sentencepiece>=0.2",
            "torch>=2.8",
            "transformers>=5,<6",
        ],
    }
    (output_dir / "sion_export.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir
