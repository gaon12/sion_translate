"""Conversion helpers for standard Transformers checkpoints."""

# Safetensors and Transformers remote-code APIs have incomplete type metadata.
# pyright: reportArgumentType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from sion_translate.artifacts import (
    MODEL_RELEASE_VERSION,
    TRANSLATION_RELEASE_NAME,
)
from sion_translate.config import ModelConfig
from sion_translate.language_tags import (
    canonicalize_language_pair,
    canonicalize_language_tags,
)
from sion_translate.tokenizer import (
    OPTIONAL_CONTROL_SYMBOLS,
    SHARED_CONTROL_SYMBOLS,
    SionTokenizer as NativeSionTokenizer,
    load_tokenizer_metadata,
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


def _canonical_pair_list(
    values: Sequence[Sequence[str]],
    *,
    field: str,
) -> list[list[str]]:
    pairs: list[list[str]] = []
    seen: set[frozenset[str]] = set()
    for index, value in enumerate(values):
        pair = list(
            canonicalize_language_pair(
                value,
                field=f"{field}[{index}]",
            )
        )
        edge = frozenset(pair)
        if edge in seen:
            raise ValueError(
                f"duplicate or reversed {field} after BCP 47 canonicalization: {value!r}"
            )
        seen.add(edge)
        pairs.append(pair)
    return pairs


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
        "import math\n"
        "import warnings\n"
        "from dataclasses import dataclass, field\n\n"
        f"{config_source[config_start:config_end]}\n"
    )
    runtime_sources = {
        "sion_language_tags.py": (package_dir / "language_tags.py").read_text(encoding="utf-8"),
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
    exported_languages = list(
        canonicalize_language_tags(
            tokenizer_languages if languages is None else list(languages),
            field="export languages",
            reject_duplicates=False,
        )
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
    pairs = _canonical_pair_list(language_pairs or (), field="export language pair")
    pair_languages = {language for pair in pairs for language in pair}
    if pairs and pair_languages != set(tokenizer.languages):
        raise ValueError(
            "translation exports must cover every reserved tokenizer language; "
            "subsetting a tokenizer requires rebuilding its control vocabulary: "
            f"{sorted(pair_languages)} != {sorted(tokenizer.languages)}"
        )
    for pair in pairs:
        if any(language not in tokenizer.language_tags for language in pair):
            raise ValueError(
                f"language pair {pair!r} is not represented by tokenizer tags "
                f"{sorted(tokenizer.language_tags)}"
            )
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
    pairs = _canonical_pair_list(language_pairs, field="export language pair")
    allowed_edges = {frozenset(pair) for pair in pairs}
    if pairs and configured is None:
        raise ValueError(
            "language pairs require authenticated translation_directions; pass them "
            "explicitly or provide tokenizer_metadata.json"
        )
    raw_directions = configured or ()
    directions: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    if pairs and not raw_directions:
        raise ValueError(
            "translation_directions cannot be empty when language pairs are configured"
        )
    for raw_direction in raw_directions:
        if isinstance(raw_direction, (str, bytes)) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            raw_direction, Sequence
        ):
            raise ValueError("each translation direction must be a two-item language sequence")
        direction = list(
            canonicalize_language_pair(
                raw_direction,
                field="export translation direction",
            )
        )
        key = tuple(direction)
        if (
            len(direction) != 2
            or direction[0] == direction[1]
            or frozenset(direction) not in allowed_edges
        ):
            raise ValueError(f"invalid translation direction: {raw_direction!r}")
        if key in seen:
            raise ValueError(
                "duplicate export translation direction after BCP 47 "
                f"canonicalization: {raw_direction!r}"
            )
        seen.add(key)
        directions.append(direction)
    covered_edges = {frozenset(direction) for direction in directions}
    missing_pairs = [pair for pair in pairs if frozenset(pair) not in covered_edges]
    if missing_pairs:
        raise ValueError(
            "every language pair must have at least one translation direction: "
            f"missing={missing_pairs!r}"
        )
    return directions


def _revision_directions(
    translation_directions: Sequence[Sequence[str]],
    configured: Sequence[Sequence[str]] | None,
    revision_trained: bool | None,
) -> list[list[str]] | None:
    allowed = {tuple(direction) for direction in translation_directions}
    if configured is None:
        if revision_trained is True:
            if len(allowed) != 1:
                raise ValueError(
                    "revision_trained=true is ambiguous unless exactly one authenticated "
                    "translation direction exists; pass revision_directions"
                )
            return [list(next(iter(allowed)))]
        if revision_trained is False:
            return []
        return None
    directions: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_direction in configured:
        if isinstance(raw_direction, (str, bytes)) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            raw_direction, Sequence
        ):
            raise ValueError("each revision direction must be a two-item language sequence")
        direction = canonicalize_language_pair(
            raw_direction,
            field="export revision direction",
        )
        if direction in seen:
            raise ValueError(
                "duplicate export revision direction after BCP 47 canonicalization: "
                f"{raw_direction!r}"
            )
        if direction not in allowed:
            raise ValueError(
                "revision_directions must be a subset of authenticated "
                f"translation_directions; got {direction!r}"
            )
        seen.add(direction)
        directions.append(list(direction))
    if revision_trained is not None and revision_trained is not bool(directions):
        raise ValueError("revision_trained disagrees with revision_directions")
    return directions


def _uses_current_capability_contract(release_version: str) -> bool:
    parts = release_version.strip().split(".")
    return (
        len(parts) in {2, 3}
        and all(part.isdigit() for part in parts)
        and tuple(int(part) for part in parts[:2]) >= (1, 5)
    )


def _metadata_language_graph(
    metadata: dict[str, object],
    *,
    source: str,
) -> tuple[list[list[str]] | None, list[list[str]] | None]:
    raw_pairs = metadata.get("language_pairs")
    if raw_pairs is None and metadata.get("language_pair") is not None:
        raw_pairs = [metadata["language_pair"]]
    if raw_pairs is None:
        return None, None
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise ValueError(f"{source} language_pairs must be a sequence")
    pairs = _canonical_pair_list(raw_pairs, field=f"{source} language pair")
    raw_directions = metadata.get("translation_directions")
    if raw_directions is None:
        return pairs, None
    if not isinstance(raw_directions, Sequence) or isinstance(raw_directions, (str, bytes)):
        raise ValueError(f"{source} translation_directions must be a sequence")
    return pairs, _translation_directions(pairs, raw_directions)


def _is_current_tokenizer_contract(metadata: dict[str, object]) -> bool:
    if metadata.get("pipeline") is not None:
        return True
    release_version = metadata.get("release_version")
    if not isinstance(release_version, str) or not release_version:
        return False
    parts = release_version.split(".")
    if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts):
        return False
    return tuple(int(part) for part in parts[:2]) >= (1, 5)


def _load_json_object(path: Path, *, source: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object: {path}")
    return value


def _authenticated_tokenizer_metadata(tokenizer_path: Path) -> dict[str, object] | None:
    """Reconcile native and Transformers tokenizer identity sidecars."""

    actual_sha256 = _file_sha256(tokenizer_path)
    native_metadata = load_tokenizer_metadata(tokenizer_path)
    if native_metadata is not None:
        native_sha256 = native_metadata.get("model_sha256")
        if native_sha256 is not None and (
            not isinstance(native_sha256, str) or native_sha256 != actual_sha256
        ):
            raise ValueError("tokenizer metadata model_sha256 does not match tokenizer.model")

    config_path = tokenizer_path.parent / "tokenizer_config.json"
    if not config_path.is_file():
        return native_metadata
    tokenizer_config = _load_json_object(config_path, source="tokenizer_config.json")
    if not _is_current_tokenizer_contract(tokenizer_config):
        return native_metadata

    config_sha256 = tokenizer_config.get("tokenizer_sha256")
    if not isinstance(config_sha256, str) or config_sha256 != actual_sha256:
        raise ValueError(
            "current tokenizer_config.json must authenticate tokenizer.model with "
            "a matching tokenizer_sha256"
        )
    config_capable = tokenizer_config.get("translation_capable")
    if not isinstance(config_capable, bool):
        raise ValueError("current tokenizer_config.json translation_capable must be a boolean")
    config_pairs, config_directions = _metadata_language_graph(
        tokenizer_config,
        source="tokenizer_config.json",
    )
    if config_capable and (not config_pairs or not config_directions):
        raise ValueError(
            "current translation-capable tokenizer_config.json requires a non-empty "
            "language_pairs and translation_directions graph"
        )
    if not config_capable and (config_pairs or config_directions):
        raise ValueError(
            "translation-incapable tokenizer_config.json cannot advertise language pairs "
            "or directions"
        )

    if native_metadata is not None:
        native_capable = native_metadata.get("translation_capable")
        if native_capable is not None and (
            not isinstance(native_capable, bool) or native_capable != config_capable
        ):
            raise ValueError(
                "tokenizer_metadata.json and tokenizer_config.json disagree on translation_capable"
            )
        native_pairs, native_directions = _metadata_language_graph(
            native_metadata,
            source="tokenizer metadata",
        )
        if native_pairs is not None and {frozenset(pair) for pair in native_pairs} != {
            frozenset(pair) for pair in config_pairs or ()
        }:
            raise ValueError(
                "tokenizer_metadata.json and tokenizer_config.json disagree on language_pairs"
            )
        if native_directions is not None and {
            tuple(direction) for direction in native_directions
        } != {tuple(direction) for direction in config_directions or ()}:
            raise ValueError(
                "tokenizer_metadata.json and tokenizer_config.json disagree on "
                "translation_directions"
            )
    return tokenizer_config


def _tokenizer_metadata_contract(
    tokenizer_path: Path,
    requested_pairs: Sequence[Sequence[str]] | None,
    *,
    translation_capable: bool,
) -> tuple[Sequence[Sequence[str]] | None, Sequence[Sequence[str]] | None]:
    """Return only an explicit, validated graph recorded beside the tokenizer."""

    metadata = _authenticated_tokenizer_metadata(tokenizer_path)
    if metadata is None:
        return requested_pairs, None
    recorded_capability = metadata.get("translation_capable")
    if recorded_capability is not None and (
        not isinstance(recorded_capability, bool) or recorded_capability != translation_capable
    ):
        raise ValueError(
            "requested translation_capable value disagrees with authenticated tokenizer metadata"
        )
    if not translation_capable:
        return requested_pairs, None
    metadata_pairs, metadata_directions = _metadata_language_graph(
        metadata,
        source="tokenizer metadata",
    )
    if metadata_pairs is None:
        return requested_pairs, None
    effective_pairs = metadata_pairs if requested_pairs is None else requested_pairs
    normalized_requested = _canonical_pair_list(
        effective_pairs,
        field="requested language pair",
    )
    metadata_edges = {frozenset(pair) for pair in metadata_pairs}
    missing_pairs = [pair for pair in normalized_requested if frozenset(pair) not in metadata_edges]
    if missing_pairs:
        raise ValueError(
            f"requested language pairs are absent from tokenizer metadata: {missing_pairs!r}"
        )
    if metadata_directions is None:
        return effective_pairs, None
    requested_edges = {frozenset(pair) for pair in normalized_requested}
    selected_directions: list[list[str]] = []
    for direction in metadata_directions:
        if frozenset(direction) in requested_edges:
            selected_directions.append(direction)
    return effective_pairs, selected_directions


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


def _save_transformers_checkpoint_unpublished(
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
    release_name: str = TRANSLATION_RELEASE_NAME,
    release_version: str = MODEL_RELEASE_VERSION,
    translation_capable: bool = True,
    revision_directions: Sequence[Sequence[str]] | None = None,
    revision_trained: bool | None = None,
    pipeline_identity: Mapping[str, Any] | None = None,
    allow_language_subset: bool = False,
    max_shard_size: str = "5GB",
) -> Path:
    """Save native Sion weights as a safe, AutoClass-compatible directory."""

    export_dtype = _state_dict_float_dtype(state_dict)
    output_dir = Path(output_dir)
    bos_id = 2
    eos_id = 3
    tokenizer: NativeSionTokenizer | None = None
    tokenizer_metadata_directions: Sequence[Sequence[str]] | None = None
    tokenizer_sha256: str | None = None
    slot_token_ids: list[int] = []
    if languages is not None:
        languages = list(
            canonicalize_language_tags(
                list(languages),
                field="export languages",
                reject_duplicates=False,
            )
        )
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)
        tokenizer = NativeSionTokenizer(tokenizer_path)
        effective_pairs, tokenizer_metadata_directions = _tokenizer_metadata_contract(
            tokenizer_path,
            language_pairs,
            translation_capable=translation_capable,
        )
        languages, pairs = _validate_language_contract(
            tokenizer,
            model_config=model_config,
            pad_id=pad_id,
            languages=languages,
            language_pairs=effective_pairs,
            allow_language_subset=allow_language_subset,
        )
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        tokenizer_sha256 = _file_sha256(tokenizer_path)
        slot_token_ids = list(tokenizer.slot_ids)
    else:
        pairs = _canonical_pair_list(language_pairs or (), field="export language pair")
        if languages is None:
            languages = list(
                canonicalize_language_tags(
                    [language for pair in pairs for language in pair],
                    field="export pair languages",
                    reject_duplicates=False,
                )
            )
    if tokenizer is not None and translation_capable and not pairs:
        raise ValueError(
            "translation-capable tokenizer exports require a non-empty authenticated "
            "language graph; explicit empty language_pairs cannot erase trained directions"
        )
    directions = _translation_directions(
        pairs,
        (
            translation_directions
            if translation_directions is not None
            else tokenizer_metadata_directions
        ),
    )
    if translation_directions is not None and tokenizer_metadata_directions is not None:
        authenticated_directions = {
            tuple(direction)
            for direction in _translation_directions(pairs, tokenizer_metadata_directions)
        }
        unauthenticated = [
            direction
            for direction in directions
            if tuple(direction) not in authenticated_directions
        ]
        if unauthenticated:
            raise ValueError(
                "requested translation directions are not authenticated by tokenizer "
                f"metadata: {unauthenticated!r}"
            )

    resolved_revision_directions = _revision_directions(
        directions,
        revision_directions,
        revision_trained,
    )
    if resolved_revision_directions is None and _uses_current_capability_contract(release_version):
        # A new current-generation checkpoint may safely underclaim revision
        # support, but it must serialize that decision as an exact empty graph.
        resolved_revision_directions = []

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
    default_reasoning_level = (
        9
        if model_config.experimental.evidence_repair_enabled
        or model_config.experimental.candidate_refinement_enabled
        else 0
    )
    config = SionConfig.from_model_config(
        model_config,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        languages=list(languages or []),
        language_pairs=pairs,
        translation_directions=directions,
        revision_directions=resolved_revision_directions,
        release_name=release_name,
        release_version=release_version,
        translation_capable=translation_capable,
        revision_trained=(
            bool(resolved_revision_directions)
            if resolved_revision_directions is not None
            else revision_trained
        ),
        default_reasoning_level=default_reasoning_level,
        slot_token_ids=slot_token_ids,
        tokenizer_sha256=tokenizer_sha256,
        token_features_sha256=token_features_sha256,
        token_features_shapes=token_features_shapes,
        pipeline=(dict(pipeline_identity) if pipeline_identity is not None else None),
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
    model.generation_config.reasoning_level = (  # pyright: ignore[reportAttributeAccessIssue]
        default_reasoning_level
    )
    model.generation_config.suppress_tokens = _generation_suppress_tokens(
        tokenizer,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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
            release_name=release_name,
            release_version=release_version,
            translation_capable=translation_capable,
            script_classes=model_config.experimental.script_classes,
            tetm_type_id=min(8, model_config.experimental.tetm_types - 1),
            tetm_mode_id=min(4, model_config.experimental.tetm_modes - 1),
        )
        hf_tokenizer.save_pretrained(output_dir)
        tokenizer_metadata = tokenizer_path.parent / "tokenizer_metadata.json"
        if tokenizer_metadata.is_file():
            metadata_destination = output_dir / tokenizer_metadata.name
            tokenizer_metadata_payload = load_tokenizer_metadata(tokenizer_metadata)
            if tokenizer_metadata_payload is None:
                raise RuntimeError("tokenizer metadata disappeared during export")
            tokenizer_metadata_payload = dict(tokenizer_metadata_payload)
            tokenizer_metadata_payload["model_file"] = "tokenizer.model"
            tokenizer_metadata_payload["model_sha256"] = tokenizer_sha256
            tokenizer_metadata_payload["translation_capable"] = translation_capable
            if pairs:
                tokenizer_metadata_payload["language_pairs"] = pairs
                tokenizer_metadata_payload["translation_directions"] = directions
                if len(pairs) == 1:
                    tokenizer_metadata_payload["language_pair"] = pairs[0]
                else:
                    tokenizer_metadata_payload.pop("language_pair", None)
            else:
                tokenizer_metadata_payload.pop("language_pair", None)
                tokenizer_metadata_payload["language_pairs"] = []
                tokenizer_metadata_payload["translation_directions"] = []
            metadata_destination.write_text(
                json.dumps(tokenizer_metadata_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
        if filename == "configuration_sion.py":
            ambient_language_tags = (
                "try:\n"
                "    from sion_translate.language_tags import (\n"
                "        canonicalize_language_pair,\n"
                "        canonicalize_language_tags,\n"
                "    )\n"
                "except ImportError:\n"
                "    from .sion_language_tags import (  # type: ignore[import-not-found]\n"
                "        canonicalize_language_pair,\n"
                "        canonicalize_language_tags,\n"
                "    )\n"
            )
            bundled_language_tags = (
                "from .sion_language_tags import (\n"
                "    canonicalize_language_pair,\n"
                "    canonicalize_language_tags,\n"
                ")\n"
            )
            if ambient_language_tags not in source:
                raise RuntimeError("could not bind configuration_sion.py to bundled language tags")
            source = source.replace(ambient_language_tags, bundled_language_tags)
        elif filename == "tokenization_sion.py":
            ambient_language_tags = (
                "try:\n"
                "    from sion_translate.language_tags import (\n"
                "        LanguageTagError,\n"
                "        canonicalize_language_pair,\n"
                "        canonicalize_language_tag,\n"
                "    )\n"
                "except ImportError:\n"
                "    # Hub remote-code checkpoints bundle this dependency as a sibling module.\n"
                "    from .sion_language_tags import (  # type: ignore[import-not-found]\n"
                "        LanguageTagError,\n"
                "        canonicalize_language_pair,\n"
                "        canonicalize_language_tag,\n"
                "    )\n"
            )
            bundled_language_tags = (
                "from .sion_language_tags import (\n"
                "    LanguageTagError,\n"
                "    canonicalize_language_pair,\n"
                "    canonicalize_language_tag,\n"
                ")\n"
            )
            if ambient_language_tags not in source:
                raise RuntimeError("could not bind tokenization_sion.py to bundled language tags")
            source = source.replace(ambient_language_tags, bundled_language_tags)
        (output_dir / filename).write_text(source, encoding="utf-8")
    runtime_files = _copy_self_contained_runtime(output_dir)
    metadata = {
        "format": "transformers-safetensors-v1",
        "dtype": str(export_dtype).removeprefix("torch."),
        "release_name": config.release_name,
        "release_version": config.release_version,
        "languages": list(languages or []),
        "language_pairs": pairs,
        "translation_directions": directions,
        "translation_capable": translation_capable,
        "generation_defaults": {
            "reasoning_level": default_reasoning_level,
        },
        "capabilities": (
            {
                "revision_directions": resolved_revision_directions,
                "revision_trained": bool(resolved_revision_directions),
            }
            if resolved_revision_directions is not None
            else {}
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
    if pipeline_identity is not None:
        metadata["pipeline"] = dict(pipeline_identity)
    (output_dir / "sion_export.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


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
    release_name: str = TRANSLATION_RELEASE_NAME,
    release_version: str = MODEL_RELEASE_VERSION,
    translation_capable: bool = True,
    revision_directions: Sequence[Sequence[str]] | None = None,
    revision_trained: bool | None = None,
    pipeline_identity: Mapping[str, Any] | None = None,
    allow_language_subset: bool = False,
    max_shard_size: str = "5GB",
    _atomic_publish: bool = True,
) -> Path:
    """Build a complete checkpoint privately, then publish it atomically."""

    destination = Path(output_dir)
    if not _atomic_publish:
        return _save_transformers_checkpoint_unpublished(
            destination,
            state_dict,
            model_config,
            pad_id=pad_id,
            tokenizer_path=tokenizer_path,
            token_features_path=token_features_path,
            languages=languages,
            language_pairs=language_pairs,
            translation_directions=translation_directions,
            release_name=release_name,
            release_version=release_version,
            translation_capable=translation_capable,
            revision_directions=revision_directions,
            revision_trained=revision_trained,
            pipeline_identity=pipeline_identity,
            allow_language_subset=allow_language_subset,
            max_shard_size=max_shard_size,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
        )
    )
    try:
        _save_transformers_checkpoint_unpublished(
            temporary,
            state_dict,
            model_config,
            pad_id=pad_id,
            tokenizer_path=tokenizer_path,
            token_features_path=token_features_path,
            languages=languages,
            language_pairs=language_pairs,
            translation_directions=translation_directions,
            release_name=release_name,
            release_version=release_version,
            translation_capable=translation_capable,
            revision_directions=revision_directions,
            revision_trained=revision_trained,
            pipeline_identity=pipeline_identity,
            allow_language_subset=allow_language_subset,
            max_shard_size=max_shard_size,
        )
        # Import lazily to keep the self-contained HF conversion module free of
        # a module-import cycle. The publisher preserves an existing complete
        # destination if installation fails on Windows or POSIX.
        from sion_translate.training.export import (
            _atomic_replace_directory,  # pyright: ignore[reportPrivateUsage]
        )

        _atomic_replace_directory(  # pyright: ignore[reportPrivateUsage]
            temporary, destination
        )
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
