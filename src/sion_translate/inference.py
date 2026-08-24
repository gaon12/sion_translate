"""Shared inference helpers.

Locate and load trained exports, then translate sequences of sentences in
batches. The interactive translator and backtranslation augmenter share this
runtime.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch

from sion_translate.generation import (
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_OUTPUT_LENGTH_MARGIN,
    DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_NUM_BEAMS,
)
from sion_translate.glossary import Glossary, apply_source_placeholders, restore_targets
from sion_translate.language_tags import canonicalize_language_pair, canonicalize_language_tag
from sion_translate.model import SionForConditionalGeneration
from sion_translate.rerank import select as rerank_select
from sion_translate.revision import DRAFT_SEPARATOR, serialize_revision_input
from sion_translate.structured import mask_structured_spans
from sion_translate.tokenizer import (
    SLOT_SYMBOLS,
    SionTokenizer,
    load_tokenizer_metadata,
    tokenizer_metadata_path,
    tokenizer_split_digits_policy,
)
from sion_translate.fp8_runtime import describe_runtime, prepare_fp8_model_for_device
from sion_translate.training.export import (
    load_exported_model,
    metadata_requires_explicit_direction_graph,
    resolve_manifest_artifact,
)


def _runtime_file_identity(path: Path, *, label: str) -> dict[str, object]:
    """Hash one runtime artifact while detecting replacement during the read."""

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    after = resolved.stat()
    before_stat = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_stat = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_stat != after_stat:
        raise ValueError(f"{label} changed while its load identity was being captured: {resolved}")
    return {
        "path": str(resolved),
        "size": after.st_size,
        "sha256": digest.hexdigest(),
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _optional_runtime_file_identity(path: Path, *, label: str) -> dict[str, object] | None:
    return _runtime_file_identity(path, label=label) if path.is_file() else None


def _verified_load_identity(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    *,
    label: str,
) -> dict[str, object] | None:
    """Require identical bytes and stat identity before and after an artifact load."""

    if before != after:
        raise ValueError(f"{label} changed while Translator was loading it")
    return dict(before) if before is not None else None


def _language_pairs_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    raw_pairs: object = metadata.get("language_pairs")
    if raw_pairs is None and metadata.get("language_pair") is not None:
        raw_pairs = [metadata["language_pair"]]
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise ValueError("language_pairs metadata must be a sequence")
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for raw_pair in cast(Sequence[object], raw_pairs):
        if not isinstance(raw_pair, Sequence) or isinstance(raw_pair, (str, bytes)):
            raise ValueError(f"invalid language pair metadata: {raw_pair!r}")
        pair_items = cast(Sequence[object], raw_pair)
        if len(pair_items) != 2:
            raise ValueError(f"invalid language pair metadata: {raw_pair!r}")
        pair = canonicalize_language_pair(
            pair_items,
            field="language pair metadata",
        )
        edge = frozenset(pair)
        if not all(pair) or len(edge) != 2:
            raise ValueError(f"invalid language pair metadata: {raw_pair!r}")
        if edge in seen:
            raise ValueError(f"duplicate or reversed language pair metadata: {raw_pair!r}")
        seen.add(edge)
        pairs.append(pair)
    return tuple(pairs)


def _translation_directions_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    pairs = _language_pairs_from_metadata(metadata)
    raw_directions: object = metadata.get("translation_directions")
    if raw_directions is None:
        if pairs and metadata_requires_explicit_direction_graph(metadata):
            raise ValueError(
                "current translation metadata with language pairs requires explicit "
                "translation_directions"
            )
        return ()
    if not isinstance(raw_directions, Sequence) or isinstance(
        raw_directions,
        (str, bytes),
    ):
        raise ValueError("translation_directions metadata must be a sequence")
    if pairs and not raw_directions:
        raise ValueError(
            "translation_directions metadata cannot be empty when language pairs are configured"
        )
    allowed_edges = {frozenset(pair) for pair in pairs}
    directions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_direction in cast(Sequence[object], raw_directions):
        if not isinstance(raw_direction, Sequence) or isinstance(raw_direction, (str, bytes)):
            raise ValueError(f"invalid translation direction metadata: {raw_direction!r}")
        direction_items = cast(Sequence[object], raw_direction)
        if len(direction_items) != 2:
            raise ValueError(f"invalid translation direction metadata: {raw_direction!r}")
        direction = canonicalize_language_pair(
            direction_items,
            field="translation direction metadata",
        )
        if direction[0] == direction[1] or frozenset(direction) not in allowed_edges:
            raise ValueError(f"invalid translation direction metadata: {raw_direction!r}")
        if direction not in seen:
            seen.add(direction)
            directions.append(direction)
    covered_edges = {frozenset(direction) for direction in directions}
    missing_pairs = [pair for pair in pairs if frozenset(pair) not in covered_edges]
    if missing_pairs:
        raise ValueError(
            "translation direction metadata must cover every language pair: "
            f"missing={missing_pairs!r}"
        )
    return tuple(directions)


def _revision_directions_from_metadata(
    metadata: Mapping[str, Any] | None,
    translation_directions: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...] | None:
    """Return authoritative revision edges, preserving legacy uncertainty."""

    if metadata is None:
        return None
    capabilities = metadata.get("capabilities")
    if capabilities is None:
        return None
    if not isinstance(capabilities, Mapping):
        raise ValueError("model capabilities metadata must be an object")
    typed_capabilities = cast(Mapping[object, object], capabilities)
    has_summary = "revision_trained" in typed_capabilities
    summary: object = typed_capabilities.get("revision_trained")
    if has_summary and not isinstance(summary, bool):
        raise ValueError("model capabilities.revision_trained must be a boolean when present")
    raw_directions: object = typed_capabilities.get("revision_directions")
    if "revision_directions" not in typed_capabilities:
        if summary is False:
            return ()
        if summary is True and len(translation_directions) == 1:
            return (translation_directions[0],)
        return None
    if not isinstance(raw_directions, Sequence) or isinstance(raw_directions, (str, bytes)):
        raise ValueError("model capabilities.revision_directions must be a sequence")
    allowed = set(translation_directions)
    directions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_direction in cast(Sequence[object], raw_directions):
        if not isinstance(raw_direction, Sequence) or isinstance(raw_direction, (str, bytes)):
            raise ValueError(f"invalid revision direction metadata: {raw_direction!r}")
        direction = canonicalize_language_pair(
            cast(Sequence[object], raw_direction),
            field="revision direction metadata",
        )
        if direction in seen:
            raise ValueError(f"duplicate revision direction metadata: {raw_direction!r}")
        if direction not in allowed:
            raise ValueError(
                "revision direction metadata must be a subset of authenticated translation "
                f"directions: {direction!r}"
            )
        seen.add(direction)
        directions.append(direction)
    if has_summary and summary is not bool(directions):
        raise ValueError("model capabilities.revision_trained disagrees with revision_directions")
    return tuple(directions)


def _manifest_artifact(directory: Path, *, int8: bool) -> Path | None:
    format_names = ("int8",) if int8 else ("fp32", "bf16", "fp16")
    return resolve_manifest_artifact(directory, format_names)


def find_exported_model(
    output_dir: str | Path,
    *,
    int8: bool = False,
) -> Path:
    """Find the best available exported model.

    Search post-training before pre-training, best before latest, and then the
    legacy single-stage directory. EMA weights take priority because they
    usually translate better. ``int8=True`` selects a quantized artifact.
    """
    output_dir = Path(output_dir)
    filenames = ["model_int8.pt"] if int8 else ["model_ema.pt", "model.pt"]
    export_roots = [
        output_dir / "posttrain" / "exports",
        output_dir / "pretrain" / "exports",
        output_dir / "exports",  # Compatibility with legacy single-stage exports.
    ]
    for exports in export_roots:
        for stage in ("best", "latest"):
            directory = exports / stage
            manifested = _manifest_artifact(directory, int8=int8)
            if manifested is not None:
                return manifested
            # A v2 manifest is authoritative. If it has no successful requested
            # format, do not silently select a stale file that it did not verify.
            if (directory / "export_manifest.json").exists():
                continue
            for filename in filenames:
                candidate = directory / filename
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(
        f"No exported model exists below {output_dir}. Train with sion-train first."
    )


class Translator:
    """Translate text with one exported model and its tokenizer."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        *,
        device: str | torch.device | None = None,
        token_features_path: str | Path | None = None,
    ):
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        tokenizer_sidecar_path = tokenizer_metadata_path(tokenizer_path)
        model_identity_before = _optional_runtime_file_identity(
            model_path,
            label="translation model",
        )
        tokenizer_identity_before = _runtime_file_identity(
            tokenizer_path,
            label="tokenizer model",
        )
        tokenizer_metadata_identity_before = _optional_runtime_file_identity(
            tokenizer_sidecar_path,
            label="tokenizer metadata",
        )
        # Queue translation verifies these load-time identities again at its
        # public library boundary. Preserve the exact paths used here instead
        # of trusting caller-supplied provenance paths.
        self.translation_model_path = str(model_path.resolve())
        self.tokenizer_model_path = str(tokenizer_path.resolve())
        self.tokenizer = SionTokenizer(tokenizer_path)
        declared_split_digits = tokenizer_split_digits_policy(tokenizer_path)
        if declared_split_digits is False or (
            declared_split_digits is None and not self.tokenizer.splits_digits
        ):
            # A tokenizer trained without split_digits can memorize numbers as
            # opaque pieces and silently change amounts, doses, or dates. Warn at
            # load time because this corruption is hard to notice in fluent text.
            warnings.warn(
                f"{tokenizer_path} does not split numbers into digits. Amounts, doses, and "
                "dates can change silently, so review number-sensitive output. Enable "
                "split_digits when retraining the tokenizer.",
                RuntimeWarning,
                stacklevel=2,
            )
        loaded = load_exported_model(model_path, return_metadata=True)
        if len(loaded) == 3:
            self.model, self.model_config, self.pad_id = cast(
                tuple[SionForConditionalGeneration, Any, int], loaded
            )
            self.export_metadata: dict[str, Any] = {}
        else:
            self.model, self.model_config, self.pad_id, raw_metadata = cast(
                tuple[SionForConditionalGeneration, Any, int, object], loaded
            )
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("model export metadata must be an object")
            self.export_metadata = cast(
                dict[str, Any], dict(cast(Mapping[object, object], raw_metadata))
            )
        # A foundation model shares the architecture and accepts direction tags
        # despite never seeing parallel supervision. Require an authenticated
        # translation capability instead of producing plausible-looking noise.
        translation_capable = self.export_metadata.get("translation_capable")
        if not isinstance(translation_capable, bool):
            raise ValueError(
                "model export metadata.translation_capable must be an explicit boolean; "
                "tokenizer metadata cannot establish that native weights were trained "
                "for translation"
            )
        if not translation_capable:
            release = self.export_metadata.get("release_name", "unknown")
            raise ValueError(
                f"This export is not a translation model (release_name={release!r}). "
                "A foundation artifact trained only on monolingual reconstruction cannot "
                "translate. Select an export from the translation stage under "
                "runs/*/pretrain or runs/*/posttrain."
            )
        self.tokenizer_metadata = cast(
            dict[str, Any] | None, load_tokenizer_metadata(tokenizer_path)
        )
        self.language_pairs = _language_pairs_from_metadata(self.export_metadata)
        if not self.language_pairs:
            self.language_pairs = _language_pairs_from_metadata(self.tokenizer_metadata)
        if not self.language_pairs and len(self.tokenizer.languages) == 2:
            self.language_pairs = ((self.tokenizer.languages[0], self.tokenizer.languages[1]),)
        if not self.language_pairs and len(self.tokenizer.languages) > 2:
            raise ValueError(
                "A multilingual translation model requires language_pairs metadata. "
                "The runtime cannot infer that every language combination was trained "
                "when the exact direction graph is unknown."
            )
        self.translation_directions = _translation_directions_from_metadata(self.export_metadata)
        if not self.translation_directions:
            raise ValueError(
                "Model export metadata does not contain exact translation_directions. "
                "Tokenizer-side language_pairs or translation_directions cannot establish "
                "which directions the native weights learned. Re-export with explicit "
                "--bidirectional, --unidirectional, or --translation-direction options."
            )
        self._translation_direction_edges: set[tuple[str, str]] = set(self.translation_directions)
        self.revision_directions = _revision_directions_from_metadata(
            self.export_metadata,
            self.translation_directions,
        )
        self._revision_direction_edges = (
            set(self.revision_directions) if self.revision_directions is not None else None
        )
        self._validate_compatibility(tokenizer_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # CPU-kernel quantized models must remain on the CPU.
        quantization = self.export_metadata.get("quantization")
        quantization_mapping = (
            cast(Mapping[object, object], quantization)
            if isinstance(quantization, Mapping)
            else None
        )
        runtime_device = (
            quantization_mapping.get("runtime_device") if quantization_mapping is not None else None
        )
        self.quantized = runtime_device == "cpu" or any(
            "quantized" in type(module).__module__ for module in self.model.modules()
        )
        # FP8 exports are not CPU-only. The current runtime immediately
        # dequantizes weights to BF16, or FP16 on unsupported CUDA devices, and
        # uses dense GEMM. Record that runtime choice separately from storage.
        fp8_export = (
            quantization_mapping is not None and quantization_mapping.get("format") == "fp8"
        )
        self.fp8_runtime: str | None = describe_runtime(self.device) if fp8_export else None
        if not self.quantized:
            if fp8_export:
                prepare_fp8_model_for_device(self.model, self.device)
            else:
                self.model.to(self.device)
        else:
            if self.device.type != "cpu":
                warnings.warn(
                    "This quantized export is CPU-only; using CPU instead of the requested "
                    "CUDA device.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.device = torch.device("cpu")
        self.model.eval()

        feature_path = self._resolve_token_features_path(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            explicit_path=token_features_path,
        )
        token_features_identity_before = _optional_runtime_file_identity(
            feature_path,
            label="token features",
        )
        self.token_features_path = (
            str(feature_path.resolve(strict=True)) if feature_path.is_file() else None
        )
        self.tokenizer_metadata_path = (
            str(tokenizer_sidecar_path.resolve(strict=True))
            if tokenizer_sidecar_path.is_file()
            else None
        )
        self.token_features = self._load_token_features(
            feature_path,
            required=self.model_config.experimental.morphoscript_enabled,
            explicit=token_features_path is not None,
            expected_identity=self.export_metadata.get("token_features"),
        )
        self.translation_model_identity = _verified_load_identity(
            model_identity_before,
            _optional_runtime_file_identity(model_path, label="translation model"),
            label="translation model",
        )
        self.tokenizer_model_identity = _verified_load_identity(
            tokenizer_identity_before,
            _runtime_file_identity(tokenizer_path, label="tokenizer model"),
            label="tokenizer model",
        )
        self.tokenizer_metadata_identity = _verified_load_identity(
            tokenizer_metadata_identity_before,
            _optional_runtime_file_identity(
                tokenizer_sidecar_path,
                label="tokenizer metadata",
            ),
            label="tokenizer metadata",
        )
        self.token_features_identity = _verified_load_identity(
            token_features_identity_before,
            _optional_runtime_file_identity(feature_path, label="token features"),
            label="token features",
        )

    def _resolve_token_features_path(
        self,
        *,
        model_path: Path,
        tokenizer_path: Path,
        explicit_path: str | Path | None,
    ) -> Path:
        if explicit_path is not None:
            return Path(explicit_path)

        filenames: list[str] = []
        export_identity = self.export_metadata.get("token_features")
        if isinstance(export_identity, Mapping):
            filename = cast(Mapping[object, object], export_identity).get("filename")
            if isinstance(filename, str) and filename and Path(filename).name == filename:
                filenames.append(filename)
        if self.tokenizer_metadata is not None:
            filename = self.tokenizer_metadata.get("token_features_file")
            if (
                isinstance(filename, str)
                and filename
                and Path(filename).name == filename
                and filename not in filenames
            ):
                filenames.append(filename)
        if "token_features.npz" not in filenames:
            filenames.append("token_features.npz")

        candidates = [
            parent / filename
            for filename in filenames
            for parent in (model_path.parent, tokenizer_path.parent)
        ]
        return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    def _validate_compatibility(self, tokenizer_path: Path) -> None:
        tokenizer_vocab = len(self.tokenizer)
        if tokenizer_vocab != self.model_config.vocab_size:
            raise ValueError(
                "tokenizer vocab does not match model config: "
                f"{tokenizer_vocab} != {self.model_config.vocab_size}"
            )
        if self.tokenizer.pad_id != self.pad_id:
            raise ValueError(
                "tokenizer pad ID does not match model export: "
                f"{self.tokenizer.pad_id} != {self.pad_id}"
            )

        tokenizer_identity = self.export_metadata.get("tokenizer")
        if isinstance(tokenizer_identity, Mapping):
            identity = cast(Mapping[object, object], tokenizer_identity)
            expected_size = identity.get("size")
            if isinstance(expected_size, int) and tokenizer_path.stat().st_size != expected_size:
                raise ValueError("tokenizer metadata size does not match the selected tokenizer")
            expected_sha256 = identity.get("sha256")
            if isinstance(expected_sha256, str):
                digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
                if digest != expected_sha256:
                    raise ValueError(
                        "tokenizer metadata SHA256 does not match the selected tokenizer"
                    )

        expected_languages = {language for pair in self.language_pairs for language in pair}
        if expected_languages and expected_languages != set(self.tokenizer.languages):
            raise ValueError(
                "tokenizer languages do not match model metadata: "
                f"{sorted(self.tokenizer.languages)} != {sorted(expected_languages)}"
            )

        feature_flags = self.export_metadata.get("feature_flags")
        if isinstance(feature_flags, Mapping):
            typed_feature_flags = cast(Mapping[object, object], feature_flags)
            experimental = self.model_config.experimental
            expected_flags = {
                "bats": bool(experimental.bats_enabled),
                "core": bool(experimental.core_enabled),
                "tetm": bool(experimental.tetm_enabled),
                "morphoscript": bool(experimental.morphoscript_enabled),
                "evidence_repair": bool(experimental.evidence_repair_enabled),
                "candidate_refinement": bool(experimental.candidate_refinement_enabled),
                "semantic_parity": bool(experimental.semantic_parity_enabled),
                "situglu": bool(experimental.situglu_enabled),
                "recurrent_block": bool(experimental.recurrent_block_layers),
            }
            mismatches = {
                name: (bool(typed_feature_flags[name]), enabled)
                for name, enabled in expected_flags.items()
                if name in typed_feature_flags and bool(typed_feature_flags[name]) != enabled
            }
            if mismatches:
                raise ValueError(
                    f"model feature metadata does not match model config: {mismatches}"
                )

        capabilities = self.export_metadata.get("capabilities")
        if capabilities is not None:
            if not isinstance(capabilities, Mapping):
                raise ValueError("model capabilities metadata must be an object")
            if "revision_trained" in capabilities and not isinstance(
                capabilities["revision_trained"], bool
            ):
                raise ValueError(
                    "model capabilities.revision_trained must be a boolean when present"
                )
            _revision_directions_from_metadata(
                self.export_metadata,
                self.translation_directions,
            )

        tokenizer_metadata = self.tokenizer_metadata
        if tokenizer_metadata is None:
            return
        metadata_vocab = tokenizer_metadata.get("vocab_size")
        if isinstance(metadata_vocab, int) and metadata_vocab != tokenizer_vocab:
            raise ValueError(
                "tokenizer metadata vocab does not match tokenizer model: "
                f"{metadata_vocab} != {tokenizer_vocab}"
            )
        metadata_sha256 = tokenizer_metadata.get("model_sha256")
        if isinstance(metadata_sha256, str):
            digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            if digest != metadata_sha256:
                raise ValueError("tokenizer sidecar model identity does not match tokenizer.model")
        sidecar_pairs = _language_pairs_from_metadata(tokenizer_metadata)
        metadata_languages = {language for pair in sidecar_pairs for language in pair}
        if metadata_languages and metadata_languages != set(self.tokenizer.languages):
            raise ValueError(
                "tokenizer sidecar languages do not match tokenizer.model: "
                f"{sorted(metadata_languages)} != {sorted(self.tokenizer.languages)}"
            )
        export_pairs = _language_pairs_from_metadata(self.export_metadata)
        if (
            sidecar_pairs
            and export_pairs
            and {frozenset(pair) for pair in sidecar_pairs}
            != {frozenset(pair) for pair in export_pairs}
        ):
            raise ValueError(
                "tokenizer sidecar language pairs do not match model metadata: "
                f"{sidecar_pairs} != {export_pairs}"
            )
        sidecar_directions = _translation_directions_from_metadata(tokenizer_metadata)
        export_directions = _translation_directions_from_metadata(self.export_metadata)
        if (
            sidecar_directions
            and export_directions
            and set(sidecar_directions) != set(export_directions)
        ):
            raise ValueError(
                "tokenizer sidecar translation directions do not match model metadata: "
                f"{sidecar_directions} != {export_directions}"
            )

    def _load_token_features(
        self,
        path: Path,
        *,
        required: bool,
        explicit: bool,
        expected_identity: object,
    ) -> dict[str, torch.Tensor] | None:
        if not required and not explicit:
            return None
        if not path.is_file():
            if required or explicit:
                raise FileNotFoundError(
                    f"MorphoScript token feature file does not exist: {path}" if required else path
                )
            return None
        identity = (
            cast(Mapping[object, object], expected_identity)
            if isinstance(expected_identity, Mapping)
            else None
        )
        if identity is None and self.tokenizer_metadata is not None:
            expected_sha = self.tokenizer_metadata.get("token_features_sha256")
            if isinstance(expected_sha, str):
                identity = {
                    "filename": self.tokenizer_metadata.get("token_features_file"),
                    "size": self.tokenizer_metadata.get("token_features_size"),
                    "sha256": expected_sha,
                }
        if identity is not None:
            expected_filename = identity.get("filename")
            if isinstance(expected_filename, str) and path.name != expected_filename:
                raise ValueError(
                    "token feature filename does not match model/tokenizer metadata: "
                    f"{path.name} != {expected_filename}"
                )
            expected_size = identity.get("size")
            if isinstance(expected_size, int) and path.stat().st_size != expected_size:
                raise ValueError("token feature size does not match model/tokenizer metadata")
            expected_sha256 = identity.get("sha256")
            if isinstance(expected_sha256, str):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != expected_sha256:
                    raise ValueError("token feature SHA256 does not match model/tokenizer metadata")
        expected_length = len(self.tokenizer)
        features: dict[str, torch.Tensor] = {}
        maximum_ids = {
            "script": self.model_config.experimental.script_classes,
            "onset": 20,
            "vowel": 22,
            "coda": 29,
        }
        with np.load(path, allow_pickle=False) as loaded:
            required_names = {"script", "onset", "vowel", "coda"}
            if set(loaded.files) != required_names:
                raise ValueError(
                    "token feature file must contain exactly "
                    f"{', '.join(sorted(required_names))}; got {sorted(loaded.files)}"
                )
            for name in ("script", "onset", "vowel", "coda"):
                values = np.asarray(loaded[name])
                if values.ndim != 1 or len(values) != expected_length:
                    raise ValueError(
                        f"token feature {name} has shape {values.shape}; "
                        f"expected ({expected_length},)"
                    )
                if not np.issubdtype(values.dtype, np.integer):
                    raise ValueError(f"token feature {name} must use an integer dtype")
                values = values.astype(np.int64, copy=True)
                if values.size and (
                    int(values.min()) < 0 or int(values.max()) >= maximum_ids[name]
                ):
                    raise ValueError(
                        f"token feature {name} contains IDs outside [0, {maximum_ids[name]})"
                    )
                features[name] = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                    values
                )
        return features

    def _generation_features(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        if self.token_features is not None:
            for source_name, target_name in (
                ("script", "src_script_ids"),
                ("onset", "src_onset_ids"),
                ("vowel", "src_vowel_ids"),
                ("coda", "src_coda_ids"),
            ):
                features[target_name] = self.token_features[source_name][input_ids].to(self.device)

        if self.model_config.experimental.tetm_enabled:
            slot_ids = set(self.tokenizer.slot_ids)
            rows = [
                [int(token_id) for token_id in row if int(token_id) in slot_ids][:64]
                for row in input_ids
            ]
            memory_length = max(1, max(map(len, rows), default=0))
            memory_token_ids = torch.full(
                (len(rows), memory_length, 1),
                self.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            memory_mask = torch.zeros(
                (len(rows), memory_length),
                dtype=torch.bool,
                device=self.device,
            )
            memory_type_ids = torch.zeros_like(memory_mask, dtype=torch.long)
            memory_mode_ids = torch.zeros_like(memory_mask, dtype=torch.long)
            for row_index, row in enumerate(rows):
                if not row:
                    continue
                length = len(row)
                memory_token_ids[row_index, :length, 0] = torch.tensor(
                    row, dtype=torch.long, device=self.device
                )
                memory_mask[row_index, :length] = True
                memory_type_ids[row_index, :length] = min(
                    8, self.model_config.experimental.tetm_types - 1
                )
                memory_mode_ids[row_index, :length] = min(
                    4, self.model_config.experimental.tetm_modes - 1
                )
            features.update(
                memory_token_ids=memory_token_ids,
                memory_mask=memory_mask,
                memory_type_ids=memory_type_ids,
                memory_mode_ids=memory_mode_ids,
            )
        return features

    @property
    def languages(self) -> tuple[str, ...]:
        """Return languages advertised by the tokenizer's target tags."""
        return self.tokenizer.languages

    def _other_language(self, target_language: str) -> str:
        """Infer the non-target source only when exactly two languages exist."""
        others = [lang for lang in self.languages if lang != target_language]
        return others[0] if len(others) == 1 else ""

    def _resolve_source_language(
        self,
        source_language: str | None,
        target_language: str,
    ) -> str:
        target_language = canonicalize_language_tag(target_language, field="target_language")
        source_language = (
            canonicalize_language_tag(source_language, field="source_language")
            if source_language is not None
            else None
        )
        if source_language is None:
            source_language = self._other_language(target_language)
            if not source_language:
                raise ValueError(
                    "Multilingual models require an explicit source_language "
                    f"(supported: {sorted(self.languages)})"
                )
        if source_language not in self.languages:
            raise ValueError(
                f"Unsupported source language: {source_language} "
                f"(supported: {sorted(self.languages)})"
            )
        if source_language == target_language:
            raise ValueError("source_language and target_language must be different")
        empty_directions: set[tuple[str, str]] = set()
        direction_edges = cast(
            set[tuple[str, str]],
            getattr(self, "_translation_direction_edges", empty_directions),
        )
        if direction_edges and (source_language, target_language) not in direction_edges:
            translation_directions = getattr(self, "translation_directions", ())
            supported = ", ".join(f"{source}→{target}" for source, target in translation_directions)
            raise ValueError(
                f"Untrained translation direction: {source_language}→{target_language} "
                f"(supported directions: {supported})"
            )
        return source_language

    @torch.no_grad()
    def _translate_internal(
        self,
        texts: Sequence[str],
        *,
        source_language: str | None = None,
        target_language: str,
        num_beams: int = DEFAULT_NUM_BEAMS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        max_new_tokens: int = 256,
        batch_size: int = 16,
        glossary: Glossary | None = None,
        append_missing_glossary: bool = True,
        append_missing_structured: bool = True,
        num_candidates: int = 0,
        rerank: str = "mbr+qe",
        temperature: float = 0.3,
        top_k: int = 0,
        seed: int | None = None,
        sampling_seed: int | None = None,
        generator: torch.Generator | None = None,
        return_rerank_details: bool = False,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_output_length_ratio: float | None = DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
        max_output_length_margin: int = DEFAULT_MAX_OUTPUT_LENGTH_MARGIN,
        reasoning_level: int | None = None,
    ) -> list[str]:
        """Translate a sequence of sentences into ``target_language``.

        ``source_language`` may be omitted only when exactly one non-target
        language is possible. The target control tag follows the same contract
        used during training.

        A ``glossary`` replaces source terms with protected slots and restores
        their required target forms after generation. If the model drops a
        glossary slot, ``append_missing_glossary`` appends the missing term.

        Numbers, units, URLs, and localization placeholders use the same slot
        protection path and retain their exact source surface forms.
        ``append_missing_structured`` preserves a value even when generation
        drops its slot.

        A positive ``num_candidates`` adds stochastic candidates to the beam
        result and selects one with ``rerank``. The beam result remains first,
        so stable tie-breaking preserves the original behavior.

        ``return_rerank_details`` returns ``RerankResult`` objects instead of
        plain strings for selection diagnostics.

        ``reasoning_level`` controls optional evidence repair and distribution
        refinement. Zero bypasses both paths, 1-9 selects configured refinement
        endpoints monotonically, and ``None`` uses the checkpoint default.

        Generation forbids training-only control tokens and blocks repeated
        n-grams of ``no_repeat_ngram_size``. ``max_output_length_ratio`` stops
        only outputs that are implausibly long relative to their sources.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_new_tokens <= 0 or max_new_tokens > self.model_config.max_seq_len:
            raise ValueError(
                "max_new_tokens must be positive and no larger than model max_seq_len "
                f"({self.model_config.max_seq_len})"
            )
        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if length_penalty <= 0:
            raise ValueError("length_penalty must be positive")
        if num_candidates < 0:
            raise ValueError("num_candidates must be non-negative")
        if return_rerank_details and num_candidates < 1:
            raise ValueError("return_rerank_details requires num_candidates to be positive")
        if num_candidates and temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        if max_output_length_ratio is not None and max_output_length_ratio <= 0:
            raise ValueError("max_output_length_ratio must be positive or None")
        if max_output_length_margin < 0:
            raise ValueError("max_output_length_margin must be non-negative")
        if reasoning_level is not None:
            if type(reasoning_level) is not int:
                raise TypeError("reasoning_level must be an integer from 0 to 9 or None")
            if not 0 <= reasoning_level <= 9:
                raise ValueError("reasoning_level must be between 0 and 9")
        else:
            generation_defaults = self.export_metadata.get("generation_defaults")
            if isinstance(generation_defaults, Mapping):
                typed_generation_defaults = cast(Mapping[object, object], generation_defaults)
                stored_level: object = typed_generation_defaults.get("reasoning_level")
                if isinstance(stored_level, bool) or not isinstance(stored_level, int):
                    raise ValueError(
                        "model generation_defaults.reasoning_level must be an integer from 0 to 9"
                    )
                if not 0 <= stored_level <= 9:
                    raise ValueError(
                        "model generation_defaults.reasoning_level must be between 0 and 9"
                    )
                reasoning_level = stored_level
        if seed is not None and sampling_seed is not None:
            raise ValueError("seed and sampling_seed are aliases; provide only one")
        resolved_seed = seed if seed is not None else sampling_seed
        if resolved_seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if resolved_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(resolved_seed))
        target_language = canonicalize_language_tag(target_language, field="target_language")
        source_language = (
            canonicalize_language_tag(source_language, field="source_language")
            if source_language is not None
            else None
        )
        tag_id = self.tokenizer.language_tags.get(target_language)
        if tag_id is None:
            raise ValueError(
                f"Unsupported target language: {target_language} "
                f"(supported: {sorted(self.languages)})"
            )
        source_language = self._resolve_source_language(
            source_language,
            target_language,
        )
        eos = self.tokenizer.eos_id
        results: list[Any] = []
        special_ids = {
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            eos,
            self.tokenizer.mask_id,
            *self.tokenizer.language_tags.values(),
            *self.tokenizer.denoise_tags.values(),
            *getattr(self.tokenizer, "reasoning_tags", {}).values(),
            *getattr(self.tokenizer, "reasoning_trace_ids", {}).values(),
        }
        if self.tokenizer.draft_id is not None:
            special_ids.add(self.tokenizer.draft_id)
        forbidden_token_ids = tuple(sorted(special_ids - {eos}))

        def restore(
            row: Sequence[int],
            structured_map: dict[str, str] | None,
            glossary_map: dict[str, str] | None,
        ) -> str:
            """Decode generated tokens and restore protected slots."""
            tokens = [token for token in row if token not in special_ids]
            text = self.tokenizer.decode(tokens)
            if structured_map:
                text, missing = restore_targets(text, structured_map)
                if missing and append_missing_structured:
                    text = f"{text} ({', '.join(missing)})"
            if glossary_map:
                text, missing = restore_targets(text, glossary_map)
                if missing and append_missing_glossary:
                    # Append required terms when generation drops their slots.
                    text = f"{text} ({', '.join(missing)})"
            return text

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            # Quality estimation compares against the unmasked source.
            sources = list(chunk)
            # Protect glossary terms and retain one restoration map per row.
            structured_maps: list[dict[str, str]] = []
            glossary_maps: list[dict[str, str]] = []
            prepared: list[str] = []
            for text in chunk:
                masked, structured_map = mask_structured_spans(
                    text,
                    slot_symbols=SLOT_SYMBOLS,
                )
                structured_maps.append(structured_map)
                glossary_map: dict[str, str] = {}
                if glossary is not None and source_language:
                    remaining_slots = SLOT_SYMBOLS[len(structured_map) :]
                    masked, slot_map = apply_source_placeholders(
                        masked,
                        glossary,
                        source_language=source_language,
                        target_language=target_language,
                        slot_symbols=remaining_slots,
                    )
                    glossary_map = slot_map
                glossary_maps.append(glossary_map)
                prepared.append(masked)
            chunk = prepared
            encoded = [[tag_id, *self.tokenizer.encode(text), eos] for text in chunk]
            longest = max(len(ids) for ids in encoded)
            if longest > self.model_config.max_seq_len:
                raise ValueError(
                    "encoded source length exceeds model max_seq_len: "
                    f"{longest} > {self.model_config.max_seq_len}"
                )
            input_ids = torch.full((len(encoded), longest), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), longest), dtype=torch.bool)
            for row, ids in enumerate(encoded):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = True
            device_inputs = input_ids.to(self.device)
            device_mask = attention_mask.to(self.device)
            generation_features = self._generation_features(input_ids)
            generation_context = (
                self.model.prepare_generation(
                    device_inputs,
                    device_mask,
                    reasoning_level=reasoning_level,
                    **generation_features,
                )
                if num_candidates > 0
                else None
            )
            chunk_max_new_tokens = max_new_tokens
            row_max_new_tokens: torch.Tensor | None = None
            if max_output_length_ratio is not None:
                row_limits = [
                    min(
                        max_new_tokens,
                        max(
                            min_new_tokens + 1,
                            math.ceil((len(ids) - 2) * max_output_length_ratio)
                            + max_output_length_margin,
                        ),
                    )
                    for ids in encoded
                ]
                chunk_max_new_tokens = max(row_limits)
                row_max_new_tokens = torch.tensor(
                    row_limits,
                    dtype=torch.long,
                    device=self.device,
                )
            chunk_min_new_tokens = min(
                min_new_tokens,
                max(0, chunk_max_new_tokens - 1),
            )
            generated = self.model.generate(
                device_inputs,
                device_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=eos,
                max_new_tokens=chunk_max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                generation_context=generation_context,
                forbidden_token_ids=forbidden_token_ids,
                min_new_tokens=chunk_min_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens_per_row=row_max_new_tokens,
                reasoning_level=reasoning_level,
                **({} if generation_context is not None else generation_features),
            )
            generated_rows = cast(
                list[list[int]],
                generated.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            beam_texts = [
                restore(row, structured_maps[index], glossary_maps[index])
                for index, row in enumerate(generated_rows)
            ]

            if num_candidates < 1:
                results.extend(beam_texts)
                continue

            # Keep the beam result first, then add stochastic candidates. Stable
            # tie-breaking therefore preserves the baseline beam result.
            sampled = self.model.sample(
                device_inputs,
                device_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=eos,
                num_samples=num_candidates,
                max_new_tokens=chunk_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                forbidden_token_ids=forbidden_token_ids,
                min_new_tokens=chunk_min_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                generator=generator,
                generation_context=generation_context,
                max_new_tokens_per_row=row_max_new_tokens,
                reasoning_level=reasoning_level,
            )
            sampled_rows = cast(
                list[list[list[int]]],
                sampled.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            for row_index, source_text in enumerate(sources):
                candidates = [beam_texts[row_index]]
                for sample_row in sampled_rows[row_index]:
                    candidate = restore(
                        sample_row,
                        structured_maps[row_index],
                        glossary_maps[row_index],
                    )
                    # Avoid scoring duplicate candidate text.
                    if candidate not in candidates:
                        candidates.append(candidate)
                outcome = rerank_select(
                    source_text,
                    candidates,
                    strategy=rerank,
                    target_language=target_language,
                )
                results.append(outcome if return_rerank_details else outcome.text)
        return results

    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str | None = None,
        target_language: str,
        num_beams: int = DEFAULT_NUM_BEAMS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        max_new_tokens: int = 256,
        batch_size: int = 16,
        glossary: Glossary | None = None,
        append_missing_glossary: bool = True,
        append_missing_structured: bool = True,
        num_candidates: int = 0,
        rerank: str = "mbr+qe",
        temperature: float = 0.3,
        top_k: int = 0,
        seed: int | None = None,
        sampling_seed: int | None = None,
        generator: torch.Generator | None = None,
        return_rerank_details: bool = False,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_output_length_ratio: float | None = DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
        max_output_length_margin: int = DEFAULT_MAX_OUTPUT_LENGTH_MARGIN,
        reasoning_level: int | None = None,
    ) -> list[str]:
        """Translate raw sources, excluding the revision-only control separator."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        raw_texts = tuple(cast(Sequence[object], texts))
        if any(not isinstance(text, str) for text in raw_texts):
            raise TypeError("texts must contain only strings")
        validated_texts = cast(tuple[str, ...], raw_texts)
        if any(DRAFT_SEPARATOR in text for text in validated_texts):
            raise ValueError(
                f"raw translation source must not contain reserved {DRAFT_SEPARATOR}; "
                "use revise() with separate source and draft values"
            )
        return self._translate_internal(
            validated_texts,
            source_language=source_language,
            target_language=target_language,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            glossary=glossary,
            append_missing_glossary=append_missing_glossary,
            append_missing_structured=append_missing_structured,
            num_candidates=num_candidates,
            rerank=rerank,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            sampling_seed=sampling_seed,
            generator=generator,
            return_rerank_details=return_rerank_details,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_output_length_ratio=max_output_length_ratio,
            max_output_length_margin=max_output_length_margin,
            reasoning_level=reasoning_level,
        )

    @torch.no_grad()
    def revise(
        self,
        texts: Sequence[str],
        drafts: Sequence[str],
        *,
        source_language: str | None = None,
        target_language: str,
        num_beams: int = DEFAULT_NUM_BEAMS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        max_new_tokens: int = 256,
        batch_size: int = 16,
        reasoning_level: int | None = None,
    ) -> list[str]:
        """Revise drafts using separate source and draft sequences.

        This operation is valid only for exact directions trained on examples
        shaped as ``source <draft> draft -> target``. An untrained model treats
        the draft as ordinary source text and cannot provide reliable revision.

        The tokenizer must contain the atomic ``<draft>`` control token. Older
        tokenizers that split it into pieces do not reproduce the training input.
        """
        target_language = canonicalize_language_tag(target_language, field="target_language")
        source_language = self._resolve_source_language(source_language, target_language)
        requested_direction = (source_language, target_language)
        capabilities = self.export_metadata.get("capabilities")
        capabilities_mapping = (
            cast(Mapping[object, object], capabilities)
            if isinstance(capabilities, Mapping)
            else None
        )
        revision_trained = (
            capabilities_mapping.get("revision_trained")
            if capabilities_mapping is not None
            else None
        )
        revision_direction_edges = getattr(self, "_revision_direction_edges", None)
        if (
            revision_direction_edges is not None
            and requested_direction not in revision_direction_edges
        ):
            raise ValueError(
                "The exported model does not declare the revision capability for "
                f"{source_language}→{target_language}; re-export with the exact "
                "revision_directions."
            )
        if revision_direction_edges is None:
            if revision_trained is None:
                raise ValueError(
                    "Legacy export metadata does not record revision_directions or "
                    "revision_trained; revision inference is disabled until the model "
                    "is re-exported with an explicit revision capability."
                )
            if len(self.translation_directions) != 1:
                raise ValueError(
                    "Legacy revision capability metadata is ambiguous for a model with multiple "
                    "translation directions; re-export with revision_directions."
                )
            if revision_trained is False:
                raise ValueError(
                    "The exported model does not declare the revision capability; "
                    "export a checkpoint trained on revision examples."
                )
        if self.tokenizer.draft_id is None:
            raise ValueError(
                f"This tokenizer does not contain the {DRAFT_SEPARATOR} control token. "
                "Retrain it with sion-train-tokenizer before using revision."
            )
        if isinstance(texts, (str, bytes)) or isinstance(drafts, (str, bytes)):
            raise TypeError("revision texts and drafts must each be a sequence of strings")
        raw_texts = tuple(cast(Sequence[object], texts))
        raw_drafts = tuple(cast(Sequence[object], drafts))
        if any(not isinstance(text, str) for text in (*raw_texts, *raw_drafts)):
            raise TypeError("revision texts and drafts must contain only strings")
        validated_texts = cast(tuple[str, ...], raw_texts)
        validated_drafts = cast(tuple[str, ...], raw_drafts)
        if len(validated_texts) != len(validated_drafts):
            raise ValueError(
                f"Revision source count {len(validated_texts)} does not match draft count "
                f"{len(validated_drafts)}"
            )
        serialized_inputs: list[str] = []
        for index, (source, draft) in enumerate(
            zip(validated_texts, validated_drafts, strict=True)
        ):
            if not source.strip() or not draft.strip():
                raise ValueError(f"revision source and draft must be non-blank at row {index}")
            if DRAFT_SEPARATOR in source or DRAFT_SEPARATOR in draft:
                raise ValueError(
                    f"revision source and draft must not contain reserved {DRAFT_SEPARATOR} "
                    f"at row {index}"
                )
            serialized = serialize_revision_input(source, draft)
            if serialized.count(DRAFT_SEPARATOR) != 1:
                raise ValueError(
                    f"revision input must contain exactly one separator at row {index}"
                )
            serialized_inputs.append(serialized)
        return self._translate_internal(
            serialized_inputs,
            source_language=source_language,
            target_language=target_language,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            reasoning_level=reasoning_level,
        )
