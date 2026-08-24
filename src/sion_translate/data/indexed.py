from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
from torch.utils.data import Dataset, Sampler

from sion_translate.synthetic import DEFAULT_SYNTHETIC_SAMPLING_WEIGHT
from sion_translate.fingerprint import PREPROCESSING_SCHEMA
from sion_translate.language_tags import canonicalize_language_tag

from .integrity import validate_dataset_artifact_inventory
from .records import (
    languages_from_pairs,
    normalize_language_pairs,
    normalize_translation_directions,
)
from .record_metadata import (
    RECORD_METADATA_DATA_SUFFIX,
    RECORD_METADATA_INDEX_DTYPE,
    RECORD_METADATA_INDEX_SUFFIX,
    decode_record_metadata,
    resolve_record_training_direction,
)


class IndexedParallelDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        bidirectional: bool = True,
        legacy_bidirectional: bool | None = None,
        legacy_language_pairs: Sequence[Sequence[str]] | None = None,
        include_metadata: bool = False,
        verify_integrity: bool = True,
        allow_unverified_legacy: bool = False,
    ):
        self.root = Path(root) / split
        self.dataset_root = Path(root)
        self.bidirectional = bidirectional
        self.include_metadata = include_metadata
        self.verify_integrity = verify_integrity
        self.allow_unverified_legacy = allow_unverified_legacy
        if verify_integrity:
            validate_dataset_artifact_inventory(
                self.dataset_root,
                require_manifest=not allow_unverified_legacy,
            )
        self._manifest = self._read_dataset_manifest()
        self._current_translation_schema = self._detect_current_translation_schema()
        self.legacy_bidirectional = legacy_bidirectional
        try:
            self.legacy_language_pairs = (
                None
                if legacy_language_pairs is None
                else normalize_language_pairs(language_pairs=legacy_language_pairs)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("legacy_language_pairs are invalid") from error
        if not self._current_translation_schema and legacy_bidirectional is not None:
            self.bidirectional = legacy_bidirectional
        self.index_paths = sorted(self.root.glob("*.idx.npy"))
        if not self.index_paths:
            raise FileNotFoundError(f"No index shards found under {self.root}")
        self.indices = self._open_indices()
        self.record_metadata_indices = self._open_record_metadata_indices()
        self._record_metadata_cache: dict[int, np.memmap] = {}
        self.has_record_metadata = any(index is not None for index in self.record_metadata_indices)
        self.is_v3 = bool(
            self.indices
            and self.indices[0].dtype.names
            and "src_offset" in self.indices[0].dtype.names
        )
        self.cumulative: list[int] = []
        total = 0
        lengths: list[np.ndarray] = []
        source_ids: list[np.ndarray] = []
        synthetic_flags: list[np.ndarray] = []
        forward_only_flags: list[np.ndarray] = []
        self.has_source_metadata = True
        self.has_synthetic_metadata = True
        self.has_forward_only_metadata = True
        for index in self.indices:
            total += len(index)
            self.cumulative.append(total)
            src_length = "src_length" if self.is_v3 else "ko_length"
            tgt_length = "tgt_length" if self.is_v3 else "ja_length"
            lengths.append(
                index[src_length].astype(np.uint32) + index[tgt_length].astype(np.uint32)
            )
            if index.dtype.names is not None and "source_id" in index.dtype.names:
                source_ids.append(index["source_id"].astype(np.uint16))
            else:
                self.has_source_metadata = False
                source_ids.append(np.zeros(len(index), dtype=np.uint16))
            if index.dtype.names is not None and "synthetic" in index.dtype.names:
                synthetic_flags.append(index["synthetic"].astype(np.bool_))
            else:
                self.has_synthetic_metadata = False
                synthetic_flags.append(np.zeros(len(index), dtype=np.bool_))
            if index.dtype.names is not None and "forward_only" in index.dtype.names:
                forward_only_flags.append(index["forward_only"].astype(np.bool_))
            else:
                # Shards written before the v5 index have no such column, and
                # every pair in them was trained in both directions.
                self.has_forward_only_metadata = False
                forward_only_flags.append(np.zeros(len(index), dtype=np.bool_))
        self.pair_count = total
        self.pair_lengths: np.ndarray | None = np.concatenate(lengths)
        self.pair_source_ids: np.ndarray | None = np.concatenate(source_ids)
        self.pair_synthetic_flags: np.ndarray | None = np.concatenate(synthetic_flags)
        self.forward_only_count = int(np.count_nonzero(np.concatenate(forward_only_flags)))
        self.source_names = self._load_source_names()
        self.synthetic_sampling_weight = self._load_synthetic_sampling_weight()
        self.language_pairs, self.languages = self._load_language_metadata()
        self.language_pair = self.language_pairs[0]
        self.observed_language_pairs = self._find_observed_language_pairs()
        self.source_only_languages = self._load_source_only_languages()
        self.translation_directions = self._load_translation_directions()
        self._validate_translation_direction_rows()
        self._token_cache: dict[tuple[int, str], np.memmap] = {}
        self._bidirectional_pairs: np.ndarray | None = None
        self._forward_only_pairs: np.ndarray | None = None
        self._build_direction_maps()

    def _read_dataset_manifest(self) -> dict[str, Any]:
        manifest_path = self.dataset_root / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"dataset manifest cannot be read: {manifest_path}") from error
        if not isinstance(raw, dict):
            raise ValueError("dataset manifest must be a JSON object")
        return cast(dict[str, Any], raw)

    def _detect_current_translation_schema(self) -> bool:
        """Refuse downgrading a current dataset by deleting one mutable marker."""

        top_level = self._manifest.get("preprocessing_schema")
        raw_nested: object = self._manifest.get("fingerprint")
        nested = (
            cast(dict[object, object], raw_nested).get("preprocessing_schema")
            if isinstance(raw_nested, dict)
            else None
        )
        raw_fingerprint_path = self.dataset_root / "raw_fingerprint.json"
        raw_fingerprint_schema: object = None
        if raw_fingerprint_path.exists():
            try:
                raw_fingerprint: object = json.loads(
                    raw_fingerprint_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"dataset raw fingerprint cannot be read: {raw_fingerprint_path}"
                ) from error
            if not isinstance(raw_fingerprint, dict):
                raise ValueError("dataset raw fingerprint must be a JSON object")
            raw_fingerprint_schema = cast(dict[object, object], raw_fingerprint).get(
                "preprocessing_schema"
            )
        markers = (top_level, nested, raw_fingerprint_schema)
        if PREPROCESSING_SCHEMA not in markers:
            return False
        if any(marker != PREPROCESSING_SCHEMA for marker in markers):
            raise ValueError("current dataset preprocessing schema markers disagree")
        return True

    def _open_indices(self) -> list[np.ndarray]:
        return [np.load(path, mmap_mode="r", allow_pickle=False) for path in self.index_paths]

    def _open_record_metadata_indices(self) -> list[np.ndarray | None]:
        result: list[np.ndarray | None] = []
        for index_path, index in zip(self.index_paths, self.indices, strict=True):
            prefix = index_path.name.removesuffix(".idx.npy")
            metadata_index_path = self.root / f"{prefix}{RECORD_METADATA_INDEX_SUFFIX}"
            metadata_data_path = self.root / f"{prefix}{RECORD_METADATA_DATA_SUFFIX}"
            if not metadata_index_path.exists() and not metadata_data_path.exists():
                result.append(None)
                continue
            if not metadata_index_path.is_file() or not metadata_data_path.is_file():
                raise ValueError(f"Incomplete record metadata sidecar for {index_path}")
            metadata_index = np.load(metadata_index_path, mmap_mode="r", allow_pickle=False)
            if metadata_index.dtype != RECORD_METADATA_INDEX_DTYPE:
                raise ValueError(
                    f"Unsupported record metadata index dtype in {metadata_index_path}: "
                    f"{metadata_index.dtype.descr!r}"
                )
            if len(metadata_index) != len(index):
                raise ValueError(
                    f"Record metadata row count does not match {index_path}: "
                    f"{len(metadata_index)} != {len(index)}"
                )
            data_size = metadata_data_path.stat().st_size
            if len(metadata_index):
                ends = metadata_index["offset"].astype(np.uint64) + metadata_index["length"].astype(
                    np.uint64
                )
                if bool((ends > data_size).any()):
                    raise ValueError(f"Record metadata offset exceeds {metadata_data_path}")
            result.append(metadata_index)
        return result

    def _build_direction_maps(self) -> None:
        """Split pairs into bidirectional and forward-only virtual index ranges.

        Nothing is allocated when no pair is forward-only, which keeps the memory
        profile of a plain ko-ja corpus unchanged. When a source-only language is
        present the two int32 maps let ``__getitem__`` resolve a virtual index in
        constant time without materializing one entry per direction.
        """

        if self.forward_only_count == 0:
            self._bidirectional_pairs = None
            self._forward_only_pairs = None
            return
        flags = np.concatenate(
            [
                (
                    index["forward_only"].astype(np.bool_)
                    if index.dtype.names is not None and "forward_only" in index.dtype.names
                    else np.zeros(len(index), dtype=np.bool_)
                )
                for index in self.indices
            ]
        )
        self._bidirectional_pairs = np.flatnonzero(~flags).astype(np.int32)
        self._forward_only_pairs = np.flatnonzero(flags).astype(np.int32)

    def __getstate__(self) -> dict[str, Any]:
        """Keep Windows spawn workers from serializing hundreds of MB of memmaps."""

        state = self.__dict__.copy()
        state["indices"] = None
        state["record_metadata_indices"] = None
        state["pair_lengths"] = None
        state["pair_source_ids"] = None
        state["pair_synthetic_flags"] = None
        state["_bidirectional_pairs"] = None
        state["_forward_only_pairs"] = None
        state["_token_cache"] = {}
        state["_record_metadata_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Pickles created before row sidecars predate the opt-in flag.
        self.include_metadata = bool(getattr(self, "include_metadata", False))
        self.verify_integrity = bool(getattr(self, "verify_integrity", True))
        self.allow_unverified_legacy = bool(getattr(self, "allow_unverified_legacy", False))
        self.legacy_bidirectional = getattr(self, "legacy_bidirectional", None)
        self.legacy_language_pairs = getattr(self, "legacy_language_pairs", None)
        self.indices = self._open_indices()
        self.record_metadata_indices = self._open_record_metadata_indices()
        self._token_cache = {}
        self._record_metadata_cache = {}
        # Rebuilding from the reopened memmaps costs one pass over a uint8
        # column and avoids shipping the maps to every spawned worker.
        self._build_direction_maps()

    def _load_language_metadata(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        if self._manifest:
            manifest = self._manifest
            raw_pairs: object = manifest.get("language_pairs")
            if manifest.get("stage") == "foundation":
                if not isinstance(raw_pairs, list) or not raw_pairs:
                    raise ValueError("foundation dataset manifest language_pairs are missing")
                foundation_pairs: list[tuple[str, str]] = []
                seen_languages: set[str] = set()
                for index, raw_pair in enumerate(cast(list[object], raw_pairs)):
                    if not isinstance(raw_pair, list):
                        raise ValueError(
                            "foundation dataset language pairs must be two-item self-pairs"
                        )
                    pair_values = cast(list[object], raw_pair)
                    if len(pair_values) != 2:
                        raise ValueError(
                            "foundation dataset language pairs must be two-item self-pairs"
                        )
                    try:
                        source = canonicalize_language_tag(
                            pair_values[0],
                            field=f"foundation language_pairs[{index}][0]",
                        )
                        target = canonicalize_language_tag(
                            pair_values[1],
                            field=f"foundation language_pairs[{index}][1]",
                        )
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "foundation dataset manifest language_pairs are invalid"
                        ) from error
                    if source != target:
                        raise ValueError("foundation dataset language pairs must be self-pairs")
                    if source in seen_languages:
                        raise ValueError(
                            "foundation dataset manifest language_pairs contain duplicates"
                        )
                    seen_languages.add(source)
                    foundation_pairs.append((source, target))
                expected_languages = tuple(pair[0] for pair in foundation_pairs)
                raw_languages = manifest.get("languages")
                if (
                    not isinstance(raw_languages, list)
                    or tuple(cast(list[object], raw_languages)) != expected_languages
                ):
                    raise ValueError("foundation dataset manifest languages are invalid")
                return tuple(foundation_pairs), expected_languages
            if self._current_translation_schema:
                if not isinstance(raw_pairs, list) or not raw_pairs:
                    raise ValueError("current dataset manifest language_pairs are missing")
                pair_values = cast(list[object], raw_pairs)
                try:
                    pairs = normalize_language_pairs(
                        language_pairs=cast(list[list[str]], pair_values)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "current dataset manifest language_pairs are invalid"
                    ) from error
                if len(pairs) != len(pair_values):
                    raise ValueError("current dataset manifest language_pairs contain duplicates")
                raw_languages: object = manifest.get("languages")
                expected_languages = languages_from_pairs(pairs)
                language_values = (
                    cast(list[object], raw_languages) if isinstance(raw_languages, list) else []
                )
                if (
                    not isinstance(raw_languages, list)
                    or not all(isinstance(language, str) for language in language_values)
                    or tuple(cast(list[str], language_values)) != expected_languages
                ):
                    raise ValueError("current dataset manifest languages are invalid")
                return pairs, expected_languages
            if isinstance(raw_pairs, list) and raw_pairs:
                pair_values = cast(list[object], raw_pairs)
                try:
                    pairs = normalize_language_pairs(
                        language_pairs=cast(list[list[str]], pair_values)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "legacy dataset manifest language_pairs are invalid"
                    ) from error
                if len(pairs) != len(pair_values):
                    raise ValueError("legacy dataset manifest language_pairs contain duplicates")
                raw_languages: object = manifest.get("languages")
                expected_languages = languages_from_pairs(pairs)
                if raw_languages is not None:
                    language_values = (
                        cast(list[object], raw_languages) if isinstance(raw_languages, list) else []
                    )
                    if (
                        not isinstance(raw_languages, list)
                        or not all(isinstance(language, str) for language in language_values)
                        or tuple(cast(list[str], language_values)) != expected_languages
                    ):
                        raise ValueError("legacy dataset manifest languages are invalid")
                return pairs, expected_languages
            pair: object = manifest.get("language_pair")
            if pair is not None:
                if not isinstance(pair, list):
                    raise ValueError("legacy dataset manifest language_pair is invalid")
                pair_items = cast(list[object], pair)
                try:
                    pairs = normalize_language_pairs(language_pair=cast(list[str], pair_items))
                except (TypeError, ValueError) as error:
                    raise ValueError("legacy dataset manifest language_pair is invalid") from error
                return pairs, languages_from_pairs(pairs)
        if self.legacy_language_pairs is None:
            raise ValueError(
                "legacy dataset has no authenticated language identity; "
                "pass legacy_language_pairs explicitly"
            )
        if not self.is_v3 and len(self.legacy_language_pairs) != 1:
            raise ValueError("legacy v2 datasets require exactly one legacy language pair")
        return self.legacy_language_pairs, languages_from_pairs(self.legacy_language_pairs)

    def _load_source_names(self) -> list[str]:
        if not self._manifest:
            if self.pair_source_ids is None:
                return ["source-0"]
            maximum = int(self.pair_source_ids.max(initial=0))
            return [f"source-{source_id}" for source_id in range(maximum + 1)]
        manifest = self._manifest
        raw_sources: object = manifest.get("sources")
        sources = cast(list[object], raw_sources) if isinstance(raw_sources, list) else []
        if not sources:
            raw_inputs: object = manifest.get("inputs")
            inputs = cast(list[object], raw_inputs) if isinstance(raw_inputs, list) else []
            return [Path(str(path)).name for path in inputs] or ["source-0"]
        source_mappings = [
            cast(dict[object, object], source) for source in sources if isinstance(source, dict)
        ]
        maximum = max(int(str(source["id"])) for source in source_mappings)
        names = [f"source-{source_id}" for source_id in range(maximum + 1)]
        for source in source_mappings:
            names[int(str(source["id"]))] = str(source["name"])
        return names

    def _load_source_only_languages(self) -> tuple[str, ...]:
        if not self._manifest:
            return ()
        manifest = self._manifest
        raw: object = manifest.get("source_only_languages")
        if not isinstance(raw, list):
            if self._current_translation_schema:
                raise ValueError("current dataset manifest source_only_languages are invalid")
            return ()
        raw_languages = cast(list[object], raw)
        if self._current_translation_schema and not all(
            isinstance(language, str) for language in raw_languages
        ):
            raise ValueError("current dataset manifest source_only_languages are invalid")
        return tuple(str(language) for language in raw_languages)

    def _find_observed_language_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return configured physical pairs that have at least one row in this split."""

        return self.observed_language_pairs_for_physical_mask(
            np.ones(self.pair_count, dtype=np.bool_)
        )

    def observed_language_pairs_for_physical_mask(
        self,
        mask: np.ndarray,
    ) -> tuple[tuple[str, str], ...]:
        """Return configured pairs represented by selected physical rows."""

        selected = np.asarray(mask, dtype=np.bool_)
        if selected.ndim != 1 or len(selected) != self.pair_count:
            raise ValueError("physical pair mask must match the dataset pair count")
        if not self.is_v3:
            return (self.language_pair,) if bool(selected.any()) else ()
        observed_edges: set[frozenset[str]] = set()
        offset = 0
        for index in self.indices:
            local_mask = selected[offset : offset + len(index)]
            offset += len(index)
            if not bool(local_mask.any()):
                continue
            source_ids = np.asarray(index["src_language_id"], dtype=np.int64)[local_mask]
            target_ids = np.asarray(index["tgt_language_id"], dtype=np.int64)[local_mask]
            packed = source_ids * len(self.languages) + target_ids
            for packed_pair in np.unique(packed):
                source_id, target_id = divmod(int(packed_pair), len(self.languages))
                observed_edges.add(
                    frozenset((self.languages[source_id], self.languages[target_id]))
                )
        return tuple(pair for pair in self.language_pairs if frozenset(pair) in observed_edges)

    def observed_translation_directions_for_physical_mask(
        self,
        mask: np.ndarray,
    ) -> tuple[tuple[str, str], ...]:
        """Return directed edges that selected physical rows can actually emit."""

        selected = np.asarray(mask, dtype=np.bool_)
        if selected.ndim != 1 or len(selected) != self.pair_count:
            raise ValueError("physical pair mask must match the dataset pair count")
        if not bool(selected.any()):
            return ()
        if not self.is_v3:
            # Legacy dense rows have no row-scoped direction identity; their
            # caller-authenticated global policy is the only available graph.
            return self.translation_directions
        observed: set[tuple[str, str]] = set()
        offset = 0
        for index in self.indices:
            local_mask = selected[offset : offset + len(index)]
            offset += len(index)
            for row in index[local_mask]:
                source_language = self.languages[int(row["src_language_id"])]
                target_language = self.languages[int(row["tgt_language_id"])]
                observed.add((source_language, target_language))
                if not bool(row["forward_only"]):
                    observed.add((target_language, source_language))
        return tuple(
            direction for direction in self.translation_directions if direction in observed
        )

    def _load_translation_directions(self) -> tuple[tuple[str, str], ...]:
        if self._manifest:
            manifest = self._manifest
            if manifest.get("stage") == "foundation":
                if any(source != target for source, target in self.language_pairs):
                    raise ValueError("foundation datasets must contain only self-restoration pairs")
                return self.language_pairs
            raw: object = manifest.get("translation_directions")
            if isinstance(raw, list):
                return normalize_translation_directions(
                    self.language_pairs,
                    cast(list[list[str]], raw),
                    source_only_languages=self.source_only_languages,
                )
            if self._current_translation_schema:
                raise ValueError("current dataset manifest translation_directions are invalid")
        # Old manifests did not authenticate the runtime policy. Preserve the
        # caller's explicit legacy choice instead of silently upgrading every
        # old dataset to a bidirectional graph.
        return normalize_translation_directions(
            self.language_pairs,
            bidirectional=self.bidirectional,
            source_only_languages=self.source_only_languages,
        )

    def _validate_translation_direction_rows(self) -> None:
        """Cross-check physical row orientation and flags against the manifest graph."""

        if self._manifest.get("stage") == "foundation":
            return
        if not self._current_translation_schema:
            return
        if not self.is_v3:
            return

        required_fields = {"src_language_id", "tgt_language_id", "forward_only"}
        language_to_id = {language: index for index, language in enumerate(self.languages)}
        direction_set = set(self.translation_directions)
        allowed = {
            (language_to_id[source], language_to_id[target]) for source, target in direction_set
        }

        language_count = len(self.languages)
        for shard, index in enumerate(self.indices):
            names = set(index.dtype.names or ())
            if not required_fields <= names:
                raise ValueError("current dataset index lacks translation direction fields")
            source_ids = np.asarray(index["src_language_id"], dtype=np.int64)
            target_ids = np.asarray(index["tgt_language_id"], dtype=np.int64)
            forward_only = np.asarray(index["forward_only"], dtype=np.int64)
            packed = (source_ids * language_count + target_ids) * 2 + forward_only
            for local, packed_row in enumerate(packed):
                pair_value, forward_flag = divmod(int(packed_row), 2)
                source_id, target_id = divmod(pair_value, language_count)
                if (source_id, target_id) not in allowed or (
                    not bool(forward_flag) and (target_id, source_id) not in allowed
                ):
                    raise ValueError(
                        "dataset index direction rows disagree with the manifest graph"
                    )
                if bool(forward_flag) and (target_id, source_id) in allowed:
                    metadata = self._metadata_for_physical_pair(shard, local)
                    expected_direction = [
                        self.languages[source_id],
                        self.languages[target_id],
                    ]
                    if metadata.get("training_direction") != expected_direction:
                        raise ValueError(
                            "dataset index direction rows disagree with the manifest graph: "
                            "row-scoped direction lacks matching metadata"
                        )

    def detect_revision_directions(
        self,
        *,
        draft_token_id: int | None,
        max_source_tokens: int,
        physical_mask: np.ndarray | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Authenticate directed revision edges from row source/provenance metadata.

        A filename marker alone identifies candidate rows, but it cannot prove
        their direction. Revision rows therefore need current index fields, a
        forward-only physical orientation, and matching row-scoped
        ``training_direction`` metadata. Provenance-marked rows use the same
        contract, so custom filenames remain supported without guessing.
        """

        selected = (
            np.ones(self.pair_count, dtype=np.bool_)
            if physical_mask is None
            else np.asarray(physical_mask, dtype=np.bool_)
        )
        if selected.ndim != 1 or len(selected) != self.pair_count:
            raise ValueError("physical pair mask must match the dataset pair count")
        trained = set(self.translation_directions)
        detected: set[tuple[str, str]] = set()
        required_fields = {
            "source_id",
            "src_language_id",
            "tgt_language_id",
            "src_offset",
            "src_length",
            "forward_only",
        }
        physical_index = 0
        for shard, index in enumerate(self.indices):
            names = set(index.dtype.names or ())
            for local, row in enumerate(index):
                row_selected = bool(selected[physical_index])
                physical_index += 1
                source_id = int(row["source_id"]) if "source_id" in names else -1
                source_name = (
                    Path(self.source_names[source_id]).name
                    if 0 <= source_id < len(self.source_names)
                    else ""
                )
                source_marked = source_name.startswith("revise_")
                metadata = self._metadata_for_physical_pair(shard, local)
                raw_provenance = metadata.get("provenance")
                provenance = (
                    cast(Mapping[object, object], raw_provenance)
                    if isinstance(raw_provenance, Mapping)
                    else None
                )
                transformation = (
                    provenance.get("transformation") if provenance is not None else None
                )
                provenance_marked = transformation == "revision"
                if source_marked and transformation is not None and not provenance_marked:
                    raise ValueError(
                        "revision source row has conflicting provenance transformation: "
                        f"source={source_name!r}, transformation={transformation!r}"
                    )
                revision_marked = source_marked or provenance_marked
                source_tokens: np.ndarray | None = None
                draft_positions: np.ndarray | None = None
                source_length = -1
                if (
                    draft_token_id is not None
                    and self.is_v3
                    and {"src_offset", "src_length"} <= names
                ):
                    source_start = int(row["src_offset"])
                    source_length = int(row["src_length"])
                    source_store = self._tokens(shard, "src")
                    source_tokens = np.asarray(
                        source_store[source_start : source_start + source_length],
                        dtype=np.int64,
                    )
                    draft_positions = np.flatnonzero(source_tokens == draft_token_id)
                if not revision_marked:
                    if draft_positions is not None and len(draft_positions):
                        raise ValueError(
                            "row contains a reserved <draft> token but lacks a revision "
                            f"filename or provenance marker: source={source_name!r}"
                        )
                    continue
                if not self._current_translation_schema or not self.is_v3:
                    raise ValueError(
                        "revision rows require a current indexed dataset with authenticated "
                        "direction metadata"
                    )
                if not required_fields <= names:
                    raise ValueError("revision row index lacks authenticated direction fields")
                if source_id < 0 or source_id >= len(self.source_names):
                    raise ValueError("revision row source_id is outside the source manifest")
                if draft_token_id is None:
                    raise ValueError(
                        "revision rows require a tokenizer with an authenticated <draft> token"
                    )
                assert source_tokens is not None
                assert draft_positions is not None
                if (
                    len(draft_positions) != 1
                    or int(draft_positions[0]) == 0
                    or int(draft_positions[0]) == source_length - 1
                ):
                    raise ValueError(
                        "revision row source must contain exactly one <draft> token with "
                        "non-empty source and draft segments"
                    )
                effective_source = source_tokens[:max_source_tokens]
                effective_draft_positions = np.flatnonzero(effective_source == draft_token_id)
                if (
                    len(effective_draft_positions) != 1
                    or int(effective_draft_positions[0]) == 0
                    or int(effective_draft_positions[0]) == len(effective_source) - 1
                ):
                    raise ValueError(
                        "revision row loses its <draft> structure after the training "
                        "collator's source truncation"
                    )
                source_language_id = int(row["src_language_id"])
                target_language_id = int(row["tgt_language_id"])
                if not (
                    0 <= source_language_id < len(self.languages)
                    and 0 <= target_language_id < len(self.languages)
                ):
                    raise ValueError("revision row language ID is outside the language manifest")
                if not bool(row["forward_only"]):
                    raise ValueError(
                        "revision rows must be forward-only so their training direction is exact"
                    )
                physical_direction = (
                    self.languages[source_language_id],
                    self.languages[target_language_id],
                )
                resolved = resolve_record_training_direction(
                    metadata,
                    physical_direction,
                    trained,
                )
                if resolved is None or resolved != physical_direction:
                    raise ValueError(
                        "revision row training_direction must exactly match its stored forward "
                        f"orientation: source={source_name!r}, stored={physical_direction!r}"
                    )
                if row_selected:
                    detected.add(resolved)
        return tuple(
            direction for direction in self.translation_directions if direction in detected
        )

    def _load_synthetic_sampling_weight(self) -> float:
        if not self._manifest:
            return DEFAULT_SYNTHETIC_SAMPLING_WEIGHT
        manifest = self._manifest
        raw_policy: object = manifest.get("synthetic_policy")
        policy = cast(dict[object, object], raw_policy) if isinstance(raw_policy, dict) else {}
        return float(str(policy.get("sampling_weight", DEFAULT_SYNTHETIC_SAMPLING_WEIGHT)))

    def __len__(self) -> int:
        if not self.bidirectional:
            return self.pair_count
        # Forward-only pairs contribute one direction instead of two.
        return 2 * self.pair_count - self.forward_only_count

    @property
    def direction_count(self) -> int:
        """Number of distinct (source, target) directions this split can yield."""

        if not self.bidirectional:
            return len(self.language_pairs)
        return len(self.translation_directions)

    def _resolve_virtual(self, index: int) -> tuple[int, int]:
        """Map a virtual index to ``(pair_index, direction)``."""

        if not self.bidirectional:
            return index, 0
        if self._bidirectional_pairs is None:
            return divmod(index, 2)
        boundary = 2 * len(self._bidirectional_pairs)
        if index < boundary:
            local, direction = divmod(index, 2)
            return int(self._bidirectional_pairs[local]), direction
        assert self._forward_only_pairs is not None
        return int(self._forward_only_pairs[index - boundary]), 0

    def _pair_index(self, index: int) -> int:
        return self._resolve_virtual(index)[0]

    def _pair_indices(self, indices: np.ndarray) -> np.ndarray:
        """Vectorized ``_pair_index`` for the batch sampler."""

        if not self.bidirectional:
            return indices
        if self._bidirectional_pairs is None:
            return indices // 2
        assert self._forward_only_pairs is not None
        boundary = 2 * len(self._bidirectional_pairs)
        result = np.empty(len(indices), dtype=np.int64)
        low = indices < boundary
        result[low] = self._bidirectional_pairs[indices[low] // 2]
        result[~low] = self._forward_only_pairs[indices[~low] - boundary]
        return result

    def _virtual_indices_for_pairs(
        self,
        pair_indices: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Map sampled physical pairs to valid virtual dataset indices.

        Forward-only rows occupy a compact, single-direction range after all
        bidirectional rows. Consequently, ``pair * 2 + direction`` is valid
        only for the legacy dense layout. This inverse mapping preserves the
        sampled physical pair while forcing forward-only rows to direction 0.
        """

        pairs = np.asarray(pair_indices, dtype=np.int64)
        requested_directions = np.asarray(directions, dtype=np.int64)
        if pairs.ndim != 1 or requested_directions.shape != pairs.shape:
            raise ValueError("pair_indices and directions must be matching 1D arrays")
        if bool(((pairs < 0) | (pairs >= self.pair_count)).any()):
            raise IndexError("physical pair index is out of range")
        if bool(((requested_directions < 0) | (requested_directions > 1)).any()):
            raise ValueError("directions must contain only 0 or 1")

        index_dtype = np.uint32 if len(self) <= np.iinfo(np.uint32).max else np.uint64
        if not self.bidirectional:
            return pairs.astype(index_dtype, copy=False)
        if self._bidirectional_pairs is None:
            return pairs.astype(index_dtype, copy=False) * index_dtype(
                2
            ) + requested_directions.astype(index_dtype, copy=False)

        assert self._forward_only_pairs is not None
        bidirectional_positions = np.searchsorted(self._bidirectional_pairs, pairs)
        is_bidirectional = bidirectional_positions < len(self._bidirectional_pairs)
        matched_positions = np.flatnonzero(is_bidirectional)
        is_bidirectional[matched_positions] &= (
            self._bidirectional_pairs[bidirectional_positions[matched_positions]]
            == pairs[matched_positions]
        )

        result = np.empty(len(pairs), dtype=index_dtype)
        result[is_bidirectional] = bidirectional_positions[is_bidirectional].astype(
            index_dtype, copy=False
        ) * index_dtype(2) + requested_directions[is_bidirectional].astype(index_dtype, copy=False)

        forward_pairs = pairs[~is_bidirectional]
        forward_positions = np.searchsorted(self._forward_only_pairs, forward_pairs)
        if bool(
            (forward_positions >= len(self._forward_only_pairs)).any()
            or (self._forward_only_pairs[forward_positions] != forward_pairs).any()
        ):
            raise RuntimeError("direction maps do not cover every physical pair")
        boundary = index_dtype(2 * len(self._bidirectional_pairs))
        result[~is_bidirectional] = boundary + forward_positions.astype(
            index_dtype,
            copy=False,
        )
        return result

    def _resolve(self, pair_index: int) -> tuple[int, int]:
        shard = bisect.bisect_right(self.cumulative, pair_index)
        previous = 0 if shard == 0 else self.cumulative[shard - 1]
        return shard, pair_index - previous

    def _tokens(self, shard: int, language: str) -> np.memmap:
        key = (shard, language)
        if key not in self._token_cache:
            prefix = self.index_paths[shard].name.removesuffix(".idx.npy")
            path = self.root / f"{prefix}.{language}.bin"
            self._token_cache[key] = np.memmap(path, dtype=np.uint32, mode="r")
        return self._token_cache[key]

    def _record_metadata_bytes(self, shard: int) -> np.memmap:
        if shard not in self._record_metadata_cache:
            prefix = self.index_paths[shard].name.removesuffix(".idx.npy")
            path = self.root / f"{prefix}{RECORD_METADATA_DATA_SUFFIX}"
            self._record_metadata_cache[shard] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._record_metadata_cache[shard]

    def _metadata_for_physical_pair(self, shard: int, local: int) -> dict[str, object]:
        metadata_index = self.record_metadata_indices[shard]
        if metadata_index is None:
            return {}
        row = metadata_index[local]
        offset, length = int(row["offset"]), int(row["length"])
        if length == 0:
            return {}
        store = self._record_metadata_bytes(shard)
        payload = np.asarray(store[offset : offset + length], dtype=np.uint8).tobytes()
        return decode_record_metadata(payload)

    def metadata_at(self, index: int) -> dict[str, object]:
        """Return preserved raw-record annotations for one virtual sample."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pair_index = self._pair_index(index)
        shard, local = self._resolve(pair_index)
        return self._metadata_for_physical_pair(shard, local)

    def length_at(self, index: int) -> int:
        pair_index = self._pair_index(index)
        if self.pair_lengths is not None:
            return int(self.pair_lengths[pair_index]) + 4
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]
        if self.is_v3:
            return int(row["src_length"] + row["tgt_length"]) + 4
        return int(row["ko_length"] + row["ja_length"]) + 4

    def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.pair_lengths is None:
            raise RuntimeError("Length metadata is unavailable inside a DataLoader worker")
        return self.pair_lengths[self._pair_indices(indices)]

    def source_id_at(self, index: int) -> int:
        pair_index = self._pair_index(index)
        if self.pair_source_ids is not None:
            return int(self.pair_source_ids[pair_index])
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]
        if row.dtype.names is not None and "source_id" in row.dtype.names:
            return int(row["source_id"])
        return 0

    def synthetic_at(self, index: int) -> bool:
        pair_index = self._pair_index(index)
        if self.pair_synthetic_flags is not None:
            return bool(self.pair_synthetic_flags[pair_index])
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]
        return bool(
            row["synthetic"]
            if row.dtype.names is not None and "synthetic" in row.dtype.names
            else False
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pair_index, direction = self._resolve_virtual(index)
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]

        if self.is_v3:
            language_a = self.languages[int(row["src_language_id"])]
            language_b = self.languages[int(row["tgt_language_id"])]
            src_store = self._tokens(shard, "src")
            tgt_store = self._tokens(shard, "tgt")
            src_start, src_length = int(row["src_offset"]), int(row["src_length"])
            tgt_start, tgt_length = int(row["tgt_offset"]), int(row["tgt_length"])
            side_a = np.asarray(
                src_store[src_start : src_start + src_length],
                dtype=np.int64,
            )
            side_b = np.asarray(
                tgt_store[tgt_start : tgt_start + tgt_length],
                dtype=np.int64,
            )
            register_a = int(row["src_register"])
            register_b = int(row["tgt_register"])
        else:
            language_a, language_b = self.language_pair
            src_store = self._tokens(shard, language_a)
            tgt_store = self._tokens(shard, language_b)
            src_start, src_length = int(row["ko_offset"]), int(row["ko_length"])
            tgt_start, tgt_length = int(row["ja_offset"]), int(row["ja_length"])
            side_a = np.asarray(
                src_store[src_start : src_start + src_length],
                dtype=np.int64,
            )
            side_b = np.asarray(
                tgt_store[tgt_start : tgt_start + tgt_length],
                dtype=np.int64,
            )
            register_a = int(row["ko_register"])
            register_b = int(row["ja_register"])

        if direction == 0:
            src, tgt = side_a, side_b
            src_language, target_language = language_a, language_b
            src_register, target_register = register_a, register_b
        else:
            src, tgt = side_b, side_a
            src_language, target_language = language_b, language_a
            src_register, target_register = register_b, register_a
        forward_only = bool(
            row["forward_only"]
            if row.dtype.names is not None and "forward_only" in row.dtype.names
            else False
        )
        item: dict[str, object] = {
            "src": src,
            "tgt": tgt,
            "src_language": src_language,
            "target_language": target_language,
            "src_register": src_register,
            "target_register": target_register,
            "pair_index": pair_index,
            "synthetic": bool(
                row["synthetic"]
                if row.dtype.names is not None and "synthetic" in row.dtype.names
                else False
            ),
            # MRT may backtranslate only when this split actually trained the
            # reverse edge. Language names alone cannot establish that for
            # source-only rows or a globally unidirectional dataset.
            "reverse_direction_trained": self.bidirectional and not forward_only,
        }
        if self.include_metadata:
            metadata = self.metadata_at(index)
            item["metadata"] = metadata
            item.update(metadata)
        return item


class DistributedBucketBatchSampler(Sampler[list[int]]):
    """Fixed-size distributed batches with local length bucketing.

    Every rank gets exactly the same number of batches, avoiding collective hangs.
    """

    def __init__(
        self,
        dataset: IndexedParallelDataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        bucket_size: int = 4096,
        seed: int = 0,
        drop_last: bool = True,
        source_sampling_alpha: float = 1.0,
        source_sampling_weights: dict[str, float] | None = None,
        source_sampling_weights_by_id: dict[int, float] | None = None,
        max_source_upsampling: float = 3.0,
        synthetic_sampling_weight: float | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.bucket_size = max(bucket_size, batch_size * world_size)
        self.seed = seed
        self.drop_last = drop_last
        if source_sampling_alpha <= 0:
            raise ValueError("source_sampling_alpha must be positive")
        if max_source_upsampling < 1.0:
            raise ValueError("max_source_upsampling must be at least 1")
        self.source_sampling_alpha = source_sampling_alpha
        self.source_sampling_weights = dict(source_sampling_weights or {})
        self.source_sampling_weights_by_id = dict(source_sampling_weights_by_id or {})
        self.max_source_upsampling = max_source_upsampling
        self.synthetic_sampling_weight = (
            float(synthetic_sampling_weight)
            if synthetic_sampling_weight is not None
            else float(
                getattr(
                    dataset,
                    "synthetic_sampling_weight",
                    DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
                )
            )
        )
        if not 0.0 <= self.synthetic_sampling_weight <= 1.0:
            raise ValueError("synthetic_sampling_weight must be in [0, 1]")
        known_sources = set(dataset.source_names)
        unknown_sources = set(self.source_sampling_weights) - known_sources
        if unknown_sources:
            raise ValueError(f"Unknown source_sampling_weights keys: {sorted(unknown_sources)}")
        if any(weight < 0 for weight in self.source_sampling_weights.values()):
            raise ValueError("source sampling weights must be non-negative")
        unknown_source_ids = sorted(
            source_id
            for source_id in self.source_sampling_weights_by_id
            if source_id < 0 or source_id >= len(dataset.source_names)
        )
        if unknown_source_ids:
            raise ValueError(f"Unknown source_sampling_weights_by_id keys: {unknown_source_ids}")
        if any(weight < 0 for weight in self.source_sampling_weights_by_id.values()):
            raise ValueError("source sampling weights by id must be non-negative")
        synthetic_flags = getattr(dataset, "pair_synthetic_flags", None)
        has_synthetic = synthetic_flags is not None and bool(np.asarray(synthetic_flags).any())
        self._balance_sources = (
            not math.isclose(source_sampling_alpha, 1.0)
            or any(
                not math.isclose(weight, 1.0) for weight in self.source_sampling_weights.values()
            )
            or any(
                not math.isclose(weight, 1.0)
                for weight in self.source_sampling_weights_by_id.values()
            )
            or (has_synthetic and not math.isclose(self.synthetic_sampling_weight, 1.0))
        )
        if self._balance_sources and not getattr(dataset, "has_source_metadata", True):
            raise ValueError(
                "Source-balanced sampling requires v2 shards with source_id metadata; "
                "re-run sion-prepare-data"
            )
        self._group_pair_indices: dict[int, np.ndarray] | None = None
        self._group_pair_probabilities: dict[int, np.ndarray] | None = None
        self._group_codes: np.ndarray | None = None
        self._group_probabilities: np.ndarray | None = None
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.start_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        """Skip already-consumed batches without fetching or collating them."""

        if start_batch < 0 or start_batch > len(self):
            raise ValueError(f"start_batch must be in [0, {len(self)}]")
        self.start_batch = int(start_batch)

    def __len__(self) -> int:
        denominator = self.batch_size * self.world_size
        if self.drop_last:
            return len(self.dataset) // denominator
        return math.ceil(len(self.dataset) / denominator)

    @staticmethod
    def _cap_source_probabilities(
        raw: np.ndarray, natural: np.ndarray, maximum_multiplier: float
    ) -> np.ndarray:
        """Project source probabilities onto capped natural-distribution bounds."""

        result = np.zeros_like(raw, dtype=np.float64)
        active = raw > 0
        remaining = 1.0
        caps = natural * maximum_multiplier
        while active.any():
            active_raw = raw[active]
            candidate = remaining * active_raw / active_raw.sum()
            active_indices = np.flatnonzero(active)
            over = candidate > caps[active]
            if not over.any():
                result[active_indices] = candidate
                break
            capped_indices = active_indices[over]
            result[capped_indices] = caps[capped_indices]
            active[capped_indices] = False
            remaining = max(0.0, 1.0 - result.sum())
        total = result.sum()
        if total <= 0:
            raise ValueError("Source sampling weights exclude every available source")
        return result / total

    def _balanced_indices(self, rng: np.random.Generator, sample_count: int) -> np.ndarray:
        if self.dataset.pair_source_ids is None:
            raise RuntimeError("Source metadata is unavailable for balanced sampling")
        self._initialize_balanced_groups()
        group_codes = self._group_codes
        probabilities = self._group_probabilities
        assert group_codes is not None
        assert probabilities is not None
        assert self._group_pair_indices is not None
        assert self._group_pair_probabilities is not None
        sampled_groups = rng.choice(
            group_codes,
            size=sample_count,
            replace=True,
            p=probabilities,
        )
        sampled_pairs = np.empty(sample_count, dtype=np.uint32)
        for group_code in group_codes:
            positions = np.flatnonzero(sampled_groups == group_code)
            candidates = self._group_pair_indices[int(group_code)]
            sampled_pairs[positions] = rng.choice(
                candidates,
                size=len(positions),
                replace=True,
                p=self._group_pair_probabilities[int(group_code)],
            )
        if not self.dataset.bidirectional:
            return sampled_pairs
        directions = rng.integers(0, 2, size=sample_count, dtype=np.uint32)
        return self.dataset._virtual_indices_for_pairs(  # pyright: ignore[reportPrivateUsage]
            sampled_pairs, directions
        )

    def _initialize_balanced_groups(self) -> None:
        if self._group_codes is not None and self._group_probabilities is not None:
            return
        if self.dataset.pair_source_ids is None:
            raise RuntimeError("Source metadata is unavailable for balanced sampling")
        source_ids_for_pairs = self.dataset.pair_source_ids.astype(np.uint32, copy=False)
        pair_synthetic_flags = getattr(self.dataset, "pair_synthetic_flags", None)
        if pair_synthetic_flags is None:
            pair_synthetic_flags = np.zeros(len(source_ids_for_pairs), dtype=np.bool_)
        synthetic_flags = np.asarray(pair_synthetic_flags, dtype=np.uint32)
        pair_group_codes = source_ids_for_pairs * np.uint32(2) + synthetic_flags
        direction_multiplicity = np.ones(len(source_ids_for_pairs), dtype=np.float64)
        if self.dataset.bidirectional:
            direction_multiplicity.fill(2.0)
            forward_only_pairs = getattr(self.dataset, "_forward_only_pairs", None)
            if forward_only_pairs is not None:
                direction_multiplicity[np.asarray(forward_only_pairs, dtype=np.int64)] = 1.0
        source_counts = np.bincount(source_ids_for_pairs)
        synthetic_counts_by_source = np.bincount(
            source_ids_for_pairs,
            weights=synthetic_flags,
        )
        counts_all = np.bincount(pair_group_codes, weights=direction_multiplicity)
        group_codes = np.flatnonzero(counts_all).astype(np.uint32, copy=False)
        counts = counts_all[group_codes].astype(np.float64)
        natural = counts / counts.sum()
        raw = np.power(counts, self.source_sampling_alpha)
        for position, group_code in enumerate(group_codes):
            source_id = int(group_code) // 2
            group_is_synthetic = bool(int(group_code) % 2)
            name = (
                self.dataset.source_names[source_id]
                if source_id < len(self.dataset.source_names)
                else f"source-{source_id}"
            )
            raw[position] *= self.source_sampling_weights.get(name, 1.0)
            raw[position] *= self.source_sampling_weights_by_id.get(source_id, 1.0)
            if group_is_synthetic:
                source_is_all_synthetic = (
                    synthetic_counts_by_source[source_id] == source_counts[source_id]
                )
                has_explicit_source_weight = (
                    name in self.source_sampling_weights
                    or source_id in self.source_sampling_weights_by_id
                )
                if not (source_is_all_synthetic and has_explicit_source_weight):
                    raw[position] *= self.synthetic_sampling_weight
        self._group_codes = group_codes
        self._group_probabilities = self._cap_source_probabilities(
            raw, natural, self.max_source_upsampling
        )
        self._group_pair_indices = {
            int(group_code): np.flatnonzero(pair_group_codes == group_code).astype(
                np.uint32, copy=False
            )
            for group_code in group_codes
        }
        self._group_pair_probabilities = {}
        for group_code, candidates in self._group_pair_indices.items():
            weights = direction_multiplicity[candidates]
            self._group_pair_probabilities[group_code] = weights / weights.sum()

    def positive_sampling_pair_mask(self) -> np.ndarray:
        """Return physical rows with non-zero probability under this sampler."""

        if not self._balance_sources:
            return np.ones(self.dataset.pair_count, dtype=np.bool_)
        self._initialize_balanced_groups()
        assert self._group_codes is not None
        assert self._group_probabilities is not None
        assert self._group_pair_indices is not None
        result = np.zeros(self.dataset.pair_count, dtype=np.bool_)
        for group_code, probability in zip(
            self._group_codes,
            self._group_probabilities,
            strict=True,
        ):
            if probability > 0:
                result[self._group_pair_indices[int(group_code)]] = True
        return result

    def __iter__(self) -> Iterator[list[int]]:
        if self._balance_sources:
            sample_count = len(self) * self.batch_size
            if sample_count == 0:
                return
            # Balanced sampling is with replacement, so each rank can generate
            # its local stream directly instead of materializing a global epoch.
            rng = np.random.default_rng(self.seed + self.epoch + self.rank * 0x9E3779B1)
            indices = self._balanced_indices(rng, sample_count)
        else:
            rng = np.random.default_rng(self.seed + self.epoch)
            index_dtype = np.uint32 if len(self.dataset) <= np.iinfo(np.uint32).max else np.uint64
            indices = np.arange(len(self.dataset), dtype=index_dtype)
            rng.shuffle(indices)
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            lengths = self.dataset.lengths_for_indices(bucket)
            bucket = bucket[np.argsort(lengths, kind="stable")]
            indices[start : start + len(bucket)] = bucket

        if self._balance_sources:
            local = indices
        else:
            group = self.batch_size * self.world_size
            if self.drop_last:
                usable = (len(indices) // group) * group
                indices = indices[:usable]
            else:
                usable = math.ceil(len(indices) / group) * group if len(indices) else 0
                if usable > len(indices):
                    indices = np.resize(indices, usable)
            local = indices[self.rank : usable : self.world_size]
        for batch_index, start in enumerate(range(0, len(local), self.batch_size)):
            if batch_index < self.start_batch:
                continue
            batch = local[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch.tolist()
