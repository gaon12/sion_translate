from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
from torch.utils.data import Dataset, Sampler

from sion_translate.synthetic import DEFAULT_SYNTHETIC_SAMPLING_WEIGHT

from .record_metadata import (
    RECORD_METADATA_DATA_SUFFIX,
    RECORD_METADATA_INDEX_DTYPE,
    RECORD_METADATA_INDEX_SUFFIX,
    decode_record_metadata,
)


class IndexedParallelDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        bidirectional: bool = True,
        include_metadata: bool = False,
    ):
        self.root = Path(root) / split
        self.dataset_root = Path(root)
        self.bidirectional = bidirectional
        self.include_metadata = include_metadata
        self.index_paths = sorted(self.root.glob("*.idx.npy"))
        if not self.index_paths:
            raise FileNotFoundError(f"No index shards found under {self.root}")
        self.indices = self._open_indices()
        self.record_metadata_indices = self._open_record_metadata_indices()
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
        self.pair_lengths = np.concatenate(lengths)
        self.pair_source_ids = np.concatenate(source_ids)
        self.pair_synthetic_flags = np.concatenate(synthetic_flags)
        self.forward_only_count = int(np.count_nonzero(np.concatenate(forward_only_flags)))
        self.source_names = self._load_source_names()
        self.synthetic_sampling_weight = self._load_synthetic_sampling_weight()
        self.language_pairs, self.languages = self._load_language_metadata()
        self.language_pair = self.language_pairs[0]
        self.source_only_languages = self._load_source_only_languages()
        self._token_cache: dict[tuple[int, str], np.memmap] = {}
        self._record_metadata_cache: dict[int, np.memmap] = {}
        self._bidirectional_pairs: np.ndarray | None = None
        self._forward_only_pairs: np.ndarray | None = None
        self._build_direction_maps()

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

    def __getstate__(self) -> dict:
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

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Pickles created before row sidecars predate the opt-in flag.
        self.include_metadata = bool(getattr(self, "include_metadata", False))
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
        manifest_path = self.dataset_root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            raw_pairs = manifest.get("language_pairs")
            if isinstance(raw_pairs, list) and raw_pairs:
                pairs = tuple(
                    (str(pair[0]), str(pair[1]))
                    for pair in raw_pairs
                    if isinstance(pair, list) and len(pair) == 2
                )
                raw_languages = manifest.get("languages")
                languages = (
                    tuple(map(str, raw_languages))
                    if isinstance(raw_languages, list)
                    else tuple(dict.fromkeys(language for pair in pairs for language in pair))
                )
                if pairs and languages:
                    return pairs, languages
            pair = manifest.get("language_pair")
            if isinstance(pair, list) and len(pair) == 2:
                normalized = (str(pair[0]), str(pair[1]))
                return (normalized,), normalized
        return (("ko", "ja"),), ("ko", "ja")

    def _load_source_names(self) -> list[str]:
        manifest_path = self.dataset_root / "manifest.json"
        if not manifest_path.exists():
            maximum = int(self.pair_source_ids.max(initial=0))
            return [f"source-{source_id}" for source_id in range(maximum + 1)]
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        sources = manifest.get("sources") or []
        if not sources:
            inputs = manifest.get("inputs") or []
            return [Path(path).name for path in inputs] or ["source-0"]
        maximum = max(int(source["id"]) for source in sources)
        names = [f"source-{source_id}" for source_id in range(maximum + 1)]
        for source in sources:
            names[int(source["id"])] = str(source["name"])
        return names

    def _load_source_only_languages(self) -> tuple[str, ...]:
        manifest_path = self.dataset_root / "manifest.json"
        if not manifest_path.exists():
            return ()
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        raw = manifest.get("source_only_languages")
        if not isinstance(raw, list):
            return ()
        return tuple(str(language) for language in raw)

    def _load_synthetic_sampling_weight(self) -> float:
        manifest_path = self.dataset_root / "manifest.json"
        if not manifest_path.exists():
            return DEFAULT_SYNTHETIC_SAMPLING_WEIGHT
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        policy = manifest.get("synthetic_policy") or {}
        return float(policy.get("sampling_weight", DEFAULT_SYNTHETIC_SAMPLING_WEIGHT))

    def __len__(self) -> int:
        if not self.bidirectional:
            return self.pair_count
        # Forward-only pairs contribute one direction instead of two.
        return 2 * self.pair_count - self.forward_only_count

    @property
    def direction_count(self) -> int:
        """Number of distinct (source, target) directions this split can yield."""

        forbidden = set(self.source_only_languages)
        return sum(
            1
            for pair in self.language_pairs
            for direction in (pair, (pair[1], pair[0]))
            if direction[1] not in forbidden
        )

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

    def metadata_at(self, index: int) -> dict[str, object]:
        """Return preserved raw-record annotations for one virtual sample."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pair_index = self._pair_index(index)
        shard, local = self._resolve(pair_index)
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

    def __getitem__(self, index: int) -> dict:
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
        item = {
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
        synthetic_flags = getattr(dataset, "pair_synthetic_flags", None)
        has_synthetic = synthetic_flags is not None and bool(np.asarray(synthetic_flags).any())
        self._balance_sources = (
            not math.isclose(source_sampling_alpha, 1.0)
            or any(
                not math.isclose(weight, 1.0) for weight in self.source_sampling_weights.values()
            )
            or (has_synthetic and not math.isclose(self.synthetic_sampling_weight, 1.0))
        )
        if self._balance_sources and not getattr(dataset, "has_source_metadata", True):
            raise ValueError(
                "Source-balanced sampling requires v2 shards with source_id metadata; "
                "re-run sion-prepare-data"
            )
        self._group_pair_indices: dict[int, np.ndarray] | None = None
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
        if self._group_codes is None or self._group_probabilities is None:
            source_ids_for_pairs = self.dataset.pair_source_ids.astype(np.uint32, copy=False)
            pair_synthetic_flags = getattr(self.dataset, "pair_synthetic_flags", None)
            if pair_synthetic_flags is None:
                pair_synthetic_flags = np.zeros(len(source_ids_for_pairs), dtype=np.bool_)
            synthetic_flags = np.asarray(pair_synthetic_flags, dtype=np.uint32)
            pair_group_codes = source_ids_for_pairs * np.uint32(2) + synthetic_flags
            source_counts = np.bincount(source_ids_for_pairs)
            synthetic_counts_by_source = np.bincount(
                source_ids_for_pairs,
                weights=synthetic_flags,
            )
            counts_all = np.bincount(pair_group_codes)
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
                if group_is_synthetic:
                    source_is_all_synthetic = (
                        synthetic_counts_by_source[source_id] == source_counts[source_id]
                    )
                    has_explicit_source_weight = name in self.source_sampling_weights
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
        group_codes = self._group_codes
        probabilities = self._group_probabilities
        assert self._group_pair_indices is not None
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
            sampled_pairs[positions] = rng.choice(candidates, size=len(positions), replace=True)
        if not self.dataset.bidirectional:
            return sampled_pairs
        directions = rng.integers(0, 2, size=sample_count, dtype=np.uint32)
        return self.dataset._virtual_indices_for_pairs(sampled_pairs, directions)

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
