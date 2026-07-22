from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
from torch.utils.data import Dataset, Sampler


class IndexedParallelDataset(Dataset):
    def __init__(self, root: str | Path, split: str, *, bidirectional: bool = True):
        self.root = Path(root) / split
        self.dataset_root = Path(root)
        self.bidirectional = bidirectional
        self.index_paths = sorted(self.root.glob("*.idx.npy"))
        if not self.index_paths:
            raise FileNotFoundError(f"No index shards found under {self.root}")
        self.indices = self._open_indices()
        self.cumulative: list[int] = []
        total = 0
        lengths: list[np.ndarray] = []
        source_ids: list[np.ndarray] = []
        self.has_source_metadata = True
        for index in self.indices:
            total += len(index)
            self.cumulative.append(total)
            lengths.append(index["ko_length"].astype(np.uint32) + index["ja_length"].astype(np.uint32))
            if index.dtype.names is not None and "source_id" in index.dtype.names:
                source_ids.append(index["source_id"].astype(np.uint16))
            else:
                self.has_source_metadata = False
                source_ids.append(np.zeros(len(index), dtype=np.uint16))
        self.pair_count = total
        self.pair_lengths = np.concatenate(lengths)
        self.pair_source_ids = np.concatenate(source_ids)
        self.source_names = self._load_source_names()
        # 언어쌍 이름 (예: ("ko","ja") / ("en","de")) — 토큰 bin 파일 이름과
        # 방향 태그가 이 이름을 따릅니다. manifest 에 없으면 ko/ja 로 간주.
        self.language_pair = self._load_language_pair()
        self._token_cache: dict[tuple[int, str], np.memmap] = {}

    def _open_indices(self) -> list[np.ndarray]:
        return [
            np.load(path, mmap_mode="r", allow_pickle=False)
            for path in self.index_paths
        ]

    def __getstate__(self) -> dict:
        """Keep Windows spawn workers from serializing hundreds of MB of memmaps."""

        state = self.__dict__.copy()
        state["indices"] = None
        state["pair_lengths"] = None
        state["pair_source_ids"] = None
        state["_token_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.indices = self._open_indices()
        self._token_cache = {}

    def _load_language_pair(self) -> tuple[str, str]:
        manifest_path = self.dataset_root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            pair = manifest.get("language_pair")
            if isinstance(pair, list) and len(pair) == 2:
                return str(pair[0]), str(pair[1])
        return ("ko", "ja")

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

    def __len__(self) -> int:
        return self.pair_count * (2 if self.bidirectional else 1)

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

    def length_at(self, index: int) -> int:
        pair_index = index // 2 if self.bidirectional else index
        if self.pair_lengths is not None:
            return int(self.pair_lengths[pair_index]) + 4
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]
        return int(row["ko_length"] + row["ja_length"]) + 4

    def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.pair_lengths is None:
            raise RuntimeError("Length metadata is unavailable inside a DataLoader worker")
        pair_indices = indices // 2 if self.bidirectional else indices
        return self.pair_lengths[pair_indices]

    def source_id_at(self, index: int) -> int:
        pair_index = index // 2 if self.bidirectional else index
        if self.pair_source_ids is not None:
            return int(self.pair_source_ids[pair_index])
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]
        if row.dtype.names is not None and "source_id" in row.dtype.names:
            return int(row["source_id"])
        return 0

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.bidirectional:
            pair_index, direction = divmod(index, 2)
        else:
            pair_index, direction = index, 0
        shard, local = self._resolve(pair_index)
        row = self.indices[shard][local]

        # 인덱스 필드 이름의 ko_/ja_ 는 '첫 번째/두 번째 언어'라는 내부 명칭이며,
        # 실제 언어 이름은 self.language_pair 를 따릅니다.
        language_a, language_b = self.language_pair
        ko_store = self._tokens(shard, language_a)
        ja_store = self._tokens(shard, language_b)
        ko_start, ko_length = int(row["ko_offset"]), int(row["ko_length"])
        ja_start, ja_length = int(row["ja_offset"]), int(row["ja_length"])
        ko = np.asarray(ko_store[ko_start : ko_start + ko_length], dtype=np.int64)
        ja = np.asarray(ja_store[ja_start : ja_start + ja_length], dtype=np.int64)

        if direction == 0:
            src, tgt = ko, ja
            src_language, target_language = language_a, language_b
            src_register, target_register = int(row["ko_register"]), int(row["ja_register"])
        else:
            src, tgt = ja, ko
            src_language, target_language = language_b, language_a
            src_register, target_register = int(row["ja_register"]), int(row["ko_register"])
        return {
            "src": src,
            "tgt": tgt,
            "src_language": src_language,
            "target_language": target_language,
            "src_register": src_register,
            "target_register": target_register,
            "pair_index": pair_index,
        }


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
        known_sources = set(dataset.source_names)
        unknown_sources = set(self.source_sampling_weights) - known_sources
        if unknown_sources:
            raise ValueError(
                f"Unknown source_sampling_weights keys: {sorted(unknown_sources)}"
            )
        if any(weight < 0 for weight in self.source_sampling_weights.values()):
            raise ValueError("source sampling weights must be non-negative")
        self._balance_sources = (
            not math.isclose(source_sampling_alpha, 1.0)
            or any(
                not math.isclose(weight, 1.0)
                for weight in self.source_sampling_weights.values()
            )
        )
        if self._balance_sources and not getattr(dataset, "has_source_metadata", True):
            raise ValueError(
                "Source-balanced sampling requires v2 shards with source_id metadata; "
                "re-run kjx-prepare-data"
        )
        self._source_pair_indices: dict[int, np.ndarray] | None = None
        self._source_ids: np.ndarray | None = None
        self._source_probabilities: np.ndarray | None = None
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

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

    def _balanced_indices(
        self, rng: np.random.Generator, sample_count: int
    ) -> np.ndarray:
        if self.dataset.pair_source_ids is None:
            raise RuntimeError("Source metadata is unavailable for balanced sampling")
        if self._source_ids is None or self._source_probabilities is None:
            counts_all = np.bincount(self.dataset.pair_source_ids)
            source_ids = np.flatnonzero(counts_all).astype(np.uint16, copy=False)
            counts = counts_all[source_ids].astype(np.float64)
            natural = counts / counts.sum()
            raw = np.power(counts, self.source_sampling_alpha)
            for position, source_id in enumerate(source_ids):
                name = (
                    self.dataset.source_names[int(source_id)]
                    if int(source_id) < len(self.dataset.source_names)
                    else f"source-{int(source_id)}"
                )
                raw[position] *= self.source_sampling_weights.get(name, 1.0)
            self._source_ids = source_ids
            self._source_probabilities = self._cap_source_probabilities(
                raw, natural, self.max_source_upsampling
            )
            self._source_pair_indices = {
                int(source_id): np.flatnonzero(
                    self.dataset.pair_source_ids == source_id
                ).astype(np.uint32, copy=False)
                for source_id in source_ids
            }
        source_ids = self._source_ids
        probabilities = self._source_probabilities
        assert self._source_pair_indices is not None
        sampled_sources = rng.choice(
            source_ids,
            size=sample_count,
            replace=True,
            p=probabilities,
        )
        sampled_pairs = np.empty(sample_count, dtype=np.uint32)
        for source_id in source_ids:
            positions = np.flatnonzero(sampled_sources == source_id)
            candidates = self._source_pair_indices[int(source_id)]
            sampled_pairs[positions] = rng.choice(
                candidates, size=len(positions), replace=True
            )
        if not self.dataset.bidirectional:
            return sampled_pairs
        directions = rng.integers(0, 2, size=sample_count, dtype=np.uint32)
        return sampled_pairs * np.uint32(2) + directions

    def __iter__(self) -> Iterator[list[int]]:
        if self._balance_sources:
            sample_count = len(self) * self.batch_size
            if sample_count == 0:
                return
            # Balanced sampling is with replacement, so each rank can generate
            # its local stream directly instead of materializing a global epoch.
            rng = np.random.default_rng(
                self.seed + self.epoch + self.rank * 0x9E3779B1
            )
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
        for start in range(0, len(local), self.batch_size):
            batch = local[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch.tolist()
