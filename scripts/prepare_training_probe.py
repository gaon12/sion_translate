"""Build a small, authenticated real-data cohort for bounded GPU measurements.

This command never initializes CUDA or changes prepared training artifacts.
It authenticates both complete indexed inventories, uses their full token mass
to resolve architecture, and copies only sampled rows and tokenizer assets.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from sion_translate.auto import apply_auto_data_settings
from sion_translate.cli.train import _foundation_source_sampling_weights
from sion_translate.config import AppConfig, config_from_raw, load_raw_config
from sion_translate.data.indexed import DistributedBucketBatchSampler, IndexedParallelDataset
from sion_translate.fingerprint import file_sha256
from sion_translate.foundation import build_foundation_config
from sion_translate.locking import artifact_lock
from sion_translate.tokenizer import SionTokenizer


SCHEMA = "sion-training-probe-data-v1"
TOKENIZER_FILES = ("sion.model", "sion.vocab", "token_features.npz", "tokenizer_metadata.json")


def json_value(value: Any) -> Any:
    """Preserve token IDs and metadata while removing NumPy-only containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def bounded_representative_indices(sampler: DistributedBucketBatchSampler, count: int) -> list[int]:
    """Draw the production distribution without materializing a full epoch.

    Balanced sampling delegates to the production sampler's weighted draw.
    The draw is distribution-equivalent, not an exact full-epoch RNG prefix.
    Four independently populated length buckets provide short and long batches.
    """
    if count < 1:
        raise ValueError("representative count must be positive")
    dataset = sampler.dataset
    rng = np.random.default_rng(sampler.seed + sampler.epoch)
    draw_count = max(count, min(4 * sampler.bucket_size, count * 4))
    if sampler._balance_sources:
        indices = sampler._balanced_indices(rng, draw_count)
    else:
        draw_count = min(draw_count, len(dataset))
        indices = rng.choice(len(dataset), size=draw_count, replace=False)
    if len(indices) < count:
        raise ValueError("the dataset is smaller than the requested representative cohort")
    ordered = []
    for start in range(0, len(indices), sampler.bucket_size):
        bucket = indices[start : start + sampler.bucket_size]
        lengths = dataset.lengths_for_indices(bucket)
        ordered.extend(bucket[np.argsort(lengths, kind="stable")].tolist())
    positions = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    return [int(ordered[int(position)]) for position in positions]


def stress_indices(sampler: DistributedBucketBatchSampler, count: int) -> list[int]:
    """Find the longest eligible physical rows with bounded scratch memory."""
    if count < 1:
        raise ValueError("stress count must be positive")
    dataset = sampler.dataset
    if dataset.pair_lengths is None:
        raise ValueError("stress sampling requires parent-process indexed lengths")
    eligible = sampler.positive_sampling_pair_mask()
    best_indices = np.empty(0, dtype=np.int64)
    best_lengths = np.empty(0, dtype=np.uint32)
    for start in range(0, dataset.pair_count, 65_536):
        stop = min(start + 65_536, dataset.pair_count)
        local = np.flatnonzero(eligible[start:stop]) + start
        lengths = dataset.pair_lengths[local]
        merged_indices = np.concatenate((best_indices, local))
        merged_lengths = np.concatenate((best_lengths, lengths))
        keep = min(count, len(merged_indices))
        positions = np.argsort(merged_lengths, kind="stable")[-keep:] if keep else []
        best_indices = merged_indices[positions]
        best_lengths = merged_lengths[positions]
    if len(best_indices) < count:
        raise ValueError("not enough eligible physical rows for the stress cohort")
    best_indices = best_indices[np.argsort(-best_lengths.astype(np.int64), kind="stable")]
    if dataset.bidirectional:
        directions = np.arange(len(best_indices), dtype=np.uint32) % 2
        return dataset._virtual_indices_for_pairs(best_indices, directions).astype(int).tolist()
    return best_indices.astype(int).tolist()


def length_statistics(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("src", "tgt"):
        values = np.asarray([len(item[name]) for item in items], dtype=np.int64)
        result[name] = {
            "min": int(values.min()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
            "max": int(values.max()),
            "mean": float(values.mean()),
        }
    directions: dict[str, int] = {}
    for item in items:
        key = f"{item['src_language']}->{item['target_language']}"
        directions[key] = directions.get(key, 0) + 1
    result["directions"] = directions
    return result


def stage_cohort(
    dataset: IndexedParallelDataset,
    sampler: DistributedBucketBatchSampler,
    config: AppConfig,
    representative_count: int,
    stress_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "examples_per_epoch": len(dataset),
        "physical_pairs": dataset.pair_count,
        "physical_tokens": dataset.physical_token_count,
        "epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps,
        "batch_size_per_gpu": config.training.batch_size_per_gpu,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "drop_last": sampler.drop_last,
        "sampling_method": "bounded production distribution; not exact full-epoch RNG prefix",
        "artifact_inventory_sha256": dataset.artifact_inventory_sha256,
    }
    for label, indices in (
        ("representative", bounded_representative_indices(sampler, representative_count)),
        ("stress", stress_indices(sampler, stress_count)),
    ):
        items = []
        for index in indices:
            item = json_value(dataset[index])
            item["probe_virtual_index"] = index
            items.append(item)
        result[label] = items
        result[f"{label}_lengths"] = length_statistics(items)
        result[f"{label}_unique_physical_rows"] = len({item["pair_index"] for item in items})
    return result


def _sampler(dataset: IndexedParallelDataset, config: AppConfig, foundation: bool):
    options: dict[str, Any] = (
        {
            "source_sampling_weights_by_id": _foundation_source_sampling_weights(dataset),
            "max_source_upsampling": math.inf,
        }
        if foundation
        else {
            "source_sampling_alpha": config.data.source_sampling_alpha,
            "source_sampling_weights": config.data.source_sampling_weights,
            "max_source_upsampling": config.data.max_source_upsampling,
            "language_pair_sampling_alpha": config.data.language_pair_sampling_alpha,
        }
    )
    return DistributedBucketBatchSampler(
        dataset,
        config.training.batch_size_per_gpu,
        bucket_size=config.data.bucket_size,
        seed=config.training.seed,
        **options,
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": file_sha256(path), "size": path.stat().st_size}


def _snapshot(dataset_root: Path) -> list[tuple[str, int, int]]:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [manifest_path] + [
        dataset_root / entry["path"] for entry in manifest["artifact_inventory"]["files"]
    ]
    return [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def verify_dataset_tokenizer(dataset_root: Path, tokenizer_sha256: str) -> None:
    """Reject individually valid assets that belong to different tokenizers."""
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest.get("fingerprint", {})
    recorded = manifest.get("tokenizer_sha256", fingerprint.get("tokenizer_sha256"))
    if recorded != tokenizer_sha256:
        raise ValueError(f"indexed dataset tokenizer SHA-256 mismatch: {dataset_root}")


def prepare(
    config_path: Path, output: Path, *, representative_count: int = 4096, stress_count: int = 128
) -> dict[str, Any]:
    if not 2048 <= representative_count <= 4096 or not 128 <= stress_count <= 512:
        raise ValueError("cohort sizes must be representative=2048..4096 and stress=128..512")
    config_path = config_path.resolve(strict=True)
    output = output.resolve()
    if any(part.casefold().startswith("onedrive") for part in output.parts):
        raise ValueError("probe output must be outside OneDrive")
    if output.exists():
        raise FileExistsError(f"probe output must be a new directory: {output}")
    raw = load_raw_config(config_path)
    config = config_from_raw(raw)
    root = config_path.parent
    tokenizer_root = (root / config.data.tokenizer_model).resolve().parent
    translation_root = (root / config.data.dataset_dir).resolve()
    foundation_root = (root / config.foundation.dataset_dir).resolve()
    started = time.time()
    output.mkdir(parents=True)
    try:
        with ExitStack() as scope:
            # Indexed cohorts require a stable source throughout extraction.
            # Tokenizer assets are copied read-only and authenticated before and
            # after copying, without acquiring a writer's tokenizer lease.
            for directory in sorted({translation_root, foundation_root}):
                scope.enter_context(artifact_lock(directory))
            before = {str(path): _snapshot(path) for path in (translation_root, foundation_root)}
            token_records = {name: _file_record(tokenizer_root / name) for name in TOKENIZER_FILES}
            metadata = json.loads((tokenizer_root / "tokenizer_metadata.json").read_text("utf-8"))
            for name, key in (
                ("sion.model", "model_sha256"),
                ("sion.vocab", "vocab_sha256"),
                ("token_features.npz", "token_features_sha256"),
            ):
                if metadata.get(key) != token_records[name]["sha256"]:
                    raise ValueError(f"tokenizer metadata SHA-256 mismatch: {name}")
            for dataset_root in (translation_root, foundation_root):
                verify_dataset_tokenizer(dataset_root, token_records["sion.model"]["sha256"])
            tokenizer = SionTokenizer(tokenizer_root / "sion.model")
            config.model.vocab_size = len(tokenizer)
            print("Authenticating the complete translation inventory.", flush=True)
            translation = IndexedParallelDataset(
                translation_root,
                config.data.train_split,
                bidirectional=config.data.bidirectional,
                legacy_language_pairs=config.data.configured_language_pairs(),
                include_metadata=True,
                verify_integrity=True,
            )
            decisions = apply_auto_data_settings(
                config,
                raw,
                train_examples=len(translation),
                physical_train_pairs=translation.pair_count,
                physical_train_tokens=translation.physical_token_count,
                source_names=translation.source_names,
            )
            config.validate()
            foundation_config = build_foundation_config(config)
            print("Authenticating the complete foundation inventory.", flush=True)
            foundation = IndexedParallelDataset(
                foundation_root,
                foundation_config.data.train_split,
                bidirectional=foundation_config.data.bidirectional,
                legacy_language_pairs=foundation_config.data.configured_language_pairs(),
                include_metadata=True,
                verify_integrity=True,
            )
            print(
                "Drawing bounded real-data cohorts using the production sampling policies.",
                flush=True,
            )
            stages = {}
            for name, dataset, stage_config in (
                ("foundation", foundation, foundation_config),
                ("sft", translation, config),
            ):
                stages[name] = stage_cohort(
                    dataset,
                    _sampler(dataset, stage_config, name == "foundation"),
                    stage_config,
                    representative_count,
                    stress_count,
                )
            tokenizer_output = output / "tokenizer"
            tokenizer_output.mkdir()
            for name in TOKENIZER_FILES:
                shutil.copyfile(tokenizer_root / name, tokenizer_output / name)
                if _file_record(tokenizer_output / name) != token_records[name]:
                    raise ValueError(f"copied tokenizer asset changed: {name}")
                if _file_record(tokenizer_root / name) != token_records[name]:
                    raise ValueError(f"source tokenizer asset changed: {name}")
            for path in (translation_root, foundation_root):
                if _snapshot(path) != before[str(path)]:
                    raise RuntimeError("prepared dataset changed during cohort extraction")
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            plan = {
                "schema": SCHEMA,
                "config": asdict(config),
                "raw_config": raw,
                "auto_data_decisions": decisions,
                "stages": stages,
                "provenance": {
                    "source_commit": commit,
                    "config_path": str(config_path),
                    "config_file": _file_record(config_path),
                    "source_manifests": {
                        name: {"root": str(path), **_file_record(path / "manifest.json")}
                        for name, path in (
                            ("foundation", foundation_root),
                            ("sft", translation_root),
                        )
                    },
                    "tokenizer_source": str(tokenizer_root),
                    "complete_indexed_inventories_verified": True,
                    "raw_sources_rehashed": False,
                    "started_unix": started,
                    "elapsed_seconds": time.time() - started,
                    "purpose": "runtime and memory measurements, not quality or full-training proof",
                },
                "files": {f"tokenizer/{name}": token_records[name] for name in TOKENIZER_FILES},
            }
            (output / "plan.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            evidence = {
                "schema": SCHEMA,
                "status": "succeeded",
                "plan": _file_record(output / "plan.json"),
                "output": str(output),
            }
            (output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n", "utf-8")
            return evidence
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps({"status": "failed", "error": repr(error)}, indent=2) + "\n", "utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("sion_translate.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representative-count", type=int, default=4096)
    parser.add_argument("--stress-count", type=int, default=128)
    args = parser.parse_args()
    result = prepare(
        args.config,
        args.output,
        representative_count=args.representative_count,
        stress_count=args.stress_count,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
