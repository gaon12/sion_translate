#!/usr/bin/env python3
"""Filter Korean-Japanese JSONL with a pinned multilingual embedding model.

The filter is intended as a conservative second pass for deterministic
localization joins. It does not create alignments. It scores an already joined
pair and removes only low-similarity tails, while preserving an audit sample of
the rejected rows.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            if not isinstance(row.get("ko"), str) or not isinstance(row.get("ja"), str):
                raise ValueError(f"{path}:{line_number}: missing ko/ja strings")
            rows.append(row)
    return rows


def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        [f"query: {text}" for text in texts],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
    pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    return functional.normalize(pooled, p=2, dim=1).cpu()


def score_rows(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    revision: str,
    batch_size: int,
    max_length: int,
    device_name: str,
) -> list[float]:
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    model.eval()
    model.to(device)

    scores = [math.nan] * len(rows)
    ordered_indices = sorted(
        range(len(rows)),
        key=lambda index: max(len(str(rows[index]["ko"])), len(str(rows[index]["ja"]))),
    )
    for start in range(0, len(ordered_indices), batch_size):
        batch_indices = ordered_indices[start : start + batch_size]
        batch = [rows[index] for index in batch_indices]
        ko_embeddings = embed(
            [str(row["ko"]) for row in batch],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
        )
        ja_embeddings = embed(
            [str(row["ja"]) for row in batch],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
        )
        batch_scores = torch.sum(ko_embeddings * ja_embeddings, dim=1)
        for index, value in zip(batch_indices, batch_scores.tolist(), strict=True):
            scores[index] = float(value)
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(
                f"semantic score: {min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    if any(math.isnan(score) for score in scores):
        raise AssertionError("Not all rows received a semantic score")
    return scores


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def score_summary(scores: list[float]) -> dict[str, float]:
    ordered = sorted(scores)
    return {
        "minimum": min(scores),
        "p01": percentile(ordered, 0.01),
        "p05": percentile(ordered, 0.05),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "median": statistics.median(scores),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "maximum": max(scores),
        "mean": statistics.fmean(scores),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def filter_file(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    threshold: float,
    model_name: str,
    revision: str,
    batch_size: int,
    max_length: int,
    device: str,
    default_domain: str | None,
    source_name: str | None,
    source_revision: str | None,
) -> dict[str, Any]:
    rows = read_rows(input_path)
    input_digest = sha256_file(input_path)
    for index, row in enumerate(rows, start=1):
        resource_id = str(row.setdefault("resource_id", f"line:{index}"))
        if source_name:
            row.setdefault("source", source_name)
        if default_domain:
            row.setdefault("domain", default_domain)
        row.setdefault("document_id", input_path.stem)
        row.setdefault("family_id", f"{input_path.stem}:{resource_id}")
        row.setdefault("original_direction", "parallel_localization")
        row.setdefault("source_revision", source_revision or input_digest)
    scores = score_rows(
        rows,
        model_name=model_name,
        revision=revision,
        batch_size=batch_size,
        max_length=max_length,
        device_name=device,
    )
    scored = list(zip(rows, scores, strict=True))
    kept: list[dict[str, Any]] = []
    rejected: list[tuple[float, dict[str, Any]]] = []
    for row, score in scored:
        enriched = dict(row)
        enriched["semantic_similarity"] = round(score, 6)
        enriched["semantic_filter_model"] = f"{model_name}@{revision}"
        if score >= threshold:
            kept.append(enriched)
        else:
            rejected.append((score, enriched))

    count = write_jsonl(output_path, kept)
    output_digest = sha256_file(output_path)
    rejected.sort(key=lambda item: item[0])
    report = {
        "schema": "sion-semantic-pair-filter-v1",
        "input": str(input_path),
        "input_rows": len(rows),
        "input_sha256": input_digest,
        "output": str(output_path),
        "written_rows": count,
        "output_sha256": output_digest,
        "threshold": threshold,
        "removed_rows": len(rejected),
        "model": model_name,
        "model_revision": revision,
        "default_domain": default_domain,
        "source_name": source_name,
        "source_revision": source_revision or input_digest,
        "max_length": max_length,
        "score_summary": score_summary(scores),
        "lowest_rejected_or_scored": [
            {
                "score": score,
                "resource_id": row.get("resource_id"),
                "source": row.get("source"),
                "ko": row["ko"],
                "ja": row["ja"],
            }
            for row, score in sorted(scored, key=lambda item: item[1])[:100]
        ],
        "highest_rejected": [
            {
                "score": score,
                "resource_id": row.get("resource_id"),
                "source": row.get("source"),
                "ko": row["ko"],
                "ja": row["ja"],
            }
            for score, row in rejected[-100:]
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-domain")
    parser.add_argument("--source-name")
    parser.add_argument("--source-revision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = filter_file(
        args.input.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        threshold=args.threshold,
        model_name=args.model,
        revision=args.revision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        default_domain=args.default_domain,
        source_name=args.source_name,
        source_revision=args.source_revision,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
