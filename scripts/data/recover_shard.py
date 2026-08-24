#!/usr/bin/env python3
"""Recover an excluded shard instead of discarding the content it carries.

Several shards were excluded wholesale, but ``QualityPolicy()`` accepts 100% of
data23, data25 and data50. Their exclusions were undocumented manual judgements,
so the defects have to be re-established by measurement before the rows are
thrown away for good.

Two of the defects are repairable rather than fatal:

``spacing``
    The target side arrives morpheme-segmented (``甘い 香り が 鼻先 を``). That
    whitespace is a segmenter artifact, and ``collapse_spurious_spaces`` removes
    it by writing system, leaving space-using scripts untouched.

``fan-out``
    One source is joined to many targets - up to 60 in data50. Only one of them
    can be the translation. Picking the best one needs a similarity score, which
    this tool does not compute, so recovery runs in two stages.

Stage ``prepare`` does every deterministic repair and drop, and writes rows that
are ready to be scored::

    python scripts/data/recover_shard.py prepare \
        --source-key ko --target-key ja \
        --source-language ko --target-language ja \
        --source-scripts ko --target-scripts ja \
        --output data/staging/data50.prepared.jsonl \
        --report reports/recover-data50-prepare.json \
        data/excluded/llm_generated_20260730/data50.jsonl

Then score them with the pinned embedding model, which attaches
``semantic_similarity`` to every row::

    python scripts/data/filter_semantic_pairs.py --threshold 0.0 \
        --input data/staging/data50.prepared.jsonl \
        --output data/staging/data50.scored.jsonl \
        --report reports/recover-data50-scores.json

Stage ``select`` then resolves the fan-out and cuts the low-similarity tail::

    python scripts/data/recover_shard.py select \
        --source-key ko --target-key ja \
        --min-similarity 0.80 --unique-source \
        --output data/data54.jsonl \
        --report reports/recover-data50-select.json \
        data/staging/data50.scored.jsonl

Splitting it this way keeps the expensive step exactly once: ``prepare`` shrinks
the input before any embedding runs, and ``select`` can be re-run at a different
threshold without re-scoring.

Exit codes: 0 written, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence
import uuid

from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.function_morphemes import (
    placeholder_hole_markers,
    rejoin_orphan_particles,
)
from sion_translate.language_tags import canonicalize_language_tag
from sion_translate.scripts_registry import (
    collapse_spurious_spaces,
    has_foreign_script,
    resolve_scripts,
    spurious_space_count,
)


def _validate_parallel_keys(source_key: str, target_key: str) -> None:
    if (
        not source_key.strip()
        or not target_key.strip()
        or source_key != source_key.strip()
        or target_key != target_key.strip()
    ):
        raise ValueError(
            "source_key and target_key must be non-empty and have no surrounding whitespace"
        )
    if source_key == target_key:
        raise ValueError("source_key and target_key must be distinct")


def _canonical_language_direction(
    source_language: str,
    target_language: str,
) -> tuple[str, str]:
    source = canonicalize_language_tag(source_language, field="source_language")
    target = canonicalize_language_tag(target_language, field="target_language")
    if source == target:
        raise ValueError(
            "source_language and target_language must identify distinct BCP 47 languages"
        )
    return source, target


def _path_identity(path: Path) -> str:
    resolved = str(path.resolve(strict=False))
    return resolved.casefold() if sys.platform == "win32" else resolved


def _validate_output_path(path: Path) -> None:
    if path.exists() and not path.is_file():
        raise ValueError(f"output path is not a regular file: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {path.parent}")


def _validate_distinct_io(path: Path, output: Path) -> None:
    _validate_output_path(output)
    if _path_identity(path) == _path_identity(output):
        raise ValueError("input and output paths must be distinct")


@dataclass
class PrepareResult:
    path: str
    output: str
    rows_in: int = 0
    rows_out: int = 0
    spacing_repaired: int = 0
    spaces_removed: int = 0
    dropped_unparsable: int = 0
    dropped_missing_side: int = 0
    dropped_empty_after_repair: int = 0
    dropped_foreign_script: int = 0
    dropped_quality: int = 0
    dropped_duplicate_pair: int = 0
    dropped_placeholder_hole: int = 0
    dropped_isolated_spacing: int = 0
    particles_rejoined: int = 0
    particles_joined: int = 0
    min_space_density: float = 0.0
    quality_reasons: dict[str, int] = field(default_factory=dict[str, int])
    foreign_script_examples: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    spacing_examples: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    placeholder_hole_examples: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    isolated_spacing_examples: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    distinct_sources: int = 0
    max_targets_per_source: int = 0


@dataclass
class SelectResult:
    path: str
    output: str
    rows_in: int = 0
    rows_out: int = 0
    min_similarity: float = 0.0
    dropped_below_threshold: int = 0
    dropped_duplicate_source: int = 0
    dropped_duplicate_target: int = 0
    dropped_missing_score: int = 0
    dropped_over_fanout: int = 0
    sources_over_fanout: int = 0
    max_targets_per_source: int | None = None
    similarity_percentiles: dict[str, float] = field(default_factory=dict[str, float])
    dropped_examples: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    resolved_fanout_examples: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    over_fanout_examples: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def read_rows(path: Path) -> Iterable[tuple[int, dict[str, Any] | None]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield number, None
                continue
            yield number, row if isinstance(row, dict) else None


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path: Path, result: PrepareResult | SelectResult) -> None:
    _validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def rank_key(source: str, target: str, seed: str) -> str:
    digest = hashlib.blake2b(
        f"{seed}\x00{source}\x00{target}".encode("utf-8"),
        digest_size=16,
    )
    return digest.hexdigest()


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return round(ordered[round((len(ordered) - 1) * fraction)], 6)

    return {
        "minimum": at(0.0),
        "p05": at(0.05),
        "p10": at(0.10),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "maximum": at(1.0),
    }


def prepare_shard(
    path: Path,
    output: Path,
    *,
    source_key: str,
    target_key: str,
    source_scripts: Sequence[str],
    target_scripts: Sequence[str],
    repair_spacing: bool,
    policy: QualityPolicy,
    apply_quality: bool,
    source_language: str,
    target_language: str,
    min_space_density: float,
    rejoin_particles: bool,
) -> PrepareResult:
    _validate_parallel_keys(source_key, target_key)
    _validate_distinct_io(path, output)
    source_language, target_language = _canonical_language_direction(
        source_language,
        target_language,
    )
    result = PrepareResult(path=str(path), output=str(output))
    result.min_space_density = min_space_density
    permitted_source: frozenset[str] = (
        resolve_scripts(source_scripts) if source_scripts else frozenset()
    )
    permitted_target: frozenset[str] = (
        resolve_scripts(target_scripts) if target_scripts else frozenset()
    )

    kept: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    targets_per_source: Counter[str] = Counter()
    quality_reasons: Counter[str] = Counter()

    for _, row in read_rows(path):
        result.rows_in += 1
        if row is None:
            result.dropped_unparsable += 1
            continue
        source = row.get(source_key)
        target = row.get(target_key)
        if not isinstance(source, str) or not isinstance(target, str):
            result.dropped_missing_side += 1
            continue

        # A particle spaced off a host that is still present is a spacing slip,
        # not a deletion, so it is repaired rather than dropped. data37 has 629
        # of these and no deletions at all.
        if rejoin_particles:
            source, source_joined = rejoin_orphan_particles(source, source_language)
            target, target_joined = rejoin_orphan_particles(target, target_language)
            if source_joined or target_joined:
                result.particles_rejoined += 1
                result.particles_joined += source_joined + target_joined

        # A deleted name placeholder leaves the same hole on both sides, so no
        # similarity check can see it. Check before collapsing spaces: the
        # collapse would weld `、 の森` into `、の森` and hide the evidence.
        holes = placeholder_hole_markers(source, source_language) + placeholder_hole_markers(
            target, target_language
        )
        if holes:
            result.dropped_placeholder_hole += 1
            if len(result.placeholder_hole_examples) < 8:
                result.placeholder_hole_examples.append(
                    {
                        "markers": " ".join(holes),
                        source_key: source[:90],
                        target_key: target[:90],
                    }
                )
            continue

        # Segmenter spacing marks every boundary, so its density is high. One or
        # two isolated spaces in otherwise unsegmented text are ambiguous: they
        # may be interpolation artifacts or an undetected hole. Drop those rather
        # than guess.
        if repair_spacing:
            source_removed = spurious_space_count(source)
            target_removed = spurious_space_count(target)
            removed = source_removed + target_removed
            if removed:
                # Per side, not pooled. Only one side is usually segmented, and
                # pooling halves its density and pushes a segmented row under the
                # floor. The segmented side is the one that has to clear it.
                density = max(
                    source_removed / max(len(source), 1),
                    target_removed / max(len(target), 1),
                )
                if density < min_space_density:
                    result.dropped_isolated_spacing += 1
                    if len(result.isolated_spacing_examples) < 8:
                        result.isolated_spacing_examples.append(
                            {
                                "density": round(density, 5),
                                source_key: source[:90],
                                target_key: target[:90],
                            }
                        )
                    continue
                if len(result.spacing_examples) < 8:
                    result.spacing_examples.append({"before": target[:90]})
                source = collapse_spurious_spaces(source)
                target = collapse_spurious_spaces(target)
                if result.spacing_examples and "after" not in result.spacing_examples[-1]:
                    result.spacing_examples[-1]["after"] = target[:90]
                result.spacing_repaired += 1
                result.spaces_removed += removed

        source = source.strip()
        target = target.strip()
        if not source or not target:
            result.dropped_empty_after_repair += 1
            continue

        if (permitted_source and has_foreign_script(source, permitted_source)) or (
            permitted_target and has_foreign_script(target, permitted_target)
        ):
            result.dropped_foreign_script += 1
            if len(result.foreign_script_examples) < 8:
                result.foreign_script_examples.append(
                    {source_key: source[:90], target_key: target[:90]}
                )
            continue

        if apply_quality:
            assessment = assess_pair(
                source,
                target,
                policy=policy,
                languages=(source_language, target_language),
            )
            if not assessment.accepted:
                result.dropped_quality += 1
                for reason in assessment.rejection_reasons:
                    quality_reasons[reason] += 1
                continue

        key = (canonical_text(source), canonical_text(target))
        if key in seen_pairs:
            result.dropped_duplicate_pair += 1
            continue
        seen_pairs.add(key)

        repaired = dict(row)
        repaired[source_key] = source
        repaired[target_key] = target
        kept.append(repaired)
        targets_per_source[canonical_text(source)] += 1

    result.quality_reasons = dict(quality_reasons.most_common())
    result.distinct_sources = len(targets_per_source)
    result.max_targets_per_source = max(targets_per_source.values(), default=0)
    result.rows_out = len(kept)
    write_rows(output, kept)
    return result


def select_alignments(
    path: Path,
    output: Path,
    *,
    source_key: str,
    target_key: str,
    score_key: str,
    min_similarity: float,
    unique_source: bool,
    unique_target: bool,
    max_targets_per_source: int | None,
    seed: str,
) -> SelectResult:
    _validate_parallel_keys(source_key, target_key)
    _validate_distinct_io(path, output)
    result = SelectResult(path=str(path), output=str(output), min_similarity=min_similarity)
    result.max_targets_per_source = max_targets_per_source

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for _, row in read_rows(path):
        result.rows_in += 1
        if row is None:
            result.dropped_missing_score += 1
            continue
        score = row.get(score_key)
        if not isinstance(score, (int, float)):
            result.dropped_missing_score += 1
            continue
        rows.append(row)
        scores.append(float(score))

    result.similarity_percentiles = percentiles(scores)

    # Some fan-out is not one translation among several renderings but a
    # cross product: in data50 all 60 targets of a source share a prefix that
    # has nothing to do with the source. Picking a winner there manufactures a
    # pair. Sources over the cap are discarded whole instead.
    if max_targets_per_source is not None:
        candidates: Counter[str] = Counter()
        for row in rows:
            candidates[canonical_text(str(row.get(source_key, "")))] += 1
        over_cap = {
            source for source, count in candidates.items() if count > max_targets_per_source
        }
        if over_cap:
            retained: list[dict[str, Any]] = []
            for row in rows:
                source = canonical_text(str(row.get(source_key, "")))
                if source in over_cap:
                    result.dropped_over_fanout += 1
                    if len(result.over_fanout_examples) < 8:
                        result.over_fanout_examples.append(
                            {
                                "candidates": candidates[source],
                                source_key: str(row.get(source_key, ""))[:70],
                                target_key: str(row.get(target_key, ""))[:70],
                            }
                        )
                    continue
                retained.append(row)
            rows = retained
            result.sources_over_fanout = len(over_cap)

    above: list[dict[str, Any]] = []
    for row in rows:
        if float(row[score_key]) < min_similarity:
            result.dropped_below_threshold += 1
            if len(result.dropped_examples) < 12:
                result.dropped_examples.append(
                    {
                        "score": round(float(row[score_key]), 6),
                        source_key: str(row.get(source_key, ""))[:80],
                        target_key: str(row.get(target_key, ""))[:80],
                    }
                )
            continue
        above.append(row)

    # Highest score wins the source, with a seeded hash as a deterministic
    # tie-break so file order never decides the corpus.
    above.sort(
        key=lambda row: (
            -float(row[score_key]),
            rank_key(str(row.get(source_key, "")), str(row.get(target_key, "")), seed),
        )
    )

    fanout: Counter[str] = Counter()
    for row in above:
        fanout[canonical_text(str(row.get(source_key, "")))] += 1

    chosen: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    used_targets: set[str] = set()
    for row in above:
        source = canonical_text(str(row.get(source_key, "")))
        target = canonical_text(str(row.get(target_key, "")))
        if unique_source and source in used_sources:
            result.dropped_duplicate_source += 1
            continue
        if unique_target and target in used_targets:
            result.dropped_duplicate_target += 1
            continue
        if unique_source and fanout[source] > 1 and len(result.resolved_fanout_examples) < 8:
            result.resolved_fanout_examples.append(
                {
                    "candidates": fanout[source],
                    "score": round(float(row[score_key]), 6),
                    source_key: str(row.get(source_key, ""))[:80],
                    target_key: str(row.get(target_key, ""))[:80],
                }
            )
        used_sources.add(source)
        used_targets.add(target)
        chosen.append(row)

    result.rows_out = len(chosen)
    write_rows(output, chosen)
    return result


def script_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    resolve_scripts(names)
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", type=Path)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--report", type=Path)
    common.add_argument("--source-key", required=True)
    common.add_argument("--target-key", required=True)

    prepare = subparsers.add_parser("prepare", parents=[common], help="deterministic repairs")
    prepare.add_argument("--source-scripts", type=script_list, default=[])
    prepare.add_argument("--target-scripts", type=script_list, default=[])
    prepare.add_argument(
        "--source-language",
        required=True,
        help="language tag used for stranded-particle detection; unknown tags are not checked",
    )
    prepare.add_argument("--target-language", required=True)
    prepare.add_argument(
        "--min-space-density",
        type=float,
        default=0.08,
        help=(
            "spurious spaces per character below which spacing counts as isolated "
            "rather than segmented. Segmented rows are repaired; isolated ones are "
            "dropped, because an isolated space may be an undetected placeholder hole. "
            "Measured on data50: segmented median 0.14, isolated median 0.02."
        ),
    )
    prepare.add_argument("--no-repair-spacing", action="store_true")
    prepare.add_argument("--no-rejoin-particles", action="store_true")
    prepare.add_argument("--no-quality-filter", action="store_true")

    select = subparsers.add_parser("select", parents=[common], help="resolve fan-out by score")
    select.add_argument("--min-similarity", type=float, required=True)
    select.add_argument("--score-key", default="semantic_similarity")
    select.add_argument("--unique-source", action="store_true")
    select.add_argument(
        "--max-targets-per-source",
        type=int,
        help=(
            "discard a source entirely when it has more than N candidate targets. "
            "Use 1 to keep only unambiguous joins. Unset resolves fan-out by score "
            "instead, which is only right when the candidates really are alternative "
            "renderings of the same source."
        ),
    )
    select.add_argument("--unique-target", action="store_true")
    select.add_argument("--seed", default="sion-recover-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_parallel_keys(args.source_key, args.target_key)
        if args.stage == "prepare":
            _canonical_language_direction(args.source_language, args.target_language)
    except ValueError as error:
        print(f"invalid parallel identity: {error}", file=sys.stderr)
        return 2
    path = args.input
    if not path.is_file():
        print(f"{path}: not a file", file=sys.stderr)
        return 2
    try:
        _validate_distinct_io(path, args.output)
        if args.report:
            _validate_output_path(args.report)
            report_identity = _path_identity(args.report)
            if report_identity in {_path_identity(path), _path_identity(args.output)}:
                raise ValueError("report path must be distinct from input and output paths")
    except (OSError, ValueError) as error:
        print(f"invalid output identity: {error}", file=sys.stderr)
        return 2

    try:
        if args.stage == "prepare":
            result: PrepareResult | SelectResult = prepare_shard(
                path,
                args.output,
                source_key=args.source_key,
                target_key=args.target_key,
                source_scripts=args.source_scripts,
                target_scripts=args.target_scripts,
                repair_spacing=not args.no_repair_spacing,
                policy=QualityPolicy(),
                apply_quality=not args.no_quality_filter,
                source_language=args.source_language,
                target_language=args.target_language,
                min_space_density=args.min_space_density,
                rejoin_particles=not args.no_rejoin_particles,
            )
            print(
                f"{path.name:34} {result.rows_in:>8,} -> {result.rows_out:>8,} rows  "
                f"spacing={result.spacing_repaired:,} ({result.spaces_removed:,} spaces)  "
                f"rejoined={result.particles_rejoined:,} "
                f"hole={result.dropped_placeholder_hole:,} "
                f"isolated={result.dropped_isolated_spacing:,} "
                f"script={result.dropped_foreign_script:,} "
                f"quality={result.dropped_quality:,} "
                f"dup={result.dropped_duplicate_pair:,}  "
                f"sources={result.distinct_sources:,} "
                f"maxFanout={result.max_targets_per_source}"
            )
        else:
            result = select_alignments(
                path,
                args.output,
                source_key=args.source_key,
                target_key=args.target_key,
                score_key=args.score_key,
                min_similarity=args.min_similarity,
                unique_source=args.unique_source,
                unique_target=args.unique_target,
                max_targets_per_source=args.max_targets_per_source,
                seed=args.seed,
            )
            print(
                f"{path.name:34} {result.rows_in:>8,} -> {result.rows_out:>8,} rows  "
                f"overFanout={result.dropped_over_fanout:,} "
                f"belowThreshold={result.dropped_below_threshold:,} "
                f"dupSource={result.dropped_duplicate_source:,} "
                f"dupTarget={result.dropped_duplicate_target:,}"
            )
            for name, value in result.similarity_percentiles.items():
                print(f"      {name:8} {value:.4f}")
    except (OSError, ValueError) as error:
        print(f"{path}: cannot recover ({error})", file=sys.stderr)
        return 2

    if args.report:
        try:
            write_report(args.report, result)
        except OSError as error:
            print(f"cannot write report ({error})", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
