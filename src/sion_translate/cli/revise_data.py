"""Build training data for iterative draft revision.

    sion-revise-data --input "data/*.jsonl" --output data/revise_synthetic.jsonl

The command corrupts reference translations with observed error patterns to
create drafts, then writes ``source <draft> draft -> reference translation``
examples. It does not require a trained model and can run before pretraining.

To use actual output from a trained model, pass draft JSONL through ``--drafts``
with one ``{"draft": ...}`` object per input row in the same order. This better
matches the inference distribution, but it requires a model first.
"""

# CLI result serializers are attached dynamically.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.concat import read_records
from sion_translate.console import configure_stdio
from sion_translate.data.prepare import DEFAULT_TRAIN_ONLY_PREFIXES
from sion_translate.language_tags import canonicalize_language_pair
from sion_translate.revision import (
    DEFAULT_CORRUPTIONS,
    RevisionExample,
    RevisionStats,
    build_revision_examples,
    serialize_revision_input,
    write_revision_examples,
)
from sion_translate.tokenizer import expand_inputs


# Above this ratio, most corruption attempts had no effect, so emit a warning.
UNCHANGED_WARNING_RATIO = 0.40


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build source+draft -> target revision training data"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or glob patterns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Maximum number of pairs to use (default: all)",
    )
    parser.add_argument(
        "--drafts",
        help=(
            'Model-generated draft JSONL (one {"draft": ...} object per line in '
            "input order). When provided, use these drafts instead of synthetic "
            "corruption."
        ),
    )
    for kind, default in DEFAULT_CORRUPTIONS.items():
        parser.add_argument(
            f"--weight-{kind.replace('_', '-')}",
            type=float,
            default=default,
            dest=f"weight_{kind}",
            help=f"Relative weight for {kind} corruption (default: {default})",
        )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        required=True,
        metavar=("LANG_A", "LANG_B"),
        help="JSONL language keys and revision direction (SOURCE TARGET)",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def _load_drafts(path: str | Path) -> list[str]:
    drafts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            draft = row.get("draft")
            if not isinstance(draft, str) or not draft.strip():
                raise SystemExit(f"{path}:{number}: 'draft' must be a non-empty string")
            drafts.append(draft.strip())
    return drafts


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"Could not find any input JSONL files: {args.input}")
    language_pair = canonicalize_language_pair(
        args.language_pair,
        field="revision CLI language_pair",
    )
    records = list(read_records(paths, language_pair))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No readable translation pairs were found.")
    incompatible = [
        record.source_identifier
        for record in records
        if record.metadata.get("training_direction") is not None
        and record.metadata.get("training_direction") != list(language_pair)
    ]
    if incompatible:
        raise SystemExit(
            "An input training_direction differs from the requested revision direction: "
            f"requested={language_pair!r}, first={incompatible[0]}"
        )
    pairs = [(record.text_a, record.text_b) for record in records]

    if args.drafts:
        drafts = _load_drafts(args.drafts)
        if len(drafts) != len(pairs):
            raise SystemExit(
                f"The {len(drafts)} drafts do not match the {len(pairs)} translation "
                "pairs; both inputs must have the same count and order."
            )
        raw_examples = [
            (serialize_revision_input(source, draft), target)
            for (source, target), draft in zip(pairs, drafts, strict=True)
        ]
        unchanged = sum(
            1 for (_, target), draft in zip(pairs, drafts, strict=True) if draft == target
        )
        stats = RevisionStats(
            len(raw_examples),
            {"model_draft": len(raw_examples)},
            unchanged,
        )
    else:
        weights = {kind: getattr(args, f"weight_{kind}") for kind in DEFAULT_CORRUPTIONS}
        weights = {kind: value for kind, value in weights.items() if value > 0}
        raw_examples, stats = build_revision_examples(pairs, weights=weights, seed=args.seed)

    examples = [
        RevisionExample(
            serialized_source=serialized,
            target=target,
            metadata=record.metadata,
            source_identifier=record.source_identifier,
        )
        for (serialized, target), record in zip(raw_examples, records, strict=True)
    ]

    written = write_revision_examples(args.output, examples, language_pair)
    output = Path(args.output)
    if not output.name.startswith(DEFAULT_TRAIN_ONLY_PREFIXES):
        print(
            f"[sion] Warning: {output.name} does not start with "
            f"{' / '.join(DEFAULT_TRAIN_ONLY_PREFIXES)}. Synthetic examples may "
            "therefore enter validation/test and inflate holdout scores."
        )
    print(f"[sion] Wrote {written} examples to {output}")
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))

    if written and stats.unchanged / written > UNCHANGED_WARNING_RATIO:
        share = 100.0 * stats.unchanged / written
        print(
            f"[sion] Warning: {share:.0f}% of drafts equal their references. "
            "Corruption often has no effect on short, simple sentences: number "
            "does nothing without digits, and drop_clause/swap do nothing with a "
            "single clause. Training on this distribution teaches the revision "
            "model mostly to leave drafts unchanged. Use inputs with longer "
            "sentences or pass actual trained-model output through --drafts."
        )


if __name__ == "__main__":
    main()
