"""Run a SentencePiece training probe behind a crash-surviving parent process.

This is a diagnostic tool, not the production trainer.  A native SIGSEGV kills
only the child, so the parent can report a return code and elapsed time.  Use a
``char`` model first: it exercises corpus loading and normalization without the
much more expensive unigram seed extraction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import sentencepiece as spm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-type", choices=("char", "unigram"), default="char")
    parser.add_argument(
        "--maximum-sentences",
        type=int,
        default=0,
        help="0 uses file input; a positive value probes that ordered prefix via an iterator",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _sentences(path: Path, maximum: int) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= maximum:
                return
            yield line.rstrip("\r\n")


def _train(args: argparse.Namespace, output_dir: Path) -> None:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source: dict[str, object]
    if args.maximum_sentences > 0:
        source = {
            "sentence_iterator": _sentences(args.corpus, args.maximum_sentences),
        }
    else:
        source = {"input": str(args.corpus)}
    spm.SentencePieceTrainer.train(
        **source,
        model_prefix=str(output_dir / "probe"),
        vocab_size=int(plan["vocab_size"]),
        model_type=args.model_type,
        character_coverage=0.9999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=list(plan["user_defined_symbols"]),
        required_chars=str(plan["required_characters"]),
        input_sentence_size=0,
        seed_sentencepiece_size=1_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        hard_vocab_limit=False,
        num_threads=args.threads,
    )


def _child_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--corpus",
        str(args.corpus.resolve()),
        "--plan",
        str(args.plan.resolve()),
        "--threads",
        str(args.threads),
        "--model-type",
        args.model_type,
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if args.maximum_sentences > 0:
        command.extend(("--maximum-sentences", str(args.maximum_sentences)))
    return command


def _run_parent(args: argparse.Namespace, output_dir: Path) -> int:
    started = time.monotonic()
    completed = subprocess.run(_child_command(args, output_dir), check=False)
    report = {
        "corpus": str(args.corpus.resolve()),
        "maximum_sentences": args.maximum_sentences or None,
        "model_type": args.model_type,
        "output_dir": str(output_dir.resolve()),
        "returncode": completed.returncode,
        "seconds": time.monotonic() - started,
        "sentencepiece_version": str(getattr(spm, "__version__", "unknown")),
        "threads": args.threads,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if completed.returncode == 0 else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.maximum_sentences < 0:
        raise ValueError("--maximum-sentences must be non-negative")
    if args.child:
        if args.output_dir is None:
            raise ValueError("the child requires --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _train(args, args.output_dir)
        return 0
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return _run_parent(args, args.output_dir)
    with tempfile.TemporaryDirectory(prefix="sion-spm-probe-") as workspace:
        return _run_parent(args, Path(workspace))


if __name__ == "__main__":
    raise SystemExit(main())
