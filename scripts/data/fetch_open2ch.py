"""Download and convert open2ch Japanese forum dialogue for foundation training.

This source fills a measured register gap. The current Japanese corpus consists
of fineweb2, Wikipedia, Kokkai, Aozora, and e-Gov data, all written or document
language, while Korean community data contributes 13.7% of that corpus. Adding
forum dialogue reduces this spoken-register imbalance.

Only the ``all-corpus-cleaned`` configuration is used. The unfiltered
``all-corpus`` version duplicates conversations and retains their noise.

The converter writes one turn per line. A whole dialogue on one line would force
foundation preparation to reconstruct its sentence boundaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from sion_translate.console import configure_stdio

REPO = "p1atdev/open2ch"
CONFIG = "all-corpus-cleaned"
_WHITESPACE = re.compile(r"\s+")
# Remove forum quote markers and anchors so the model does not learn them as content.
_FORUM_NOISE = re.compile(r">>\d+|^>+|https?://\S+")


def clean(text: str) -> str:
    return _WHITESPACE.sub(" ", _FORUM_NOISE.sub(" ", text)).strip()


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="open2ch → foundation JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-characters", type=int, default=8)
    parser.add_argument("--maximum-characters", type=int, default=4000)
    parser.add_argument(
        "--max-bytes", type=float, default=1.0e9, help="approximate output size limit"
    )
    args = parser.parse_args()

    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith(CONFIG + "/")]
    if not files:
        raise SystemExit(f"No files were found for the {CONFIG} configuration")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = seen_dupes = 0
    seen: set[int] = set()
    with output.open("w", encoding="utf-8") as sink:
        for name in sorted(files):
            if output.exists() and output.stat().st_size >= args.max_bytes:
                print("  output size limit reached; skipping remaining files", flush=True)
                break
            local = hf_hub_download(REPO, name, repo_type="dataset")
            table = pq.read_table(local)
            column = "dialogue" if "dialogue" in table.column_names else table.column_names[0]
            for value in table.column(column).to_pylist():
                # `dialogue` is a {"speaker": [...], "content": [...]} structure.
                # Treating it as a string list silently produced zero rows in practice.
                if isinstance(value, dict):
                    turns = value.get("content") or []
                elif isinstance(value, list):
                    turns = value
                else:
                    turns = [value]
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    text = clean(turn)
                    if not args.minimum_characters <= len(text) <= args.maximum_characters:
                        continue
                    digest = hash(text)
                    if digest in seen:
                        seen_dupes += 1
                        continue
                    seen.add(digest)
                    sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
            sink.flush()
            print(
                f"  {name}: {written:,} cumulative rows, {output.stat().st_size / 1e9:.2f} GB",
                flush=True,
            )
    print(
        f"{output}: {written:,} rows, {output.stat().st_size / 1e9:.2f} GB "
        f"({seen_dupes:,} duplicates)"
    )


if __name__ == "__main__":
    main()
