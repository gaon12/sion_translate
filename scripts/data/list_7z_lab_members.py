#!/usr/bin/env python3
"""Write a JSON list of .lab member paths from a 7z archive without extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def parse_paths(output: str) -> list[str]:
    paths: list[str] = []
    for line in output.splitlines():
        if not line.startswith("Path = "):
            continue
        path = line.removeprefix("Path = ").strip()
        if path.lower().endswith(".lab"):
            paths.append(path.replace("\\", "/"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seven-zip", default="7z.exe")
    args = parser.parse_args()

    completed = subprocess.run(
        [args.seven_zip, "l", "-slt", "-sccUTF-8", str(args.archive.resolve())],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    paths = parse_paths(completed.stdout)
    if not paths:
        raise SystemExit(f"no .lab members found in {args.archive}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(args.archive.resolve()),
                "lab_members": len(paths),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
