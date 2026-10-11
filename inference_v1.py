"""Run the public v1 EMA with the numeric-protection maintenance runtime."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--target-language", choices=("ko", "ja"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    from sion_translate.inference import Translator

    if "strict_numbers" not in inspect.signature(Translator.translate).parameters:
        parser.error("Install the v1-numeric-hotfix runtime listed in the model card first.")
    manifest = json.loads((args.model_dir / "v1_numeric_hotfix.json").read_text(encoding="utf-8"))
    for filename, identity in manifest["artifacts"].items():
        path = args.model_dir / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        if path.stat().st_size != identity["size"] or digest.hexdigest() != identity["sha256"]:
            raise ValueError(f"Public v1 artifact identity mismatch: {filename}")
    translator = Translator(
        args.model_dir / "model_ema.pt", args.model_dir / "kjx.model", device=args.device
    )
    for output in translator.translate(
        args.text,
        target_language=args.target_language,
        num_beams=args.num_beams,
        strict_numbers=True,
    ):
        print(output)


if __name__ == "__main__":
    main()
