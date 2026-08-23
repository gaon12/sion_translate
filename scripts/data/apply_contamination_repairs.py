#!/usr/bin/env python3
"""확정적으로 고칠 수 있는 오염만 shard 에 적용한다.

`씨발` 이 `種まき`(씨 뿌리기)로 옮겨진 행은 대체어가 문맥에 의존하지 않으므로
규칙으로 고칠 수 있습니다. 반면 관용구 직역과 욕설 강도 소실은 대체할 일본어를
새로 써야 하고, 그것은 번역이지 규칙이 아닙니다. 이 도구는 전자만 건드리고
후자는 `build_review_queue.py` 의 사람 검수 queue 로 남깁니다.

    # 무엇이 바뀔지 먼저 봅니다 (아무것도 쓰지 않음)
    python scripts/data/apply_contamination_repairs.py --input "data/*.jsonl" \
        --source-language ko --target-language ja

    # 실제로 적용합니다
    python scripts/data/apply_contamination_repairs.py --input "data/*.jsonl" \
        --source-language ko --target-language ja \
        --apply --report reports/contamination_repairs.json

``--apply`` 없이는 아무 파일도 쓰지 않습니다. 코퍼스를 제자리에서 고치는
일이라 기본값이 쓰기여서는 안 됩니다.

원본은 ``data/excluded/contamination_repair_<날짜>/`` 아래에 그대로 보존하고,
바뀐 행은 전부 보고서에 원문·수정문과 함께 남습니다. 되돌리려면 보존본을
제자리에 돌려놓으면 됩니다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import cast

from sion_translate.console import configure_stdio
from sion_translate.contamination import repair_pair, supported_direction
from sion_translate.data.quality import canonical_text
from sion_translate.tokenizer import expand_inputs

# 보고서에 남길 표본 수. 전량은 queue 파일에 있고, 여기서는 사람이 눈으로
# 확인할 만큼만 봅니다.
SAMPLE_LIMIT = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="확정적인 오염만 수정합니다 (기본은 미리보기, 쓰지 않음)"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 파일 또는 glob")
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 파일을 고칩니다. 없으면 무엇이 바뀔지 보고만 합니다.",
    )
    parser.add_argument(
        "--backup-root",
        default=None,
        help="원본 보존 위치 (기본: data/excluded/contamination_repair_<날짜>)",
    )
    parser.add_argument("--report", help="JSON 보고서 경로")
    return parser


def _default_backup_root() -> Path:
    return Path("data/excluded") / f"contamination_repair_{date.today():%Y%m%d}"


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    source_language = args.source_language
    target_language = args.target_language
    if not supported_direction(source_language, target_language):
        raise SystemExit(
            f"{source_language}->{target_language} 규칙이 없습니다. 규칙 없는 방향을 "
            "'고칠 것 없음' 으로 보고하지 않기 위해 여기서 중단합니다."
        )

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"입력과 일치하는 JSONL 이 없습니다: {args.input}")

    backup_root = Path(args.backup_root) if args.backup_root else _default_backup_root()
    by_file: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    scanned = 0
    repaired_rows = 0

    for path in paths:
        original_lines = path.read_text(encoding="utf-8-sig").splitlines()
        rewritten: list[str] = []
        changed_here = 0

        for line_number, raw_line in enumerate(original_lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                rewritten.append(raw_line)
                continue
            try:
                payload: object = json.loads(stripped)
            except json.JSONDecodeError:
                rewritten.append(raw_line)
                continue
            if not isinstance(payload, dict):
                rewritten.append(raw_line)
                continue
            row = cast(dict[str, object], payload)

            source = row.get(source_language)
            target = row.get(target_language)
            if not isinstance(source, str) or not isinstance(target, str):
                rewritten.append(raw_line)
                continue

            scanned += 1
            repair = repair_pair(
                canonical_text(source),
                canonical_text(target),
                source_language=source_language,
                target_language=target_language,
            )
            if repair is None:
                rewritten.append(raw_line)
                continue

            changed_here += 1
            repaired_rows += 1
            by_file[path.name] += 1
            if len(samples) < SAMPLE_LIMIT:
                samples.append(
                    {
                        "file": path.name,
                        "line": line_number,
                        "source": canonical_text(source),
                        "before": repair.original_target,
                        "after": repair.target,
                        "replacements": [list(pair) for pair in repair.replacements],
                    }
                )
            row[target_language] = repair.target
            # 되돌릴 수 있도록 행 자체에도 흔적을 남깁니다. 보존본이 사라져도
            # 무엇이 규칙으로 고쳐진 행인지 알 수 있어야 합니다.
            row["contamination_repaired"] = True
            rewritten.append(json.dumps(row, ensure_ascii=False))

        if changed_here and args.apply:
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_root / path.name)
            path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    report = {
        "applied": bool(args.apply),
        "scanned_pairs": scanned,
        "repaired_rows": repaired_rows,
        "by_file": dict(by_file.most_common()),
        "backup_root": str(backup_root) if args.apply and repaired_rows else None,
        "samples": samples,
        "note": (
            "규칙으로 확정 수정한 행만 집계합니다. 관용구 직역과 욕설 강도 "
            "소실은 대체어를 사람이 써야 하므로 build_review_queue.py 의 "
            "검수 queue 에 남아 있습니다."
        ),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"검사한 쌍 {scanned:,} / 수정 {repaired_rows:,}")
    for name, count in by_file.most_common():
        print(f"  {name}: {count:,}")
    if not args.apply:
        print("\n미리보기입니다. 아무 파일도 쓰지 않았습니다. 적용하려면 --apply 를 주십시오.")
    elif repaired_rows:
        print(f"\n원본 보존: {backup_root}")


if __name__ == "__main__":
    main()
