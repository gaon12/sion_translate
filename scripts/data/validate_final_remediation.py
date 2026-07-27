#!/usr/bin/env python3
"""Validate the final 2026-07-27 dataset remediation without modifying data."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from rebuild_verified_parallel import DEFAULT_SOURCES


CORRECTED_ROWS = {
    "data16.jsonl": {
        "vp.d:349": ("26달러 5센트입니다.", "二十六ドル五セントです。"),
    },
    "data18.jsonl": {
        "AMO.7.10": (
            "베델의 아마샤 제사장이 이스라엘의 여로보암 왕에게 사람을 보내서 알렸다. "
            "“아모스가 이스라엘 나라 한가운데서 임금님께 대한 반란을 선동하고 있습니다. "
            "그가 하는 모든 말을 이 나라가 더 이상 참을 수 없습니다.",
            "ベテルの祭司アマツヤは、イスラエルの王ヤロブアムに人を遣わして言った。"
            "「イスラエルの家の真ん中で、アモスがあなたに背きました。"
            "この国は彼のすべての言葉に耐えられません。",
        ),
    },
    "data39.jsonl": {
        "browser/browser/sanitize.ftl::clear-time-duration-prefix.value": (
            "지우는 시간 범위:",
            "消去する履歴の期間:",
        ),
        "extensions/vscode.css-language-features.i18n.json::bundle/colon expected": (
            "콜론이 필요합니다",
            "コロンが必要です",
        ),
        "toolkit/toolkit/intl/languageNames.ftl::language-name-su": (
            "순다어",
            "スンダ語",
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_jsonl(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    rows = 0
    selected: dict[str, dict[str, Any]] = {}
    wanted = CORRECTED_ROWS.get(path.name, {})
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            rows += 1
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if not isinstance(value.get("ko"), str) or not isinstance(value.get("ja"), str):
                raise ValueError(f"{path}:{line_number}: missing ko/ja strings")
            resource_id = value.get("resource_id")
            if resource_id in wanted:
                if resource_id in selected:
                    raise ValueError(f"{path}: duplicate resource_id {resource_id}")
                selected[str(resource_id)] = value
    return rows, selected


def validate(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "dataset_remediation_20260727.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_state = manifest["final_state"]
    root = final_state["training_root"]
    results: dict[str, Any] = {}

    for name, expected in root.items():
        path = data_dir / name
        if not expected["included"]:
            if path.exists():
                raise ValueError(f"Excluded source was recreated in training root: {path}")
            continue
        rows, selected = scan_jsonl(path)
        digest = sha256_file(path)
        if rows != expected["rows"]:
            raise ValueError(f"{path}: expected {expected['rows']} rows, got {rows}")
        if digest != expected["sha256"]:
            raise ValueError(f"{path}: expected SHA-256 {expected['sha256']}, got {digest}")
        for resource_id, pair in CORRECTED_ROWS.get(name, {}).items():
            row = selected.get(resource_id)
            if row is None:
                raise ValueError(f"{path}: missing corrected row {resource_id}")
            if (row["ko"], row["ja"]) != pair:
                raise ValueError(f"{path}: corrected row changed: {resource_id}")
        results[name] = {"rows": rows, "sha256": digest}

    removed_audit = json.loads((data_dir / "data27.remediation.json").read_text(encoding="utf-8"))
    removed_ids = {item["resource_id"] for item in removed_audit["removed"]}
    present_ids: set[str] = set()
    with (data_dir / "data27.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            resource_id = json.loads(line).get("resource_id")
            if resource_id in removed_ids:
                present_ids.add(str(resource_id))
    if present_ids:
        raise ValueError(f"Removed data27 rows are still present: {sorted(present_ids)}")

    root_rows = 0
    for path in data_dir.glob("*.jsonl"):
        with path.open("rb") as handle:
            root_rows += sum(1 for _ in handle)
    if root_rows != final_state["training_root_rows"]:
        raise ValueError(
            f"Training-root total mismatch: {root_rows} != {final_state['training_root_rows']}"
        )
    if "massive" in DEFAULT_SOURCES:
        raise ValueError("MASSIVE must remain opt-in, not a default rebuild source")

    data_notes = (data_dir / "data.txt").read_text(encoding="utf-8")
    required_notes = (
        f"현재 data/*.jsonl 훈련 루트 합계: {root_rows:,}쌍",
        "data27.jsonl, 5,587행",
        "data37.jsonl 전체",
        "synthetic_numeric_data38.jsonl, 240,000쌍",
        "data39.jsonl, 31,101쌍",
    )
    missing_notes = [note for note in required_notes if note not in data_notes]
    if missing_notes:
        raise ValueError(f"data.txt is missing final-state notes: {missing_notes}")

    results["training_root_rows"] = root_rows
    results["default_rebuild_sources"] = list(DEFAULT_SOURCES)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.data_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
