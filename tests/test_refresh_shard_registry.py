from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "refresh_shard_registry.py"
SPEC = importlib.util.spec_from_file_location("refresh_shard_registry_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REFRESH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFRESH
SPEC.loader.exec_module(REFRESH)


COLUMNS = ("file", "rows", "bytes", "sha256", "original_file", "notes")


def write_registry(path: Path, rows: list[dict[str, str]]) -> Path:
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(row.get(column, "") for column in COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_shard(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def read_registry(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]


def test_stale_row_is_recomputed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    shard = write_shard(data_root / "data01.jsonl", [{"ko": "가"}, {"ko": "나"}])
    registry = write_registry(
        tmp_path / "data.tsv",
        [
            {
                "file": "ko-ja/data01.jsonl",
                "rows": "99",
                "bytes": "1",
                "sha256": "stale",
                "original_file": "main/data01.jsonl",
                "notes": "원본 기록",
            }
        ],
    )

    assert REFRESH.main(["--registry", str(registry), "--data-root", str(data_root)]) == 0

    row = read_registry(registry)[0]
    assert row["rows"] == "2"
    assert row["bytes"] == str(shard.stat().st_size)
    assert row["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert row["notes"] == "원본 기록"


def test_note_is_appended_only_to_changed_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    stale = write_shard(data_root / "data01.jsonl", [{"ko": "가"}])
    fresh = write_shard(data_root / "data02.jsonl", [{"ko": "나"}])
    registry = write_registry(
        tmp_path / "data.tsv",
        [
            {
                "rows": "99",
                "bytes": "1",
                "sha256": "stale",
                "original_file": "main/data01.jsonl",
                "notes": "기존",
            },
            {
                "rows": str(len(fresh.read_text(encoding="utf-8").splitlines())),
                "bytes": str(fresh.stat().st_size),
                "sha256": hashlib.sha256(fresh.read_bytes()).hexdigest(),
                "original_file": "main/data02.jsonl",
                "notes": "기존",
            },
        ],
    )

    REFRESH.main(
        ["--registry", str(registry), "--data-root", str(data_root), "--note", "중복 제거"]
    )

    rows = read_registry(registry)
    assert stale.exists()
    assert rows[0]["notes"] == "기존; 중복 제거"
    assert rows[1]["notes"] == "기존"


def test_missing_shard_is_reported_and_can_be_marked(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    registry = write_registry(
        tmp_path / "data.tsv",
        [
            {
                "rows": "70024",
                "bytes": "123",
                "sha256": "abc",
                "original_file": "main/data52.jsonl",
                "notes": "기존",
            }
        ],
    )
    report = tmp_path / "report.json"

    REFRESH.main(
        [
            "--registry",
            str(registry),
            "--data-root",
            str(data_root),
            "--mark-missing",
            "--note",
            "아카이브로 이동",
            "--report",
            str(report),
        ]
    )

    row = read_registry(registry)[0]
    assert row["rows"] == "0"
    assert row["sha256"] == ""
    assert row["notes"] == "기존; 아카이브로 이동"
    assert json.loads(report.read_text(encoding="utf-8"))["registry_rows_missing_file"] == [
        "data52.jsonl"
    ]


def test_check_reports_drift_without_writing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_shard(data_root / "data01.jsonl", [{"ko": "가"}])
    registry = write_registry(
        tmp_path / "data.tsv",
        [{"rows": "99", "bytes": "1", "sha256": "stale", "original_file": "main/data01.jsonl"}],
    )
    before = registry.read_text(encoding="utf-8")

    code = REFRESH.main(["--registry", str(registry), "--data-root", str(data_root), "--check"])

    assert code == 1
    assert registry.read_text(encoding="utf-8") == before


def test_check_passes_when_everything_matches(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    shard = write_shard(data_root / "data01.jsonl", [{"ko": "가"}])
    registry = write_registry(
        tmp_path / "data.tsv",
        [
            {
                "rows": "1",
                "bytes": str(shard.stat().st_size),
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "original_file": "main/data01.jsonl",
            }
        ],
    )

    assert (
        REFRESH.main(["--registry", str(registry), "--data-root", str(data_root), "--check"]) == 0
    )


def test_manifest_measurements_are_refreshed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    shard = write_shard(data_root / "data01.jsonl", [{"ko": "가"}, {"ko": "나"}])
    manifest_path = data_root / "data01.jsonl.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "strategy": "kept as written",
                "row_count": 99,
                "output_bytes": 1,
                "sha256": "stale",
                "output": "D:\\\\elsewhere\\\\data\\\\data01.jsonl",
            }
        ),
        encoding="utf-8",
    )
    registry = write_registry(tmp_path / "data.tsv", [])

    REFRESH.main(["--registry", str(registry), "--data-root", str(data_root)])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 2
    assert manifest["output_bytes"] == shard.stat().st_size
    assert manifest["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert manifest["strategy"] == "kept as written"


def test_aggregate_manifest_totals_its_members(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_shard(data_root / "data66.jsonl", [{"en": "a"}, {"en": "b"}])
    write_shard(data_root / "data67.jsonl", [{"en": "c"}])
    manifest_path = data_root / "opus.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rows": 999,
                "corpora": [
                    {"name": "kftt", "output": "X:\\\\data\\\\data66.jsonl", "rows": 500},
                    {"name": "tatoeba", "output": "X:\\\\data\\\\data67.jsonl", "rows": 400},
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = write_registry(tmp_path / "data.tsv", [])

    REFRESH.main(["--registry", str(registry), "--data-root", str(data_root)])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [member["rows"] for member in manifest["corpora"]] == [2, 1]
    assert manifest["rows"] == 3


def test_missing_registry_is_rejected(tmp_path: Path) -> None:
    assert REFRESH.main(["--registry", str(tmp_path / "nope.tsv"), "--data-root", str(tmp_path)]) == 2
