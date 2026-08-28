from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "build_aihub_koen.py"
SPEC = importlib.util.spec_from_file_location("build_aihub_koen_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


def write_archive(path: Path, records: list[dict[str, object]], member: str = "set.json") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, json.dumps({"data": records}, ensure_ascii=False))
    return path


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ko": "그 말을 들으니 기쁩니다.",
        "en": "I'm glad to hear that.",
        "mt": "그 소식을 들으니 기쁩니다.",
        "en_original": "I am glad to hear that.",
        "domain": "해외고객과의채팅",
        "subdomain": "숙박,음식점",
        "style": "구어체",
    }
    base.update(overrides)
    return base


def test_machine_translation_and_drafts_never_reach_the_shard(tmp_path: Path) -> None:
    write_archive(tmp_path / "dist" / "TL1.zip", [record()])
    output = tmp_path / "data79_koen.jsonl"

    assert BUILD.main(["--input", str(tmp_path / "dist"), "--output", str(output)]) == 0

    rows = read_shard(output)
    assert rows == [
        {
            "ko": "그 말을 들으니 기쁩니다.",
            "en": "I'm glad to hear that.",
            "domain": "해외고객과의채팅",
            "subdomain": "숙박,음식점",
            "style": "구어체",
        }
    ]
    assert "mt" not in rows[0]
    assert "en_original" not in rows[0]


def test_labels_can_be_dropped(tmp_path: Path) -> None:
    write_archive(tmp_path / "dist" / "TL1.zip", [record()])
    output = tmp_path / "shard.jsonl"

    BUILD.main(
        ["--input", str(tmp_path / "dist"), "--output", str(output), "--drop-labels"]
    )

    assert set(read_shard(output)[0]) == {"ko", "en"}


def test_rows_missing_a_human_side_are_counted_not_written(tmp_path: Path) -> None:
    write_archive(
        tmp_path / "dist" / "TL1.zip",
        [record(), record(ko="   "), record(en=None), {"mt": "only machine output"}],
    )
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(tmp_path / "dist"), "--output", str(output)])

    manifest = json.loads(
        (tmp_path / "shard.jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scanned_records"] == 4
    assert manifest["row_count"] == 1
    assert manifest["unusable_records"] == 3


def test_every_archive_in_the_tree_is_read(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    write_archive(distribution / "1.Training" / "라벨링데이터" / "TL1.zip", [record()])
    write_archive(distribution / "1.Training" / "라벨링데이터" / "TL2.zip", [record(ko="둘")])
    write_archive(distribution / "2.Validation" / "라벨링데이터" / "VL1.zip", [record(ko="셋")])
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(distribution), "--output", str(output)])

    assert len(read_shard(output)) == 3
    manifest = json.loads((tmp_path / "shard.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert [entry["archive"] for entry in manifest["archives"]] == ["TL1.zip", "TL2.zip", "VL1.zip"]


def test_manifest_records_domain_and_style_distribution(tmp_path: Path) -> None:
    write_archive(
        tmp_path / "dist" / "TL1.zip",
        [record(), record(ko="둘", domain="세계", style="문어체"), record(ko="셋")],
    )
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(tmp_path / "dist"), "--output", str(output)])

    manifest = json.loads((tmp_path / "shard.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["domains"] == {"해외고객과의채팅": 2, "세계": 1}
    assert manifest["styles"] == {"구어체": 2, "문어체": 1}
    assert manifest["sha256"] and manifest["output_bytes"] > 0


def test_bare_array_documents_are_accepted(tmp_path: Path) -> None:
    archive = tmp_path / "dist" / "TL1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("set.json", json.dumps([record()], ensure_ascii=False))
    output = tmp_path / "shard.jsonl"

    assert BUILD.main(["--input", str(tmp_path / "dist"), "--output", str(output)]) == 0
    assert len(read_shard(output)) == 1


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    assert (
        BUILD.main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out.jsonl")])
        == 2
    )


def test_a_distribution_with_no_usable_row_publishes_nothing(tmp_path: Path) -> None:
    write_archive(tmp_path / "dist" / "TL1.zip", [{"mt": "machine only"}])
    output = tmp_path / "shard.jsonl"

    assert BUILD.main(["--input", str(tmp_path / "dist"), "--output", str(output)]) == 2
    assert not output.exists()
