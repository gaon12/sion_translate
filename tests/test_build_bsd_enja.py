from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "build_bsd_enja.py"
SPEC = importlib.util.spec_from_file_location("build_bsd_enja_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


def conversation(identifier: str = "c1", turns: int = 2, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": identifier,
        "tag": "phone call",
        "title": "Calling a client",
        "original_language": "ja",
        "conversation": [
            {
                "no": index + 1,
                "en_speaker": "Doi-san",
                "ja_speaker": "土井さん",
                "en_sentence": f"Hello, this is turn {index + 1}.",
                "ja_sentence": f"もしもし、{index + 1}番目の発話です。",
            }
            for index in range(turns)
        ],
    }
    base.update(overrides)
    return base


def write_corpus(root: Path, **splits: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, conversations in splits.items():
        (root / f"{name}.json").write_text(
            json.dumps(conversations, ensure_ascii=False), encoding="utf-8"
        )
    return root


def read_shard(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_conversation_becomes_one_record_with_aligned_turn_lists(tmp_path: Path) -> None:
    root = write_corpus(
        tmp_path / "BSD",
        train=[conversation(turns=3)],
        dev=[],
        test=[],
    )
    output = tmp_path / "data81_enja.jsonl"

    assert BUILD.main(["--input", str(root), "--output", str(output)]) == 0

    rows = read_shard(output)
    assert len(rows) == 1
    assert len(rows[0]["en"]) == 3
    assert len(rows[0]["ja"]) == 3
    assert rows[0]["scene"] == "phone call"
    assert rows[0]["conversation_id"] == "c1"
    assert rows[0]["original_language"] == "ja"


def test_every_split_is_included_by_default(tmp_path: Path) -> None:
    root = write_corpus(
        tmp_path / "BSD",
        train=[conversation("a")],
        dev=[conversation("b")],
        test=[conversation("c")],
    )
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(root), "--output", str(output)])

    manifest = json.loads((tmp_path / "shard.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["conversations_per_split"] == {"train": 1, "dev": 1, "test": 1}
    assert manifest["turns"] == 6


def test_a_single_split_can_be_selected(tmp_path: Path) -> None:
    root = write_corpus(
        tmp_path / "BSD",
        train=[conversation("a")],
        dev=[conversation("b")],
        test=[conversation("c")],
    )
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(root), "--output", str(output), "--split", "train"])

    assert len(read_shard(output)) == 1


def test_turns_missing_a_side_are_dropped_but_the_lists_stay_aligned(tmp_path: Path) -> None:
    broken = conversation(turns=2)
    turns = broken["conversation"]
    assert isinstance(turns, list)
    turns.append({"no": 3, "en_sentence": "No Japanese here."})
    turns.append({"no": 4, "ja_sentence": "英語がない。"})
    root = write_corpus(tmp_path / "BSD", train=[broken], dev=[], test=[])
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(root), "--output", str(output)])

    row = read_shard(output)[0]
    assert len(row["en"]) == len(row["ja"]) == 2


def test_manifest_records_the_noncommercial_licence(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "BSD", train=[conversation()], dev=[], test=[])
    output = tmp_path / "shard.jsonl"

    BUILD.main(["--input", str(root), "--output", str(output)])

    manifest = json.loads((tmp_path / "shard.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert "NC" in manifest["license"]
    assert manifest["source_url"] == "https://github.com/tsuruoka-lab/BSD"


def test_missing_split_file_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "BSD"
    root.mkdir()
    (root / "train.json").write_text("[]", encoding="utf-8")

    assert BUILD.main(["--input", str(root), "--output", str(tmp_path / "out.jsonl")]) == 2


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    assert (
        BUILD.main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out.jsonl")])
        == 2
    )
