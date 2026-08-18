from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any, Iterator

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_foundation_expansion.py"
SPEC = importlib.util.spec_from_file_location("build_foundation_expansion_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


def test_korean_reasoning_allows_math_and_technical_identifiers() -> None:
    think = (
        "먼저 전체 학생 수를 계산합니다. 학생 252명과 교사 8명을 더하면 260명입니다. "
        "버스 수를 x라고 두면 x = ceil(260 / 41) = 7입니다. 마지막으로 비용을 더합니다."
    )

    assert BUILD.reasoning_language_issue("ko", think) is None


def test_korean_reasoning_rejects_an_english_prose_switch() -> None:
    think = (
        "먼저 전체 학생 수를 계산합니다. 학생과 교사의 수를 더하면 260명입니다.\n"
        "Now we need to calculate how many buses are required for all of the students.\n"
        "따라서 버스는 모두 7대가 필요합니다."
    )

    assert BUILD.reasoning_language_issue("ko", think) == "think_english_prose_switch"


def test_japanese_reasoning_rejects_an_english_prose_switch() -> None:
    think = (
        "まず、問題で与えられた人数を合計します。生徒と教師を合わせると260人です。\n"
        "Then we divide the total number by the capacity of each bus and round up.\n"
        "したがって、必要なバスは7台です。"
    )

    assert BUILD.reasoning_language_issue("ja", think) == "think_english_prose_switch"


def test_japanese_reasoning_allows_latin_product_names() -> None:
    think = (
        "まずAPIの応答を確認します。Pythonのlistに値を保存し、JSONとして変換します。"
        "次にHTTPの状態コードが200であることを確かめれば、処理が成功したと判断できます。"
    )

    assert BUILD.reasoning_language_issue("ja", think) is None


def test_english_output_is_split_at_the_final_answer_marker() -> None:
    think, answer = BUILD._split_english_output(
        "There are 24 clips in May. Therefore, the final answer is 72."
    )

    assert think == "There are 24 clips in May."
    assert answer == "72."


def test_a_script_neutral_short_answer_is_not_rejected() -> None:
    assert BUILD.text_language_issue("ko", "2,152,500", field="answer") is None
    assert BUILD.text_language_issue("en", "(C)", field="answer") is None


def test_category_assignment_prefers_the_least_filled_matching_group() -> None:
    used = {name: 0 for name in BUILD.ENGLISH_CATEGORY_BUDGETS}
    used["mathematics"] = BUILD.ENGLISH_CATEGORY_BUDGETS["mathematics"] // 2

    assert BUILD._choose_group(["Mathematics", "Computer Science"], used) == "computer_science"


def test_jamard_replaces_contextless_grounded_qa_with_pinned_train_data() -> None:
    japanese_sources = [source for source in BUILD.REASONING_SOURCES if source.language == "ja"]
    repos = {source.repo for source in japanese_sources}
    jamard = next(source for source in japanese_sources if source.repo == "elyza/JaMARD")

    assert "hotchpotch/japanese-qa-reasoning-100k" not in repos
    assert jamard.path == "data/train.parquet"
    assert jamard.revision == "82e107d209dec19e17a76d76425452c81b192755"
    assert jamard.sha256 == "f05f958e30a4e67baf3fb2b1ef6d68017ecaf8a5ea9d6bbde6e4e09817ae7810"


def test_jamard_keeps_shortest_verified_trace_per_unique_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "instruction": "2 + 2 はいくつですか？",
            "gold_answer": "4",
            "true_responses": [
                "順番に計算すると、2 に 2 を足します。\n\n回答: 4",
                "2 と 2 を足すと 4 です。\n\n答え: 4",
            ],
            "source": "gsm8k",
        },
        {
            "instruction": "2 + 2 はいくつですか？",
            "gold_answer": "4",
            "true_responses": ["2+2=4 と計算できます。\n回答: 4"],
            "source": "gsm8k",
        },
        {
            "instruction": "根拠のない問題",
            "gold_answer": "",
            "true_responses": [],
            "source": "prm800k",
        },
    ]

    def iter_rows(_path: Path) -> Iterator[dict[str, Any]]:
        yield from rows

    monkeypatch.setattr(BUILD, "_iter_parquet", iter_rows)

    normalized, stats = BUILD._jamard_reasoning_rows(tmp_path / "train.parquet")

    assert normalized == [
        {
            "prompt": "2 + 2 はいくつですか？",
            "think": "2+2=4 と計算できます。",
            "answer": "4",
            "category": "math_reasoning",
            "source_id": "train:1:0",
            "seed_source": "gsm8k",
        }
    ]
    assert stats == {
        "seen": 3,
        "duplicate_prompt": 1,
        "no_usable_verified_response": 1,
        "selected_marker_回答:": 1,
        "selected_unique_prompts": 1,
    }


def test_source_file_validation_checks_the_pinned_sha256(tmp_path: Path) -> None:
    payload = b"verified-source"
    source_path = tmp_path / "train.parquet"
    source_path.write_bytes(payload)
    source = BUILD.SourceFile(
        repo="example/reasoning",
        revision="a" * 40,
        path="train.parquet",
        size=len(payload),
        license="MIT",
        language="ja",
        category="math_reasoning",
        sha256=sha256(payload).hexdigest(),
    )

    BUILD._validate_source_file(source, source_path)
    source_path.write_bytes(b"tampered-source")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        BUILD._validate_source_file(source, source_path)


def test_reasoning_manifest_hashes_the_normalized_output(tmp_path: Path) -> None:
    source_path = tmp_path / "reasoning.jsonl"
    source_path.write_text(
        '{"instruction":"2と2を足すといくつですか？",'
        '"reasoning":"まず2と2を順番に足すと、合計は4になります。",'
        '"final_output":"4"}\n',
        encoding="utf-8",
    )
    source = BUILD.SourceFile(
        repo="DeL-example/tiny-ja",
        revision="b" * 40,
        path="reasoning.jsonl",
        size=source_path.stat().st_size,
        license="MIT",
        language="ja",
        category="math_reasoning",
    )

    manifest = BUILD.build_reasoning_corpora([(source, source_path)], tmp_path / "corpus")
    output = tmp_path / "corpus" / "ja" / "reasoning_tiny-ja.jsonl"

    assert manifest["outputs"]["ja/tiny-ja"] == {
        "path": output.as_posix(),
        "size": output.stat().st_size,
        "sha256": sha256(output.read_bytes()).hexdigest(),
    }
