from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "audit_generated_shards.py"
SPEC = importlib.util.spec_from_file_location("audit_generated_shards_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def write_shard(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, target in rows:
            handle.write(json.dumps({"ko": source, "ja": target}, ensure_ascii=False) + "\n")
    return path


# ``skeleton()`` blanks digits, so a fixture that varies only a number
# collapses to a single frame. Vary the words instead.
_KO_SYLLABLES = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
_JA_SYLLABLES = "カキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモ"


def _word(index: int, alphabet: str) -> str:
    high, low = divmod(index, len(alphabet))
    return alphabet[high % len(alphabet)] + alphabet[low]


def distinct_rows(count: int) -> list[tuple[str, str]]:
    """Rows whose sentence frames differ lexically, not only numerically."""

    if count > len(_KO_SYLLABLES) ** 2:
        raise ValueError("fixture alphabet is too small for the requested row count")
    return [
        (
            f"{_word(index, _KO_SYLLABLES)} 마을의 기록관은 자료를 다시 검토했다.",
            f"{_word(index, _JA_SYLLABLES)}村の資料館は資料を改めて検討した。",
        )
        for index in range(count)
    ]


def test_skeleton_blanks_quoted_spans_and_digits() -> None:
    assert AUDIT.skeleton('값은 "0.5 mg"이고 3개다.') == "값은 <Q>이고 #개다."
    assert AUDIT.skeleton("따옴표가 없으면 그대로다.") == "따옴표가 없으면 그대로다."


def test_diverse_shard_passes(tmp_path: Path) -> None:
    report = AUDIT.audit_shard(write_shard(tmp_path / "good.jsonl", distinct_rows(400)))

    assert report.rows == 400
    assert report.unreadable_rows == 0
    assert report.skeleton_ttr == pytest.approx(1.0)
    assert report.max_targets_per_source == 1
    assert report.passed, report.violations


def test_template_collapse_is_rejected(tmp_path: Path) -> None:
    rows = [
        (f'이 표현은 "관용구{index % 3}"로 옮긴다.', f"この表現は「慣用句{index % 3}」と訳す。")
        for index in range(300)
    ]
    report = AUDIT.audit_shard(write_shard(tmp_path / "template.jsonl", rows))

    assert not report.passed
    assert any(name.startswith("skeleton_ttr") for name in report.violations)
    assert any(name.startswith("quoted_ttr") for name in report.violations)


def test_one_to_many_targets_are_rejected(tmp_path: Path) -> None:
    rows = [("같은 원문이 계속 반복된다.", f"別の訳{index}です。") for index in range(300)]
    report = AUDIT.audit_shard(write_shard(tmp_path / "many.jsonl", rows))

    assert report.max_targets_per_source == 300
    assert report.conflicting_source == pytest.approx(1.0)
    assert any(name.startswith("duplicate_source") for name in report.violations)
    assert any(name.startswith("conflicting_source") for name in report.violations)


def test_hangul_in_japanese_target_is_rejected(tmp_path: Path) -> None:
    rows = [
        (source, f"{_word(index, _JA_SYLLABLES)}村の 문장 は違う。")
        for index, (source, _) in enumerate(distinct_rows(300))
    ]
    report = AUDIT.audit_shard(write_shard(tmp_path / "leak.jsonl", rows))

    assert report.foreign_script_target == pytest.approx(1.0)
    assert any(name.startswith("foreign_script_target") for name in report.violations)


def test_korean_target_is_not_reported_as_foreign(tmp_path: Path) -> None:
    """Auditing kj->ko must not flag the Korean target as contaminated."""

    rows = [
        (
            f"{_word(index, _JA_SYLLABLES)}ノ 마을은 조용데스네",
            f"{_word(index, _KO_SYLLABLES)} 마을은 조용하네요",
        )
        for index in range(300)
    ]
    path = tmp_path / "kj.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, target in rows:
            handle.write(json.dumps({"kj": source, "ko": target}, ensure_ascii=False) + "\n")

    japanese = AUDIT.audit_shard(path, source_key="kj", target_key="ko", target_language="ja")
    korean = AUDIT.audit_shard(path, source_key="kj", target_key="ko", target_language="ko")
    ignored = AUDIT.audit_shard(path, source_key="kj", target_key="ko", target_language="none")

    assert japanese.foreign_script_target == pytest.approx(1.0)
    assert korean.foreign_script_target == pytest.approx(0.0)
    assert ignored.foreign_script_target == pytest.approx(0.0)
    assert korean.passed, korean.violations


def test_kana_in_a_korean_target_is_gated(tmp_path: Path) -> None:
    rows = [
        (source, f"{_word(index, _KO_SYLLABLES)} 마을은 やっぱり 조용하다")
        for index, (source, _) in enumerate(distinct_rows(300))
    ]
    path = write_shard(tmp_path / "kanaleak.jsonl", rows)

    report = AUDIT.audit_shard(path, target_language="ko")

    assert report.foreign_script_target == pytest.approx(1.0)
    assert any(name.startswith("foreign_script_target") for name in report.violations)


def test_unknown_target_language_is_rejected(tmp_path: Path) -> None:
    path = write_shard(tmp_path / "x.jsonl", distinct_rows(10))

    with pytest.raises(ValueError, match="target_language must be"):
        AUDIT.audit_shard(path, target_language="en")


def test_kana_in_korean_source_is_measured_but_not_gated(tmp_path: Path) -> None:
    rows = [
        (
            f"{_word(index, _KO_SYLLABLES)} 마을은 やっぱり 조용했다.",
            f"{_word(index, _JA_SYLLABLES)}村はやっぱり静かだった。",
        )
        for index in range(300)
    ]
    report = AUDIT.audit_shard(write_shard(tmp_path / "mixed.jsonl", rows))

    assert report.foreign_script_source == pytest.approx(1.0)
    assert report.passed, report.violations


def test_near_duplicate_leak_is_detected(tmp_path: Path) -> None:
    # One frame with a varying quoted span: every held-out row shares its
    # skeleton with a training row, which is exactly the leak we gate on.
    rows = [
        (
            f'{_word(index % 576, _KO_SYLLABLES)} 표현 "{index}"를 옮긴다.',
            f"表現「{index}」を訳す。",
        )
        for index in range(4000)
    ]
    report = AUDIT.audit_shard(write_shard(tmp_path / "leaky.jsonl", rows))

    assert report.held_out_rows > 0
    assert report.near_duplicate_leak == pytest.approx(1.0)
    assert any(name.startswith("near_duplicate_leak") for name in report.violations)


def test_unreadable_rows_are_counted_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, target in distinct_rows(300):
            handle.write(json.dumps({"ko": source, "ja": target}, ensure_ascii=False) + "\n")
        handle.write("not json\n")
        handle.write(json.dumps(["array", "not", "object"]) + "\n")
        handle.write(json.dumps({"ko": 5, "ja": "あ"}) + "\n")

    report = AUDIT.audit_shard(path)

    assert report.rows == 303
    assert report.unreadable_rows == 3
    assert report.passed, report.violations


def test_shard_without_usable_rows_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    report = AUDIT.audit_shard(path)

    assert not report.passed
    assert report.violations == ["usable_rows 0 < 1"]


def test_custom_keys_are_honoured(tmp_path: Path) -> None:
    path = tmp_path / "custom.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(300):
            handle.write(
                json.dumps(
                    {"kj": f"{index}번째 ぶんしょう 이다.", "ko": f"{index}번째 문장이다."},
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = AUDIT.audit_shard(path, source_key="kj", target_key="ko")

    assert report.rows == 300
    assert report.unreadable_rows == 0


def test_thresholds_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        AUDIT.Thresholds(min_skeleton_ttr=2.0).validate()
    with pytest.raises(ValueError):
        AUDIT.Thresholds(max_near_duplicate_leak=-0.1).validate()


def test_main_returns_one_when_a_shard_fails(tmp_path: Path, capsys) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(300))
    bad = write_shard(
        tmp_path / "bad.jsonl",
        [("같은 원문이 계속 반복된다.", f"別の訳{index}です。") for index in range(300)],
    )
    report_path = tmp_path / "report.json"

    assert AUDIT.main(["--json", str(report_path), str(good), str(bad)]) == 1

    printed = capsys.readouterr().out
    assert "good.jsonl" in printed and "PASS" in printed
    assert "bad.jsonl" in printed and "FAIL" in printed
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert [Path(entry["path"]).name for entry in payload] == ["good.jsonl", "bad.jsonl"]


def test_main_returns_zero_when_every_shard_passes(tmp_path: Path) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(300))

    assert AUDIT.main([str(good)]) == 0


def test_main_reports_bad_input_with_exit_code_two(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(10))

    assert AUDIT.main([str(missing)]) == 2
    assert AUDIT.main(["--min-skeleton-ttr", "1.5", str(good)]) == 2
    assert AUDIT.main(["--examples", "-1", str(good)]) == 2


def test_relaxed_threshold_lets_a_known_shard_through(tmp_path: Path) -> None:
    # A shard the default thresholds reject: 100 frames restated four times.
    rows = [
        (
            f"{_word(index // 4, _KO_SYLLABLES)} 마을의 기록관은 자료를 검토했다.",
            "村の資料館は資料を検討した。",
        )
        for index in range(400)
    ]
    path = write_shard(tmp_path / "frames.jsonl", rows)

    assert AUDIT.main([str(path)]) == 1
    assert (
        AUDIT.main(
            [
                "--min-skeleton-ttr",
                "0.2",
                "--max-duplicate-source",
                "0.8",
                "--max-near-duplicate-leak",
                "1.0",
                str(path),
            ]
        )
        == 0
    )
