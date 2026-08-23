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


def audit_shard(
    path: Path,
    *args: object,
    source_key: str = "ko",
    target_key: str = "ja",
    **kwargs: object,
):
    return AUDIT.audit_shard(
        path,
        *args,
        source_key=source_key,
        target_key=target_key,
        **kwargs,
    )


def audit_main(args: list[str]) -> int:
    return AUDIT.main(["--source-key", "ko", "--target-key", "ja", *args])


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
    report = audit_shard(write_shard(tmp_path / "good.jsonl", distinct_rows(400)))

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
    report = audit_shard(write_shard(tmp_path / "template.jsonl", rows))

    assert not report.passed
    assert any(name.startswith("skeleton_ttr") for name in report.violations)
    assert any(name.startswith("quoted_ttr") for name in report.violations)


def test_one_to_many_targets_are_rejected(tmp_path: Path) -> None:
    rows = [("같은 원문이 계속 반복된다.", f"別の訳{index}です。") for index in range(300)]
    report = audit_shard(write_shard(tmp_path / "many.jsonl", rows))

    assert report.max_targets_per_source == 300
    assert report.conflicting_source == pytest.approx(1.0)
    assert any(name.startswith("duplicate_source") for name in report.violations)
    assert any(name.startswith("conflicting_source") for name in report.violations)


def test_hangul_in_japanese_target_is_rejected(tmp_path: Path) -> None:
    rows = [
        (source, f"{_word(index, _JA_SYLLABLES)}村の 문장 は違う。")
        for index, (source, _) in enumerate(distinct_rows(300))
    ]
    report = audit_shard(write_shard(tmp_path / "leak.jsonl", rows), target_scripts=("ja",))

    assert report.foreign_script_target == pytest.approx(1.0)
    assert report.foreign_target_scripts == ["hangul"]
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

    japanese = audit_shard(path, source_key="kj", target_key="ko", target_scripts=("ja",))
    korean = audit_shard(path, source_key="kj", target_key="ko", target_scripts=("ko",))
    ignored = audit_shard(path, source_key="kj", target_key="ko")

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

    report = audit_shard(path, target_scripts=("ko",))

    assert report.foreign_script_target == pytest.approx(1.0)
    assert report.foreign_target_scripts == ["kana"]
    assert any(name.startswith("foreign_script_target") for name in report.violations)


def test_unknown_script_name_is_rejected(tmp_path: Path) -> None:
    path = write_shard(tmp_path / "x.jsonl", distinct_rows(10))

    with pytest.raises(ValueError, match="unknown script or language"):
        audit_shard(path, target_scripts=("klingon",))


def test_script_list_parses_a_comma_separated_value() -> None:
    assert AUDIT.script_list("ko") == ("ko",)
    assert AUDIT.script_list("kana, han") == ("kana", "han")
    assert AUDIT.script_list("") == ()
    with pytest.raises(ValueError):
        AUDIT.script_list("klingon")


def test_scripts_present_are_reported_for_both_sides(tmp_path: Path) -> None:
    path = write_shard(tmp_path / "scripts.jsonl", distinct_rows(300))

    report = audit_shard(path)

    assert report.source_scripts == ["hangul"]
    assert report.target_scripts == ["han", "kana"]
    assert report.foreign_target_scripts == []


def test_kana_in_korean_source_is_measured_but_not_gated(tmp_path: Path) -> None:
    rows = [
        (
            f"{_word(index, _KO_SYLLABLES)} 마을은 やっぱり 조용했다.",
            f"{_word(index, _JA_SYLLABLES)}村はやっぱり静かだった。",
        )
        for index in range(300)
    ]
    path = write_shard(tmp_path / "mixed.jsonl", rows)

    # Declaring the source as Korean measures the kana but never gates on it.
    measured = audit_shard(path, source_scripts=("ko",))
    # Declaring it as 한본어 says the mixture is expected, so nothing is foreign.
    expected = audit_shard(path, source_scripts=("kj",))

    assert measured.foreign_script_source == pytest.approx(1.0)
    assert measured.foreign_source_scripts == ["kana"]
    assert measured.passed, measured.violations
    assert expected.foreign_script_source == pytest.approx(0.0)


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
    report = audit_shard(write_shard(tmp_path / "leaky.jsonl", rows))

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

    report = audit_shard(path)

    assert report.rows == 303
    assert report.unreadable_rows == 3
    assert report.passed, report.violations


def test_shard_without_usable_rows_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    report = audit_shard(path)

    assert not report.passed
    assert report.violations == ["usable_rows 0 < 1"]


def test_custom_keys_are_honoured(tmp_path: Path) -> None:
    path = tmp_path / "custom.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(300):
            handle.write(
                json.dumps(
                    {
                        "pt-BR": f"Frase de origem número {index}.",
                        "zh-Hant": f"第 {index} 個目標句子。",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = audit_shard(path, source_key="pt-BR", target_key="zh-Hant")

    assert report.rows == 300
    assert report.unreadable_rows == 0


def test_audit_api_requires_explicit_distinct_nonempty_keys_before_reading(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(TypeError, match="source_key.*target_key"):
        AUDIT.audit_shard(missing)
    with pytest.raises(ValueError, match="source_key must be a non-empty"):
        AUDIT.audit_shard(missing, source_key="", target_key="fr")
    with pytest.raises(ValueError, match="target_key must be a non-empty"):
        AUDIT.audit_shard(missing, source_key="de", target_key=" ")
    with pytest.raises(ValueError, match="must be distinct"):
        AUDIT.audit_shard(missing, source_key="de", target_key="de")


def test_audit_split_configuration_fails_before_reading(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(ValueError, match="split fractions must be non-negative"):
        AUDIT.audit_shard(
            missing,
            source_key="de",
            target_key="fr",
            validation_fraction=-0.1,
        )
    with pytest.raises(ValueError, match="must be below 0.5"):
        AUDIT.audit_shard(
            missing,
            source_key="de",
            target_key="fr",
            validation_fraction=0.3,
            test_fraction=0.2,
        )


def test_thresholds_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        AUDIT.Thresholds(min_skeleton_ttr=2.0).validate()
    with pytest.raises(ValueError):
        AUDIT.Thresholds(max_near_duplicate_leak=-0.1).validate()


def test_main_returns_one_when_a_shard_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(300))
    bad = write_shard(
        tmp_path / "bad.jsonl",
        [("같은 원문이 계속 반복된다.", f"別の訳{index}です。") for index in range(300)],
    )
    report_path = tmp_path / "report.json"

    assert audit_main(["--json", str(report_path), str(good), str(bad)]) == 1

    printed = capsys.readouterr().out
    assert "good.jsonl" in printed and "PASS" in printed
    assert "bad.jsonl" in printed and "FAIL" in printed
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert [Path(entry["path"]).name for entry in payload] == ["good.jsonl", "bad.jsonl"]


def test_main_returns_zero_when_every_shard_passes(tmp_path: Path) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(300))

    assert audit_main([str(good)]) == 0


def test_audit_cli_requires_explicit_keys(tmp_path: Path) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(10))

    with pytest.raises(SystemExit) as error:
        AUDIT.main([str(good)])

    assert error.value.code == 2


def test_audit_cli_preflights_inputs_and_report_without_mutation(tmp_path: Path) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(10))
    original = good.read_bytes()
    missing = tmp_path / "missing.jsonl"
    report = tmp_path / "report.json"
    report.write_text("sentinel\n", encoding="utf-8")

    assert audit_main(["--json", str(report), str(good), str(missing)]) == 2
    assert report.read_text(encoding="utf-8") == "sentinel\n"
    assert audit_main(["--json", str(good), str(good)]) == 2
    assert good.read_bytes() == original
    assert audit_main([str(good), str(good)]) == 2


def test_audit_json_report_atomically_replaces_an_existing_file(tmp_path: Path) -> None:
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(300))
    report = tmp_path / "nested" / "report.json"
    report.parent.mkdir()
    report.write_text("old report\n", encoding="utf-8")

    assert audit_main(["--json", str(report), str(good)]) == 0

    assert json.loads(report.read_text(encoding="utf-8"))[0]["rows"] == 300
    assert not list(report.parent.glob(f".{report.name}.*.tmp"))


def test_main_reports_bad_input_with_exit_code_two(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    good = write_shard(tmp_path / "good.jsonl", distinct_rows(10))

    assert audit_main([str(missing)]) == 2
    assert audit_main(["--min-skeleton-ttr", "1.5", str(good)]) == 2
    assert audit_main(["--examples", "-1", str(good)]) == 2


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

    assert audit_main([str(path)]) == 1
    assert (
        audit_main(
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
