from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.cli.audit_data import main as audit_main
from sion_translate.data.audit import audit_dataset


def _write_rows(path: Path, rows: list[object | str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    good = {"ko": "안녕하세요", "ja": "こんにちは"}
    _write_rows(
        first,
        [
            good,
            good,
            {"ko": "漢字", "ja": "漢字"},
            {"ko": "한국어문장", "ja": "あ"},
            {"ko": "こんにちは", "ja": "안녕하세요"},
            {"ko": "정상\u0001문장", "ja": "正常な文"},
            {"ko": "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ", "ja": "わかりました"},
            {"ko": "누락"},
            {"ko": "", "ja": "空"},
            {"ko": 123, "ja": "数字"},
            "{broken json",
            [],
        ],
    )
    _write_rows(
        second,
        [
            {"ko": "좋은 아침입니다", "ja": "おはようございます"},
            {"ko": "감사합니다", "ja": "ありがとうございます"},
        ],
    )
    return first, second


def test_streaming_audit_reports_deterministic_quality_counters(tmp_path: Path) -> None:
    first, second = _write_fixture(tmp_path)
    options = {
        "language_pair": ("ko", "ja"),
        "max_length_ratio": 4.0,
        "sample_size": 100,
        "seed": 17,
        "hll_precision": 8,
        "exact_unique_limit": 100,
        "max_issue_examples": 20,
        "min_language_check_chars": 2,
    }
    report = audit_dataset([str(tmp_path / "*.jsonl")], **options)
    repeated = audit_dataset([str(second), str(first)], **options)

    assert report == repeated
    assert report["schema"] == "sion-raw-dataset-audit-v2"
    assert report["global"]["file_count"] == 2
    assert report["global"]["bytes"] == first.stat().st_size + second.stat().st_size
    assert report["global"]["rows"] == 14
    assert report["global"]["valid"] == 9
    assert report["global"]["invalid"] == 2
    assert report["global"]["missing"] == 2
    assert report["global"]["non_string"] == 1
    assert report["global"]["invalid_breakdown"] == {
        "invalid_json": 1,
        "invalid_record_type": 1,
    }

    assert report["global"]["unique_pairs"]["exact_count"] == 8
    estimate = report["global"]["unique_pairs"]["hyperloglog_estimate"]
    assert estimate["label"] == "approximate_unique_pair_estimate"
    assert estimate["registers"] == 256
    assert 6 <= estimate["count"] <= 10

    assert report["global"]["signals"] == {
        "control_characters": 1,
        "excessive_repetition": 1,
        "identical_text": 1,
        "ja_script_mismatch": 1,
        "ko_script_mismatch": 2,
        "length_ratio": 1,
        "too_short": 1,
    }
    assert report["global"]["quality_pass_count"] == 4
    assert report["global"]["quality_pass_rate"] == 0.44444444
    assert len(report["global"]["issue_examples"]) == 10

    lengths = report["global"]["character_lengths"]
    assert lengths["ko"]["count"] == 9
    assert lengths["ko"]["sample_count"] == 9
    assert lengths["ko"]["total_chars"] > 0
    assert set(lengths["ko"]["sampled_percentiles_nearest_rank"]) == {
        "p50",
        "p95",
        "p99",
    }

    files = {Path(item["source"]).name: item for item in report["files"]}
    assert files["a.jsonl"]["rows"] == 12
    assert files["a.jsonl"]["valid"] == 7
    assert files["b.jsonl"]["rows"] == 2
    assert files["b.jsonl"]["quality_pass_count"] == 2
    assert files["a.jsonl"]["source_share"]["valid"] == 0.77777778
    assert files["b.jsonl"]["source_share"]["valid"] == 0.22222222


def test_audit_bounds_samples_and_marks_exact_count_unavailable(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = audit_dataset(
        [str(tmp_path)],
        language_pair=("ko", "ja"),
        sample_size=3,
        seed=99,
        hll_precision=8,
        exact_unique_limit=2,
        min_language_check_chars=2,
    )

    unique = report["global"]["unique_pairs"]
    assert unique["exact_count"] is None
    assert unique["exact_count_available"] is False
    assert unique["exact_tracking_limit"] == 2
    assert report["global"]["character_lengths"]["pair"]["sample_count"] == 3


def test_audit_cli_writes_json_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    output = tmp_path / "reports" / "audit.json"

    audit_main(
        [
            "--input",
            str(tmp_path / "*.jsonl"),
            "--language-pair",
            "ko",
            "ja",
            "--output",
            str(output),
            "--max-ratio",
            "4",
            "--sample-size",
            "3",
            "--seed",
            "23",
            "--hll-precision",
            "8",
            "--script-min-chars",
            "2",
        ]
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["global"]["rows"] == 14
    assert written["parameters"]["sample_size"] == 3
    assert written["parameters"]["max_length_ratio"] == 4.0
    assert written["parameters"]["language_pairs"] == [["ko", "ja"]]


def test_audit_supports_arbitrary_canonical_language_pairs(tmp_path: Path) -> None:
    source = tmp_path / "multilingual.jsonl"
    _write_rows(
        source,
        [
            {"PT-br": "Olá, mundo inteiro.", "zh-hant": "這是一個完整的句子。"},
            {
                "pt-BR": "Primeiro valor.",
                "PT-br": "Segundo valor.",
                "zh-Hant": "另一個句子。",
            },
            {"pt-BR": "Somente a origem."},
            {"pt-BR": "Texto normal.", "zh-Hant": 123},
        ],
    )
    output = tmp_path / "arbitrary-audit.json"

    audit_main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--language-pair",
            "PT-br",
            "zh-hant",
            "--hll-precision",
            "8",
            "--script-min-chars",
            "2",
            "--max-issue-examples",
            "10",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parameters"]["language_pairs"] == [["pt-BR", "zh-Hant"]]
    assert report["global"]["input_rows"] == 4
    assert report["global"]["rows"] == 4
    assert report["global"]["valid"] == 1
    assert report["global"]["invalid"] == 1
    assert report["global"]["missing"] == 1
    assert report["global"]["non_string"] == 1
    assert report["global"]["invalid_breakdown"] == {"duplicate_language_key": 1}
    assert set(report["global"]["character_lengths"]) == {
        "pt-BR",
        "zh-Hant",
        "pair",
    }
    assert "pt-BR_script_mismatch" in report["global"]["signals"]
    assert "zh-Hant_script_mismatch" in report["global"]["signals"]
    assert "ko_script_mismatch" not in report["global"]["signals"]
    examples = report["global"]["issue_examples"]
    assert examples[0]["issues"] == ["duplicate_language_key"]
    assert "pt-BR_preview" in examples[0]
    assert "zh-Hant_preview" in examples[0]


def test_audit_requires_and_separates_a_multigraph(tmp_path: Path) -> None:
    source = tmp_path / "multigraph.jsonl"
    _write_rows(
        source,
        [
            {"EN": "A complete English sentence.", "de": "Ein vollständiger deutscher Satz."},
            {"sw": "Hii ni sentensi kamili.", "AR": "هذه جملة عربية كاملة."},
        ],
    )

    with pytest.raises(ValueError, match="explicit language_pair"):
        audit_dataset([str(source)], hll_precision=8)

    report = audit_dataset(
        [str(source)],
        language_pairs=(("en", "de"), ("sw", "ar")),
        hll_precision=8,
        min_language_check_chars=2,
    )

    assert report["parameters"]["language_pairs"] == [
        ["en", "de"],
        ["sw", "ar"],
    ]
    assert report["global"]["input_rows"] == 2
    assert report["global"]["rows"] == 2
    assert report["global"]["valid"] == 2
    assert report["global"]["missing"] == 0
    assert set(report["global"]["character_lengths"]) == {
        "en",
        "de",
        "sw",
        "ar",
        "pair",
    }


def test_audit_uses_the_training_record_expansion_contract(tmp_path: Path) -> None:
    source = tmp_path / "heterogeneous.jsonl"
    _write_rows(
        source,
        [
            {
                "source_language": "PT-br",
                "target_language": "ZH-hant",
                "source": "Uma frase explícita completa.",
                "translation": "這是一個完整的明確句子。",
                "synthetic": True,
                "training_direction": ["pt-BR", "zh-Hant"],
            },
            {
                "records": [
                    {"pt-BR": "Primeira frase aninhada.", "zh-Hant": "第一個巢狀句子。"},
                    {"pt-BR": "Segunda frase aninhada.", "zh-Hant": "第二個巢狀句子。"},
                ]
            },
            {
                "pt-BR": ["Primeira frase em lista.", "Segunda frase em lista."],
                "zh-Hant": ["第一個列表句子。", "第二個列表句子。"],
            },
        ],
    )

    report = audit_dataset(
        [str(source)],
        language_pair=("pt-br", "ZH-hant"),
        hll_precision=8,
        min_language_check_chars=2,
    )

    assert report["global"]["input_rows"] == 3
    assert report["global"]["rows"] == 5
    assert report["global"]["valid"] == 5
    assert report["global"]["missing"] == 0
    assert report["global"]["non_string"] == 0
    assert report["global"]["invalid"] == 0
    assert report["global"]["unique_pairs"]["exact_count"] == 5
