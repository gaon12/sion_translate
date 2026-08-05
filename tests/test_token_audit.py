from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sion_translate.cli.audit_tokens import main as audit_tokens_main
from sion_translate.data.prepare import INDEX_DTYPE
from sion_translate.token_audit import audit_indexed_token_exposure, audit_token_exposure
from sion_translate.tokenizer import SionTokenizer, train_tokenizer


def _corpus(path: Path) -> None:
    rows = [
        {"ko": "헉, 정말 깜짝 놀랐어!", "ja": "えっ、本当にびっくりした！"},
        {"ko": "둘은 붕어빵처럼 닮았다.", "ja": "二人は瓜二つだ。"},
        {"ko": "젠장, 또 늦었네.", "ja": "ちくしょう、また遅れた。"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def corpus_and_tokenizer(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "parallel.jsonl"
    _corpus(corpus)
    tokenizer = train_tokenizer(
        [str(corpus)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=0,
        seed_sentencepiece_size=100,
        validation_fraction=0.0,
        test_fraction=0.0,
        num_workers=1,
        num_threads=1,
    )
    return corpus, tokenizer


def test_audit_reports_each_translation_direction(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure([str(corpus)], tokenizer, rare_threshold=2)

    assert report["complete_scan"] is True
    assert report["physical_pairs"] == 3
    assert report["virtual_translation_examples"] == 6
    assert report["directions"]["ko-ja"]["examples"] == 3
    assert report["directions"]["ja-ko"]["examples"] == 3
    assert report["languages"]["ko"]["target_enabled"] is True
    assert report["languages"]["ja"]["target_enabled"] is True
    assert report["languages"]["ko"]["byte_fallback_rate"] >= 0.0
    assert report["global_target_frequency"]["observed_pieces"] > 0
    assert (
        report["global_target_frequency"]["eligible_pieces"]
        == report["global_target_frequency"]["observed_pieces"]
        + report["global_target_frequency"]["unused_pieces"]
    )
    assert report["lowest_global_target_exposure"]


def test_source_only_language_is_not_misreported_as_undertrained_target(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        source_only_languages=["ko"],
        rare_threshold=2,
    )

    assert set(report["directions"]) == {"ko-ja"}
    assert report["virtual_translation_examples"] == 3
    korean = report["languages"]["ko"]
    assert korean["target_enabled"] is False
    assert korean["target_frequency"]["eligible_pieces"] == 0
    assert korean["lowest_target_exposure"] == []


def test_prefix_scan_is_labelled_incomplete(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        max_physical_pairs=1,
    )
    assert report["physical_pairs"] == 1
    assert report["complete_scan"] is False


def test_invalid_audit_limits_are_rejected(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    with pytest.raises(ValueError, match="rare_threshold"):
        audit_token_exposure([str(corpus)], tokenizer, rare_threshold=0)
    with pytest.raises(ValueError, match="max_physical_pairs"):
        audit_token_exposure([str(corpus)], tokenizer, max_physical_pairs=-1)


def _write_legacy_v2_dataset(
    root: Path,
    side_a: list[int],
    side_b: list[int],
) -> None:
    split = root / "train"
    split.mkdir(parents=True)
    legacy_dtype = np.dtype(
        [
            ("ko_offset", "<u8"),
            ("ko_length", "<u4"),
            ("ja_offset", "<u8"),
            ("ja_length", "<u4"),
            ("ko_register", "u1"),
            ("ja_register", "u1"),
            ("source_id", "<u2"),
            ("quality_score", "u1"),
        ]
    )
    np.save(
        split / "00000.idx.npy",
        np.asarray([(0, len(side_a), 0, len(side_b), 0, 0, 0, 100)], dtype=legacy_dtype),
        allow_pickle=False,
    )
    np.asarray(side_a, dtype=np.uint32).tofile(split / "00000.ko.bin")
    np.asarray(side_b, dtype=np.uint32).tofile(split / "00000.ja.bin")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "sion-indexed-parallel-v2",
                "language_pair": ["ko", "ja"],
            }
        ),
        encoding="utf-8",
    )


def _write_v5_dataset(
    root: Path,
    side_a: list[int],
    side_b: list[int],
    *,
    forward_only: bool,
    source_only_languages: list[str],
) -> None:
    split = root / "train"
    split.mkdir(parents=True)
    row = (
        0,
        len(side_a),
        0,
        len(side_b),
        0,
        0,
        0,
        1,
        0,
        100,
        0,
        int(forward_only),
    )
    np.save(
        split / "00000.idx.npy",
        np.asarray([row], dtype=INDEX_DTYPE),
        allow_pickle=False,
    )
    np.asarray(side_a, dtype=np.uint32).tofile(split / "00000.src.bin")
    np.asarray(side_b, dtype=np.uint32).tofile(split / "00000.tgt.bin")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "sion-indexed-parallel-v5",
                "language_pair": ["ko", "ja"],
                "language_pairs": [["ko", "ja"]],
                "languages": ["ko", "ja"],
                "source_only_languages": source_only_languages,
            }
        ),
        encoding="utf-8",
    )


def test_indexed_v2_audit_counts_both_decoder_directions_exactly(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    korean = tokenizer.encode("젠장 또 늦었네")
    japanese = tokenizer.encode("ちくしょう また遅れた")
    dataset = tmp_path / "legacy"
    _write_legacy_v2_dataset(dataset, korean, japanese)

    report = audit_indexed_token_exposure(dataset, tokenizer_model)

    assert report["complete_scan"] is True
    assert report["count_basis"] == "stored_target_content_tokens"
    assert report["physical_pairs"] == 1
    assert report["virtual_translation_examples"] == 2
    assert report["directions"]["ko-ja"]["target_tokens"] == len(japanese)
    assert report["directions"]["ja-ko"]["target_tokens"] == len(korean)
    assert report["global_target_frequency"]["all_target_tokens"] == len(korean) + len(japanese)


@pytest.mark.parametrize(
    ("forward_only", "source_only_languages"),
    [(True, []), (False, ["ko"])],
)
def test_indexed_v5_audit_never_counts_a_suppressed_reverse_target(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    forward_only: bool,
    source_only_languages: list[str],
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    korean = tokenizer.encode("둘은 정말 똑 닮았다")
    japanese = tokenizer.encode("二人は本当に瓜二つだ")
    dataset = tmp_path / f"v5-{forward_only}-{len(source_only_languages)}"
    _write_v5_dataset(
        dataset,
        korean,
        japanese,
        forward_only=forward_only,
        source_only_languages=source_only_languages,
    )

    report = audit_indexed_token_exposure(dataset, tokenizer_model)

    assert set(report["directions"]) == {"ko-ja"}
    assert report["virtual_translation_examples"] == 1
    assert report["directions"]["ko-ja"]["target_tokens"] == len(japanese)
    assert report["global_target_frequency"]["all_target_tokens"] == len(japanese)
    assert report["languages"]["ko"]["target_frequency"]["total_occurrences"] == 0


def test_indexed_cli_mode_writes_a_json_report(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "legacy-cli"
    _write_legacy_v2_dataset(
        dataset,
        tokenizer.encode("정말 깜짝 놀랐어"),
        tokenizer.encode("本当にびっくりした"),
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--dataset",
            str(dataset),
            "--tokenizer",
            str(tokenizer_model),
            "--output",
            str(output),
        ],
    )

    audit_tokens_main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "sion-indexed-token-exposure-audit-v1"
    assert report["parameters"]["split"] == "train"


def test_indexed_audit_rejects_a_tokenizer_identity_mismatch(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "mismatched-tokenizer"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("둘은 똑 닮았다"),
        tokenizer.encode("二人は瓜二つだ"),
        forward_only=False,
        source_only_languages=[],
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"] = {"tokenizer_sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Tokenizer SHA-256"):
        audit_indexed_token_exposure(dataset, tokenizer_model)
