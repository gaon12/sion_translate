from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sion_translate.data.prepare as prepare_module
import sion_translate.token_audit as token_audit_module
from sion_translate.cli.audit_tokens import main as audit_tokens_main
from sion_translate.data.integrity import build_dataset_artifact_inventory
from sion_translate.data.prepare import INDEX_DTYPE, prepare_dataset
from sion_translate.data.record_metadata import RECORD_METADATA_INDEX_DTYPE
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


def _reauthenticate_current_payload(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_inventory"] = build_dataset_artifact_inventory(root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / prepare_module.PREPARE_COMPLETION_FILENAME).write_text(
        json.dumps(prepare_module._completion_payload(root, manifest)),
        encoding="utf-8",
    )


def _rewrite_current_preprocessing_option(root: Path, name: str, value: object) -> None:
    manifest_path = root / "manifest.json"
    raw_path = root / prepare_module.RAW_FINGERPRINT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_fingerprint = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest["preprocessing_options"][name] = value
    raw_fingerprint["preprocessing_options"][name] = value
    manifest["fingerprint"] = raw_fingerprint
    if name in manifest:
        manifest[name] = value
    raw_path.write_text(json.dumps(raw_fingerprint), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(root)


def _write_record_metadata_sidecar(index_path: Path, payloads: list[bytes]) -> None:
    index = np.load(index_path, allow_pickle=False)
    assert len(index) == len(payloads)
    offsets: list[int] = []
    offset = 0
    for payload in payloads:
        offsets.append(offset)
        offset += len(payload)
    prefix = index_path.name.removesuffix(".idx.npy")
    np.save(
        index_path.parent / f"{prefix}.meta.npy",
        np.asarray(
            [
                (row_offset, len(payload))
                for row_offset, payload in zip(offsets, payloads, strict=True)
            ],
            dtype=RECORD_METADATA_INDEX_DTYPE,
        ),
        allow_pickle=False,
    )
    (index_path.parent / f"{prefix}.meta.bin").write_bytes(b"".join(payloads))


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
        language_pair=("ko", "ja"),
        num_workers=1,
        num_threads=1,
    )
    return corpus, tokenizer


def test_audit_reports_each_translation_direction(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        rare_threshold=2,
    )

    assert report["schema"] == "sion-token-exposure-audit-v2"
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


def test_one_way_source_language_is_not_misreported_as_undertrained_target(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        translation_directions=(("ko", "ja"),),
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
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        max_physical_pairs=1,
    )
    assert report["physical_pairs"] == 1
    assert report["complete_scan"] is False


def test_invalid_audit_limits_are_rejected(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    with pytest.raises(ValueError, match="rare_threshold"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"),),
            rare_threshold=0,
        )
    with pytest.raises(ValueError, match="max_physical_pairs"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"),),
            max_physical_pairs=-1,
        )


def test_row_scoped_direction_counts_only_its_authenticated_target(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / "mixed-directions.jsonl"
    rows = [
        {"ko": "첫 번째 원문입니다.", "ja": "一つ目の原文です。"},
        {
            "ko": "두 번째 원문입니다.",
            "ja": "二つ目の原文です。",
            "training_direction": ["JA", "KO"],
        },
    ]
    corpus.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        filter_quality=False,
    )

    assert report["physical_pairs"] == 2
    assert report["virtual_translation_examples"] == 3
    assert report["directions"]["ko-ja"]["examples"] == 1
    assert report["directions"]["ja-ko"]["examples"] == 2


def test_row_direction_outside_authenticated_graph_fails_closed(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / "unauthenticated-direction.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "ko": "정방향만 학습했습니다.",
                "ja": "順方向だけを学習しました。",
                "training_direction": ["ja", "ko"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unauthenticated-direction\.jsonl:1:.*absent"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"),),
            filter_quality=False,
        )


def test_unscoped_synthetic_row_fails_closed(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / "generated.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "ko": "합성 입력입니다.",
                "ja": "合成入力です。",
                "metadata": {"synthetic": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic records require an explicit"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"), ("ja", "ko")),
            filter_quality=False,
        )


def test_unscoped_real_concat_output_expands_across_the_authenticated_graph(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / "concat_real_bitext.jsonl"
    corpus.write_text(
        json.dumps(
            {"ko": "실제 병렬 문장을 이어 붙였습니다.", "ja": "実並列文を連結しました。"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        filter_quality=False,
    )

    assert report["physical_pairs"] == 1
    assert report["virtual_translation_examples"] == 2
    assert set(report["directions"]) == {"ko-ja", "ja-ko"}


@pytest.mark.parametrize("prefix", ["bt_", "revise_", "synthetic_", "queue_bt_"])
def test_unscoped_generated_file_prefixes_fail_closed(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    prefix: str,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / f"{prefix}legacy.jsonl"
    corpus.write_text(
        json.dumps({"ko": "합성 원문입니다.", "ja": "合成原文です。"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic records require an explicit"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"), ("ja", "ko")),
            filter_quality=False,
        )


def test_custom_train_only_prefix_fails_closed_and_is_reported(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / "vendor_generated_parallel.jsonl"
    corpus.write_text(
        json.dumps(
            {"ko": "사용자 합성 원문입니다.", "ja": "利用者の合成原文です。"}, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic records require an explicit"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"),),
            train_only_prefixes=("vendor_generated_",),
            filter_quality=False,
        )

    corpus.write_text(
        json.dumps(
            {
                "ko": "사용자 합성 원문입니다.",
                "ja": "利用者の合成原文です。",
                "training_direction": ["KO", "JA"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        translation_directions=(("ko", "ja"),),
        train_only_prefixes=("vendor_generated_",),
        filter_quality=False,
    )

    assert "vendor_generated_" in report["parameters"]["direction_required_synthetic_prefixes"]


def test_legacy_raw_policy_arguments_only_validate_the_explicit_graph(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    report = audit_token_exposure(
        [str(corpus)],
        tokenizer,
        translation_directions=(("ko", "ja"),),
        source_only_languages=("KO",),
        bidirectional=True,
    )
    assert set(report["directions"]) == {"ko-ja"}

    with pytest.raises(ValueError, match="legacy.*contradicts"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("ko", "ja"),),
            bidirectional=True,
        )


def test_raw_cli_applies_custom_train_only_prefixes(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer = corpus_and_tokenizer
    corpus = tmp_path / "vendor_generated_cli.jsonl"
    corpus.write_text(
        json.dumps({"ko": "합성 원문입니다.", "ja": "合成原文です。"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tokenizer),
            "--translation-direction",
            "ko",
            "ja",
            "--train-only-prefix",
            "vendor_generated_",
            "--no-filter-quality",
        ],
    )

    with pytest.raises(ValueError, match="synthetic records require an explicit"):
        audit_tokens_main()


def test_duplicate_canonical_translation_directions_are_rejected(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer

    with pytest.raises(ValueError, match="duplicate translation direction"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            translation_directions=(("KO", "JA"), ("ko", "ja")),
        )


def test_raw_audit_rejects_conflicting_pair_arguments(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer

    with pytest.raises(ValueError, match="mutually exclusive"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            language_pair=("ko", "ja"),
            language_pairs=(("ko", "ja"),),
            translation_directions=(("ko", "ja"),),
        )


def test_raw_audit_rejects_reversed_duplicate_physical_pairs(
    corpus_and_tokenizer: tuple[Path, Path],
) -> None:
    corpus, tokenizer = corpus_and_tokenizer

    with pytest.raises(ValueError, match="duplicate or reversed physical pair"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer,
            language_pairs=(("KO", "ja"), ("JA", "ko")),
            translation_directions=(("ko", "ja"),),
        )


def test_bcp47_direction_graph_is_canonical_and_unambiguously_labelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcessor:
        def id_to_piece(self, token_id: int) -> str:
            return ("<unk>", "word")[token_id]

    class FakeTokenizer:
        languages = ("pt-BR", "zh-Hant")
        unk_id = 0
        processor = FakeProcessor()

        def __init__(self, _path: object) -> None:
            pass

        def __len__(self) -> int:
            return 2

        def encode(self, _text: str) -> list[int]:
            return [1]

    monkeypatch.setattr("sion_translate.token_audit.SionTokenizer", FakeTokenizer)
    corpus = tmp_path / "bcp47.jsonl"
    corpus.write_text(
        json.dumps({"PT-br": "Olá, mundo!", "zh-hant": "你好，世界！"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tokenizer_model = tmp_path / "fake.model"
    tokenizer_model.write_bytes(b"fake-tokenizer")

    report = audit_token_exposure(
        [str(corpus)],
        tokenizer_model,
        translation_directions=(("PT-br", "zh-hant"),),
        filter_quality=False,
    )

    assert report["parameters"]["language_pairs"] == [["pt-BR", "zh-Hant"]]
    assert report["parameters"]["translation_directions"] == [["pt-BR", "zh-Hant"]]
    assert set(report["directions"]) == {"pt-BR/zh-Hant"}
    assert report["languages"]["pt-BR"]["target_enabled"] is False
    assert report["languages"]["zh-Hant"]["target_enabled"] is True


def test_raw_cli_requires_an_explicit_translation_graph(
    corpus_and_tokenizer: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tokenizer),
        ],
    )

    with pytest.raises(SystemExit, match="require at least one --translation-direction"):
        audit_tokens_main()


def test_raw_cli_accepts_repeated_ordered_translation_directions(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, tokenizer = corpus_and_tokenizer
    output = tmp_path / "raw-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tokenizer),
            "--translation-direction",
            "ko",
            "ja",
            "--translation-direction",
            "ja",
            "ko",
            "--output",
            str(output),
        ],
    )

    audit_tokens_main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parameters"]["translation_directions"] == [
        ["ko", "ja"],
        ["ja", "ko"],
    ]
    assert set(report["directions"]) == {"ko-ja", "ja-ko"}


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
    language_pair: tuple[str, str] = ("ko", "ja"),
    translation_directions: list[list[str]] | None = None,
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
    if translation_directions is None:
        translation_directions = [list(language_pair)]
        if not forward_only and not source_only_languages:
            translation_directions.append([language_pair[1], language_pair[0]])
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "sion-indexed-parallel-v5",
                "language_pair": list(language_pair),
                "language_pairs": [list(language_pair)],
                "translation_directions": translation_directions,
                "languages": list(language_pair),
                "source_only_languages": source_only_languages,
            }
        ),
        encoding="utf-8",
    )


def test_bare_v6_format_claim_is_not_accepted_as_a_current_or_legacy_artifact(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "hand-built-v6"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("인증되지 않은 원문입니다."),
        tokenizer.encode("認証されていない原文です。"),
        forward_only=True,
        source_only_languages=[],
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "sion-indexed-parallel-v6"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unauthenticated v6 dataset"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_legacy_v4_generic_src_tgt_layout_allows_explicit_compatibility_policy(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    korean = tokenizer.encode("레거시 일반 원문입니다.")
    japanese = tokenizer.encode("旧形式の一般原文です。")
    dataset = tmp_path / "legacy-v4-generic"
    _write_v5_dataset(
        dataset,
        korean,
        japanese,
        forward_only=False,
        source_only_languages=[],
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "sion-indexed-parallel-v4"
    manifest.pop("translation_directions")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_indexed_token_exposure(
        dataset,
        tokenizer_model,
        bidirectional=True,
    )

    assert report["parameters"]["dataset_contract"] == "legacy-unverified-explicit-policy"
    assert report["parameters"]["integrity_assurance"]["level"] == ("legacy-payload-unverified")
    assert report["parameters"]["legacy_bidirectional_override"] is True
    assert report["virtual_translation_examples"] == 2
    assert set(report["directions"]) == {"ko-ja", "ja-ko"}


def test_current_indexed_audit_accepts_only_a_fully_published_prepare_artifact(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-published"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )

    report = audit_indexed_token_exposure(dataset, tokenizer_model)

    assert report["parameters"]["dataset_contract"] == "current-integrity-verified"
    assert report["parameters"]["integrity_assurance"] == {
        "level": "self-consistent-hashes-not-signed",
        "payload_sha256_reverified_after_scan": True,
        "manifest_sha256_reverified_after_scan": True,
        "tokenizer_sha256_reverified_after_scan": True,
        "cryptographically_signed": False,
    }
    assert report["physical_pairs"] == 3
    assert report["virtual_translation_examples"] == 6


def test_current_indexed_audit_accepts_authenticated_legacy_storage_side_stats(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-legacy-stats"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("stats_schema")
    for stats in (manifest["stats"], manifest["sources"][0]["stats"]):
        stats["ko_tokens"] = stats.pop("src_tokens")
        stats["ja_tokens"] = stats.pop("tgt_tokens")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(dataset)

    report = audit_indexed_token_exposure(dataset, tokenizer_model)

    assert report["parameters"]["dataset_contract"] == "current-integrity-verified"
    assert report["physical_pairs"] == 3


def test_current_indexed_audit_rejects_an_explicit_null_stats_schema(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-null-stats-schema"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stats_schema"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="stats_schema is unsupported"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_current_indexed_audit_rejects_per_source_legacy_keys_under_the_new_schema(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-mixed-source-stats"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_stats = manifest["sources"][0]["stats"]
    source_stats["ko_tokens"] = source_stats.pop("src_tokens")
    source_stats["ja_tokens"] = source_stats.pop("tgt_tokens")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="stats fields do not match their schema"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_current_indexed_audit_rejects_rebound_split_stats(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-bad-stats"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for stats in (manifest["stats"], manifest["sources"][0]["stats"]):
        stats["train"] += 1
        stats["valid_pairs"] += 1
    manifest["mean_quality_score"] = (
        manifest["stats"]["quality_score_sum"] / manifest["stats"]["valid_pairs"]
    )
    manifest["sources"][0]["mean_quality_score"] = (
        manifest["sources"][0]["stats"]["quality_score_sum"]
        / manifest["sources"][0]["stats"]["valid_pairs"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="split counts differ"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_current_indexed_audit_rejects_rebound_noncanonical_index_dtype(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-bad-index"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    index_path = next((dataset / "train").glob("*.idx.npy"))
    index = np.load(index_path, allow_pickle=False)
    wrong_dtype = np.dtype([*INDEX_DTYPE.descr[:-1], ("forward_only", "<u2")])
    rewritten = np.zeros(len(index), dtype=wrong_dtype)
    for name in INDEX_DTYPE.names or ():
        rewritten[name] = index[name]
    np.save(index_path, rewritten, allow_pickle=False)
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="index dtype is invalid"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_current_indexed_audit_rejects_rebound_incomplete_metadata_sidecar(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    corpus = tmp_path / "scoped-current.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "ko": "행 단위 방향이 있는 원문입니다.",
                "ja": "行単位の方向がある原文です。",
                "training_direction": ["ko", "ja"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "current-incomplete-metadata"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    metadata_path = next((dataset / "train").glob("*.meta.bin"))
    metadata_path.unlink()
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="Incomplete record metadata sidecar"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


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

    with pytest.raises(ValueError, match="requires an explicit bidirectional"):
        audit_indexed_token_exposure(dataset, tokenizer_model)

    report = audit_indexed_token_exposure(dataset, tokenizer_model, bidirectional=True)

    assert report["complete_scan"] is True
    assert report["count_basis"] == "stored_target_content_tokens"
    assert report["physical_pairs"] == 1
    assert report["virtual_translation_examples"] == 2
    assert report["directions"]["ko-ja"]["target_tokens"] == len(japanese)
    assert report["directions"]["ja-ko"]["target_tokens"] == len(korean)
    assert report["global_target_frequency"]["all_target_tokens"] == len(korean) + len(japanese)


def test_indexed_audit_does_not_invent_a_korean_japanese_pair(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "legacy-without-languages"
    _write_legacy_v2_dataset(
        dataset,
        tokenizer.encode("언어 메타데이터가 필요합니다"),
        tokenizer.encode("言語メタデータが必要です"),
    )
    (dataset / "manifest.json").write_text(
        json.dumps({"format": "sion-indexed-parallel-v2"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy indexed manifest language_pair"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


@pytest.mark.parametrize(
    ("forward_only", "source_only_languages"),
    [(True, []), (True, ["ko"])],
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
    assert report["languages"]["ko"]["target_enabled"] is False


def test_modern_indexed_audit_rejects_reverse_exposure_outside_manifest_graph(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "invalid-reverse"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("정방향 원문입니다"),
        tokenizer.encode("順方向の原文です"),
        forward_only=False,
        source_only_languages=[],
        translation_directions=[["ko", "ja"]],
    )

    with pytest.raises(ValueError, match="unauthenticated reverse direction"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_modern_indexed_audit_rejects_a_stored_forward_edge_outside_the_graph(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "invalid-stored-forward"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("저장된 한국어입니다"),
        tokenizer.encode("保存された日本語です"),
        forward_only=True,
        source_only_languages=[],
        translation_directions=[["ja", "ko"]],
    )

    with pytest.raises(ValueError, match="stored direction is absent"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_indexed_source_only_alias_collision_fails_closed(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "source-only-alias-collision"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("원문입니다"),
        tokenizer.encode("原文です"),
        forward_only=True,
        source_only_languages=["KO", "ko"],
    )

    with pytest.raises(ValueError, match="duplicate language identities"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_raw_and_indexed_one_way_bcp47_audits_have_matching_direction_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcessor:
        def id_to_piece(self, token_id: int) -> str:
            return ("<unk>", "word")[token_id]

    class FakeTokenizer:
        languages = ("pt-BR", "zh-Hant")
        unk_id = 0
        processor = FakeProcessor()

        def __init__(self, _path: object) -> None:
            pass

        def __len__(self) -> int:
            return 2

        def encode(self, _text: str) -> list[int]:
            return [1]

    monkeypatch.setattr("sion_translate.token_audit.SionTokenizer", FakeTokenizer)
    tokenizer_model = tmp_path / "fake.model"
    tokenizer_model.write_bytes(b"fake-tokenizer")
    corpus = tmp_path / "bcp47-parallel.jsonl"
    corpus.write_text(
        json.dumps({"PT-br": "Olá, mundo!", "zh-hant": "你好，世界！"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raw_report = audit_token_exposure(
        [str(corpus)],
        tokenizer_model,
        translation_directions=(("PT-br", "zh-hant"),),
        filter_quality=False,
    )
    dataset = tmp_path / "bcp47-indexed"
    _write_v5_dataset(
        dataset,
        [1],
        [1],
        forward_only=True,
        source_only_languages=[],
        language_pair=("PT-br", "zh-hant"),
        translation_directions=[["PT-br", "zh-hant"]],
    )

    indexed_report = audit_indexed_token_exposure(dataset, tokenizer_model)

    assert indexed_report["parameters"]["language_pairs"] == [["pt-BR", "zh-Hant"]]
    assert indexed_report["parameters"]["translation_directions"] == [["pt-BR", "zh-Hant"]]
    assert set(indexed_report["directions"]) == set(raw_report["directions"])
    for direction in indexed_report["directions"]:
        for field in ("examples", "source_tokens", "target_tokens", "mean_target_tokens"):
            assert (
                indexed_report["directions"][direction][field]
                == raw_report["directions"][direction][field]
            )
    assert (
        indexed_report["virtual_translation_examples"] == raw_report["virtual_translation_examples"]
    )
    assert (
        indexed_report["global_target_frequency"]["all_target_tokens"]
        == raw_report["global_target_frequency"]["all_target_tokens"]
    )
    assert (
        {
            language: values["target_enabled"]
            for language, values in indexed_report["languages"].items()
        }
        == {
            language: values["target_enabled"]
            for language, values in raw_report["languages"].items()
        }
        == {"pt-BR": False, "zh-Hant": True}
    )


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
            "--bidirectional",
            "--output",
            str(output),
        ],
    )

    audit_tokens_main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "sion-indexed-token-exposure-audit-v2"
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


@pytest.mark.parametrize(
    ("generation", "accepted"),
    [
        ("sion-prepare-v8", True),
        ("sion-prepare-v9", True),
        ("sion-prepare-v999-forged", False),
    ],
)
def test_legacy_v6_accepts_only_known_historical_preprocessing_generations(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    generation: str,
    accepted: bool,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / f"legacy-v6-{accepted}"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("과거 세대 원문입니다."),
        tokenizer.encode("過去世代の原文です。"),
        forward_only=True,
        source_only_languages=[],
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "format": prepare_module.INDEX_FORMAT,
            "preprocessing_schema": generation,
            "fingerprint": {"preprocessing_schema": generation},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (dataset / prepare_module.RAW_FINGERPRINT_FILENAME).write_text(
        json.dumps({"preprocessing_schema": generation}),
        encoding="utf-8",
    )

    if not accepted:
        with pytest.raises(ValueError, match="unknown historical preprocessing"):
            audit_indexed_token_exposure(dataset, tokenizer_model)
        return

    report = audit_indexed_token_exposure(dataset, tokenizer_model)
    assert report["parameters"]["dataset_contract"] == ("legacy-unverified-explicit-policy")


@pytest.mark.parametrize("field", ["src_offset", "src_language_id"])
def test_legacy_generic_index_rejects_coercible_wrong_field_dtypes(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    field: str,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / f"legacy-wrong-{field}"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("잘못된 자료형의 원문입니다."),
        tokenizer.encode("誤った型の原文です。"),
        forward_only=True,
        source_only_languages=[],
    )
    index_path = dataset / "train" / "00000.idx.npy"
    index = np.load(index_path, allow_pickle=False)
    wrong_dtype = np.dtype(
        [
            (
                name,
                np.dtype("<f8") if name == field else index.dtype.fields[name][0],
            )
            for name in index.dtype.names or ()
        ]
    )
    rewritten = np.zeros(len(index), dtype=wrong_dtype)
    for name in index.dtype.names or ():
        rewritten[name] = index[name]
    np.save(index_path, rewritten, allow_pickle=False)

    with pytest.raises(ValueError, match=rf"field '{field}' has invalid dtype"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_legacy_generic_index_without_forward_flag_infers_one_way_from_graph(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "legacy-inferred-one-way"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("정방향만 학습합니다."),
        tokenizer.encode("正方向だけを学習します。"),
        forward_only=True,
        source_only_languages=[],
        translation_directions=[["ko", "ja"]],
    )
    index_path = dataset / "train" / "00000.idx.npy"
    index = np.load(index_path, allow_pickle=False)
    no_forward_dtype = np.dtype([item for item in INDEX_DTYPE.descr if item[0] != "forward_only"])
    rewritten = np.zeros(len(index), dtype=no_forward_dtype)
    for name in no_forward_dtype.names or ():
        rewritten[name] = index[name]
    np.save(index_path, rewritten, allow_pickle=False)

    report = audit_indexed_token_exposure(dataset, tokenizer_model)

    assert report["forward_only_pairs"] == 1
    assert report["virtual_translation_examples"] == 1
    assert set(report["directions"]) == {"ko-ja"}


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("src_offset", "offsets are not contiguous"),
        ("src_length", "token count does not match"),
    ],
)
def test_legacy_generic_index_rejects_tampered_offsets_and_lengths(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / f"legacy-tampered-{field}"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("오프셋과 길이를 확인합니다."),
        tokenizer.encode("オフセットと長さを確認します。"),
        forward_only=True,
        source_only_languages=[],
    )
    index_path = dataset / "train" / "00000.idx.npy"
    rewritten = np.load(index_path, allow_pickle=False).copy()
    rewritten[field][0] += 1
    np.save(index_path, rewritten, allow_pickle=False)

    with pytest.raises(ValueError, match=message):
        audit_indexed_token_exposure(dataset, tokenizer_model)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("filter_quality", 1, "filter_quality must be boolean"),
        ("validation_fraction", -0.1, "finite and non-negative"),
        ("max_tokens_per_side", 0, "must be a positive integer"),
        ("max_tokens_per_side", 1, "exceeds configured max_tokens_per_side"),
    ],
)
def test_current_indexed_audit_validates_preprocessing_values_and_row_limits(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    option: str,
    value: object,
    message: str,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / f"current-option-{option}-{value}"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    _rewrite_current_preprocessing_option(dataset, option, value)

    with pytest.raises(ValueError, match=message):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_current_indexed_audit_revalidates_payload_inventory_after_scan(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / "current-toctou-payload"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    token_path = next((dataset / "train").glob("*.tgt.bin"))
    tokenizer = SionTokenizer(tokenizer_model)
    original_validator = token_audit_module.validate_dataset_artifact_inventory
    calls = 0

    def mutate_after_first_validation(root: Path, manifest: object) -> str | None:
        nonlocal calls
        calls += 1
        digest = original_validator(root, manifest)  # type: ignore[arg-type]
        if calls == 1:
            tokens = np.fromfile(token_path, dtype=np.uint32)
            assert tokens.size
            tokens[0] = np.uint32((int(tokens[0]) + 1) % len(tokenizer))
            tokens.tofile(token_path)
        return digest

    monkeypatch.setattr(
        token_audit_module,
        "validate_dataset_artifact_inventory",
        mutate_after_first_validation,
    )

    with pytest.raises(RuntimeError, match="dataset contract changed") as captured:
        audit_indexed_token_exposure(dataset, tokenizer_model)
    assert "artifact SHA-256 mismatch" in str(captured.value.__cause__)
    assert calls == 2


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("manifest", "dataset contract changed during token audit"),
        ("tokenizer", "Tokenizer changed during indexed token audit"),
    ],
)
def test_current_indexed_audit_revalidates_manifest_and_tokenizer_after_scan(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / f"current-toctou-{target}"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    mutation_path = dataset / "manifest.json" if target == "manifest" else tokenizer_model
    original_add_direction_totals = token_audit_module._add_direction_totals
    mutated = False

    def mutate_during_scan(*args: object, **kwargs: object) -> int:
        nonlocal mutated
        if not mutated:
            mutation_path.write_bytes(mutation_path.read_bytes() + b" ")
            mutated = True
        return original_add_direction_totals(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(token_audit_module, "_add_direction_totals", mutate_during_scan)

    with pytest.raises(RuntimeError, match=message):
        audit_indexed_token_exposure(dataset, tokenizer_model)
    assert mutated is True


def test_raw_audit_revalidates_tokenizer_bytes_after_scan(
    corpus_and_tokenizer: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    original_file_sha256 = token_audit_module.file_sha256
    tokenizer_hash_calls = 0

    def mutate_before_second_hash(path: str | Path) -> str:
        nonlocal tokenizer_hash_calls
        candidate = Path(path)
        if candidate == Path(tokenizer_model):
            tokenizer_hash_calls += 1
            if tokenizer_hash_calls == 2:
                candidate.write_bytes(candidate.read_bytes() + b"tampered")
        return original_file_sha256(candidate)

    monkeypatch.setattr(token_audit_module, "file_sha256", mutate_before_second_hash)

    with pytest.raises(RuntimeError, match="Tokenizer changed"):
        audit_token_exposure(
            [str(corpus)],
            tokenizer_model,
            translation_directions=(("ko", "ja"),),
            filter_quality=False,
        )


def test_mixed_nested_and_list_records_classify_synthetic_scope_per_expanded_pair(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    corpus = tmp_path / "mixed-records.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_language": "ko",
                        "target_language": "ja",
                        "source": "실제 데이터입니다.",
                        "target": "実データです。",
                    },
                    {
                        "metadata": {"synthetic": True},
                        "ko": ["합성 원문 하나.", "합성 원문 둘."],
                        "ja": ["合成原文一。", "合成原文二。"],
                        "training_direction": ["ko", "ja"],
                    },
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_token_exposure(
        [str(corpus)],
        tokenizer_model,
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        filter_quality=False,
    )

    assert report["physical_pairs"] == 3
    assert report["virtual_translation_examples"] == 4
    assert report["directions"]["ko-ja"]["examples"] == 3
    assert report["directions"]["ja-ko"]["examples"] == 1


def test_current_indexed_audit_preseeds_zero_example_authenticated_directions(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    corpus = tmp_path / "scoped-one-way.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "ko": "이 행은 정방향만 학습합니다.",
                "ja": "この行は正方向だけ学習します。",
                "training_direction": ["ko", "ja"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "current-zero-edge"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )

    report = audit_indexed_token_exposure(dataset, tokenizer_model)
    assert report["directions"]["ko-ja"]["examples"] == 1
    assert report["directions"]["ja-ko"]["examples"] == 0

    output = tmp_path / "zero-edge-report.json"
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
    with pytest.raises(SystemExit, match="zero-example-directions=ja-ko"):
        audit_tokens_main()
    assert json.loads(output.read_text(encoding="utf-8"))["directions"]["ja-ko"]["examples"] == 0


@pytest.mark.parametrize(
    ("bad_payload", "message"),
    [
        (b"{", "record metadata is invalid"),
        (
            b'{"training_direction":["ko","ja"]}',
            "record metadata direction contradicts its index flags",
        ),
    ],
)
def test_current_indexed_audit_decodes_every_sidecar_row_and_checks_direction_parity(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    bad_payload: bytes,
    message: str,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    dataset = tmp_path / f"current-sidecar-{len(bad_payload)}"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    index_path = next((dataset / "train").glob("*.idx.npy"))
    row_count = len(np.load(index_path, allow_pickle=False))
    payloads = [b""] * row_count
    payloads[-1] = bad_payload
    _write_record_metadata_sidecar(index_path, payloads)
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match=message):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_current_indexed_audit_rejects_synthetic_rows_outside_train(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    corpus = tmp_path / "bt_move_to_validation.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "ko": "합성 학습 원문입니다.",
                "ja": "合成学習の原文です。",
                "training_direction": ["ko", "ja"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "current-synthetic-validation"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    train_index = next((dataset / "train").glob("*.idx.npy"))
    moved_rows = len(np.load(train_index, allow_pickle=False))
    for artifact in list((dataset / "train").iterdir()):
        artifact.replace(dataset / "validation" / artifact.name)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for stats in (manifest["stats"], manifest["sources"][0]["stats"]):
        stats["train"] -= moved_rows
        stats["validation"] += moved_rows
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="synthetic rows must be train-only"):
        audit_indexed_token_exposure(dataset, tokenizer_model, split="validation")


def test_current_indexed_audit_requires_synthetic_flag_for_synthetic_source_file(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    corpus = tmp_path / "bt_missing_flag.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "ko": "합성 파일의 원문입니다.",
                "ja": "合成ファイルの原文です。",
                "training_direction": ["ko", "ja"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "current-synthetic-source"
    prepare_dataset(
        [str(corpus)],
        tokenizer_model,
        dataset,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        prevent_target_leakage=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        num_workers=1,
    )
    index_path = next((dataset / "train").glob("*.idx.npy"))
    index = np.load(index_path, allow_pickle=False)
    removed = int(np.count_nonzero(index["synthetic"]))
    rewritten = index.copy()
    rewritten["synthetic"] = 0
    np.save(index_path, rewritten, allow_pickle=False)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stats"]["synthetic_pairs"] -= removed
    manifest["sources"][0]["stats"]["synthetic_pairs"] -= removed
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reauthenticate_current_payload(dataset)

    with pytest.raises(ValueError, match="synthetic-file source contains a row"):
        audit_indexed_token_exposure(dataset, tokenizer_model)


def test_legacy_path_only_tokenizer_identity_is_reported_as_unverified(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = tmp_path / "legacy-path-tokenizer"
    _write_v5_dataset(
        dataset,
        tokenizer.encode("경로만 저장된 토크나이저입니다."),
        tokenizer.encode("パスだけ保存されたトークナイザーです。"),
        forward_only=True,
        source_only_languages=[],
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_model"] = str(tokenizer_model)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    identity = audit_indexed_token_exposure(dataset, tokenizer_model)["parameters"][
        "tokenizer_identity"
    ]
    assert identity["verified_against_manifest"] is False
    assert identity["mutable_path_match"] is True
    assert identity["assurance"] == "mutable-path-match-unverified"


def test_cli_fails_closed_for_global_zero_scan_and_counts_unused_as_below_threshold(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer_model = corpus_and_tokenizer
    corpus = tmp_path / "wrong-language.jsonl"
    corpus.write_text(
        json.dumps({"en": "No configured pair.", "de": "Kein konfiguriertes Paar."}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "zero-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tokenizer_model),
            "--translation-direction",
            "ko",
            "ja",
            "--no-filter-quality",
            "--fail-rare-pieces",
            "0",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as captured:
        audit_tokens_main()

    message = str(captured.value)
    assert "physical_pairs=0" in message
    assert "zero-example-directions=ko-ja" in message
    assert "below_threshold_pieces=" in message
    report = json.loads(output.read_text(encoding="utf-8"))
    frequency = report["global_target_frequency"]
    assert frequency["below_threshold_pieces"] == (
        frequency["unused_pieces"] + frequency["rare_observed_pieces"]
    )
    assert frequency["unused_pieces"] > 0


def test_cli_combined_rare_gate_uses_remaining_below_threshold_pieces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel_report = {
        "physical_pairs": 1,
        "directions": {"ko-ja": {"examples": 1}},
        "languages": {
            "ko": {"byte_fallback_rate": 0.0},
            "ja": {"byte_fallback_rate": 0.0},
        },
        "global_target_frequency": {"below_threshold_pieces": 50},
        "global_target_counts": np.zeros(2, dtype=np.uint64),
    }
    monkeypatch.setattr(
        "sion_translate.cli.audit_tokens.audit_token_exposure",
        lambda *_args, **_kwargs: dict(parallel_report),
    )
    monkeypatch.setattr(
        "sion_translate.cli.audit_tokens.discover_monolingual_sources",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "sion_translate.cli.audit_tokens.audit_monolingual_token_exposure",
        lambda *_args, **_kwargs: {"counts": np.zeros(2, dtype=np.uint64)},
    )
    monkeypatch.setattr(
        "sion_translate.cli.audit_tokens.combine_target_exposure",
        lambda *_args, **_kwargs: {"still_below_threshold": 0},
    )
    output = tmp_path / "combined-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--input",
            "dummy.jsonl",
            "--tokenizer",
            "dummy.model",
            "--translation-direction",
            "ko",
            "ja",
            "--monolingual-corpus",
            "dummy-corpus",
            "--fail-rare-pieces",
            "0",
            "--output",
            str(output),
        ],
    )

    audit_tokens_main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["global_target_frequency"]["below_threshold_pieces"] == 50
    assert report["combined_stages"]["still_below_threshold"] == 0


def test_cli_report_replacement_is_atomic(
    corpus_and_tokenizer: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, tokenizer_model = corpus_and_tokenizer
    output = tmp_path / "atomic-report.json"
    output.write_text("previous-report\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "sion-audit-tokens",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tokenizer_model),
            "--translation-direction",
            "ko",
            "ja",
            "--no-filter-quality",
            "--output",
            str(output),
        ],
    )

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("sion_translate.cli.audit_tokens.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        audit_tokens_main()
    assert output.read_text(encoding="utf-8") == "previous-report\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--input", "raw.jsonl", "--split", "train"],
            "--split applies only to indexed --dataset scans",
        ),
        (
            ["--dataset", "indexed", "--no-filter-quality"],
            "--filter-quality applies only to raw --input scans",
        ),
        (
            ["--input", "raw.jsonl", "--monolingual-max-lines", "1"],
            "--monolingual-max-lines requires --monolingual-corpus",
        ),
    ],
)
def test_cli_rejects_mode_inapplicable_flags_before_opening_inputs(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["sion-audit-tokens", *arguments, "--tokenizer", "missing.model"],
    )

    with pytest.raises(SystemExit, match=message):
        audit_tokens_main()
