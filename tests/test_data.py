from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import sion_translate.data.prepare as prepare_module
from sion_translate.data import (
    DistributedBucketBatchSampler,
    IndexedParallelDataset,
    SionBatchCollator,
    prepare_dataset,
)
from sion_translate.fingerprint import DatasetFingerprint, build_dataset_fingerprint
from sion_translate.data.quality import (
    QualityPolicy,
    apply_record_quality_profile,
    assess_pair,
    language_fraction,
)
from sion_translate.data.prepare import infer_register, protect_shared_spans
from sion_translate.glossary import restore_targets
from sion_translate.structured import (
    extract_structured_spans,
    mask_structured_spans,
    structured_similarity,
)
from sion_translate.tokenizer import SLOT_SYMBOLS, SionTokenizer, train_tokenizer


def write_tiny_jsonl(path: Path) -> None:
    rows = []
    for index in range(80):
        rows.append(
            {
                "ko": f"테스트 문장 {index}개입니다. 제품 코드 A-{index}를 확인하세요.",
                "ja": f"テスト文は{index}個です。製品コードA-{index}を確認してください。",
            }
        )
    rows.append(rows[0])
    rows.append({"ko": "", "ja": "空の翻訳は除外します。"})
    rows.append(
        {
            "ko": "오류문자열오류문자열오류문자열오류문자열오류문자열",
            "ja": "エラー文字列エラー文字列エラー文字列エラー文字列エラー文字列",
        }
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write("{broken json\n")
    with path.open("ab") as handle:
        handle.write(b"\xff\n")
        handle.write(b'{"ko":"\\ud800","ja":"\\u65e5\\u672c\\u8a9e\\u3067\\u3059"}\n')


class _FakePrepareTokenizer:
    languages = ("ko", "ja")

    def __init__(self, _model_path: str | Path) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return [4 + byte for byte in text.encode("utf-8")]


@pytest.mark.parametrize(
    ("text", "base_language", "variant_language"),
    [
        ("안녕하세요.", "ko", "KO-kr"),
        ("こんにちはです。", "ja", "JA-jp"),
    ],
)
def test_register_inference_inherits_the_primary_language_policy(
    text: str,
    base_language: str,
    variant_language: str,
) -> None:
    assert infer_register(text, variant_language) == infer_register(text, base_language) == 2


def _write_atomic_prepare_source(path: Path, *, marker: str = "A") -> None:
    path.write_text(
        json.dumps(
            {
                "ko": f"충분히 긴 한국어 원문 {marker}1입니다.",
                "ja": f"十分に長い日本語の原文{marker}1です。",
                "marker": marker,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _atomic_prepare_fingerprint(source: Path, tokenizer: Path) -> DatasetFingerprint:
    policy = QualityPolicy()
    options = prepare_module.prepare_preprocessing_options(
        shard_size=8,
        validation_fraction=0.0,
        test_fraction=0.0,
        max_tokens_per_side=510,
        quality_policy=policy,
        filter_quality=False,
        prevent_target_leakage=True,
        approximate_split=False,
        dedup_backend="memory",
        source_only_languages=(),
        translation_directions=(("ko", "ja"), ("ja", "ko")),
        train_only_prefixes=prepare_module.DEFAULT_TRAIN_ONLY_PREFIXES,
        synthetic_sampling_weight=0.25,
        language_pair_count=1,
    )
    return build_dataset_fingerprint(
        [source],
        language_pairs=(("ko", "ja"),),
        tokenizer_model=tokenizer,
        preprocessing_options=options,
    )


def _prepare_atomic_dataset(
    source_pattern: str,
    tokenizer: Path,
    output: Path,
    *,
    expected_fingerprint: DatasetFingerprint,
) -> prepare_module.PrepareStats:
    return prepare_module.prepare_dataset(
        [source_pattern],
        tokenizer,
        output,
        shard_size=8,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        synthetic_sampling_weight=0.25,
        num_workers=1,
        expected_fingerprint=expected_fingerprint,
    )


def test_legacy_unidirectional_runtime_does_not_require_current_forward_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "parallel.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=_atomic_prepare_fingerprint(source, tokenizer),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing_schema"] = "sion-prepare-v8"
    manifest["fingerprint"]["preprocessing_schema"] = "sion-prepare-v8"
    manifest.pop("translation_directions")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    raw_fingerprint_path = output / "raw_fingerprint.json"
    raw_fingerprint = json.loads(raw_fingerprint_path.read_text(encoding="utf-8"))
    raw_fingerprint["preprocessing_schema"] = "sion-prepare-v8"
    raw_fingerprint_path.write_text(json.dumps(raw_fingerprint), encoding="utf-8")

    dataset = IndexedParallelDataset(
        output,
        "train",
        bidirectional=True,
        legacy_bidirectional=False,
        verify_integrity=False,
    )

    assert dataset.translation_directions == (("ko", "ja"),)
    assert len(dataset) == dataset.pair_count == 1


def test_tokenizer_prepare_dataset_and_collate(tmp_path: Path) -> None:
    # FLORES 원천으로 쓰는 실제 파일명도 특별 취급 없이 일반 세 분할을 따른다.
    source = tmp_path / "data22.jsonl"
    write_tiny_jsonl(source)
    tokenizer_dir = tmp_path / "tokenizer"
    model_path = train_tokenizer(
        [str(source)],
        tokenizer_dir,
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
    )
    tokenizer = SionTokenizer(model_path)
    assert len(tokenizer) >= 300
    assert tokenizer.decode(tokenizer.encode("한일翻訳テスト"))

    dataset_dir = tmp_path / "dataset"
    stats = prepare_dataset(
        [str(source)],
        model_path,
        dataset_dir,
        shard_size=17,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    assert stats.valid_pairs == 80
    assert stats.duplicates == 1
    assert stats.invalid_json == 1
    assert stats.invalid_utf8 == 1
    assert stats.control_characters == 1
    assert stats.missing_text == 1
    assert stats.excessive_repetition == 1
    assert stats.train > 0 and stats.validation > 0 and stats.test > 0

    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)
    assert len(dataset) == stats.train * 2
    forward = dataset[0]
    reverse = dataset[1]
    assert forward["src_language"] == "ko"
    assert reverse["src_language"] == "ja"
    assert dataset.source_names == ["data22.jsonl"]
    assert dataset.source_id_at(0) == 0

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.pair_lengths is None
    assert restored.pair_source_ids is None
    assert restored.pair_synthetic_flags is None
    assert restored[0]["src_language"] == "ko"

    collator = SionBatchCollator(
        tokenizer,
        max_source_length=64,
        max_target_length=64,
        pad_to_multiple_of=8,
        denoise_probability=0.0,
        token_features=tokenizer_dir / "token_features.npz",
    )
    batch = collator([forward, reverse])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["decoder_input_ids"].shape[1] % 8 == 0
    assert batch["labels"].shape == batch["decoder_input_ids"].shape
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["source_language_tag_ids"].tolist() == [
        tokenizer.language_tags["ko"],
        tokenizer.language_tags["ja"],
    ]
    assert batch["reverse_direction_trained"].tolist() == [True, True]
    assert "src_script_ids" in batch

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_dataset(
            [str(source)],
            model_path,
            dataset_dir,
            validation_fraction=0.1,
            test_fraction=0.1,
        )


def test_parallel_preparation_matches_single_worker(tmp_path: Path) -> None:
    source = tmp_path / "tiny.jsonl"
    write_tiny_jsonl(source)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        num_workers=1,
        num_threads=1,
    )
    single = prepare_dataset(
        [str(source)],
        model_path,
        tmp_path / "single",
        validation_fraction=0.1,
        test_fraction=0.1,
        num_workers=1,
    )
    parallel = prepare_dataset(
        [str(source)],
        model_path,
        tmp_path / "parallel",
        validation_fraction=0.1,
        test_fraction=0.1,
        num_workers=2,
    )
    assert single == parallel
    for split in ("train", "validation", "test"):
        single_index = IndexedParallelDataset(tmp_path / "single", split)
        parallel_index = IndexedParallelDataset(tmp_path / "parallel", split)
        assert len(single_index) == len(parallel_index)


def test_prepare_publishes_one_caller_fingerprint_and_preprocessing_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    observed_inventory_boundary = False
    original_inventory = prepare_module.build_dataset_artifact_inventory

    def inspect_inventory_boundary(root: Path) -> dict[str, object]:
        nonlocal observed_inventory_boundary
        observed_inventory_boundary = True
        assert (root / prepare_module.RAW_FINGERPRINT_FILENAME).is_file()
        assert not (root / "manifest.json").exists()
        return original_inventory(root)

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(
        prepare_module,
        "build_dataset_artifact_inventory",
        inspect_inventory_boundary,
    )

    stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert stats.valid_pairs == 1
    assert observed_inventory_boundary
    raw_fingerprint = json.loads(
        (output / prepare_module.RAW_FINGERPRINT_FILENAME).read_text(encoding="utf-8")
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert raw_fingerprint == expected.to_dict() == manifest["fingerprint"]
    assert manifest["preprocessing_options"] == raw_fingerprint["preprocessing_options"]
    assert manifest["preprocessing_schema"] == raw_fingerprint["preprocessing_schema"]
    assert (output / prepare_module.PREPARE_COMPLETION_FILENAME).is_file()


def test_prepare_rejects_same_size_same_mtime_source_mutation_after_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source, marker="A")
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    original_process = prepare_module._process_prepare_batch
    mutated = False

    def mutate_after_processing(args: object) -> list[prepare_module._PrepareEvent]:
        nonlocal mutated
        events = original_process(args)  # type: ignore[arg-type]
        if not mutated:
            before = source.stat()
            changed = source.read_bytes().replace(b'"marker": "A"', b'"marker": "B"')
            assert len(changed) == before.st_size
            source.write_bytes(changed)
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert source.stat().st_size == before.st_size
            assert source.stat().st_mtime_ns == before.st_mtime_ns
            mutated = True
        return events

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "_process_prepare_batch", mutate_after_processing)

    with pytest.raises(RuntimeError, match="input file set or bytes changed"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_rejects_tokenizer_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"tokenizer-version-A")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    original_inventory = prepare_module.build_dataset_artifact_inventory

    def mutate_tokenizer(root: Path) -> dict[str, object]:
        inventory = original_inventory(root)
        before = tokenizer.stat()
        tokenizer.write_bytes(b"tokenizer-version-B")
        os.utime(tokenizer, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert tokenizer.stat().st_size == before.st_size
        return inventory

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "build_dataset_artifact_inventory", mutate_tokenizer)

    with pytest.raises(RuntimeError, match="Tokenizer bytes changed"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert not output.exists()


def test_prepare_rechecks_inputs_after_staging_fsync_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source, marker="A")
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    original_fsync = prepare_module._fsync_staging_tree
    mutated = False

    def mutate_after_fsync(root: Path) -> None:
        nonlocal mutated
        original_fsync(root)
        if not mutated:
            before = source.stat()
            changed = source.read_bytes().replace(b'"marker": "A"', b'"marker": "B"')
            assert len(changed) == before.st_size
            source.write_bytes(changed)
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            mutated = True

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "_fsync_staging_tree", mutate_after_fsync)

    with pytest.raises(RuntimeError, match="input file set or bytes changed"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_rejects_a_new_wildcard_source_discovered_during_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    original_process = prepare_module._process_prepare_batch
    added = False

    def add_source_after_processing(args: object) -> list[prepare_module._PrepareEvent]:
        nonlocal added
        events = original_process(args)  # type: ignore[arg-type]
        if not added:
            _write_atomic_prepare_source(tmp_path / "new-pairs.jsonl", marker="B")
            added = True
        return events

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "_process_prepare_batch", add_source_after_processing)

    with pytest.raises(RuntimeError, match="input file set or bytes changed"):
        _prepare_atomic_dataset(
            str(tmp_path / "*.jsonl"),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert not output.exists()


def test_prepare_recovers_a_complete_exact_orphan_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    original_stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )
    orphan = tmp_path / ".dataset.staging-recoverable"
    output.rename(orphan)

    def unexpected_rebuild(_args: object) -> list[prepare_module._PrepareEvent]:
        raise AssertionError("complete authenticated staging should be recovered")

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", unexpected_rebuild)
    recovered = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert recovered == original_stats
    assert output.is_dir()
    assert not orphan.exists()


def test_prepare_quarantines_a_contradictory_orphan_then_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )
    orphan = tmp_path / ".dataset.staging-contradictory"
    output.rename(orphan)
    (orphan / prepare_module.RAW_FINGERPRINT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )

    rebuilt = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert rebuilt.valid_pairs == 1
    assert output.is_dir()
    assert not orphan.exists()
    assert not list(tmp_path.glob(".dataset.rejected-*"))


def test_prepare_rejects_an_authenticated_manifest_with_wrong_index_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )
    orphan = tmp_path / ".dataset.staging-wrong-index-totals"
    output.rename(orphan)
    manifest_path = orphan / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stats"]["valid_pairs"] += 1
    manifest["stats"]["train"] += 1
    manifest["sources"][0]["stats"]["valid_pairs"] += 1
    manifest["sources"][0]["stats"]["train"] += 1
    manifest["mean_quality_score"] = (
        manifest["stats"]["quality_score_sum"] / manifest["stats"]["valid_pairs"]
    )
    manifest["sources"][0]["mean_quality_score"] = (
        manifest["sources"][0]["stats"]["quality_score_sum"]
        / manifest["sources"][0]["stats"]["valid_pairs"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completion_path = orphan / prepare_module.PREPARE_COMPLETION_FILENAME
    completion_path.unlink()
    completion_path.write_text(
        json.dumps(prepare_module._completion_payload(orphan, manifest)),
        encoding="utf-8",
    )
    rebuilds = 0
    original_process = prepare_module._process_prepare_batch

    def count_rebuild(args: object) -> list[prepare_module._PrepareEvent]:
        nonlocal rebuilds
        rebuilds += 1
        return original_process(args)  # type: ignore[arg-type]

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_rebuild)

    rebuilt = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert rebuilt.valid_pairs == 1
    assert rebuilds == 1
    assert not orphan.exists()


def test_prepare_refuses_a_file_at_an_orphan_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    unsafe = tmp_path / ".dataset.staging-not-a-directory"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    unsafe.write_bytes(b"do-not-delete")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)

    with pytest.raises(RuntimeError, match="unsafe orphan staging path"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert unsafe.read_bytes() == b"do-not-delete"
    assert not output.exists()


def test_pair_quality_rejects_obvious_damage() -> None:
    policy = QualityPolicy(max_length_ratio=3.0)
    clean = assess_pair("오늘 날씨가 좋습니다.", "今日は天気が良いです。", policy)
    assert clean.accepted
    assert clean.score == 100

    identical = assess_pair("OpenAI 123", "OpenAI 123", policy)
    assert not identical.accepted
    assert "identical_text" in identical.rejection_reasons

    wrong_scripts = assess_pair("This is not Korean", "이것은 일본어가 아니다", policy)
    assert not wrong_scripts.accepted
    assert "ko_script_mismatch" in wrong_scripts.rejection_reasons
    assert "ja_script_mismatch" in wrong_scripts.rejection_reasons

    repeated = assess_pair(
        "오류문자열오류문자열오류문자열오류문자열오류문자열",
        "エラー文字列エラー文字列エラー文字列エラー文字列エラー文字列",
        policy,
    )
    assert not repeated.accepted
    assert "excessive_repetition" in repeated.rejection_reasons


def test_quality_profiles_apply_to_bcp47_variants_and_both_pair_orientations() -> None:
    wrong_scripts = assess_pair(
        "This is not Korean",
        "이것은 일본어가 아니다",
        languages=("ko-KR", "ja-JP"),
    )
    assert not wrong_scripts.accepted
    assert {"ko_script_mismatch", "ja_script_mismatch"} <= set(wrong_scripts.rejection_reasons)

    reverse = assess_pair(
        "人工知能技術発展研究社会文化交流",
        "한국어 문장입니다.",
        languages=("ja-JP", "ko-KR"),
    )
    assert "ja_no_kana" in reverse.warning_reasons


def test_unprofiled_languages_are_unchecked_instead_of_scored_as_perfect() -> None:
    assessment = assess_pair(
        "same-script source text",
        "same-script target text",
        languages=("qaa", "qab"),
    )

    assert assessment.ko_language_fraction is None
    assert assessment.ja_language_fraction is None
    assert "ko_script_mismatch" not in assessment.rejection_reasons
    assert "ja_script_mismatch" not in assessment.rejection_reasons
    assert language_fraction("Latin text", "qaa") is None
    assert language_fraction("Latin text", "qaa-Latn") == 1.0


def test_expressive_quality_profile_only_waives_short_and_repeated_reactions() -> None:
    short = assess_pair("아", "あ")
    assert short.rejection_reasons == ("too_short",)
    profiled_short = apply_record_quality_profile(short, "expressive_v1")
    assert profiled_short.accepted
    assert profiled_short.score == 100

    repeated = assess_pair("아" * 12, "あ" * 12)
    assert repeated.rejection_reasons == ("excessive_repetition",)
    assert apply_record_quality_profile(repeated, "expressive_v1").accepted

    unsafe = assess_pair("아\u0001", "あ")
    profiled_unsafe = apply_record_quality_profile(unsafe, "expressive_v1")
    assert not profiled_unsafe.accepted
    assert profiled_unsafe.rejection_reasons == ("control_characters",)


def test_source_temperature_sampling_is_deterministic_and_balanced() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 100
        pair_source_ids = np.asarray([0] * 90 + [1] * 10, dtype=np.uint16)
        source_names = ["large.jsonl", "small.jsonl"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    dataset = DummyDataset()
    kwargs = dict(
        batch_size=10,
        bucket_size=100,
        seed=7,
        source_sampling_alpha=0.5,
        max_source_upsampling=3.0,
    )
    first = DistributedBucketBatchSampler(dataset, **kwargs)
    second = DistributedBucketBatchSampler(dataset, **kwargs)
    first_indices = [index for batch in first for index in batch]
    second_indices = [index for batch in second for index in batch]
    assert first_indices == second_indices
    sampled_small = sum(index >= 90 for index in first_indices)
    assert 10 < sampled_small <= 30


def test_source_id_weights_do_not_depend_on_unique_file_names() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 10_000
        pair_source_ids = np.asarray([0] * 9_000 + [1] * 1_000, dtype=np.uint16)
        source_names = ["shared.txt", "shared.txt"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    sampler = DistributedBucketBatchSampler(
        DummyDataset(),
        batch_size=100,
        bucket_size=10_000,
        seed=13,
        source_sampling_weights_by_id={0: 1.0 / 9.0, 1: 1.0},
        max_source_upsampling=10.0,
    )
    sampled = [index for batch in sampler for index in batch]
    small_share = sum(index >= 9_000 for index in sampled) / len(sampled)
    assert 0.48 < small_share < 0.52


def test_source_id_weights_can_apply_an_uncapped_prepared_distribution() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 100_000
        pair_source_ids = np.asarray([0] * 99_000 + [1] * 1_000, dtype=np.uint16)
        source_names = ["major.txt", "minor.txt"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    alpha = 0.7
    major_target = 99_000**alpha
    minor_target = 1_000**alpha
    minor_share = minor_target / (major_target + minor_target)
    sampler = DistributedBucketBatchSampler(
        DummyDataset(),
        batch_size=1_000,
        bucket_size=100_000,
        seed=17,
        source_sampling_weights_by_id={
            0: (1.0 - minor_share) / 99_000,
            1: minor_share / 1_000,
        },
        max_source_upsampling=math.inf,
    )

    sampled = [index for batch in sampler for index in batch]
    sampled_minor_share = sum(index >= 99_000 for index in sampled) / len(sampled)

    assert minor_share > 0.03  # The generic 3x cap would stop exactly here.
    assert sampled_minor_share == pytest.approx(minor_share, abs=0.003)


def test_record_level_synthetic_samples_are_downweighted() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 10_000
        pair_source_ids = np.zeros(pair_count, dtype=np.uint16)
        pair_synthetic_flags = np.asarray(
            [False] * (pair_count // 2) + [True] * (pair_count // 2),
            dtype=np.bool_,
        )
        source_names = ["mixed.jsonl"]
        synthetic_sampling_weight = 0.5

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    sampler = DistributedBucketBatchSampler(
        DummyDataset(),
        batch_size=100,
        bucket_size=10_000,
        seed=17,
    )
    sampled = [index for batch in sampler for index in batch]
    synthetic_share = sum(index >= 5_000 for index in sampled) / len(sampled)
    assert 0.31 < synthetic_share < 0.36


def test_distributed_sampler_pads_equal_batches_when_not_dropping() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 7
        pair_source_ids = np.zeros(7, dtype=np.uint16)
        source_names = ["only.jsonl"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    batch_counts = []
    for rank in range(3):
        sampler = DistributedBucketBatchSampler(
            DummyDataset(),
            batch_size=2,
            rank=rank,
            world_size=3,
            drop_last=False,
            seed=11,
        )
        batches = list(sampler)
        batch_counts.append(len(batches))
        assert all(len(batch) == 2 for batch in batches)
    assert batch_counts == [2, 2, 2]


def test_distributed_sampler_resumes_from_batch_cursor_without_refetching() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 12
        pair_source_ids = np.zeros(12, dtype=np.uint16)
        source_names = ["only.jsonl"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    sampler = DistributedBucketBatchSampler(
        DummyDataset(),
        batch_size=2,
        seed=19,
    )
    sampler.set_epoch(4)
    full_epoch = list(sampler)
    sampler.set_epoch(4)
    sampler.set_start_batch(2)
    assert list(sampler) == full_epoch[2:]
    sampler.set_epoch(4)
    assert list(sampler) == full_epoch


def test_protected_span_replacement_is_single_pass_and_boundary_safe() -> None:
    ko, ja = protect_shared_spans(
        "코드 A-0, 값 0과 10을 확인한다.",
        "コードA-0、値0と10を確認する。",
    )
    assert "<slot_<slot_" not in ko
    assert "<slot_<slot_" not in ja
    assert ko.count("<slot_") == 3
    assert ja.count("<slot_") == 3


def test_structured_parser_handles_localization_placeholders_as_whole_spans() -> None:
    text = (
        '{ DATETIME($date, month: "long", year: "numeric") } '
        '{ -brand(case: "accusative") } %CasterName %2$Spx &#39; &#x27;'
    )
    spans = extract_structured_spans(text)
    surfaces = {(span.kind, span.surface) for span in spans}
    assert (
        "placeable",
        '{ DATETIME($date, month: "long", year: "numeric") }',
    ) in surfaces
    assert ("placeable", '{ -brand(case: "accusative") }') in surfaces
    assert ("percent_placeholder", "%CasterName") in surfaces
    assert ("printf", "%2$S") in surfaces
    assert ("entity", "&#39;") in surfaces
    assert ("entity", "&#x27;") in surfaces
    assert ("percent_placeholder", "%C") not in surfaces


def test_structured_parser_preserves_case_sensitive_placeholder_identity() -> None:
    score, mismatch = structured_similarity(
        "{ $VERSION } ${BUILD_ID} &Aacute;",
        "{ $version } ${build_id} &aacute;",
    )
    assert score == 0.0
    assert mismatch


def test_structured_similarity_counts_duplicates_and_nested_html_placeholders() -> None:
    score, mismatch = structured_similarity(
        '<a title="{ $name }">{ $name }</a>',
        '<a title="{ $other }">{ $name }</a>',
    )
    assert score < 1.0
    assert mismatch


def test_structured_mask_covers_units_and_trims_url_particles() -> None:
    source = "용량은 50MB, 약은 250mg이며 https://example.com/issues/102181에서 확인한다."
    masked, mapping = mask_structured_spans(
        source,
        slot_symbols=SLOT_SYMBOLS,
    )
    assert "50MB" not in masked
    assert "250mg" not in masked
    assert "https://" not in masked
    assert "에서 확인한다" in masked
    assert set(mapping.values()) == {
        "50MB",
        "250mg",
        "https://example.com/issues/102181",
    }
    restored, missing = restore_targets(masked, mapping)
    assert restored == source
    assert not missing


def test_structured_mask_leaves_natural_language_units_translatable() -> None:
    source = "배송비는 12,500원이고 수량은 3개이며 용량은 250mg이다."
    masked, mapping = mask_structured_spans(
        source,
        slot_symbols=SLOT_SYMBOLS,
    )

    assert "원" in masked
    assert "개" in masked
    assert "mg" not in masked
    assert set(mapping.values()) == {"12,500", "3", "250mg"}
    restored, missing = restore_targets(masked, mapping)
    assert restored == source
    assert not missing

    score, critical_mismatch = structured_similarity(
        "배송비는 12,500원이다.",
        "送料は12,500ウォンです。",
    )
    assert score == 1.0
    assert not critical_mismatch


def test_shared_complex_placeable_is_replaced_once() -> None:
    placeable = '{ DATETIME($date, month: "long", year: "numeric") }'
    left, right = protect_shared_spans(
        f"날짜: {placeable}",
        f"日付: {placeable}",
    )
    assert left.count("<slot_0>") == 1
    assert right.count("<slot_0>") == 1
    assert "numeric" not in left
    assert "numeric" not in right
