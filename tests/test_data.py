from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from sion_translate.data import (
    DistributedBucketBatchSampler,
    IndexedParallelDataset,
    SionBatchCollator,
    prepare_dataset,
)
from sion_translate.data.quality import QualityPolicy, assess_pair
from sion_translate.data.prepare import protect_shared_spans
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
    assert restored[0]["src_language"] == "ko"

    collator = SionBatchCollator(
        tokenizer,
        max_source_length=64,
        max_target_length=64,
        denoise_probability=0.0,
        token_features=tokenizer_dir / "token_features.npz",
    )
    batch = collator([forward, reverse])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["decoder_input_ids"].shape
    assert batch["attention_mask"].dtype == torch.bool
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
