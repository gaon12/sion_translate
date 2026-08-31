from __future__ import annotations

import errno
import gzip
from itertools import combinations
import json
import math
import os
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
import torch

import sion_translate.data.prepare as prepare_module
from sion_translate.data import (
    DirectionCompleteValidationBatchSampler,
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
from sion_translate.locking import artifact_lock
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


def _initialize_fake_parallel_prepare_worker(_model_path: str) -> None:
    prepare_module._PREPARE_WORKER_TOKENIZER = _FakePrepareTokenizer("unused")


def _fail_first_parallel_prepare_job(
    job: prepare_module._PrepareBatchJob,
) -> list[prepare_module._PrepareEvent]:
    if not job.descriptor.synthetic_pass and job.descriptor.batch_index == 0:
        time.sleep(0.25)
        raise RuntimeError("simulated out-of-order parallel interruption")
    # A spawned interpreter imports the production module without the parent's
    # monkeypatch, so this resolves to the real worker implementation there.
    return prepare_module._process_prepare_job(job)


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


def test_prepare_stats_normalize_only_the_exact_unmarked_legacy_representation() -> None:
    neutral = vars(prepare_module.PrepareStats(src_tokens=17, tgt_tokens=23))
    neutral.pop("refinement_evidence")
    neutral.pop("reserved_draft_separator")
    legacy = dict(neutral)
    legacy["ko_tokens"] = legacy.pop("src_tokens")
    legacy["ja_tokens"] = legacy.pop("tgt_tokens")

    schema = prepare_module.prepare_stats_schema_from_manifest({}, role="Test manifest")
    normalized = prepare_module.validated_prepare_stats(
        legacy,
        stats_schema=schema,
        role="Test manifest",
    )

    assert normalized.src_tokens == 17
    assert normalized.tgt_tokens == 23


def test_prepare_stats_v2_defaults_the_new_reserved_separator_counter() -> None:
    v2 = vars(prepare_module.PrepareStats(src_tokens=17, tgt_tokens=23))
    v2.pop("reserved_draft_separator")

    normalized = prepare_module.validated_prepare_stats(
        v2,
        stats_schema=prepare_module.PREPARE_STATS_SCHEMA_V2,
        role="Test manifest",
    )

    assert normalized.reserved_draft_separator == 0


@pytest.mark.parametrize("marker", [None, False, 0, [], {}, "unknown-stats-schema"])
def test_prepare_stats_reject_an_explicit_or_unknown_schema_marker(marker: object) -> None:
    with pytest.raises(ValueError, match="stats_schema is unsupported"):
        prepare_module.prepare_stats_schema_from_manifest(
            {"stats_schema": marker},
            role="Test manifest",
        )


def test_prepare_stats_reject_schema_field_mismatches_and_invalid_counts() -> None:
    neutral = dict(vars(prepare_module.PrepareStats(src_tokens=17, tgt_tokens=23)))
    legacy = dict(neutral)
    legacy["ko_tokens"] = legacy.pop("src_tokens")
    legacy["ja_tokens"] = legacy.pop("tgt_tokens")

    for payload, schema in (
        ({key: value for key, value in neutral.items() if key != "refinement_evidence"}, None),
        (legacy, prepare_module.PREPARE_STATS_SCHEMA),
        ({**neutral, "ko_tokens": 17}, prepare_module.PREPARE_STATS_SCHEMA),
        ({**legacy, "src_tokens": 17}, None),
    ):
        with pytest.raises(ValueError, match="stats fields do not match their schema"):
            prepare_module.validated_prepare_stats(
                payload,
                stats_schema=schema,
                role="Test manifest",
            )

    for invalid_count in (True, -1, 1.5, "17"):
        payload = {**neutral, "src_tokens": invalid_count}
        with pytest.raises(ValueError, match="stat is invalid: src_tokens"):
            prepare_module.validated_prepare_stats(
                payload,
                stats_schema=prepare_module.PREPARE_STATS_SCHEMA,
                role="Test manifest",
            )


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


def _atomic_prepare_fingerprint(
    source: Path,
    tokenizer: Path,
    *,
    language_pair: tuple[str, str] = ("ko", "ja"),
) -> DatasetFingerprint:
    policy = QualityPolicy()
    reverse_pair = (language_pair[1], language_pair[0])
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
        translation_directions=(language_pair, reverse_pair),
        train_only_prefixes=prepare_module.DEFAULT_TRAIN_ONLY_PREFIXES,
        synthetic_sampling_weight=0.25,
        language_pair_count=1,
    )
    return build_dataset_fingerprint(
        [source],
        language_pairs=(language_pair,),
        tokenizer_model=tokenizer,
        preprocessing_options=options,
    )


def _prepare_atomic_dataset(
    source_pattern: str,
    tokenizer: Path,
    output: Path,
    *,
    expected_fingerprint: DatasetFingerprint,
    language_pair: tuple[str, str] = ("ko", "ja"),
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
        language_pair=language_pair,
        synthetic_sampling_weight=0.25,
        num_workers=1,
        expected_fingerprint=expected_fingerprint,
    )


def _write_resumable_prepare_source(path: Path) -> None:
    rows = [
        {
            "ko": "재시작해도 첫 번째 문장의 결과를 안전하게 재사용합니다.",
            "ja": "再起動後も最初の文の結果を安全に再利用します。",
            "marker": "real-1",
        },
        {
            "ko": "합성 문장은 실제 문장 뒤에서 처리되어야 합니다.",
            "ja": "合成文は実文の後で処理されなければなりません。",
            "marker": "synthetic-1",
            "synthetic": True,
        },
        {
            "ko": "마지막 문장까지 결정적인 순서를 유지합니다.",
            "ja": "最後の文まで決定的な順序を維持します。",
            "marker": "real-2",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _leave_resumable_prepare_progress(
    source: Path,
    tokenizer: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    DatasetFingerprint,
    Path,
    Callable[[prepare_module._PrepareBatchInput], list[prepare_module._PrepareEvent]],
]:
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    original_process = prepare_module._process_prepare_batch
    process_calls = 0

    def fail_on_second_batch(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal process_calls
        process_calls += 1
        if process_calls == 2:
            raise RuntimeError("simulated worker interruption")
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", fail_on_second_batch)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    progress = list(output.parent.glob(f".{output.name}.prepare-progress-*"))
    assert len(progress) == 1
    assert len(list((progress[0] / "chunks").glob("*.json.gz"))) == 1
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    return expected, progress[0], original_process


def test_prepare_resumes_worker_chunks_and_matches_a_clean_build_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    resumed = tmp_path / "resumed"
    clean = tmp_path / "clean"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    expected, _, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        resumed,
        monkeypatch,
    )

    resumed_calls = 0

    def count_uncached_batches(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal resumed_calls
        resumed_calls += 1
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_uncached_batches)
    resumed_stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        resumed,
        expected_fingerprint=expected,
    )
    assert resumed_calls == 2
    assert not list(tmp_path.glob(".resumed.prepare-progress-*"))

    clean_stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        clean,
        expected_fingerprint=expected,
    )

    assert resumed_stats == clean_stats
    assert _directory_bytes(resumed) == _directory_bytes(clean)


@pytest.mark.parametrize("dedup_backend", ["memory", "sqlite"])
def test_prepare_resume_preserves_an_arbitrary_graph_and_real_row_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dedup_backend: str,
) -> None:
    class GraphTokenizer(_FakePrepareTokenizer):
        languages = ("de", "fr", "es")

    source = tmp_path / "graph.jsonl"
    tokenizer = tmp_path / "sion.model"
    resumed = tmp_path / f"resumed-{dedup_backend}"
    clean = tmp_path / f"clean-{dedup_backend}"
    duplicate = {
        "de": "Eine ausreichend lange deutsche Zeile.",
        "fr": "Une ligne française suffisamment longue.",
        "es": "Una línea española suficientemente larga.",
    }
    rows = [
        {**duplicate, "synthetic": True},
        duplicate,
        {
            "de": "Eine zweite echte deutsche Zeile.",
            "fr": "Une deuxième vraie ligne française.",
            "es": "Una segunda línea española real.",
        },
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"stable-graph-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", GraphTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    original_process = prepare_module._process_prepare_batch
    process_calls = 0

    def fail_on_second_batch(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal process_calls
        process_calls += 1
        if process_calls == 2:
            raise RuntimeError("simulated graph worker interruption")
        return original_process(batch)

    def build(output: Path) -> prepare_module.PrepareStats:
        return prepare_module.prepare_dataset(
            [str(source)],
            tokenizer,
            output,
            shard_size=8,
            validation_fraction=0.0,
            test_fraction=0.0,
            filter_quality=False,
            dedup_backend=dedup_backend,
            language_pairs=(("de", "fr"), ("fr", "es")),
            translation_directions=(("de", "fr"), ("fr", "es"), ("es", "fr")),
            num_workers=1,
        )

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", fail_on_second_batch)
    with pytest.raises(RuntimeError, match="graph worker interruption"):
        build(resumed)
    assert (
        len(
            list(
                next(tmp_path.glob(f".{resumed.name}.prepare-progress-*"))
                .joinpath("chunks")
                .glob("*.json.gz")
            )
        )
        == 1
    )

    resumed_calls = 0

    def count_uncached_batches(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal resumed_calls
        resumed_calls += 1
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_uncached_batches)
    resumed_stats = build(resumed)
    assert resumed_calls == 2
    clean_stats = build(clean)

    assert resumed_stats == clean_stats
    assert resumed_stats.physical_lines == 3
    assert resumed_stats.valid_pairs == 4
    assert resumed_stats.duplicates == 2
    assert resumed_stats.synthetic_pairs == 0
    assert resumed_stats.forward_only_pairs == 2
    assert _directory_bytes(resumed) == _directory_bytes(clean)


def test_prepare_expands_a_six_language_complete_graph_without_language_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    languages = ("de", "fr", "es", "it", "pt", "nl")

    class SixLanguageTokenizer(_FakePrepareTokenizer):
        pass

    SixLanguageTokenizer.languages = languages
    source = tmp_path / "six-language.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    source.write_text(
        json.dumps(
            {
                language: (
                    f"A sufficiently long sentence for configurable language {language}. "
                    + "Deliberately unequal expansion. " * (position + 1)
                )
                for position, language in enumerate(languages)
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"six-language-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", SixLanguageTokenizer)
    graph = tuple(combinations(languages, 2))

    stats = prepare_module.prepare_dataset(
        [str(source)],
        tokenizer,
        output,
        shard_size=32,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        language_pairs=graph,
        num_workers=1,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index = np.load(next((output / "train").glob("*.idx.npy")), allow_pickle=False)
    expected_src_tokens = int(index["src_length"].sum(dtype=np.uint64))
    expected_tgt_tokens = int(index["tgt_length"].sum(dtype=np.uint64))

    assert expected_tgt_tokens > expected_src_tokens
    assert stats.physical_lines == 1
    assert stats.valid_pairs == len(graph) == 15
    assert manifest["language_pairs"] == [list(pair) for pair in graph]
    assert manifest["stats_schema"] == prepare_module.PREPARE_STATS_SCHEMA
    for serialized in (manifest["stats"], manifest["sources"][0]["stats"]):
        assert "ko_tokens" not in serialized
        assert "ja_tokens" not in serialized
        assert serialized["src_tokens"] == expected_src_tokens
        assert serialized["tgt_tokens"] == expected_tgt_tokens
    assert stats.src_tokens == expected_src_tokens
    assert stats.tgt_tokens == expected_tgt_tokens


def test_direction_complete_validation_covers_an_imbalanced_arbitrary_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    languages = ("de", "fr", "sw", "ar")
    graph = (("de", "fr"), ("fr", "de"), ("sw", "ar"))

    class GraphTokenizer(_FakePrepareTokenizer):
        pass

    GraphTokenizer.languages = languages
    source = tmp_path / "imbalanced-graph.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    rows = [
        {
            "de": f"A sufficiently long German validation sentence number {index}.",
            "fr": f"Une phrase française de validation assez longue numéro {index}.",
        }
        for index in range(20)
    ]
    rows.append(
        {
            "sw": "Sentensi ndefu ya uthibitishaji wa Kiswahili kwa ukingo adimu.",
            "ar": "جملة تحقق عربية طويلة بما يكفي للاتجاه النادر في الرسم.",
        }
    )
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"arbitrary-graph-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", GraphTokenizer)
    prepare_module.prepare_dataset(
        [str(source)],
        tokenizer,
        output,
        shard_size=32,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        language_pairs=(("de", "fr"), ("sw", "ar")),
        translation_directions=graph,
        num_workers=1,
    )
    verified_inventory_sha256 = prepare_module.validate_dataset_artifact_inventory(output)
    dataset = IndexedParallelDataset(
        output,
        "train",
        bidirectional=True,
        verify_integrity=False,
        verified_artifact_inventory_sha256=verified_inventory_sha256,
    )
    assert dataset.verify_integrity is False
    assert dataset.artifact_inventory_sha256 == verified_inventory_sha256

    grouped = dataset.virtual_indices_by_translation_direction(graph)
    assert {direction: len(indices) for direction, indices in grouped.items()} == {
        ("de", "fr"): 20,
        ("fr", "de"): 20,
        ("sw", "ar"): 1,
    }

    samplers = [
        DirectionCompleteValidationBatchSampler(
            dataset,
            batch_size=1,
            directions=graph,
            max_batches=1,
            rank=rank,
            world_size=2,
            seed=19,
        )
        for rank in range(2)
    ]
    assert [len(sampler) for sampler in samplers] == [2, 2]
    assert samplers[0].cohort_fingerprint == samplers[1].cohort_fingerprint
    assert samplers[0].cohort_identity["dataset_split"] == "train"
    assert (
        samplers[0].cohort_identity["dataset_artifact_inventory_sha256"]
        == dataset.artifact_inventory_sha256
    )
    selected_directions = {
        (str(dataset[index]["src_language"]), str(dataset[index]["target_language"]))
        for sampler in samplers
        for batch in sampler
        for index in batch
    }
    assert selected_directions == set(graph)
    assert list(samplers[0]) == list(samplers[0])

    with pytest.raises(
        ValueError,
        match=r"requires 2 distinct examples for direction sw->ar.*contains only 1",
    ):
        DirectionCompleteValidationBatchSampler(
            dataset,
            batch_size=1,
            directions=graph,
            max_batches=1,
            minimum_examples_per_direction=2,
            require_unique_examples=True,
        )

    unique_sampler = DirectionCompleteValidationBatchSampler(
        dataset,
        batch_size=2,
        directions=graph[:2],
        max_batches=1,
        minimum_examples_per_direction=10,
        require_unique_examples=True,
    )
    unique_indices = [index for batch in unique_sampler for index in batch]
    assert len(unique_indices) == len(set(unique_indices)) == 20
    assert unique_sampler.cohort_identity["minimum_examples_per_direction"] == 10
    assert unique_sampler.cohort_identity["unique_examples_required"] is True


def test_prepare_capacity_plan_covers_large_supported_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "metadata.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    source.write_text(
        json.dumps(
            {
                "ko": "용량 계획을 검증할 만큼 충분히 긴 한국어 문장입니다.",
                "ja": "容量計画を検証するための十分に長い日本語文です。",
                "metadata": {"provenance": "p" * 4096},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"metadata-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    actual_gate = prepare_module._ensure_prepare_final_capacity
    planned_bytes = 0

    def capture_plan(*args: object, **kwargs: object) -> int:
        nonlocal planned_bytes
        planned_bytes = actual_gate(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        return planned_bytes

    monkeypatch.setattr(prepare_module, "_ensure_prepare_final_capacity", capture_plan)
    stats = prepare_module.prepare_dataset(
        [str(source)],
        tokenizer,
        output,
        shard_size=8,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        num_workers=1,
    )

    actual_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    assert stats.valid_pairs == 1
    assert planned_bytes >= actual_bytes
    assert any(path.name.endswith(".meta.bin") for path in output.rglob("*"))


def test_prepare_rejects_unbounded_record_fanout_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GraphTokenizer(_FakePrepareTokenizer):
        languages = ("de", "fr", "es")

    source = tmp_path / "fanout.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    source.write_text(
        json.dumps(
            {
                "de": "Eine ausreichend lange deutsche Zeile.",
                "fr": "Une ligne française suffisamment longue.",
                "es": "Una línea española suficientemente larga.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"fanout-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", GraphTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_MAX_EXPANDED_PAIRS_PER_LINE", 1)

    with pytest.raises(ValueError, match="expands beyond.*source_id=0.*byte_offset=0"):
        prepare_module.prepare_dataset(
            [str(source)],
            tokenizer,
            output,
            validation_fraction=0.0,
            test_fraction=0.0,
            filter_quality=False,
            dedup_backend="memory",
            language_pairs=(("de", "fr"), ("fr", "es")),
            num_workers=1,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_reports_an_oversized_raw_line_with_its_exact_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "oversized-line.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    source.write_text(
        json.dumps({"ko": "가" * 100, "ja": "あ" * 100}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"raw-line-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_MAX_RAW_LINE_BYTES", 64)

    with pytest.raises(ValueError, match=r"path=.*oversized-line\.jsonl.*byte_offset=0"):
        prepare_module.prepare_dataset(
            [str(source)],
            tokenizer,
            output,
            validation_fraction=0.0,
            test_fraction=0.0,
            filter_quality=False,
            dedup_backend="memory",
            language_pair=("ko", "ja"),
            num_workers=1,
        )

    assert not output.exists()


def test_prepare_reports_oversized_supported_metadata_before_checkpointing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "oversized-metadata.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    source.write_text(
        json.dumps(
            {
                "ko": "충분히 긴 한국어 메타데이터 검사 문장입니다.",
                "ja": "十分に長い日本語のメタデータ検査文です。",
                "metadata": {"provenance": "p" * 128},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"metadata-limit-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_MAX_RECORD_METADATA_BYTES", 64)

    with pytest.raises(ValueError, match="supported metadata.*source_id=0.*byte_offset=0"):
        prepare_module.prepare_dataset(
            [str(source)],
            tokenizer,
            output,
            validation_fraction=0.0,
            test_fraction=0.0,
            filter_quality=False,
            dedup_backend="memory",
            language_pair=("ko", "ja"),
            num_workers=1,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


@pytest.mark.parametrize("dedup_backend", ["memory", "sqlite"])
def test_prepare_resumes_out_of_order_parallel_chunks_across_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dedup_backend: str,
) -> None:
    source = tmp_path / "parallel.jsonl"
    tokenizer = tmp_path / "sion.model"
    resumed = tmp_path / f"resumed-{dedup_backend}"
    clean = tmp_path / f"clean-{dedup_backend}"
    rows = [
        {
            "ko": f"병렬 재개를 검증하는 충분히 긴 한국어 문장 {index}입니다.",
            "ja": f"並列再開を検証するための十分に長い日本語文{index}です。",
        }
        for index in range(4)
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"parallel-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    monkeypatch.setattr(
        prepare_module,
        "_initialize_prepare_worker",
        _initialize_fake_parallel_prepare_worker,
    )
    original_job = prepare_module._process_prepare_job

    def build(output: Path) -> prepare_module.PrepareStats:
        return prepare_module.prepare_dataset(
            [str(source)],
            tokenizer,
            output,
            shard_size=8,
            validation_fraction=0.0,
            test_fraction=0.0,
            filter_quality=False,
            dedup_backend=dedup_backend,
            language_pair=("ko", "ja"),
            num_workers=2,
        )

    monkeypatch.setattr(prepare_module, "_process_prepare_job", _fail_first_parallel_prepare_job)
    with pytest.raises(RuntimeError, match="out-of-order parallel interruption"):
        build(resumed)
    progress = next(tmp_path.glob(f".{resumed.name}.prepare-progress-*"))
    assert list((progress / "chunks").glob("*.json.gz"))

    monkeypatch.setattr(prepare_module, "_process_prepare_job", original_job)
    resumed_stats = build(resumed)
    clean_stats = build(clean)

    assert resumed_stats == clean_stats
    assert _directory_bytes(resumed) == _directory_bytes(clean)


def test_prepare_rejects_a_corrupt_worker_chunk_before_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    expected, progress, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        output,
        monkeypatch,
    )
    chunk = next((progress / "chunks").glob("*.json.gz"))
    chunk_document = json.loads(gzip.decompress(chunk.read_bytes()).decode("utf-8"))
    chunk_document["events"][0][0] = "invalid_json"
    chunk.write_bytes(
        gzip.compress(
            json.dumps(chunk_document, ensure_ascii=True, sort_keys=True).encode("utf-8"),
            mtime=0,
        )
    )
    monkeypatch.setattr(
        prepare_module,
        "_process_prepare_batch",
        original_process,
    )

    with pytest.raises(prepare_module._PrepareProgressError, match="integrity digest"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.prepare-progress-*"))
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_generation_fences_orphan_workers_and_reuses_identical_winners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    _, progress_root, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        output,
        monkeypatch,
    )
    contract = json.loads(
        (progress_root / prepare_module.PREPARE_PROGRESS_CONTRACT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    contract_sha256 = contract["contract_sha256"]
    batches = list(
        prepare_module._prepare_batch_records(
            (source,),
            QualityPolicy(),
            False,
            (("ko", "ja"),),
            prepare_module.DEFAULT_TRAIN_ONLY_PREFIXES,
            510,
            (("ko", "ja"), ("ja", "ko")),
        )
    )
    stale_epoch = json.loads(
        (progress_root / prepare_module.PREPARE_PROGRESS_EPOCH_FILENAME).read_text(encoding="utf-8")
    )["epoch"]
    missing_descriptor, missing_batch = batches[1]
    stale_job = prepare_module._PrepareBatchJob(
        descriptor=missing_descriptor,
        batch=missing_batch,
        progress_chunks_dir=str(progress_root / "chunks"),
        progress_contract_sha256=contract_sha256,
        generation_epoch=stale_epoch,
    )
    resumed_progress = prepare_module._initialize_prepare_progress(
        output,
        contract,
        contract_sha256,
    )
    missing_events = original_process(missing_batch)

    with pytest.raises(prepare_module._PrepareProgressError, match="superseded parent"):
        prepare_module._write_prepare_chunk(stale_job, missing_events)
    assert not (
        progress_root / "chunks" / prepare_module._prepare_chunk_filename(missing_descriptor)
    ).exists()

    winning_descriptor, winning_batch = batches[0]
    winning_job = prepare_module._PrepareBatchJob(
        descriptor=winning_descriptor,
        batch=winning_batch,
        progress_chunks_dir=str(progress_root / "chunks"),
        progress_contract_sha256=contract_sha256,
        generation_epoch=resumed_progress.generation_epoch,
    )
    prepare_module._write_prepare_chunk(winning_job, original_process(winning_batch))
    assert (
        progress_root / "chunks" / prepare_module._prepare_chunk_filename(winning_descriptor)
    ).is_file()


def test_prepare_rejects_an_unexpected_checkpoint_inventory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    expected, progress, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        output,
        monkeypatch,
    )
    chunk = next((progress / "chunks").glob("*.json.gz"))
    unexpected = progress / "chunks" / "pass-0-source-00000-batch-999999999.json.gz"
    unexpected.write_bytes(chunk.read_bytes())
    monkeypatch.setattr(prepare_module, "_process_prepare_batch", original_process)

    with pytest.raises(prepare_module._PrepareProgressError, match="inventory differs"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert not output.exists()
    assert not progress.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_discards_progress_when_the_exact_input_contract_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"tokenizer-version-A")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    _, old_progress, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        output,
        monkeypatch,
    )
    tokenizer.write_bytes(b"tokenizer-version-B")
    changed_fingerprint = _atomic_prepare_fingerprint(source, tokenizer)
    processed_batches = 0

    def count_recomputed_batches(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal processed_batches
        processed_batches += 1
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_recomputed_batches)
    stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=changed_fingerprint,
    )

    assert stats.valid_pairs == 3
    assert processed_batches == 3
    assert not old_progress.exists()
    assert not list(tmp_path.glob(".dataset.prepare-progress-*"))


def test_prepare_never_reuses_progress_after_source_bytes_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    _, old_progress, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        output,
        monkeypatch,
    )
    _write_resumable_prepare_source(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace("마지막 문장", "변경된 문장"),
        encoding="utf-8",
    )
    changed_fingerprint = _atomic_prepare_fingerprint(source, tokenizer)
    processed_batches = 0

    def count_recomputed_batches(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal processed_batches
        processed_batches += 1
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_recomputed_batches)
    stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=changed_fingerprint,
    )

    assert stats.valid_pairs == 3
    assert processed_batches == 3
    assert not old_progress.exists()
    assert not list(tmp_path.glob(".dataset.prepare-progress-*"))


def test_prepare_never_reuses_progress_after_the_language_graph_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    _, old_progress, original_process = _leave_resumable_prepare_progress(
        source,
        tokenizer,
        output,
        monkeypatch,
    )
    processed_batches = 0

    def count_recomputed_batches(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal processed_batches
        processed_batches += 1
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_recomputed_batches)
    stats = prepare_module.prepare_dataset(
        [str(source)],
        tokenizer,
        output,
        shard_size=8,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        translation_directions=(("ko", "ja"),),
        synthetic_sampling_weight=0.25,
        num_workers=1,
    )

    assert stats.valid_pairs == 3
    assert stats.forward_only_pairs == 3
    assert processed_batches == 3
    assert not old_progress.exists()
    assert not list(tmp_path.glob(".dataset.prepare-progress-*"))


def test_prepare_retains_completed_chunks_across_enospc_and_reuses_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_resumable_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "PREPARE_BATCH_SIZE", 1)
    original_reserve = prepare_module._ensure_prepare_write_reserve
    original_process = prepare_module._process_prepare_batch
    checkpoint_writes = 0

    def exhaust_during_second_checkpoint(
        directory: Path,
        pending_bytes: int,
        *,
        role: str = "prepared worker checkpoint",
    ) -> None:
        nonlocal checkpoint_writes
        if role == "prepared worker checkpoint":
            checkpoint_writes += 1
            if checkpoint_writes == 2:
                raise OSError(errno.ENOSPC, "simulated checkpoint capacity exhaustion")
        original_reserve(directory, pending_bytes, role=role)

    monkeypatch.setattr(
        prepare_module,
        "_ensure_prepare_write_reserve",
        exhaust_during_second_checkpoint,
    )
    with pytest.raises(OSError) as captured:
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )
    assert captured.value.errno == errno.ENOSPC
    progress = next(tmp_path.glob(".dataset.prepare-progress-*"))
    assert len(list((progress / "chunks").glob("*.json.gz"))) == 1
    assert not list(tmp_path.glob(".dataset.staging-*"))

    resumed_calls = 0

    def count_recomputed_batches(
        batch: prepare_module._PrepareBatchInput,
    ) -> list[prepare_module._PrepareEvent]:
        nonlocal resumed_calls
        resumed_calls += 1
        return original_process(batch)

    monkeypatch.setattr(prepare_module, "_ensure_prepare_write_reserve", original_reserve)
    monkeypatch.setattr(prepare_module, "_process_prepare_batch", count_recomputed_batches)
    stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert stats.valid_pairs == 3
    assert resumed_calls == 2
    assert not progress.exists()


def test_prepare_refuses_insufficient_disk_space_before_worker_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    disk_usage_type = type(prepare_module.shutil.disk_usage(tmp_path))

    def forbid_processing(_batch: prepare_module._PrepareBatchInput) -> NoReturn:
        raise AssertionError("disk preflight must run before worker processing")

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module, "_process_prepare_batch", forbid_processing)
    monkeypatch.setattr(
        prepare_module.shutil,
        "disk_usage",
        lambda _path: disk_usage_type(1_000_000, 999_999, 1),
    )

    with pytest.raises(OSError) as captured:
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert captured.value.errno == errno.ENOSPC
    assert "Insufficient free disk space" in str(captured.value)
    assert "estimated required=" in str(captured.value)
    assert "reserve=" in str(captured.value)
    assert "available=" in str(captured.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_keeps_completed_worker_chunks_when_final_capacity_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    disk_usage_type = type(prepare_module.shutil.disk_usage(tmp_path))
    usage_calls = 0

    def capacity_then_exhausted(_path: object) -> object:
        nonlocal usage_calls
        usage_calls += 1
        # Initial planning and the serialized chunk publication both pass. The
        # post-worker exact payload gate then refuses to create staging.
        free = 10**12 if usage_calls <= 2 else 1
        return disk_usage_type(10**12, 10**12 - free, free)

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)
    monkeypatch.setattr(prepare_module.shutil, "disk_usage", capacity_then_exhausted)

    with pytest.raises(OSError, match="after translation worker checkpointing") as captured:
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert captured.value.errno == errno.ENOSPC
    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))
    progress = list(tmp_path.glob(".dataset.prepare-progress-*"))
    assert len(progress) == 1
    assert list((progress[0] / "chunks").glob("*.json.gz"))


def test_prepare_output_lock_rejects_a_competing_builder_without_stale_cleanup(
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

    with prepare_module._prepare_output_lock(output):
        with pytest.raises(RuntimeError, match="output is locked by another process"):
            _prepare_atomic_dataset(
                str(source),
                tokenizer,
                output,
                expected_fingerprint=expected,
            )

    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.prepare-progress-*"))
    lock_path = tmp_path / prepare_module._prepare_output_lock_filename(output)
    assert lock_path.is_file()

    stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )
    assert stats.valid_pairs == 1
    assert lock_path.is_file()


def test_prepare_output_lock_nests_under_the_cli_artifact_lock(
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

    with artifact_lock(tmp_path):
        stats = _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    assert stats.valid_pairs == 1
    assert output.is_dir()


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
    # A filename used by the real FLORES source follows the normal three-way split.
    source = tmp_path / "data22.jsonl"
    write_tiny_jsonl(source)
    tokenizer_dir = tmp_path / "tokenizer"
    model_path = train_tokenizer(
        [str(source)],
        tokenizer_dir,
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pair=("ko", "ja"),
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
        language_pair=("ko", "ja"),
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
            language_pair=("ko", "ja"),
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
        language_pair=("ko", "ja"),
        num_workers=1,
        num_threads=1,
    )
    single = prepare_dataset(
        [str(source)],
        model_path,
        tmp_path / "single",
        validation_fraction=0.1,
        test_fraction=0.1,
        language_pair=("ko", "ja"),
        num_workers=1,
    )
    parallel = prepare_dataset(
        [str(source)],
        model_path,
        tmp_path / "parallel",
        validation_fraction=0.1,
        test_fraction=0.1,
        language_pair=("ko", "ja"),
        num_workers=2,
    )
    assert single == parallel
    assert _directory_bytes(tmp_path / "single") == _directory_bytes(tmp_path / "parallel")
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


def test_prepare_reuses_an_authenticated_unmarked_v10_legacy_stats_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    language_pair = ("de", "fr")

    class ConfigurableGraphTokenizer(_FakePrepareTokenizer):
        languages = language_pair

    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    source.write_text(
        json.dumps(
            {
                "de": "Eine absichtlich längere deutsche Quellsequenz für den Speichertest.",
                "fr": "Une phrase française suffisamment longue.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(
        source,
        tokenizer,
        language_pair=language_pair,
    )
    monkeypatch.setattr(prepare_module, "SionTokenizer", ConfigurableGraphTokenizer)
    original_stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
        language_pair=language_pair,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("stats_schema")
    for serialized in (manifest["stats"], manifest["sources"][0]["stats"]):
        serialized.pop("refinement_evidence")
        serialized.pop("reserved_draft_separator")
        serialized["ko_tokens"] = serialized.pop("src_tokens")
        serialized["ja_tokens"] = serialized.pop("tgt_tokens")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completion_path = output / prepare_module.PREPARE_COMPLETION_FILENAME
    completion_path.write_text(
        json.dumps(prepare_module._completion_payload(output, manifest)),
        encoding="utf-8",
    )

    def forbid_rebuild(_args: object) -> list[prepare_module._PrepareEvent]:
        raise AssertionError("an authenticated legacy manifest must be reused")

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", forbid_rebuild)
    reused_stats = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
        language_pair=language_pair,
    )

    assert reused_stats == original_stats
    assert reused_stats.src_tokens == original_stats.src_tokens
    assert reused_stats.tgt_tokens == original_stats.tgt_tokens
    assert reused_stats.src_tokens != reused_stats.tgt_tokens


def test_prepare_publication_failure_preserves_a_complete_resumable_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    actual_publish = prepare_module._publish_staged_directory

    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)

    def fail_publication(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated dataset publication failure")

    monkeypatch.setattr(
        prepare_module,
        "_publish_staged_directory",
        fail_publication,
    )
    with pytest.raises(OSError, match="simulated dataset publication failure"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )

    staging = list(tmp_path.glob(".dataset.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / prepare_module.PREPARE_COMPLETION_FILENAME).is_file()

    monkeypatch.setattr(prepare_module, "_publish_staged_directory", actual_publish)

    def forbid_rebuild(_args: object) -> list[prepare_module._PrepareEvent]:
        raise AssertionError("a complete generation must resume without tokenization")

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", forbid_rebuild)
    recovered = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert recovered.valid_pairs == 1
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_prepare_recovers_an_exact_output_stranded_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    _write_atomic_prepare_source(source)
    tokenizer.write_bytes(b"stable-tokenizer")
    expected = _atomic_prepare_fingerprint(source, tokenizer)
    actual_publish = prepare_module._publish_staged_directory
    monkeypatch.setattr(prepare_module, "SionTokenizer", _FakePrepareTokenizer)

    def strand_after_rename(
        staging_dir: Path,
        output_dir: Path,
        *,
        before_rename: Callable[[], None] | None = None,
    ) -> None:
        if before_rename is not None:
            before_rename()
        os.rename(staging_dir, output_dir)
        raise RuntimeError("simulated directory-fsync and rollback failure")

    monkeypatch.setattr(prepare_module, "_publish_staged_directory", strand_after_rename)
    with pytest.raises(RuntimeError, match="rollback failure"):
        _prepare_atomic_dataset(
            str(source),
            tokenizer,
            output,
            expected_fingerprint=expected,
        )
    assert (output / prepare_module.PREPARE_COMPLETION_FILENAME).is_file()

    monkeypatch.setattr(prepare_module, "_publish_staged_directory", actual_publish)

    def forbid_rebuild(_args: object) -> list[prepare_module._PrepareEvent]:
        raise AssertionError("an exact already-published output must not be rebuilt")

    monkeypatch.setattr(prepare_module, "_process_prepare_batch", forbid_rebuild)
    recovered = _prepare_atomic_dataset(
        str(source),
        tokenizer,
        output,
        expected_fingerprint=expected,
    )

    assert recovered.valid_pairs == 1
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".dataset.prepare-progress-*"))


def test_prepare_does_not_preserve_an_incomplete_staging_generation(tmp_path: Path) -> None:
    staging = tmp_path / ".dataset.staging-incomplete"
    output = tmp_path / "dataset"
    staging.mkdir()

    assert not prepare_module._publication_failure_is_resumable(
        OSError("simulated build failure"),
        staging,
        output,
    )


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
    clean = assess_pair(
        "오늘 날씨가 좋습니다.",
        "今日は天気が良いです。",
        policy,
        languages=("ko", "ja"),
    )
    assert clean.accepted
    assert clean.score == 100

    identical = assess_pair("OpenAI 123", "OpenAI 123", policy, languages=("ko", "ja"))
    assert not identical.accepted
    assert "identical_text" in identical.rejection_reasons

    wrong_scripts = assess_pair(
        "This is not Korean",
        "이것은 일본어가 아니다",
        policy,
        languages=("ko", "ja"),
    )
    assert not wrong_scripts.accepted
    assert "ko_script_mismatch" in wrong_scripts.rejection_reasons
    assert "ja_script_mismatch" in wrong_scripts.rejection_reasons

    repeated = assess_pair(
        "오류문자열오류문자열오류문자열오류문자열오류문자열",
        "エラー文字列エラー文字列エラー文字列エラー文字列エラー文字列",
        policy,
        languages=("ko", "ja"),
    )
    assert not repeated.accepted
    assert "excessive_repetition" in repeated.rejection_reasons


def test_pair_quality_rejects_critical_structured_token_corruption() -> None:
    assessment = assess_pair(
        "Pay {amount} to user@example.com",
        "{total} を user@example.com に支払う",
        languages=("en", "ja"),
    )

    assert not assessment.accepted
    assert assessment.score == 90
    assert assessment.rejection_reasons == ("structured_span_mismatch",)
    assert assessment.warning_reasons == ()


def test_prepare_rejects_critical_structured_corruption_when_quality_filter_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_module, "_PREPARE_WORKER_TOKENIZER", _FakePrepareTokenizer("unused")
    )
    row = json.dumps(
        {
            "en": "Pay {amount} to user@example.com",
            "ja": "{total} を user@example.com に支払う",
        },
        ensure_ascii=False,
    ).encode("utf-8")

    events = prepare_module._process_prepare_batch(  # pyright: ignore[reportPrivateUsage]
        (
            0,
            [(0, row)],
            QualityPolicy(),
            False,
            (("en", "ja"),),
            510,
            "pairs.jsonl",
            (("en", "ja"), ("ja", "en")),
        )
    )

    quality_events = [event for event in events if event[0] == "quality_filtered"]
    assert len(quality_events) == 1
    assert quality_events[0][1][1] == ("structured_span_mismatch",)
    assert not any(event[0] == "candidate" for event in events)


def test_prepare_filters_reserved_revision_syntax_but_keeps_authenticated_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ArbitraryLanguageTokenizer(_FakePrepareTokenizer):
        languages = ("sw", "ar")

    source = tmp_path / "arbitrary_parallel.jsonl"
    tokenizer = tmp_path / "sion.model"
    output = tmp_path / "dataset"
    rows = [
        {
            "sw": "Sentensi salama ya chanzo kwa jaribio hili.",
            "ar": "هذا نص عادي يتضمن <<draft>> داخل اقتباس طويل.",
        },
        {
            "sw": "Chanzo cha kawaida <draft> chenye alama iliyohifadhiwa.",
            "ar": "هذه ترجمة عادية لا تملك بيانات مراجعة موثقة.",
        },
        {
            "sw": "Chanzo cha kutafsiri <draft> rasimu inayohitaji kusahihishwa.",
            "ar": "هذه هي الترجمة المصححة بعد مراجعة المسودة.",
            "synthetic": True,
            "training_direction": ["sw", "ar"],
            "provenance": {"transformation": "revision"},
        },
        {
            "sw": "Hili ndilo toleo lililosahihishwa baada ya kupitia rasimu.",
            "ar": "هذا مصدر للترجمة <draft> وهذه مسودة تحتاج إلى تصحيح.",
            "synthetic": True,
            "training_direction": ["ar", "sw"],
            "provenance": {"transformation": "revision"},
        },
        {
            "sw": "Chanzo cha kawaida kisicho na muundo wa rasimu.",
            "ar": "هذه مسودة في الهدف <draft> ولا يجوز قبولها أبدا.",
            "synthetic": True,
            "training_direction": ["sw", "ar"],
            "provenance": {"transformation": "revision"},
        },
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"arbitrary-language-tokenizer")
    monkeypatch.setattr(prepare_module, "SionTokenizer", ArbitraryLanguageTokenizer)

    stats = prepare_module.prepare_dataset(
        [str(source)],
        tokenizer,
        output,
        shard_size=8,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        language_pairs=(("sw", "ar"),),
        translation_directions=(("sw", "ar"), ("ar", "sw")),
        num_workers=1,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert stats.physical_lines == 5
    assert stats.quality_filtered == 3
    assert stats.reserved_draft_separator == 3
    assert stats.valid_pairs == 2
    assert stats.forward_only_pairs == 2
    assert manifest["stats_schema"] == "sion-prepare-stats-src-tgt-v3"
    assert manifest["preprocessing_schema"] == "sion-prepare-v12"
    assert manifest["stats"]["reserved_draft_separator"] == 3


@pytest.mark.parametrize(
    ("text_a", "text_b", "metadata", "source_name", "rejected"),
    [
        (
            "A safe ordinary source sentence.",
            "An ordinary target with <<draft>> in a quotation.",
            {},
            "ordinary.jsonl",
            True,
        ),
        (
            "A corrected target sentence.",
            "A reverse source <draft> followed by its draft.",
            {
                "training_direction": ["ar", "sw"],
                "provenance": {"transformation": "revision"},
            },
            "generated.jsonl",
            False,
        ),
        (
            "A marked source without revision syntax.",
            "A forbidden target <draft> containing control syntax.",
            {
                "training_direction": ["sw", "ar"],
                "provenance": {"transformation": "revision"},
            },
            "generated.jsonl",
            True,
        ),
        (
            "A source <draft> one draft <draft> another draft.",
            "A corrected target sentence.",
            {
                "training_direction": ["sw", "ar"],
                "provenance": {"transformation": "revision"},
            },
            "generated.jsonl",
            True,
        ),
        (
            " <draft> a draft without a source.",
            "A corrected target sentence.",
            {
                "training_direction": ["sw", "ar"],
                "provenance": {"transformation": "revision"},
            },
            "generated.jsonl",
            True,
        ),
        (
            "A source without a draft <draft> ",
            "A corrected target sentence.",
            {
                "training_direction": ["sw", "ar"],
                "provenance": {"transformation": "revision"},
            },
            "generated.jsonl",
            True,
        ),
        (
            "A source <draft> followed by its draft.",
            "A corrected target sentence.",
            {
                "training_direction": ["sw", "ar"],
                "provenance": {"transformation": "backtranslation"},
            },
            "revise_conflict.jsonl",
            True,
        ),
    ],
)
def test_reserved_draft_separator_authentication_is_directional_and_structural(
    text_a: str,
    text_b: str,
    metadata: dict[str, object],
    source_name: str,
    rejected: bool,
) -> None:
    assert (
        prepare_module._pair_has_unauthenticated_draft_separator(  # pyright: ignore[reportPrivateUsage]
            text_a,
            text_b,
            language_pair=("sw", "ar"),
            metadata=metadata,
            source_name=source_name,
            translation_directions=(("sw", "ar"), ("ar", "sw")),
        )
        is rejected
    )


def test_pair_quality_requires_explicit_language_identity() -> None:
    with pytest.raises(TypeError, match="languages"):
        assess_pair("source", "target")  # type: ignore[call-arg]


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
    short = assess_pair("아", "あ", languages=("ko", "ja"))
    assert short.rejection_reasons == ("too_short",)
    profiled_short = apply_record_quality_profile(short, "expressive_v1")
    assert profiled_short.accepted
    assert profiled_short.score == 100

    repeated = assess_pair("아" * 12, "あ" * 12, languages=("ko", "ja"))
    assert repeated.rejection_reasons == ("excessive_repetition",)
    assert apply_record_quality_profile(repeated, "expressive_v1").accepted

    unsafe = assess_pair("아\u0001", "あ", languages=("ko", "ja"))
    profiled_unsafe = apply_record_quality_profile(unsafe, "expressive_v1")
    assert not profiled_unsafe.accepted
    assert profiled_unsafe.rejection_reasons == ("control_characters",)

    structured = assess_pair(
        "Use {account} now",
        "今すぐ {user} を使用します",
        languages=("en", "ja"),
    )
    profiled_structured = apply_record_quality_profile(structured, "expressive_v1")
    assert not profiled_structured.accepted
    assert profiled_structured.rejection_reasons == ("structured_span_mismatch",)


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


def test_language_pair_temperature_lifts_the_edge_that_fell_behind() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 10_000
        # Two edges, one source per edge: 9,000 ko-ja rows against 1,000 ko-en.
        pair_source_ids = np.asarray([0] * 9_000 + [1] * 1_000, dtype=np.uint16)
        source_names = ["koja.jsonl", "koen.jsonl"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

        def language_pair_ids_for_pairs(self) -> np.ndarray:
            return np.asarray([0] * 9_000 + [1] * 1_000, dtype=np.uint16)

    dataset = DummyDataset()
    natural = DistributedBucketBatchSampler(dataset, batch_size=10, bucket_size=100, seed=3)
    balanced = DistributedBucketBatchSampler(
        dataset,
        batch_size=10,
        bucket_size=100,
        seed=3,
        language_pair_sampling_alpha=0.0,
        max_source_upsampling=3.0,
    )

    def small_edge_share(sampler: DistributedBucketBatchSampler) -> float:
        indices = [index for batch in sampler for index in batch]
        return sum(index >= 9_000 for index in indices) / len(indices)

    assert small_edge_share(natural) < 0.15
    # The cap holds the rare edge to three times its natural share rather than
    # letting equal-edge sampling take half the batch.
    assert 0.25 < small_edge_share(balanced) <= 0.32


def test_language_pair_temperature_of_one_changes_nothing() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 1_000
        pair_source_ids = np.asarray([0] * 700 + [1] * 300, dtype=np.uint16)
        source_names = ["a.jsonl", "b.jsonl"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

        def language_pair_ids_for_pairs(self) -> np.ndarray:
            return np.asarray([0] * 700 + [1] * 300, dtype=np.uint16)

    dataset = DummyDataset()
    plain = DistributedBucketBatchSampler(dataset, batch_size=10, bucket_size=100, seed=11)
    explicit = DistributedBucketBatchSampler(
        dataset,
        batch_size=10,
        bucket_size=100,
        seed=11,
        language_pair_sampling_alpha=1.0,
    )

    assert [batch for batch in plain] == [batch for batch in explicit]


def test_language_pair_alpha_is_validated() -> None:
    class DummyDataset:
        bidirectional = False
        pair_count = 10
        pair_source_ids = np.zeros(10, dtype=np.uint16)
        source_names = ["a.jsonl"]

        def __len__(self) -> int:
            return self.pair_count

        def lengths_for_indices(self, indices: np.ndarray) -> np.ndarray:
            return np.ones_like(indices)

    with pytest.raises(ValueError, match="language_pair_sampling_alpha"):
        DistributedBucketBatchSampler(
            DummyDataset(), batch_size=2, language_pair_sampling_alpha=1.5
        )
