"""Raw row annotations survive preparation without changing the core index."""

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from sion_translate.data.indexed import IndexedParallelDataset
from sion_translate.data.prepare import INDEX_DTYPE, prepare_dataset
from sion_translate.data.record_metadata import (
    RECORD_METADATA_FIELDS,
    RECORD_METADATA_FORMAT,
    decode_record_metadata,
    encode_record_metadata,
    resolve_record_training_direction,
)
from sion_translate.data.records import expand_parallel_record


def test_metadata_codec_preserves_unicode_and_permissive_json_surrogates() -> None:
    metadata = {
        "category": "감탄사",
        "provenance": {"raw_label": "\ud800"},
    }

    assert decode_record_metadata(encode_record_metadata(metadata)) == metadata


def test_record_expansion_inherits_and_overrides_supported_metadata() -> None:
    expansion = expand_parallel_record(
        {
            "metadata": {
                "domain": "conversation",
                "category": "general",
                "ignored": "not-indexed",
            },
            "provenance": {"dataset": "fixture", "revision": 3},
            "records": [
                {
                    "ko": "둘은 붕어빵처럼 닮았습니다.",
                    "ja": "二人は瓜二つです。",
                    "category": "idiom",
                    "original_direction": "ko_to_ja",
                },
                {
                    "source_language": "ja",
                    "target_language": "ko",
                    "source": "本当にそっくりです。",
                    "target": "정말 똑 닮았습니다.",
                    "metadata": {"domain": "literary"},
                    "original_direction": "ja_to_ko",
                },
            ],
        },
        (("ko", "ja"),),
    )

    assert len(expansion.pairs) == 2
    assert expansion.pairs[0].metadata == {
        "provenance": {"dataset": "fixture", "revision": 3},
        "domain": "conversation",
        "category": "idiom",
        "original_direction": "ko_to_ja",
    }
    assert expansion.pairs[1].metadata == {
        "provenance": {"dataset": "fixture", "revision": 3},
        "domain": "literary",
        "category": "general",
        "original_direction": "ja_to_ko",
    }
    assert expansion.pairs[1].language_a == "ko"
    assert expansion.pairs[1].language_b == "ja"

    provenance = expansion.pairs[0].metadata["provenance"]
    assert isinstance(provenance, dict)
    provenance["dataset"] = "mutated"
    assert expansion.pairs[1].metadata["provenance"] == {
        "dataset": "fixture",
        "revision": 3,
    }


def test_prepare_round_trips_optional_metadata_sidecars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")
    indexed_module = importlib.import_module("sion_translate.data.indexed")

    class StubTokenizer:
        languages = ("ko", "ja")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    source_path = tmp_path / "metadata.jsonl"
    rows = [
        {
            "ko": "오늘 날씨가 정말 좋습니다.",
            "ja": "今日は本当に良い天気です。",
            "provenance": {"dataset": "verified", "row": 7},
            "domain": "conversation",
            "category": "expressive",
            "original_direction": "ko_to_ja",
        },
        {
            "ko": "내일 다시 연락하겠습니다.",
            "ja": "明日またご連絡いたします。",
        },
        {
            "metadata": {"domain": "culture", "category": "idiom"},
            "records": [
                {
                    "ko": "둘은 붕어빵처럼 닮았습니다.",
                    "ja": "二人は瓜二つです。",
                    "original_direction": "parallel",
                }
            ],
        },
        {
            "ko": "천천히 말씀해 주세요.",
            "ja": "ゆっくり話してください。",
        },
    ]
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset_root = tmp_path / "dataset"
    stats = prepare_dataset(
        [str(source_path)],
        tokenizer_path,
        dataset_root,
        shard_size=3,
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        language_pair=("ko", "ja"),
        num_workers=1,
    )

    assert stats.valid_pairs == 4
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "sion-indexed-parallel-v6"
    assert manifest["record_metadata"] == {
        "format": RECORD_METADATA_FORMAT,
        "fields": list(RECORD_METADATA_FIELDS),
        "optional": True,
        "index_suffix": ".meta.npy",
        "data_suffix": ".meta.bin",
        "index_dtype": [["offset", "<u8"], ["length", "<u4"]],
    }
    assert not set(RECORD_METADATA_FIELDS).intersection(INDEX_DTYPE.names or ())
    assert len(list((dataset_root / "train").glob("*.idx.npy"))) == 2
    assert len(list((dataset_root / "train").glob("*.meta.npy"))) == 1

    decode_calls = 0
    decode_metadata = indexed_module.decode_record_metadata

    def track_decode(payload: bytes) -> dict[str, object]:
        nonlocal decode_calls
        decode_calls += 1
        return decode_metadata(payload)

    monkeypatch.setattr(indexed_module, "decode_record_metadata", track_decode)
    default_dataset = IndexedParallelDataset(dataset_root, "train", bidirectional=True)
    default_sample = default_dataset[0]
    assert "metadata" not in default_sample
    assert not set(RECORD_METADATA_FIELDS).intersection(default_sample)
    assert decode_calls == 0
    assert default_dataset.metadata_at(0)["domain"] == "conversation"
    assert decode_calls == 1

    restored_default = pickle.loads(pickle.dumps(default_dataset))
    assert restored_default.include_metadata is False
    assert "metadata" not in restored_default[0]
    assert decode_calls == 1

    dataset = IndexedParallelDataset(
        dataset_root,
        "train",
        bidirectional=True,
        include_metadata=True,
    )
    assert dataset.has_record_metadata
    expected = [
        {
            "provenance": {"dataset": "verified", "row": 7},
            "domain": "conversation",
            "category": "expressive",
            "original_direction": "ko_to_ja",
        },
        {},
        {
            "domain": "culture",
            "category": "idiom",
            "original_direction": "parallel",
        },
        {},
    ]
    assert [dataset.metadata_at(index * 2) for index in range(4)] == expected
    for pair_index, metadata in enumerate(expected):
        forward = dataset[pair_index * 2]
        reverse = dataset[pair_index * 2 + 1]
        assert forward["metadata"] == metadata
        assert reverse["metadata"] == metadata
        for field, value in metadata.items():
            assert forward[field] == value
            assert reverse[field] == value

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.include_metadata is True
    assert restored[0]["metadata"] == expected[0]
    assert restored[-1]["metadata"] == {}


def test_row_scoped_training_direction_prevents_synthetic_reverse_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")

    class StubTokenizer:
        languages = ("sw", "ar")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    source_path = tmp_path / "bt_fixture.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "sw": "sentensi ya chanzo sintetiki",
                "ar": "هذه جملة هدف حقيقية",
                "synthetic": True,
                "training_direction": ["sw", "ar"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"

    stats = prepare_dataset(
        [str(source_path)],
        tokenizer_path,
        dataset_root,
        language_pairs=(("sw", "ar"),),
        translation_directions=(("sw", "ar"), ("ar", "sw")),
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        num_workers=1,
    )
    dataset = IndexedParallelDataset(
        dataset_root,
        "train",
        bidirectional=True,
        include_metadata=True,
    )

    assert stats.valid_pairs == 1
    assert stats.forward_only_pairs == 1
    assert len(dataset) == 1
    sample = dataset[0]
    assert (sample["src_language"], sample["target_language"]) == ("sw", "ar")
    assert sample["reverse_direction_trained"] is False
    assert sample["training_direction"] == ["sw", "ar"]


def test_row_scoped_direction_is_canonicalized_before_dataset_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")

    class StubTokenizer:
        languages = ("pt-BR", "en")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    source_path = tmp_path / "canonical-direction.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "PT-br": "Esta é uma frase paralela em português.",
                "EN": "This is a parallel sentence in English.",
                "training_direction": ["PT-br", "EN"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"

    stats = prepare_dataset(
        [str(source_path)],
        tokenizer_path,
        dataset_root,
        language_pairs=(("pt-br", "EN"),),
        translation_directions=(("PT-br", "en"),),
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        num_workers=1,
    )
    dataset = IndexedParallelDataset(
        dataset_root,
        "train",
        bidirectional=True,
        include_metadata=True,
    )

    assert stats.valid_pairs == 1
    assert len(dataset) == 1
    sample = dataset[0]
    assert (sample["src_language"], sample["target_language"]) == ("pt-BR", "en")
    assert sample["training_direction"] == ["pt-BR", "en"]
    assert resolve_record_training_direction(
        {"training_direction": ["PT-br", "EN"]},
        ("pt-BR", "en"),
        frozenset({("pt-BR", "en")}),
    ) == ("pt-BR", "en")


def test_real_duplicate_precedes_a_row_scoped_synthetic_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")

    class StubTokenizer:
        languages = ("sw", "ar")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    parallel = {
        "sw": "sentensi ile ile ya majaribio",
        "ar": "هذه هي جملة الاختبار نفسها",
    }
    (tmp_path / "bt_a.jsonl").write_text(
        json.dumps(
            {
                **parallel,
                "synthetic": True,
                "training_direction": ["sw", "ar"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "real.jsonl").write_text(
        json.dumps(parallel, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"

    stats = prepare_dataset(
        [str(tmp_path)],
        tokenizer_path,
        dataset_root,
        language_pairs=(("sw", "ar"),),
        translation_directions=(("sw", "ar"), ("ar", "sw")),
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        num_workers=1,
    )
    dataset = IndexedParallelDataset(dataset_root, "train", bidirectional=True)

    assert stats.valid_pairs == 1
    assert stats.synthetic_pairs == 0
    assert stats.forward_only_pairs == 0
    assert dataset.pair_count == 1
    assert len(dataset) == 2
    assert dataset.pair_synthetic_flags is not None
    assert not bool(dataset.pair_synthetic_flags[0])


@pytest.mark.parametrize("dedup_backend", ["memory", "sqlite"])
def test_opposite_row_scoped_directions_are_distinct_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dedup_backend: str,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")

    class StubTokenizer:
        languages = ("sw", "ar")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    parallel = {
        "sw": "sentensi ile ile ya majaribio",
        "ar": "هذه هي جملة الاختبار نفسها",
        "synthetic": True,
    }
    for index, direction in enumerate((("sw", "ar"), ("ar", "sw"))):
        (tmp_path / f"bt_{index}.jsonl").write_text(
            json.dumps(
                {**parallel, "training_direction": list(direction)},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    dataset_root = tmp_path / "dataset"

    stats = prepare_dataset(
        [str(tmp_path)],
        tokenizer_path,
        dataset_root,
        language_pairs=(("sw", "ar"),),
        translation_directions=(("sw", "ar"), ("ar", "sw")),
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend=dedup_backend,
        num_workers=1,
    )
    dataset = IndexedParallelDataset(dataset_root, "train", bidirectional=True)
    observed_directions = {
        (sample["src_language"], sample["target_language"])
        for sample in (dataset[index] for index in range(len(dataset)))
    }

    assert stats.valid_pairs == 2
    assert stats.synthetic_pairs == 2
    assert stats.forward_only_pairs == 2
    assert dataset.pair_count == 2
    assert len(dataset) == 2
    assert observed_directions == {("sw", "ar"), ("ar", "sw")}


def test_prepare_rejects_row_direction_outside_the_training_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")

    class StubTokenizer:
        languages = ("sw", "ar")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    source_path = tmp_path / "bt_invalid.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "sw": "sentensi ya chanzo sintetiki",
                "ar": "هذه جملة هدف حقيقية",
                "training_direction": ["ar", "sw"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"

    with pytest.raises(ValueError, match="absent from the configured training graph"):
        prepare_dataset(
            [str(source_path)],
            tokenizer_path,
            dataset_root,
            language_pairs=(("sw", "ar"),),
            translation_directions=(("sw", "ar"),),
            validation_fraction=0.0,
            test_fraction=0.0,
            filter_quality=False,
            dedup_backend="memory",
            num_workers=1,
        )

    assert not dataset_root.exists()


def test_legacy_v2_index_without_metadata_sidecars_still_loads(tmp_path: Path) -> None:
    dataset_root = tmp_path / "legacy-v2"
    split_root = dataset_root / "train"
    split_root.mkdir(parents=True)
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
        split_root / "00000.idx.npy",
        np.asarray([(0, 2, 0, 3, 1, 2, 0, 100)], dtype=legacy_dtype),
        allow_pickle=False,
    )
    np.asarray([11, 12], dtype=np.uint32).tofile(split_root / "00000.ko.bin")
    np.asarray([21, 22, 23], dtype=np.uint32).tofile(split_root / "00000.ja.bin")
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "sion-indexed-parallel-v2",
                "language_pair": ["ko", "ja"],
                "inputs": ["legacy.jsonl"],
            }
        ),
        encoding="utf-8",
    )

    dataset = IndexedParallelDataset(
        dataset_root,
        "train",
        bidirectional=True,
        allow_unverified_legacy=True,
    )

    assert not dataset.has_record_metadata
    assert dataset.metadata_at(0) == {}
    assert "metadata" not in dataset[0]
    assert "metadata" not in dataset[1]
    assert dataset[0]["src"].tolist() == [11, 12]
    assert dataset[1]["src"].tolist() == [21, 22, 23]

    with_metadata = IndexedParallelDataset(
        dataset_root,
        "train",
        bidirectional=True,
        include_metadata=True,
        allow_unverified_legacy=True,
    )
    assert with_metadata[0]["metadata"] == {}
