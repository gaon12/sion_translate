from __future__ import annotations

import json
from pathlib import Path

from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
)
from sion_translate.data import IndexedParallelDataset
from sion_translate.data.prepare import prepare_dataset
from sion_translate.inference import Translator
from sion_translate.tokenizer import SionTokenizer, train_tokenizer


PAIRS = (("ko", "ja"), ("en", "ru"))


def test_record_expansion_supports_mixed_and_list_valued_language_keys() -> None:
    expansion = expand_parallel_record(
        {
            "ko": ["안녕하세요.", "감사합니다."],
            "ja": ["こんにちは。", "ありがとうございます。"],
            "en": "Good morning.",
            "ru": "Доброе утро.",
            "metadata": {"source": "fixture"},
        },
        PAIRS,
    )
    assert [(pair.language_a, pair.language_b) for pair in expansion.pairs] == [
        ("ko", "ja"),
        ("ko", "ja"),
        ("en", "ru"),
    ]
    assert not expansion.issues


def test_record_expansion_supports_nested_and_explicit_items() -> None:
    expansion = expand_parallel_record(
        {
            "records": [
                {"ko": "첫 문장입니다.", "ja": "最初の文です。"},
                {
                    "source_language": "ru",
                    "target_language": "en",
                    "source": "Вторая строка.",
                    "target": "The second line.",
                },
            ],
            "ko-ja": [{"source": "세 번째 문장입니다.", "target": "三番目の文です。"}],
            "en-ru": {"items": [{"source": "A fourth line.", "target": "Четвёртая строка."}]},
        },
        PAIRS,
    )
    assert len(expansion.pairs) == 4
    assert expansion.pairs[1].language_a == "en"
    assert expansion.pairs[1].text_a == "The second line."
    assert expansion.pairs[1].text_b == "Вторая строка."


def test_record_expansion_reports_unaligned_lists_without_dropping_other_pairs() -> None:
    expansion = expand_parallel_record(
        {
            "ko": ["하나", "둘"],
            "ja": ["一", "二", "三"],
            "en": "kept English",
            "ru": "сохранённый русский",
        },
        PAIRS,
    )
    assert expansion.issues == ("unaligned_lists",)
    assert len(expansion.pairs) == 1
    assert expansion.pairs[0].language_a == "en"

    scalar_list = expand_parallel_record(
        {"ko": "한 문장", "ja": ["一文", "二文"]},
        PAIRS,
    )
    assert not scalar_list.pairs
    assert scalar_list.issues == ("unaligned_lists",)


def test_language_pair_normalization_removes_reverse_duplicates() -> None:
    pairs = normalize_language_pairs(language_pairs=(("ko", "ja"), ("ja", "ko"), ("en", "ru")))
    assert pairs == PAIRS
    assert languages_from_pairs(pairs) == ("ko", "ja", "en", "ru")


def test_multilingual_inference_requires_and_validates_source_language() -> None:
    translator = Translator.__new__(Translator)
    translator.tokenizer = type(
        "TokenizerStub",
        (),
        {"languages": ("en", "ja", "ko", "ru")},
    )()
    assert translator._resolve_source_language("ko", "ja") == "ko"
    try:
        translator._resolve_source_language(None, "ja")
    except ValueError as error:
        assert "source_language" in str(error)
    else:
        raise AssertionError("multilingual source omission must fail")


def test_multilingual_tokenizer_reads_heterogeneous_rows(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    rows = []
    for index in range(40):
        rows.append(
            {
                "records": [
                    {
                        "ko": f"한국어 예문 {index}입니다.",
                        "ja": f"日本語の例文{index}です。",
                    },
                    {
                        "en": f"English training sentence number {index}.",
                        "ru": f"Русское учебное предложение номер {index}.",
                    },
                ]
            }
        )
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=640,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pairs=PAIRS,
        num_workers=1,
        num_threads=1,
    )
    tokenizer = SionTokenizer(model_path)
    assert tokenizer.languages == ("en", "ja", "ko", "ru")
    assert set(tokenizer.language_tags) == {"ko", "ja", "en", "ru"}
    assert set(tokenizer.denoise_tags) == {"ko", "ja", "en", "ru"}

    dataset_dir = tmp_path / "dataset"
    stats = prepare_dataset(
        [str(source)],
        model_path,
        dataset_dir,
        validation_fraction=0.1,
        test_fraction=0.1,
        dedup_backend="memory",
        language_pairs=PAIRS,
        num_workers=1,
    )
    assert stats.valid_pairs == 80
    with (dataset_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["format"] == "sion-indexed-parallel-v4"
    assert manifest["language_pairs"] == [["ko", "ja"], ["en", "ru"]]
    assert manifest["languages"] == ["ko", "ja", "en", "ru"]

    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)
    assert dataset.language_pairs == PAIRS
    observed_directions = {
        (dataset[index]["src_language"], dataset[index]["target_language"])
        for index in range(len(dataset))
    }
    assert observed_directions == {
        ("ko", "ja"),
        ("ja", "ko"),
        ("en", "ru"),
        ("ru", "en"),
    }


def test_prepare_rejects_tokenizer_missing_configured_languages(tmp_path: Path) -> None:
    source = tmp_path / "two-language.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "ko": f"한국어 문장 {index}입니다.",
                    "ja": f"日本語の文{index}です。",
                },
                ensure_ascii=False,
            )
            for index in range(30)
        )
        + "\n",
        encoding="utf-8",
    )
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "two-language-tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        num_workers=1,
        num_threads=1,
    )
    try:
        prepare_dataset(
            [str(source)],
            model_path,
            tmp_path / "invalid-dataset",
            language_pairs=PAIRS,
            num_workers=1,
        )
    except ValueError as error:
        assert "missing configured language tags" in str(error)
    else:
        raise AssertionError("stale two-language tokenizer must be rejected")
