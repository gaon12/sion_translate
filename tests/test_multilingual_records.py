from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_record_expansion_supports_hugging_face_translation_container() -> None:
    expansion = expand_parallel_record(
        {
            "translation": {
                "ko": "안녕하세요.",
                "ja": "こんにちは。",
                "en": "Hello.",
                "ru": "Здравствуйте.",
            }
        },
        PAIRS,
    )
    assert [
        (pair.language_a, pair.text_a, pair.language_b, pair.text_b) for pair in expansion.pairs
    ] == [
        ("ko", "안녕하세요.", "ja", "こんにちは。"),
        ("en", "Hello.", "ru", "Здравствуйте."),
    ]
    assert not expansion.issues


def test_record_expansion_keeps_scalar_translation_as_explicit_target() -> None:
    expansion = expand_parallel_record(
        {
            "source_language": "ko",
            "target_language": "ja",
            "source": "안녕하세요.",
            "translation": "こんにちは。",
        },
        PAIRS,
    )
    assert len(expansion.pairs) == 1
    assert expansion.pairs[0].text_b == "こんにちは。"
    assert not expansion.issues


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


def test_language_pair_normalization_requires_one_explicit_shape() -> None:
    with pytest.raises(ValueError, match="explicit language_pair"):
        normalize_language_pairs()
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize_language_pairs(("sw", "ar"), (("sw", "ar"),))


def test_language_pair_normalization_canonicalizes_script_and_region_tags() -> None:
    assert normalize_language_pairs(
        language_pairs=(("pt-br", "ZH-hant"), ("sr-latn-rs", "de"))
    ) == (("pt-BR", "zh-Hant"), ("sr-Latn-RS", "de"))


def test_record_expansion_resolves_canonical_alias_keys() -> None:
    expansion = expand_parallel_record(
        {"PT-br": "Olá.", "zh-hant": "您好。"},
        (("pt-BR", "zh-Hant"),),
    )

    assert expansion.issues == ()
    assert [
        (pair.language_a, pair.text_a, pair.language_b, pair.text_b) for pair in expansion.pairs
    ] == [("pt-BR", "Olá.", "zh-Hant", "您好。")]


def test_record_expansion_rejects_duplicate_canonical_language_keys() -> None:
    expansion = expand_parallel_record(
        {"pt-BR": "primeiro", "pt-br": "segundo", "en": "English"},
        (("pt-BR", "en"),),
    )

    assert expansion.pairs == ()
    assert "duplicate_language_key" in expansion.issues


def test_hyphenated_language_pair_containers_use_unambiguous_separators() -> None:
    expansion = expand_parallel_record(
        {
            "zh-Hant/en": {"source": "您好。", "target": "Hello."},
            "en_to_sr-Latn-RS": {"source": "Hello.", "target": "Zdravo."},
        },
        (("zh-Hant", "en"), ("en", "sr-Latn-RS")),
    )

    assert expansion.issues == ()
    assert [(pair.language_a, pair.language_b) for pair in expansion.pairs] == [
        ("zh-Hant", "en"),
        ("en", "sr-Latn-RS"),
    ]


def test_pair_container_labels_resolve_canonical_language_aliases() -> None:
    expansion = expand_parallel_record(
        {"pt-br/EN": {"source": "Olá.", "target": "Hello."}},
        (("pt-BR", "en"),),
    )

    assert expansion.issues == ()
    assert [(pair.language_a, pair.language_b) for pair in expansion.pairs] == [("pt-BR", "en")]


@pytest.mark.parametrize(
    ("label", "expected_texts"),
    [
        ("EN-DE", ("source text", "target text")),
        ("DE-EN", ("target text", "source text")),
    ],
)
def test_legacy_simple_pair_containers_are_case_canonicalized(
    label: str,
    expected_texts: tuple[str, str],
) -> None:
    expansion = expand_parallel_record(
        {label: {"source": "source text", "target": "target text"}},
        (("en", "de"),),
    )

    assert expansion.issues == ()
    assert [
        (pair.language_a, pair.text_a, pair.language_b, pair.text_b) for pair in expansion.pairs
    ] == [("en", expected_texts[0], "de", expected_texts[1])]


def test_canonical_duplicate_pair_containers_are_rejected() -> None:
    expansion = expand_parallel_record(
        {
            "pt-BR/en": {"source": "primeiro", "target": "first"},
            "pt-br/EN": {"source": "segundo", "target": "second"},
        },
        (("pt-BR", "en"),),
    )

    assert expansion.pairs == ()
    assert expansion.issues == ("duplicate_pair_container",)


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


def test_prepare_canonicalizes_source_only_languages_before_direction_gating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubTokenizer:
        languages = ("pt-BR", "zh-Hant")

        def __init__(self, _model_path: str | Path):
            pass

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

    monkeypatch.setattr("sion_translate.data.prepare.SionTokenizer", StubTokenizer)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    source = tmp_path / "parallel.jsonl"
    source.write_text(
        json.dumps({"PT-br": "Olá.", "zh-hant": "您好。"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"

    prepare_dataset(
        [str(source)],
        tokenizer_path,
        dataset_dir,
        language_pairs=(("pt-br", "zh-hant"),),
        source_only_languages=("PT-br",),
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        num_workers=1,
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_only_languages"] == ["pt-BR"]
    assert manifest["translation_directions"] == [["pt-BR", "zh-Hant"]]


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
    assert manifest["format"] == "sion-indexed-parallel-v6"
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
        language_pair=("ko", "ja"),
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
