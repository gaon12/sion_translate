from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import sentencepiece as spm

from sion_translate.fingerprint import file_sha256
from sion_translate.splitting import choose_split_for_key, endpoint_split_key
from sion_translate.tokenizer import (
    SENTENCEPIECE_RESERVED_CHARACTERS,
    TOKENIZER_METADATA_VERSION,
    CorpusCounts,
    SionTokenizer,
    control_symbols,
    iter_parallel_text,
    load_tokenizer_metadata,
    tokenizer_split_digits_policy,
    train_tokenizer,
)


def test_foundation_only_language_reserves_only_a_denoise_tag() -> None:
    symbols = control_symbols(
        ["ko", "ja"],
        denoise_languages=["ko", "ja", "en"],
    )

    assert "<denoise_en>" in symbols
    assert "<2en>" not in symbols


def test_deprecated_aliases_cannot_create_duplicate_language_controls() -> None:
    symbols = control_symbols(
        ["iw", "HE", "i-klingon", "tlh", "en-BU", "en-MM", "zh-cmn", "cmn"],
        denoise_languages=["in", "id"],
        reasoning_languages=["ji", "yi"],
    )

    assert symbols.count("<2he>") == 1
    assert symbols.count("<2tlh>") == 1
    assert symbols.count("<2en-MM>") == 1
    assert symbols.count("<2cmn>") == 1
    assert symbols.count("<denoise_id>") == 1
    assert symbols.count("<reason_yi>") == 1
    assert not {
        "<2iw>",
        "<2i-klingon>",
        "<2en-BU>",
        "<2zh-cmn>",
        "<denoise_in>",
        "<reason_ji>",
    } & set(symbols)


def test_reasoning_controls_are_reserved_only_for_structured_corpus_languages() -> None:
    without_reasoning = control_symbols(["ko", "ja"], denoise_languages=["ko", "ja", "en"])
    with_reasoning = control_symbols(
        ["ko", "ja"],
        denoise_languages=["ko", "ja", "en"],
        reasoning_languages=["ja", "en"],
    )

    assert "<reason_ja>" not in without_reasoning
    assert "<think>" not in without_reasoning
    assert "<reason_ja>" in with_reasoning
    assert "<reason_en>" in with_reasoning
    assert "<reason_ko>" not in with_reasoning
    assert {"<think>", "</think>", "<answer>", "</answer>"} <= set(with_reasoning)


def _pair_for_split(split: str) -> tuple[str, str]:
    for index in range(100_000):
        ko = f"분할 검증을 위한 한국어 문장 {index}입니다."
        if choose_split_for_key(endpoint_split_key("ko", ko)) == split:
            return ko, f"分割検証用の日本語文{index}です。"
    raise AssertionError(f"Could not find text assigned to {split}")


def _text_for_language_split(language: str, split: str, template: str) -> str:
    for index in range(100_000):
        text = template.format(index=index)
        if choose_split_for_key(endpoint_split_key(language, text)) == split:
            return text
    raise AssertionError(f"Could not find {language} text assigned to {split}")


def _damaged_pair_in_train_split() -> tuple[str, str]:
    for index in range(100_000):
        text = f"OpenAI identical text {index}"
        if choose_split_for_key(endpoint_split_key("ko", text)) == "train":
            return text, text
    raise AssertionError("Could not find damaged text assigned to train")


def _repeated_pair_in_train_split() -> tuple[str, str]:
    for index in range(100_000):
        ko = f"오류오류오류오류오류오류 {index}"
        ja = f"エラーエラーエラーエラーエラーエラー {index}"
        if choose_split_for_key(endpoint_split_key("ko", ko)) == "train":
            return ko, ja
    raise AssertionError("Could not find repeated text assigned to train")


def test_iter_parallel_text_excludes_holdouts_and_invalid_rows(tmp_path: Path) -> None:
    train_pair = _pair_for_split("train")
    validation_pair = _pair_for_split("validation")
    test_pair = _pair_for_split("test")
    damaged_pair = _damaged_pair_in_train_split()
    repeated_pair = _repeated_pair_in_train_split()
    source = tmp_path / "parallel.jsonl"
    rows = [
        {"ko": train_pair[0], "ja": train_pair[1]},
        {"ko": validation_pair[0], "ja": validation_pair[1]},
        {"ko": test_pair[0], "ja": test_pair[1]},
        {"ko": 123, "ja": "文字列ではない入力です。"},
        {"ko": "", "ja": "空の入力です。"},
        {"ko": damaged_pair[0], "ja": damaged_pair[1]},
        {"ko": repeated_pair[0], "ja": repeated_pair[1]},
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert list(iter_parallel_text([source], language_pair=("ko", "ja"))) == list(train_pair)
    assert list(
        iter_parallel_text(
            [source],
            validation_fraction=0.0,
            test_fraction=0.0,
            language_pair=("ko", "ja"),
        )
    ) == [
        *train_pair,
        *validation_pair,
        *test_pair,
    ]


def test_row_scoped_reverse_direction_controls_tokenizer_split_ownership(
    tmp_path: Path,
) -> None:
    korean_train = _text_for_language_split(
        "ko",
        "train",
        "행별 방향 분할을 검증하는 한국어 문장 {index}입니다.",
    )
    japanese_validation = _text_for_language_split(
        "ja",
        "validation",
        "行単位の方向分割を検証する日本語文{index}です。",
    )
    source = tmp_path / "reverse-direction.jsonl"
    source.write_text(
        json.dumps(
            {
                "ko": korean_train,
                "ja": japanese_validation,
                "training_direction": ["ja", "ko"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        list(
            iter_parallel_text(
                [source],
                language_pairs=(("ko", "ja"),),
                translation_directions=(("ko", "ja"), ("ja", "ko")),
                num_workers=1,
            )
        )
        == []
    )


def test_kj_tokenizer_rejects_models_without_required_symbols(
    tmp_path: Path,
) -> None:
    model_prefix = tmp_path / "plain"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["한국어 문장입니다.", "日本語の文です。"] * 20),
        model_prefix=str(model_prefix),
        vocab_size=64,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
        minloglevel=2,
    )

    # Reject a plain SentencePiece model that has no <2xx> language tags.
    with pytest.raises(ValueError, match="<2xx>"):
        SionTokenizer(model_prefix.with_suffix(".model"))


def _numeric_corpus(tmp_path: Path) -> Path:
    """Build a number-heavy corpus where digit runs merge without ``split_digits``."""
    source = tmp_path / "numeric.jsonl"
    rows = []
    for index in range(400):
        amount = 38720 + index
        rows.append(
            {
                "ko": f"청구 금액은 {amount}원이고 용량은 250mg입니다. 문서 {index}번.",
                "ja": f"請求金額は{amount}ウォンで、容量は250mgです。文書{index}番。",
            }
        )
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return source


def test_corpus_counts_report_both_dimensions_from_one_pass() -> None:
    counts = CorpusCounts(characters=Counter("가나다"), sentences=1)
    assert counts.character_total == 3
    assert counts.sentences == 1


def test_train_tokenizer_rejects_missing_language_graph_without_output_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parallel.jsonl"
    source.write_text('{"de": "Hallo", "fr": "Bonjour"}\n', encoding="utf-8")
    output_dir = tmp_path / "tokenizer"

    with pytest.raises(ValueError, match="explicit language_pair or language_pairs graph"):
        train_tokenizer([str(source)], output_dir)

    assert not output_dir.exists()


def test_the_block_elements_are_reserved_and_never_required() -> None:
    """SentencePiece discards an **entire sentence** containing block elements.

    Its log says ``Reserved chars are found. Skipped: ...``. Passing such a
    character through ``required_chars`` is contradictory: it must enter the
    vocabulary, but no sentence remains to teach it. These characters are ASCII
    art residue from scraped documents, not content worth a vocabulary slot.
    """

    for codepoint in (0x2580, 0x2585, 0x2592, 0x259F):
        assert chr(codepoint) in SENTENCEPIECE_RESERVED_CHARACTERS
    assert "⁇" in SENTENCEPIECE_RESERVED_CHARACTERS
    # A neighboring geometric shape (U+25A0) is not reserved and must remain.
    assert chr(0x25A0) not in SENTENCEPIECE_RESERVED_CHARACTERS


def test_train_tokenizer_splits_digits_by_default(tmp_path: Path) -> None:
    source = _numeric_corpus(tmp_path)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "split",
        vocab_size=600,
        validation_fraction=0.0,
        test_fraction=0.0,
        language_pair=("ko", "ja"),
        num_workers=1,
        num_threads=1,
    )

    tokenizer = SionTokenizer(model_path)
    assert tokenizer.splits_digits
    # Split an amount into individual digits so the model can copy each place exactly.
    pieces = tokenizer.processor.encode("38720원", out_type=str)
    digit_pieces = [piece.replace("▁", "") for piece in pieces if piece.strip("▁").isdigit()]
    assert digit_pieces == ["3", "8", "7", "2", "0"]
    metadata = load_tokenizer_metadata(model_path)
    assert metadata is not None
    assert metadata["version"] >= TOKENIZER_METADATA_VERSION
    assert metadata["split_digits"] is True
    assert metadata["language_pair"] == ["ko", "ja"]
    assert metadata["language_pairs"] == [["ko", "ja"]]
    assert metadata["translation_directions"] == [["ko", "ja"], ["ja", "ko"]]
    assert metadata["vocab_size"] == len(tokenizer)
    assert metadata["model_file"] == model_path.name
    assert metadata["model_sha256"] == file_sha256(model_path)
    vocab_path = model_path.with_suffix(".vocab")
    assert metadata["vocab_file"] == vocab_path.name
    assert metadata["vocab_sha256"] == file_sha256(vocab_path)
    features_path = model_path.parent / "token_features.npz"
    assert metadata["token_features_file"] == features_path.name
    assert metadata["token_features_size"] == features_path.stat().st_size
    assert metadata["token_features_sha256"] == file_sha256(features_path)
    assert metadata["sentencepiece_version"] == spm.__version__
    assert metadata["required_character_count"] > 0
    assert len(metadata["required_characters_sha256"]) == 64
    assert tokenizer_split_digits_policy(model_path) is True


def test_tokenizer_round_trips_arbitrary_canonical_bcp47_controls(tmp_path: Path) -> None:
    source = tmp_path / "bcp47-parallel.jsonl"
    rows = [
        {
            "PT-br": f"Esta é uma frase de treinamento em português número {index}.",
            "zh-hant": f"這是第 {index} 個繁體中文訓練句子。",
        }
        for index in range(40)
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "bcp47-tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        required_character_min_occurrences=0,
        validation_fraction=0.0,
        test_fraction=0.0,
        language_pairs=(("PT-br", "zh-hant"),),
        translation_directions=(("PT-br", "zh-hant"),),
        foundation_languages=("sr-latn-rs",),
        reasoning_languages=("ZH-hant",),
        num_workers=1,
        num_threads=1,
    )

    tokenizer = SionTokenizer(model_path)
    metadata = load_tokenizer_metadata(model_path)
    assert tuple(tokenizer.languages) == ("pt-BR", "zh-Hant")
    assert set(tokenizer.denoise_tags) == {"pt-BR", "zh-Hant", "sr-Latn-RS"}
    assert set(tokenizer.reasoning_tags) == {"zh-Hant"}
    assert metadata is not None
    assert metadata["language_pairs"] == [["pt-BR", "zh-Hant"]]
    assert metadata["translation_directions"] == [["pt-BR", "zh-Hant"]]
    assert metadata["denoise_languages"] == ["pt-BR", "zh-Hant", "sr-Latn-RS"]
    assert metadata["reasoning_languages"] == ["zh-Hant"]


def test_train_tokenizer_can_disable_digit_splitting(tmp_path: Path) -> None:
    source = _numeric_corpus(tmp_path)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "merged",
        vocab_size=600,
        validation_fraction=0.0,
        test_fraction=0.0,
        language_pair=("ko", "ja"),
        num_workers=1,
        num_threads=1,
        split_digits=False,
    )

    tokenizer = SionTokenizer(model_path)
    # Detect that a repeated digit run such as "250" merged into one token.
    assert not tokenizer.splits_digits
    metadata = load_tokenizer_metadata(model_path)
    assert metadata is not None
    assert metadata["split_digits"] is False
    assert tokenizer_split_digits_policy(model_path) is False


def test_split_digits_policy_requires_v2_metadata(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "tokenizer_metadata.json").write_text(
        json.dumps({"version": 1, "split_digits": True}),
        encoding="utf-8",
    )
    assert tokenizer_split_digits_policy(legacy) is None

    (legacy / "tokenizer_metadata.json").write_text(
        json.dumps({"version": TOKENIZER_METADATA_VERSION, "split_digits": "true"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split_digits"):
        tokenizer_split_digits_policy(legacy)


def test_tokenizer_guard_excludes_target_owned_by_holdout(tmp_path: Path) -> None:
    validation_ko, _ = _pair_for_split("validation")
    train_ko, _ = _pair_for_split("train")
    shared_target = "これは分割をまたいで共有された日本語文です。"
    source = tmp_path / "target-leakage.jsonl"
    source.write_text(
        "".join(
            json.dumps({"ko": ko, "ja": shared_target}, ensure_ascii=False) + "\n"
            for ko in (validation_ko, train_ko)
        ),
        encoding="utf-8",
    )

    assert list(iter_parallel_text([source], language_pair=("ko", "ja"))) == []


class _StubProcessor:
    """Model a vocabulary whose control-token range extends beyond ID 256.

    SentencePiece orders pieces as pad/unk/bos/eos, user-defined symbols, 256
    byte-fallback pieces, then learned pieces.
    """

    def __init__(self, languages: list[str]) -> None:
        from sion_translate.tokenizer import (
            OPTIONAL_CONTROL_SYMBOLS,
            SHARED_CONTROL_SYMBOLS,
            SLOT_SYMBOLS,
        )

        self._pieces = ["<pad>", "<unk>", "<s>", "</s>"]
        self._pieces += [f"<2{language}>" for language in languages]
        self._pieces += [f"<denoise_{language}>" for language in languages]
        self._pieces += SHARED_CONTROL_SYMBOLS + OPTIONAL_CONTROL_SYMBOLS + SLOT_SYMBOLS
        self._pieces += [f"<0x{value:02X}>" for value in range(256)]
        # Include a learned piece that looks like a control token outside the reserved range.
        self._pieces += ["▁가", "▁나", "<2zz>", "▁다"]
        self._index = {piece: identifier for identifier, piece in enumerate(self._pieces)}

    def vocab_size(self) -> int:
        return len(self._pieces)

    def id_to_piece(self, identifier: int) -> str:
        return self._pieces[identifier]

    def piece_to_id(self, piece: str) -> int:
        return self._index.get(piece, 1)

    def pad_id(self) -> int:
        return 0

    def unk_id(self) -> int:
        return 1

    def bos_id(self) -> int:
        return 2

    def eos_id(self) -> int:
        return 3


def test_control_tokens_are_found_past_the_first_256_ids(monkeypatch) -> None:
    """Find every language tag even when the reserved range exceeds 256 IDs.

    A fixed scan limit stops halfway through the reserved range. The symptom is
    not an exception but silent discovery of only some languages, which trains
    an incorrect model.
    """
    import sion_translate.tokenizer as tokenizer_module

    languages = [f"qaa-x-l{index:03d}" for index in range(150)]
    stub = _StubProcessor(languages)
    # A control token itself must cross ID 256 for this regression test to work.
    assert stub.piece_to_id(f"<denoise_{languages[-1]}>") > 256

    monkeypatch.setattr(tokenizer_module.spm, "SentencePieceProcessor", lambda model_file: stub)
    tokenizer = SionTokenizer("stub.model")

    assert set(tokenizer.languages) == set(languages)
    assert set(tokenizer.denoise_tags) == set(languages)


def test_a_learned_piece_that_looks_like_a_tag_is_not_mistaken_for_one(monkeypatch) -> None:
    """Do not scan the learned range that follows byte-fallback pieces."""
    import sion_translate.tokenizer as tokenizer_module

    stub = _StubProcessor(["ko", "ja"])
    monkeypatch.setattr(tokenizer_module.spm, "SentencePieceProcessor", lambda model_file: stub)
    tokenizer = SionTokenizer("stub.model")

    # '<2zz>' is a learned piece outside the reserved range.
    assert "zz" not in tokenizer.language_tags
    assert set(tokenizer.languages) == {"ko", "ja"}


def _parallel_shard(path, count=1500):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "ko": f"한국어 병렬 문장 {index} 입니다",
                        "ja": f"日本語の対訳文 {index} です",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def test_monolingual_vocabulary_reaches_the_tokenizer_under_a_per_language_cap(
    tmp_path,
) -> None:
    """Foundation training needs vocabulary from its own corpus.

    Without monolingual input, corpus-specific words all use byte fallback and
    foundation training reads its own data as bytes. Adding all monolingual data
    would let the largest language monopolize the vocabulary, so each language's
    cap is based on its sentence count in the parallel corpus.
    """
    from sion_translate.data.monolingual import discover_monolingual_sources

    shard = _parallel_shard(tmp_path / "pairs.jsonl")
    corpus = tmp_path / "corpus"
    (corpus / "ko").mkdir(parents=True)
    # Vocabulary present only in the monolingual corpus, not the parallel corpus.
    (corpus / "ko" / "mono.txt").write_text(
        "\n".join(
            f"광합성 엽록체 미토콘드리아 리보솜 {index} 세포소기관 연구" for index in range(2000)
        )
        + "\n",
        encoding="utf-8",
    )
    discovery = discover_monolingual_sources(corpus, ["ko", "ja"])

    without = train_tokenizer(
        [str(shard)],
        tmp_path / "without",
        vocab_size=800,
        num_workers=1,
        language_pair=["ko", "ja"],
        monolingual=discovery,
        monolingual_sample_ratio=0.0,
    )
    with_mono = train_tokenizer(
        [str(shard)],
        tmp_path / "with",
        vocab_size=800,
        num_workers=1,
        language_pair=["ko", "ja"],
        monolingual=discovery,
        monolingual_sample_ratio=1.0,
    )

    def byte_pieces(model):
        pieces = spm.SentencePieceProcessor(model_file=str(model)).encode(
            "미토콘드리아", out_type=str
        )
        return sum(1 for piece in pieces if piece.startswith("<0x"))

    assert byte_pieces(without) > 5
    assert byte_pieces(with_mono) == 0

    # Confirm the cap: only about the parallel ko sentence count enters from 2,000 lines.
    metadata = load_tokenizer_metadata(with_mono)
    sampled = metadata["monolingual_sentences"]["ko"]
    assert 0 < sampled <= 1500
    assert metadata["monolingual_sample_ratio"] == 1.0

    # No monolingual sample is recorded when that input is disabled.
    assert load_tokenizer_metadata(without)["monolingual_sentences"] == {}
