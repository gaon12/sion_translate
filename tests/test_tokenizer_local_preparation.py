from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

import sion_translate.tokenizer as tokenizer_module
from sion_translate.fingerprint import file_sha256
from sion_translate.tokenizer import (
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_STAGING_PREFIX,
    TokenizerSentence,
    iter_stratified_tokenizer_sentences,
    stratified_sentence_quotas,
    train_tokenizer,
)


def _parallel_corpus(path: Path, *, rows: int = 60) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "pt-BR": f"Portuguese training sentence number {index} with enough context.",
                    "zh-Hant": f"這是第 {index} 個有足夠上下文的繁體中文訓練句子。",
                },
                ensure_ascii=False,
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return path


def _train_small(corpus: Path, output: Path) -> Path:
    return train_tokenizer(
        [str(corpus)],
        output,
        vocab_size=420,
        input_sentence_size=80,
        character_coverage=1.0,
        required_character_min_occurrences=0,
        validation_fraction=0.0,
        test_fraction=0.0,
        language_pairs=(("pt-BR", "zh-Hant"),),
        translation_directions=(("pt-BR", "zh-Hant"),),
        num_workers=1,
        num_threads=1,
    )


def test_stratified_quotas_are_exact_and_language_generic() -> None:
    counts = {"pt-BR": 10_000, "zh-Hant": 400, "sr-Latn-RS": 100}

    quotas = stratified_sentence_quotas(counts, 1_000, alpha=0.7)

    assert sum(quotas.values()) == 1_000
    assert set(quotas) == set(counts)
    assert all(quota > 0 for quota in quotas.values())
    assert quotas["sr-Latn-RS"] > 1_000 * counts["sr-Latn-RS"] / sum(counts.values())
    assert all(quotas[language] <= counts[language] for language in counts)


def test_stratified_iterator_is_exact_reproducible_and_corpus_wide() -> None:
    records = [TokenizerSentence("qaa-Latn", f"qaa-{index}") for index in range(100)] + [
        TokenizerSentence("qbb-Cyrl", f"qbb-{index}") for index in range(20)
    ]
    counts = {"qaa-Latn": 100, "qbb-Cyrl": 20}
    quotas = {"qaa-Latn": 10, "qbb-Cyrl": 5}

    first = list(iter_stratified_tokenizer_sentences(records, counts, quotas, seed=17))
    second = list(iter_stratified_tokenizer_sentences(records, counts, quotas, seed=17))

    assert first == second
    assert len(first) == 15
    qaa_indices = [int(text.split("-")[-1]) for text in first if text.startswith("qaa-")]
    assert max(qaa_indices) - min(qaa_indices) >= 70


def test_stratified_iterator_detects_a_changed_second_pass() -> None:
    records = [TokenizerSentence("de", f"sentence-{index}") for index in range(9)]

    with pytest.raises(RuntimeError, match="changed between counting and sampling"):
        list(iter_stratified_tokenizer_sentences(records, {"de": 10}, {"de": 5}))


def test_failed_training_never_publishes_a_model_or_leaves_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _parallel_corpus(tmp_path / "parallel.jsonl")
    output = tmp_path / "tokenizer"

    def fail_training(**kwargs: object) -> None:
        # Consume the bounded iterator so count drift checks still execute.
        list(cast(Iterable[str], kwargs["sentence_iterator"]))
        raise RuntimeError("simulated trainer failure")

    monkeypatch.setattr(tokenizer_module.spm.SentencePieceTrainer, "train", fail_training)

    with pytest.raises(RuntimeError, match="simulated trainer failure"):
        _train_small(corpus, output)

    assert not (output / "sion.model").exists()
    assert list(output.glob(".sion-tokenizer-staging-*")) == []


def test_completed_generation_resumes_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _parallel_corpus(tmp_path / "parallel.jsonl")
    output = tmp_path / "tokenizer"
    first = _train_small(corpus, output)
    first_digest = file_sha256(first)

    def unexpected_training(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("a matching complete tokenizer must be reused")

    monkeypatch.setattr(tokenizer_module.spm.SentencePieceTrainer, "train", unexpected_training)
    resumed = _train_small(corpus, output)

    assert resumed == first
    assert file_sha256(resumed) == first_digest
    metadata = json.loads((output / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sampled_sentences"] == 80
    assert set(metadata["sampled_sentences_per_language"]) == {"pt-BR", "zh-Hant"}
    assert metadata["training_contract"]["sampling_alpha"] == 0.7


def test_completed_private_staging_is_recovered_before_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _parallel_corpus(tmp_path / "parallel.jsonl")
    output = tmp_path / "tokenizer"
    model = _train_small(corpus, output)
    expected_digest = file_sha256(model)
    staging = output / f"{TOKENIZER_STAGING_PREFIX}recoverable"
    staging.mkdir()
    for name in TOKENIZER_ARTIFACT_FILENAMES:
        (output / name).replace(staging / name)

    def unexpected_training(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("completed staging should be published without retraining")

    monkeypatch.setattr(tokenizer_module.spm.SentencePieceTrainer, "train", unexpected_training)
    recovered = _train_small(corpus, output)

    assert file_sha256(recovered) == expected_digest
    assert not staging.exists()


def test_existing_generation_rejects_changed_source_content(tmp_path: Path) -> None:
    corpus = _parallel_corpus(tmp_path / "parallel.jsonl")
    output = tmp_path / "tokenizer"
    model = _train_small(corpus, output)
    original_digest = file_sha256(model)
    corpus.write_text(
        corpus.read_text(encoding="utf-8")
        + json.dumps(
            {
                "pt-BR": "A later source sentence changes the authenticated input.",
                "zh-Hant": "後來加入的句子會改變經過驗證的輸入。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different source or training contract"):
        _train_small(corpus, output)

    assert file_sha256(model) == original_digest


def test_full_corpus_opt_out_still_disables_sentencepiece_resampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _parallel_corpus(tmp_path / "parallel.jsonl", rows=10)
    observed: dict[str, object] = {}

    def inspect_arguments(**kwargs: object) -> None:
        observed.update(kwargs)
        list(cast(Iterable[str], kwargs["sentence_iterator"]))
        raise RuntimeError("argument inspection complete")

    monkeypatch.setattr(tokenizer_module.spm.SentencePieceTrainer, "train", inspect_arguments)
    with pytest.raises(RuntimeError, match="argument inspection complete"):
        train_tokenizer(
            [str(corpus)],
            tmp_path / "tokenizer",
            vocab_size=420,
            input_sentence_size=0,
            character_coverage=1.0,
            required_character_min_occurrences=0,
            validation_fraction=0.0,
            test_fraction=0.0,
            language_pairs=(("pt-BR", "zh-Hant"),),
            num_workers=1,
            num_threads=1,
        )

    assert observed["input_sentence_size"] == 0
    assert observed["shuffle_input_sentence"] is False
