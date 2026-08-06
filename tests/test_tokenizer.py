from __future__ import annotations

import json
from pathlib import Path

import pytest
import sentencepiece as spm

from sion_translate.fingerprint import file_sha256
from sion_translate.splitting import choose_split_for_key, endpoint_split_key
from sion_translate.tokenizer import (
    TOKENIZER_METADATA_VERSION,
    SionTokenizer,
    iter_parallel_text,
    load_tokenizer_metadata,
    tokenizer_split_digits_policy,
    train_tokenizer,
)


def _pair_for_split(split: str) -> tuple[str, str]:
    for index in range(100_000):
        ko = f"분할 검증을 위한 한국어 문장 {index}입니다."
        if choose_split_for_key(endpoint_split_key("ko", ko)) == split:
            return ko, f"分割検証用の日本語文{index}です。"
    raise AssertionError(f"Could not find text assigned to {split}")


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

    assert list(iter_parallel_text([source])) == list(train_pair)
    assert list(iter_parallel_text([source], validation_fraction=0.0, test_fraction=0.0)) == [
        *train_pair,
        *validation_pair,
        *test_pair,
    ]


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

    # 언어 태그(<2xx>)가 없는 일반 SentencePiece 모델은 거부해야 한다.
    with pytest.raises(ValueError, match="<2xx>"):
        SionTokenizer(model_prefix.with_suffix(".model"))


def _numeric_corpus(tmp_path: Path) -> Path:
    """숫자가 자주 반복되는 코퍼스. split_digits 가 없으면 숫자열이 병합된다."""
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


def test_train_tokenizer_splits_digits_by_default(tmp_path: Path) -> None:
    source = _numeric_corpus(tmp_path)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "split",
        vocab_size=600,
        validation_fraction=0.0,
        test_fraction=0.0,
        num_workers=1,
        num_threads=1,
    )

    tokenizer = SionTokenizer(model_path)
    assert tokenizer.splits_digits
    # 금액이 한 자리씩 분리되어야 모델이 자릿수를 그대로 옮길 수 있다.
    pieces = tokenizer.processor.encode("38720원", out_type=str)
    digit_pieces = [piece.replace("▁", "") for piece in pieces if piece.strip("▁").isdigit()]
    assert digit_pieces == ["3", "8", "7", "2", "0"]
    metadata = load_tokenizer_metadata(model_path)
    assert metadata is not None
    assert metadata["version"] >= TOKENIZER_METADATA_VERSION
    assert metadata["split_digits"] is True
    assert metadata["language_pair"] == ["ko", "ja"]
    assert metadata["language_pairs"] == [["ko", "ja"]]
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
    assert tokenizer_split_digits_policy(model_path) is True


def test_train_tokenizer_can_disable_digit_splitting(tmp_path: Path) -> None:
    source = _numeric_corpus(tmp_path)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "merged",
        vocab_size=600,
        validation_fraction=0.0,
        test_fraction=0.0,
        num_workers=1,
        num_threads=1,
        split_digits=False,
    )

    tokenizer = SionTokenizer(model_path)
    # 반복되는 "250" 같은 숫자열이 하나의 토큰으로 병합되므로 감지에 걸린다.
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

    assert list(iter_parallel_text([source])) == []


class _StubProcessor:
    """제어 토큰 구간이 256 ID 를 넘는 vocab 을 흉내 냅니다.

    SentencePiece 배치 순서: pad/unk/bos/eos → user_defined_symbols →
    byte fallback 256개 → 학습된 조각.
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
        # 학습된 조각. 예약 구간 밖에서 제어 토큰처럼 보이는 것도 하나 섞습니다.
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
    """예약 구간이 256 ID 를 넘어도 모든 언어 태그를 찾아야 한다.

    고정 상한을 쓰면 스캔이 예약 구간 중간에서 끊기고, 증상이 예외가 아니라
    '일부 언어만 감지됨' 이라 조용히 잘못된 모델을 학습하게 됩니다.
    """
    import sion_translate.tokenizer as tokenizer_module

    languages = [f"l{index:03d}" for index in range(150)]
    stub = _StubProcessor(languages)
    # 제어 토큰 **자체**가 256 ID 를 넘어가야 회귀를 잡는다.
    assert stub.piece_to_id(f"<denoise_{languages[-1]}>") > 256

    monkeypatch.setattr(tokenizer_module.spm, "SentencePieceProcessor", lambda model_file: stub)
    tokenizer = SionTokenizer("stub.model")

    assert set(tokenizer.languages) == set(languages)
    assert set(tokenizer.denoise_tags) == set(languages)


def test_a_learned_piece_that_looks_like_a_tag_is_not_mistaken_for_one(monkeypatch) -> None:
    """byte fallback 조각 이후는 학습된 구간이므로 스캔하지 않는다."""
    import sion_translate.tokenizer as tokenizer_module

    stub = _StubProcessor(["ko", "ja"])
    monkeypatch.setattr(tokenizer_module.spm, "SentencePieceProcessor", lambda model_file: stub)
    tokenizer = SionTokenizer("stub.model")

    # '<2zz>' 는 예약 구간 밖의 학습된 조각이다.
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
    """foundation 단계가 자기 코퍼스에 없는 어휘로 학습하면 안 된다.

    단일어를 넣지 않으면 그 코퍼스에만 있는 낱말이 전부 byte fallback 이 되고,
    foundation 은 자기 데이터를 바이트로 읽게 됩니다. 전량 넣으면 분량이 큰
    언어가 vocab 을 독식하므로, 상한은 병렬 코퍼스의 해당 언어 문장 수를
    기준으로 겁니다.
    """
    from sion_translate.data.monolingual import discover_monolingual_sources

    shard = _parallel_shard(tmp_path / "pairs.jsonl")
    corpus = tmp_path / "corpus"
    (corpus / "ko").mkdir(parents=True)
    # 병렬 코퍼스에는 없고 단일어에만 있는 어휘.
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

    # 상한이 실제로 걸렸는지: 단일어 2,000줄 중 병렬 ko 문장 수 언저리만 들어간다.
    metadata = load_tokenizer_metadata(with_mono)
    sampled = metadata["monolingual_sentences"]["ko"]
    assert 0 < sampled <= 1500
    assert metadata["monolingual_sample_ratio"] == 1.0

    # 넣지 않았을 때는 표본이 기록되지 않는다.
    assert load_tokenizer_metadata(without)["monolingual_sentences"] == {}
