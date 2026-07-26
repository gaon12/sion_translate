from __future__ import annotations

import json
from pathlib import Path

import pytest
import sentencepiece as spm

from sion_translate.splitting import choose_split_for_text
from sion_translate.tokenizer import SionTokenizer, iter_parallel_text, train_tokenizer


def _pair_for_split(split: str) -> tuple[str, str]:
    for index in range(100_000):
        ko = f"분할 검증을 위한 한국어 문장 {index}입니다."
        if choose_split_for_text(ko) == split:
            return ko, f"分割検証用の日本語文{index}です。"
    raise AssertionError(f"Could not find text assigned to {split}")


def _damaged_pair_in_train_split() -> tuple[str, str]:
    for index in range(100_000):
        text = f"OpenAI identical text {index}"
        if choose_split_for_text(text) == "train":
            return text, text
    raise AssertionError("Could not find damaged text assigned to train")


def _repeated_pair_in_train_split() -> tuple[str, str]:
    for index in range(100_000):
        ko = f"오류오류오류오류오류오류 {index}"
        ja = f"エラーエラーエラーエラーエラーエラー {index}"
        if choose_split_for_text(ko) == "train":
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
    assert list(
        iter_parallel_text(
            [source], validation_fraction=0.0, test_fraction=0.0
        )
    ) == [*train_pair, *validation_pair, *test_pair]


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
