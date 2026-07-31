from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import sion_translate.data.prepare as prepare_module
from sion_translate.splitting import (
    TargetSplitGuard,
    choose_split_for_key,
    endpoint_split_digest,
    endpoint_split_key,
)
from sion_translate.cli.train_tokenizer import build_parser as tokenizer_argument_parser
from sion_translate.tokenizer import iter_parallel_text


VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2
SHARED_ENGLISH = "The shared endpoint carries the same English sentence."


def _line_for_split(split: str, make_row: Callable[[int], dict[str, str]]) -> str:
    for index in range(10_000):
        line = json.dumps(make_row(index), ensure_ascii=False, sort_keys=True)
        record_digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        if (
            choose_split_for_key(
                f"record\0{record_digest}",
                VALIDATION_FRACTION,
                TEST_FRACTION,
            )
            == split
        ):
            return line
    raise AssertionError(f"could not construct a multilingual row assigned to {split}")


def _case(
    endpoint_roles: str,
) -> tuple[
    tuple[tuple[str, str], ...],
    Callable[[int], dict[str, str]],
    Callable[[int], dict[str, str]],
]:
    if endpoint_roles == "source-source":
        return (
            (("en", "de"), ("en", "fr")),
            lambda index: {
                "en": SHARED_ENGLISH,
                "de": f"Deutscher Validierungssatz Nummer {index}.",
            },
            lambda index: {
                "en": SHARED_ENGLISH,
                "fr": f"Phrase française d'entraînement numéro {index}.",
            },
        )
    if endpoint_roles == "target-target":
        return (
            (("de", "en"), ("fr", "en")),
            lambda index: {
                "de": f"Deutscher Validierungssatz Nummer {index}.",
                "en": SHARED_ENGLISH,
            },
            lambda index: {
                "fr": f"Phrase française d'entraînement numéro {index}.",
                "en": SHARED_ENGLISH,
            },
        )
    if endpoint_roles == "source-target":
        return (
            (("en", "de"), ("fr", "en")),
            lambda index: {
                "en": SHARED_ENGLISH,
                "de": f"Deutscher Validierungssatz Nummer {index}.",
            },
            lambda index: {
                "fr": f"Phrase française d'entraînement numéro {index}.",
                "en": SHARED_ENGLISH,
            },
        )
    raise AssertionError(f"unknown endpoint role fixture: {endpoint_roles}")


class _TokenizerStub:
    languages = ("de", "en", "fr")

    def __init__(self, _model_path: str | Path):
        pass

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def test_endpoint_keys_are_always_scoped_by_language() -> None:
    text = "Ａ shared   surface"

    assert endpoint_split_key("en", text) == "en\0A shared surface"
    assert endpoint_split_key("en", text) != endpoint_split_key("de", text)
    assert endpoint_split_key("en", text, approximate=True).startswith("en\0")
    assert endpoint_split_digest("en", text) != endpoint_split_digest("de", text)


def test_accept_many_rejection_does_not_partially_register_a_row() -> None:
    guard = TargetSplitGuard(
        estimated_pairs=10,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    occupied = endpoint_split_digest("en", "Already owned by validation.")
    unowned = endpoint_split_digest("de", "This endpoint must remain unowned.")

    assert guard.accept("validation", occupied)
    assert not guard.accept_many("train", (unowned, occupied))
    assert guard.accept("test", unowned)


@pytest.mark.parametrize(
    "endpoint_roles",
    ("source-source", "target-target", "source-target"),
)
def test_prepare_rejects_same_language_endpoint_across_splits(
    endpoint_roles: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    language_pairs, make_validation, make_train = _case(endpoint_roles)
    source = tmp_path / f"{endpoint_roles}.jsonl"
    source.write_text(
        "\n".join(
            (
                _line_for_split("validation", make_validation),
                _line_for_split("train", make_train),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_model = tmp_path / "tokenizer.model"
    tokenizer_model.write_bytes(b"test tokenizer identity")
    monkeypatch.setattr(prepare_module, "SionTokenizer", _TokenizerStub)
    monkeypatch.setattr(prepare_module, "_PREPARE_WORKER_TOKENIZER", None)

    output = tmp_path / "dataset"
    stats = prepare_module.prepare_dataset(
        [str(source)],
        tokenizer_model,
        output,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        filter_quality=False,
        dedup_backend="memory",
        language_pairs=language_pairs,
        num_workers=1,
    )

    assert stats.valid_pairs == 1
    assert stats.validation == 1
    assert stats.train == 0
    assert stats.split_conflicts == 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_leakage_guard"] == "language-endpoint-bloom-v2"
    assert manifest["endpoint_leakage_key"] == "language-prefixed-exact-v1"
    assert manifest["split_key"] == "record-sha256-v1"


def test_tokenizer_excludes_train_candidate_reusing_holdout_source_as_target(
    tmp_path: Path,
) -> None:
    language_pairs, make_validation, make_train = _case("source-target")
    source = tmp_path / "cross-role.jsonl"
    source.write_text(
        "\n".join(
            (
                _line_for_split("validation", make_validation),
                _line_for_split("train", make_train),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        list(
            iter_parallel_text(
                [source],
                validation_fraction=VALIDATION_FRACTION,
                test_fraction=TEST_FRACTION,
                language_pairs=language_pairs,
                num_workers=1,
            )
        )
        == []
    )


def _text_with_split(
    language: str,
    split: str,
    *,
    approximate: bool,
    prefix: str,
) -> str:
    for index in range(100_000):
        text = f"{prefix} {index} 문장입니다."
        key = endpoint_split_key(language, text, approximate=approximate)
        if choose_split_for_key(key, VALIDATION_FRACTION, TEST_FRACTION) == split:
            return text
    raise AssertionError(f"could not find {language} text assigned to {split}")


def test_tokenizer_honors_the_same_approximate_split_policy_as_preparation(
    tmp_path: Path,
) -> None:
    for index in range(100_000):
        source_text = f"근사 분할 정책을 확인하는 충분히 긴 한국어 문장 {index}입니다."
        exact = choose_split_for_key(
            endpoint_split_key("ko", source_text),
            VALIDATION_FRACTION,
            TEST_FRACTION,
        )
        approximate = choose_split_for_key(
            endpoint_split_key("ko", source_text, approximate=True),
            VALIDATION_FRACTION,
            TEST_FRACTION,
        )
        if exact == "validation" and approximate == "train":
            break
    else:
        raise AssertionError("could not construct divergent exact/approximate assignments")

    source = tmp_path / "approximate.jsonl"
    source.write_text(
        json.dumps(
            {"ko": source_text, "ja": "近似分割方針を確認する十分に長い日本語の文です。"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    common = {
        "validation_fraction": VALIDATION_FRACTION,
        "test_fraction": TEST_FRACTION,
        "num_workers": 1,
    }
    assert list(iter_parallel_text([source], approximate_split=False, **common)) == []
    assert len(list(iter_parallel_text([source], approximate_split=True, **common))) == 2


def test_tokenizer_keeps_configured_synthetic_files_train_only(tmp_path: Path) -> None:
    source_text = _text_with_split(
        "ko",
        "validation",
        approximate=False,
        prefix="합성 파일 분할을 확인하는",
    )
    source = tmp_path / "generated_custom_corpus.jsonl"
    target_text = "合成ファイルの分割方針を確認する日本語文です。"
    source.write_text(
        json.dumps({"ko": source_text, "ja": target_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert list(
        iter_parallel_text(
            [source],
            validation_fraction=VALIDATION_FRACTION,
            test_fraction=TEST_FRACTION,
            train_only_prefixes=("generated_",),
            num_workers=1,
        )
    ) == [source_text, target_text]


def test_tokenizer_applies_source_only_swapping_before_split_assignment(
    tmp_path: Path,
) -> None:
    ko_text = _text_with_split(
        "ko",
        "validation",
        approximate=False,
        prefix="원래 첫 번째 언어인",
    )
    kj_text = _text_with_split(
        "kj",
        "train",
        approximate=False,
        prefix="입력 전용 혼합 언어인",
    )
    source = tmp_path / "source-only.jsonl"
    source.write_text(
        json.dumps({"ko": ko_text, "kj": kj_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    common = {
        "validation_fraction": VALIDATION_FRACTION,
        "test_fraction": TEST_FRACTION,
        "language_pairs": (("ko", "kj"),),
        "num_workers": 1,
    }

    assert list(iter_parallel_text([source], **common)) == []
    assert list(
        iter_parallel_text(
            [source],
            source_only_languages=("kj",),
            **common,
        )
    ) == [kj_text, ko_text]


def test_tokenizer_cli_uses_safe_partition_defaults() -> None:
    parser = tokenizer_argument_parser()
    required = ["--input", "data/*.jsonl", "--output-dir", "artifacts/tokenizer"]
    safe = parser.parse_args(required)
    exact = parser.parse_args([*required, "--exact-split"])

    assert safe.approximate_split is True
    assert safe.train_only_prefix
    assert exact.approximate_split is False
