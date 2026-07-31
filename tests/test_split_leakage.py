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
