from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sion_translate.data.prepare as prepare_module
import sion_translate.tokenizer as tokenizer_module


class PoolConstructionObserved(RuntimeError):
    pass


def _raising_pool_capture(captured: dict[str, object]):
    def construct(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise PoolConstructionObserved

    return construct


def test_tokenizer_preprocessing_pool_uses_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        json.dumps({"ko": "충분히 긴 한국어 문장입니다", "ja": "十分に長い日本語文です"}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tokenizer_module,
        "ProcessPoolExecutor",
        _raising_pool_capture(captured),
    )

    with pytest.raises(PoolConstructionObserved):
        list(
            tokenizer_module.iter_parallel_text_with_languages(
                [source],
                validation_fraction=0.0,
                test_fraction=0.0,
                language_pair=("ko", "ja"),
                num_workers=2,
            )
        )

    context = captured["mp_context"]
    assert context.get_start_method() == "spawn"  # type: ignore[union-attr]


def test_dataset_preparation_pool_uses_spawn_and_cleans_failed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        json.dumps({"ko": "충분히 긴 한국어 문장입니다", "ja": "十分に長い日本語文です"}) + "\n",
        encoding="utf-8",
    )
    tokenizer = tmp_path / "sion.model"
    tokenizer.write_bytes(b"test-tokenizer")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        prepare_module,
        "SionTokenizer",
        lambda _path: SimpleNamespace(languages=("ko", "ja")),
    )
    monkeypatch.setattr(
        prepare_module,
        "ProcessPoolExecutor",
        _raising_pool_capture(captured),
    )
    output = tmp_path / "dataset"

    with pytest.raises(PoolConstructionObserved):
        prepare_module.prepare_dataset(
            [str(source)],
            tokenizer,
            output,
            language_pair=("ko", "ja"),
            num_workers=2,
        )

    context = captured["mp_context"]
    assert context.get_start_method() == "spawn"  # type: ignore[union-attr]
    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.tmp-*"))
