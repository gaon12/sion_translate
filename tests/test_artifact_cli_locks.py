from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sion_translate.cli import prepare_data as prepare_cli
from sion_translate.cli import train_tokenizer as tokenizer_cli


@dataclass
class _Stats:
    records: int = 1


def test_prepare_cli_holds_tokenizer_and_dataset_parent_locks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = tmp_path / "tokenizer" / "sion.model"
    output = tmp_path / "prepared" / "dataset"
    expected_roots = {
        tmp_path.resolve(),
        tokenizer.resolve().parent,
        output.resolve().parent,
    }
    locked = False

    @contextmanager
    def fake_locks(roots):
        nonlocal locked
        assert {Path(root).resolve() for root in roots} == expected_roots
        locked = True
        try:
            yield tuple(expected_roots)
        finally:
            locked = False

    def fake_prepare(*_args, **kwargs) -> _Stats:
        assert locked
        assert kwargs["managed_augmentation_prefix"] == "bt_"
        return _Stats()

    monkeypatch.setattr(prepare_cli, "configure_stdio", lambda: None)
    monkeypatch.setattr(prepare_cli, "artifact_locks", fake_locks)
    monkeypatch.setattr(prepare_cli, "prepare_dataset", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sion-prepare-data",
            "--input",
            str(tmp_path / "raw.jsonl"),
            "--tokenizer",
            str(tokenizer),
            "--output-dir",
            str(output),
        ],
    )

    prepare_cli.main()

    assert not locked


def test_tokenizer_cli_holds_the_canonical_output_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "tokenizer"
    locked = False

    @contextmanager
    def fake_locks(roots):
        nonlocal locked
        assert tuple(Path(root).resolve() for root in roots) == (output.resolve(),)
        locked = True
        try:
            yield (output.resolve(),)
        finally:
            locked = False

    def fake_train(*_args, **_kwargs) -> Path:
        assert locked
        return output / "sion.model"

    monkeypatch.setattr(tokenizer_cli, "configure_stdio", lambda: None)
    monkeypatch.setattr(tokenizer_cli, "artifact_locks", fake_locks)
    monkeypatch.setattr(tokenizer_cli, "train_tokenizer", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sion-train-tokenizer",
            "--input",
            str(tmp_path / "raw.jsonl"),
            "--output-dir",
            str(output),
        ],
    )

    tokenizer_cli.main()

    assert not locked
