from argparse import Namespace
from pathlib import Path

import pytest

import inferences


def test_best_preset_uses_validated_beam_width() -> None:
    args = Namespace(
        quality="best",
        thinking=None,
        num_beams=None,
        batch_size=None,
        length_penalty=None,
        max_new_tokens=64,
    )

    beams, batch_size, length_penalty = inferences.generation_options(args)

    assert beams == 4
    assert batch_size == 8
    assert length_penalty == 1.0


def test_resolve_config_path_uses_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "custom.yaml"

    resolved = inferences.resolve_config_path(
        "artifacts/tokenizer.model",
        config_path=config_path,
    )

    assert resolved == tmp_path / "configs" / "artifacts" / "tokenizer.model"


def test_require_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="파일이어야"):
        inferences.require_file(tmp_path, value_name="용어집")


@pytest.mark.parametrize(
    ("source", "translation", "expected_reason"),
    [
        ("マジ", "안 돼, 안 돼, 안 돼, 안 돼, 안 돼", "phrase_repetition"),
        ("ああ", "아~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "character_repetition"),
        ("短い", "가" * 60, "excessive_length"),
    ],
)
def test_degeneration_reasons_detects_generation_collapse(
    source: str,
    translation: str,
    expected_reason: str,
) -> None:
    assert expected_reason in inferences.degeneration_reasons(source, translation)


def test_degeneration_reasons_accepts_normal_translation() -> None:
    assert not inferences.degeneration_reasons(
        "ひとりじゃないって信じたいのに",
        "혼자가 아니라는 걸 믿고 싶은데",
    )
