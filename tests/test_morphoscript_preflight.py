from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sion_translate.cli.train import preflight_morphoscript_token_features
from sion_translate.config import AppConfig
from sion_translate.data.collate import load_morphoscript_token_features


VOCAB_SIZE = 5


class TinyTokenizer:
    def __len__(self) -> int:
        return VOCAB_SIZE


def valid_features() -> dict[str, np.ndarray]:
    return {
        "script": np.array([0, 1, 2, 3, 8], dtype=np.int16),
        "onset": np.array([0, 1, 2, 3, 19], dtype=np.int16),
        "vowel": np.array([0, 1, 2, 3, 21], dtype=np.int16),
        "coda": np.array([0, 1, 2, 3, 28], dtype=np.int16),
    }


def write_features(path: Path, features: dict[str, np.ndarray]) -> None:
    np.savez(path, **features)


def test_morphoscript_preflight_accepts_model_compatible_features(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "token_features.npz"
    write_features(feature_path, valid_features())
    config = AppConfig()
    config.model.experimental.morphoscript_enabled = True
    config.model.experimental.script_classes = 9
    config.data.tokenizer_features = str(feature_path)

    preflight_morphoscript_token_features(config, TinyTokenizer())  # type: ignore[arg-type]

    loaded = load_morphoscript_token_features(
        feature_path,
        vocab_size=VOCAB_SIZE,
        script_classes=9,
    )
    assert set(loaded) == {"script", "onset", "vowel", "coda"}
    assert all(values.dtype == torch.int64 for values in loaded.values())


def test_disabled_morphoscript_does_not_require_a_sidecar(tmp_path: Path) -> None:
    config = AppConfig()
    config.model.experimental.morphoscript_enabled = False
    config.data.tokenizer_features = str(tmp_path / "missing.npz")

    preflight_morphoscript_token_features(config, TinyTokenizer())  # type: ignore[arg-type]


def test_enabled_morphoscript_requires_an_existing_sidecar(tmp_path: Path) -> None:
    config = AppConfig()
    config.model.experimental.morphoscript_enabled = True
    config.data.tokenizer_features = str(tmp_path / "missing.npz")

    with pytest.raises(FileNotFoundError, match="MorphoScript is enabled"):
        preflight_morphoscript_token_features(
            config,
            TinyTokenizer(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda features: features.pop("coda"),
            "must contain exactly",
        ),
        (
            lambda features: features.update({"extra": np.zeros(VOCAB_SIZE, dtype=np.int16)}),
            "must contain exactly",
        ),
        (
            lambda features: features.update({"script": np.zeros((VOCAB_SIZE, 1), dtype=np.int16)}),
            r"script has shape .* expected",
        ),
        (
            lambda features: features.update({"onset": np.zeros(VOCAB_SIZE - 1, dtype=np.int16)}),
            r"onset has shape .* expected",
        ),
        (
            lambda features: features.update({"vowel": np.zeros(VOCAB_SIZE, dtype=np.float32)}),
            "vowel must use an integer dtype",
        ),
        (
            lambda features: features["coda"].__setitem__(0, -1),
            r"coda contains IDs outside \[0, 29\)",
        ),
        (
            lambda features: features["script"].__setitem__(0, 9),
            r"script contains IDs outside \[0, 9\)",
        ),
        (
            lambda features: features["onset"].__setitem__(0, 20),
            r"onset contains IDs outside \[0, 20\)",
        ),
        (
            lambda features: features["vowel"].__setitem__(0, 22),
            r"vowel contains IDs outside \[0, 22\)",
        ),
        (
            lambda features: features["coda"].__setitem__(0, 29),
            r"coda contains IDs outside \[0, 29\)",
        ),
    ],
)
def test_morphoscript_preflight_rejects_invalid_feature_tables(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    features = valid_features()
    mutate(features)
    feature_path = tmp_path / "token_features.npz"
    write_features(feature_path, features)

    with pytest.raises(ValueError, match=message):
        load_morphoscript_token_features(
            feature_path,
            vocab_size=VOCAB_SIZE,
            script_classes=9,
        )


def test_morphoscript_preflight_never_loads_pickled_object_arrays(
    tmp_path: Path,
) -> None:
    features = valid_features()
    features["script"] = np.array([object()] * VOCAB_SIZE, dtype=object)
    feature_path = tmp_path / "token_features.npz"
    write_features(feature_path, features)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_morphoscript_token_features(
            feature_path,
            vocab_size=VOCAB_SIZE,
            script_classes=9,
        )
