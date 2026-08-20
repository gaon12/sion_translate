"""Distribution metadata must agree with the package's single version source."""

from __future__ import annotations

from importlib.metadata import version

from sion_translate import __version__
from sion_translate.artifacts import MODEL_RELEASE_VERSION


def test_installed_distribution_uses_the_package_version() -> None:
    assert version("sion-translate") == __version__ == "1.5.0"


def test_new_model_exports_use_the_1_5_generation() -> None:
    assert MODEL_RELEASE_VERSION == "1.5"
