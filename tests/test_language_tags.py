from __future__ import annotations

import pytest

from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_tag,
    canonicalize_language_tags,
    is_well_formed_language_tag,
    parse_language_tag,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("KO", "ko"),
        ("pt-br", "pt-BR"),
        ("ZH-hant-tw", "zh-Hant-TW"),
        ("sr-latn-rs", "sr-Latn-RS"),
        ("de-CH-1901", "de-CH-1901"),
        ("en-u-ca-gregory", "en-u-ca-gregory"),
        ("x-Sion-Mixed", "x-sion-mixed"),
        ("en-GB-OED", "en-GB-oed"),
    ],
)
def test_language_tags_use_one_canonical_identity(raw: str, canonical: str) -> None:
    assert canonicalize_language_tag(raw) == canonical
    assert canonicalize_language_tag(canonical) == canonical


def test_parsed_language_exposes_script_region_and_private_use() -> None:
    tag = parse_language_tag("zh-cmn-Hans-CN-x-sion")
    assert tag.canonical == "zh-cmn-Hans-CN-x-sion"
    assert tag.language == "zh"
    assert tag.extlangs == ("cmn",)
    assert tag.script == "Hans"
    assert tag.region == "CN"
    assert tag.private_use == ("sion",)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " en",
        "en_US",
        "en--US",
        "e",
        "en-a",
        "en-x",
        "en-u-ca-u-nu-latn",
        "de-1901-1901",
        "日本語",
    ],
)
def test_malformed_language_tags_are_rejected(raw: str) -> None:
    with pytest.raises(LanguageTagError, match="BCP 47"):
        canonicalize_language_tag(raw)
    assert not is_well_formed_language_tag(raw)


def test_language_lists_reject_duplicates_after_canonicalization() -> None:
    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_tags(["pt-BR", "pt-br"])

    assert canonicalize_language_tags(["pt-BR", "pt-br"], reject_duplicates=False) == ("pt-BR",)
