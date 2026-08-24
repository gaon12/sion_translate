from __future__ import annotations

import pytest

import sion_translate.language_tags as language_tags
from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_pair,
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
        ("en-b-ccc-bbb-a-aaa", "en-a-aaa-b-ccc-bbb"),
        ("x-Sion-Mixed", "x-sion-mixed"),
        ("en-GB-OED", "en-GB-oxendict"),
        ("en-bu", "en-MM"),
        ("de-DD", "de-DE"),
        ("fr-fx", "fr-FR"),
        ("zh-cmn", "cmn"),
        ("ja-latn-hepburn-heploc", "ja-Latn-hepburn-alalc97"),
        (
            "zh-cmn-hans-BU-u-ca-gregory-x-Sion",
            "cmn-Hans-MM-u-ca-gregory-x-sion",
        ),
    ],
)
def test_language_tags_use_one_canonical_identity(raw: str, canonical: str) -> None:
    assert canonicalize_language_tag(raw) == canonical
    assert canonicalize_language_tag(canonical) == canonical


def test_parsed_language_exposes_script_region_and_private_use() -> None:
    tag = parse_language_tag("zh-cmn-Hans-CN-x-sion")
    assert tag.canonical == "cmn-Hans-CN-x-sion"
    assert tag.language == "cmn"
    assert tag.extlangs == ()
    assert tag.script == "Hans"
    assert tag.region == "CN"
    assert tag.private_use == ("sion",)


@pytest.mark.parametrize(
    ("raw", "preferred"),
    [
        ("IW", "he"),
        ("iw-hebr-IL", "he-Hebr-IL"),
        ("in-ID", "id-ID"),
        ("ji", "yi"),
        ("jw-Latn", "jv-Latn"),
        ("mo-Cyrl-MD", "ro-Cyrl-MD"),
        ("AJP", "apc"),
        ("i-KLINGON", "tlh"),
        ("en-gb-OED", "en-GB-oxendict"),
    ],
)
def test_deprecated_language_identities_use_iana_preferred_values(
    raw: str,
    preferred: str,
) -> None:
    parsed = parse_language_tag(raw)

    assert parsed.canonical == preferred
    assert parsed.language == preferred.split("-", 1)[0]
    assert canonicalize_language_tag(preferred) == preferred


def test_registry_canonicalization_is_limited_to_exact_language_identities() -> None:
    # ``i-default`` has no registry Preferred-Value and private-use subtags are
    # project-owned, so neither is rewritten by the explicit registry mappings.
    assert canonicalize_language_tag("I-DEFAULT") == "i-default"
    assert canonicalize_language_tag("x-IW-IN-I-KLINGON") == "x-iw-in-i-klingon"
    assert canonicalize_language_tag("en-cmn") == "en-cmn"


def test_preferred_value_alias_chains_reach_a_fixed_point_and_reject_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        language_tags,
        "_REGION_PREFERRED_VALUES",
        {"AA": "BB", "BB": "CC"},
    )
    assert canonicalize_language_tag("en-AA") == "en-CC"

    monkeypatch.setattr(
        language_tags,
        "_REGION_PREFERRED_VALUES",
        {"AA": "BB", "BB": "AA"},
    )
    with pytest.raises(RuntimeError, match=r"cyclic.*AA -> BB -> AA"):
        canonicalize_language_tag("en-AA")


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

    with pytest.raises(LanguageTagError, match="duplicate language identities"):
        canonicalize_language_tags(["iw", "HE"])

    assert canonicalize_language_tags(["in", "id"], reject_duplicates=False) == ("id",)


def test_language_pair_requires_two_distinct_canonical_identities() -> None:
    assert canonicalize_language_pair(["PT-br", "zh-hant"]) == ("pt-BR", "zh-Hant")
    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_pair(["PT-br", "pt-BR"])
    with pytest.raises(LanguageTagError, match="two-item language sequence"):
        canonicalize_language_pair(["en"])

    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_pair(["iw", "he"])
    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_pair(["i-klingon", "tlh"])
    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_pair(["en-BU", "en-MM"])
    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_pair(["zh-cmn", "cmn"])


def test_extension_order_variants_collapse_to_one_language_identity() -> None:
    with pytest.raises(LanguageTagError, match="after canonicalization"):
        canonicalize_language_pair(["en-b-ccc-bbb-a-aaa", "en-a-aaa-b-ccc-bbb"])
