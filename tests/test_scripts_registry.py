"""The script registry keeps tooling language-generic.

The stack follows ``data.language_pairs``, so a quality check must not decide
"Hangul is foreign" from a hardcoded pair. Callers name the scripts a side may
use; language shorthands are convenience names for script sets.
"""

from __future__ import annotations

import pytest

from sion_translate.scripts_registry import (
    LANGUAGE_SCRIPTS,
    SCRIPT_RANGES,
    foreign_scripts,
    has_foreign_script,
    is_monolingual,
    known_languages,
    known_scripts,
    resolve_scripts,
    script_of,
    scripts_in,
)


@pytest.mark.parametrize(
    ("char", "script"),
    [
        ("가", "hangul"),
        ("ㅋ", "hangul"),
        ("あ", "kana"),
        ("ア", "kana"),
        ("ｱ", "kana"),
        ("人", "han"),
        ("A", "latin"),
        ("ｗ", "latin"),
        ("Я", "cyrillic"),
        ("α", "greek"),
        ("ت", "arabic"),
        ("अ", "devanagari"),
        ("ก", "thai"),
        ("א", "hebrew"),
    ],
)
def test_script_of_classifies_each_writing_system(char: str, script: str) -> None:
    assert script_of(char) == script


@pytest.mark.parametrize("char", ["1", " ", ".", "!", "。", "、", "％", "±", "…"])
def test_shared_characters_belong_to_no_script(char: str) -> None:
    """Digits, punctuation and symbols must never count as foreign."""

    assert script_of(char) is None


def test_scripts_in_reports_every_system_present() -> None:
    assert scripts_in("오늘 スケジュール 어때") == frozenset({"hangul", "kana"})
    assert scripts_in("人間の百合") == frozenset({"han", "kana"})
    assert scripts_in("12345 !?") == frozenset()


def test_language_shorthands_resolve_to_script_sets() -> None:
    assert resolve_scripts(["ko"]) == frozenset({"hangul"})
    assert resolve_scripts(["ja"]) == frozenset({"kana", "han"})
    assert resolve_scripts(["kana", "han"]) == resolve_scripts(["ja"])
    assert resolve_scripts(["kj"]) == frozenset({"hangul", "kana", "han"})
    assert resolve_scripts(["ko", "ja"]) == frozenset({"hangul", "kana", "han"})


def test_any_resolves_to_every_script() -> None:
    assert resolve_scripts(["any"]) == frozenset(SCRIPT_RANGES)
    assert resolve_scripts(["ko", "any"]) == frozenset(SCRIPT_RANGES)


def test_empty_and_blank_names_are_ignored() -> None:
    assert resolve_scripts([]) == frozenset()
    assert resolve_scripts(["", "  "]) == frozenset()


def test_unknown_names_are_rejected_with_the_valid_options() -> None:
    with pytest.raises(ValueError, match="unknown script or language"):
        resolve_scripts(["klingon"])
    message = str(pytest.raises(ValueError, resolve_scripts, ["klingon"]).value)
    assert "hangul" in message and "ko" in message


def test_names_are_case_and_space_insensitive() -> None:
    assert resolve_scripts([" KO "]) == resolve_scripts(["ko"])


def test_foreign_scripts_reports_which_system_intruded() -> None:
    assert foreign_scripts("人間は강하다", ["ja"]) == frozenset({"hangul"})
    assert foreign_scripts("인간은 강하다", ["ko"]) == frozenset()
    assert foreign_scripts("인간은 やっぱり 강하다", ["ko"]) == frozenset({"kana"})
    # Fullwidth w laughter is Latin, which is why the report names the script
    # rather than only counting rows.
    assert foreign_scripts("強いですねｗｗ", ["ja"]) == frozenset({"latin"})
    assert foreign_scripts("強いですねｗｗ", ["ja", "latin"]) == frozenset()


def test_an_empty_allowed_set_disables_checking() -> None:
    assert foreign_scripts("人間は강하다", []) == frozenset()
    assert not has_foreign_script("anything 가나 あア", [])


def test_is_monolingual_requires_some_permitted_script() -> None:
    assert is_monolingual("인간은 강하다", ["ko"])
    assert not is_monolingual("人間は강하다", ["ja"])
    # Digits and punctuation alone carry no script, so nothing is confirmed.
    assert not is_monolingual("12345", ["ko"])
    assert is_monolingual("12345", [])


def test_every_language_shorthand_names_known_scripts() -> None:
    for language, scripts in LANGUAGE_SCRIPTS.items():
        assert scripts, language
        for script in scripts:
            assert script in SCRIPT_RANGES, (language, script)


def test_script_ranges_are_ordered_and_non_empty() -> None:
    for name, ranges in SCRIPT_RANGES.items():
        assert ranges, name
        for low, high in ranges:
            assert low <= high, (name, low, high)


def test_listing_helpers_are_sorted() -> None:
    assert known_scripts() == tuple(sorted(known_scripts()))
    assert known_languages() == tuple(sorted(known_languages()))
    assert "hangul" in known_scripts()
    assert "kj" in known_languages()
