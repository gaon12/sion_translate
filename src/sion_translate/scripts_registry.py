"""Writing-system membership tests, keyed by script rather than by language.

The translation stack is language-generic: pairs, tags and preprocessing all
follow ``data.language_pairs``. Tooling that wants to check "is this target
monolingual" must not hardcode ko/ja, so it names the scripts a language is
allowed to use and treats anything else as foreign.

Language shorthands exist as a convenience for the pairs this repository happens
to work on, and they are just names for script sets. Any other language works by
listing scripts explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

# Unicode ranges per script. Kept explicit rather than pulled from
# unicodedata.name lookups, which are far slower per character.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "hangul": (
        (0x1100, 0x11FF),  # Jamo
        (0x3130, 0x318F),  # Compatibility jamo, including ㅋㅋ and ㅎㅎ
        (0xA960, 0xA97F),  # Jamo Extended-A
        (0xAC00, 0xD7A3),  # Syllables
        (0xD7B0, 0xD7FF),  # Jamo Extended-B
    ),
    "kana": (
        (0x3040, 0x30FF),  # Hiragana and katakana
        (0x31F0, 0x31FF),  # Katakana phonetic extensions
        (0xFF66, 0xFF9D),  # Halfwidth katakana
    ),
    "han": (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2FA1F),
    ),
    "latin": (
        (0x0041, 0x005A),
        (0x0061, 0x007A),
        (0x00C0, 0x024F),
        (0xFF21, 0xFF3A),
        (0xFF41, 0xFF5A),
    ),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "devanagari": ((0x0900, 0x097F),),
    "thai": ((0x0E00, 0x0E7F),),
    "hebrew": ((0x0590, 0x05FF),),
}

# Shorthands for the scripts a language writes in. Convenience only; the
# checkers accept explicit script names for anything not listed here.
LANGUAGE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "ko": ("hangul",),
    "ja": ("kana", "han"),
    "zh": ("han",),
    "en": ("latin",),
    "de": ("latin",),
    "fr": ("latin",),
    "es": ("latin",),
    "ru": ("cyrillic",),
    "el": ("greek",),
    "ar": ("arabic",),
    "hi": ("devanagari",),
    "th": ("thai",),
    "he": ("hebrew",),
    # 한본어: a code-mixed variety, so both writing systems are expected.
    "kj": ("hangul", "kana", "han"),
}

# Writing systems that do not separate words with spaces. A space between two
# characters that both belong to such a script carries no information, so it is
# a segmentation artifact from whatever tool produced the text. Hangul is not
# listed: Korean does use inter-word spaces, and collapsing them changes meaning.
SPACELESS_SCRIPTS: frozenset[str] = frozenset({"han", "kana", "thai"})

_WHITESPACE_RUN = re.compile(r"\s+")


def known_scripts() -> tuple[str, ...]:
    return tuple(sorted(SCRIPT_RANGES))


def known_languages() -> tuple[str, ...]:
    return tuple(sorted(LANGUAGE_SCRIPTS))


def resolve_scripts(names: Iterable[str]) -> frozenset[str]:
    """Resolve script names and language shorthands to a set of script names.

    ``resolve_scripts(["ja"])`` and ``resolve_scripts(["kana", "han"])`` are the
    same thing. ``"any"`` disables checking by resolving to every script.
    """

    resolved: set[str] = set()
    for raw in names:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name == "any":
            return frozenset(SCRIPT_RANGES)
        if name in SCRIPT_RANGES:
            resolved.add(name)
        elif name in LANGUAGE_SCRIPTS:
            resolved.update(LANGUAGE_SCRIPTS[name])
        else:
            raise ValueError(
                f"unknown script or language {raw!r}; "
                f"scripts={known_scripts()} languages={known_languages()}"
            )
    return frozenset(resolved)


def script_of(char: str) -> str | None:
    """Return the script a character belongs to, or None when it has none.

    Digits, punctuation and symbols return None: they are shared across
    languages and must never count as foreign.
    """

    codepoint = ord(char)
    for name, ranges in SCRIPT_RANGES.items():
        for low, high in ranges:
            if low <= codepoint <= high:
                return name
    return None


def scripts_in(text: str) -> frozenset[str]:
    """The set of scripts present in ``text``."""

    found: set[str] = set()
    for char in text:
        name = script_of(char)
        if name is not None:
            found.add(name)
    return frozenset(found)


def foreign_scripts(text: str, allowed: Iterable[str]) -> frozenset[str]:
    """Scripts present in ``text`` that ``allowed`` does not permit.

    An empty ``allowed`` set means "do not check", so nothing is foreign.
    """

    permitted = resolve_scripts(allowed)
    if not permitted:
        return frozenset()
    return scripts_in(text) - permitted


def has_foreign_script(text: str, allowed: Iterable[str]) -> bool:
    return bool(foreign_scripts(text, allowed))


def is_spaceless(script: str | None) -> bool:
    """True when ``script`` writes words without separating spaces."""

    return script in SPACELESS_SCRIPTS


def collapse_spurious_spaces(text: str) -> str:
    """Remove whitespace that sits between two characters of a spaceless script.

    Morpheme-segmented corpora often arrive with the segmenter's spaces left in
    (``甘い 香り が 鼻先 を``). Those spaces are not content, and a model trained on
    them learns to emit them. Whitespace next to punctuation, digits or a
    space-using script is left alone, because there it may well be intentional.
    """

    if not text:
        return text

    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        match = _WHITESPACE_RUN.match(text, index)
        if match is None:
            result.append(text[index])
            index += 1
            continue
        previous = script_of(result[-1]) if result else None
        following = script_of(text[match.end()]) if match.end() < length else None
        if is_spaceless(previous) and is_spaceless(following):
            index = match.end()
            continue
        result.append(match.group())
        index = match.end()
    return "".join(result)


def spurious_space_count(text: str) -> int:
    """How many whitespace runs ``collapse_spurious_spaces`` would remove."""

    if not text:
        return 0
    removed = 0
    kept: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        match = _WHITESPACE_RUN.match(text, index)
        if match is None:
            kept.append(text[index])
            index += 1
            continue
        previous = script_of(kept[-1]) if kept else None
        following = script_of(text[match.end()]) if match.end() < length else None
        if is_spaceless(previous) and is_spaceless(following):
            removed += 1
        else:
            kept.append(match.group())
        index = match.end()
    return removed


def is_monolingual(text: str, allowed: Sequence[str]) -> bool:
    """True when ``text`` uses only permitted scripts and at least one of them."""

    permitted = resolve_scripts(allowed)
    if not permitted:
        return True
    present = scripts_in(text)
    return bool(present) and present <= permitted


__all__ = [
    "LANGUAGE_SCRIPTS",
    "SCRIPT_RANGES",
    "SPACELESS_SCRIPTS",
    "collapse_spurious_spaces",
    "foreign_scripts",
    "has_foreign_script",
    "is_monolingual",
    "is_spaceless",
    "known_languages",
    "known_scripts",
    "resolve_scripts",
    "script_of",
    "scripts_in",
    "spurious_space_count",
]
