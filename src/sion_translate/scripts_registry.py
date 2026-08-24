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
import unicodedata

from sion_translate.language_tags import parse_language_tag

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
    # Hanboneo is a code-mixed variety, so both writing systems are expected.
    "kj": ("hangul", "kana", "han"),
    # Regional varieties use the same writing systems as their standard forms.
    # script checks are identical; they exist as separate tags only so that
    # ``data.source_only_languages`` can stop the model learning to *produce*
    # dialect from a standard prompt. The region is row metadata, not a tag.
    "kd": ("hangul",),
    "jd": ("kana", "han"),
}

# Writing systems that do not separate words with spaces. A space between two
# characters that both belong to such a script carries no information, so it is
# a segmentation artifact from whatever tool produced the text. Hangul is not
# listed: Korean does use inter-word spaces, and collapsing them changes meaning.
SPACELESS_SCRIPTS: frozenset[str] = frozenset({"han", "kana", "thai"})

# ISO 15924 script subtags that can be checked with the Unicode ranges above.
# This is script metadata rather than a closed language list: an arbitrary tag
# such as ``az-Arab`` or ``sr-Latn`` works without adding its primary language.
SCRIPT_SUBTAG_SCRIPTS: dict[str, tuple[str, ...]] = {
    "Arab": ("arabic",),
    "Cyrl": ("cyrillic",),
    "Deva": ("devanagari",),
    "Grek": ("greek",),
    "Hang": ("hangul",),
    "Hani": ("han",),
    "Hans": ("han",),
    "Hant": ("han",),
    "Hebr": ("hebrew",),
    "Hira": ("kana",),
    "Jpan": ("kana", "han"),
    "Kana": ("kana",),
    "Kore": ("hangul", "han"),
    "Latn": ("latin",),
    "Thai": ("thai",),
}

# Character-tokenized metrics and substring glossary matching are properties
# of writing systems, not of one hard-coded language pair. Hangul uses spaces,
# but its productive particles attach directly to terms, so a Python ``\w``
# boundary would still miss legitimate glossary occurrences.
CHARACTER_TOKENIZATION_SCRIPTS: frozenset[str] = SPACELESS_SCRIPTS | {"hangul"}
SUBSTRING_MATCH_SCRIPTS: frozenset[str] = SPACELESS_SCRIPTS | {"hangul"}

_WHITESPACE_RUN = re.compile(r"\s+")
_SCRIPT_POLICY_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
_UNSAFE_GENERIC_SCRIPT_NAMES = frozenset(
    {
        "alpha",
        "character",
        "digit",
        "letter",
        "mark",
        "number",
        "sign",
        "small",
        "capital",
        "symbol",
    }
)


def known_scripts() -> tuple[str, ...]:
    return tuple(sorted(SCRIPT_RANGES))


def canonicalize_script_policy_name(value: object) -> str:
    """Return a safe script-policy name, including an open Unicode-name script.

    Built-in names use fast range checks. Other names, such as ``bengali`` or
    ``georgian``, match the script word in Unicode character names. Rejecting
    generic Unicode words prevents a rule such as ``letter`` from matching
    unrelated writing systems.
    """

    if not isinstance(value, str) or value != value.strip():
        raise ValueError("script policy names must be non-empty strings without outer spaces")
    normalized = value.casefold().replace("-", "_")
    if _SCRIPT_POLICY_NAME.fullmatch(normalized) is None:
        raise ValueError(
            "script policy names must contain 2-32 lowercase ASCII letters, digits, or underscores"
        )
    words = frozenset(normalized.split("_"))
    if words & _UNSAFE_GENERIC_SCRIPT_NAMES:
        raise ValueError(f"script policy name is too generic to be safe: {value!r}")
    return normalized


def script_letter_count(text: str, script: str) -> int:
    """Count actual letters in one built-in or Unicode-name writing system.

    Modifier letters and punctuation do not satisfy a minimum. This excludes
    Japanese prolonged-sound marks and Arabic punctuation while retaining
    syllabic letters, ideographs, and ordinary alphabetic letters.
    """

    normalized = canonicalize_script_policy_name(script)
    name_words = tuple(word.upper() for word in normalized.split("_") if word)
    total = 0
    for character in text:
        if unicodedata.category(character) not in {"Lu", "Ll", "Lt", "Lo"}:
            continue
        if normalized in SCRIPT_RANGES and script_of(character) == normalized:
            total += 1
            continue
        unicode_name = unicodedata.name(character, "")
        if name_words and all(word in unicode_name for word in name_words):
            total += 1
    return total


def known_languages() -> tuple[str, ...]:
    return tuple(sorted(LANGUAGE_SCRIPTS))


def primary_language(language: str) -> str:
    """Return the canonical primary subtag for any well-formed BCP 47 tag."""

    return parse_language_tag(language, field="language").language


def scripts_for_language(language: str) -> frozenset[str] | None:
    """Resolve a BCP 47 tag to checkable scripts, or ``None`` if unknown.

    Explicit script subtags take precedence over convenience defaults. This
    keeps the operation open-ended: new primary languages work immediately when
    their tag carries a supported ISO 15924 script subtag.
    """

    parsed = parse_language_tag(language, field="language")
    if parsed.script is not None:
        scripts = SCRIPT_SUBTAG_SCRIPTS.get(parsed.script)
        return frozenset(scripts) if scripts is not None else None
    scripts = LANGUAGE_SCRIPTS.get(parsed.canonical.casefold())
    if scripts is None:
        scripts = LANGUAGE_SCRIPTS.get(parsed.language)
    return frozenset(scripts) if scripts is not None else None


def uses_character_tokenization(language: str) -> bool:
    """Whether a tag's writing system should use character-level BLEU."""

    scripts = scripts_for_language(language)
    return bool(scripts and scripts & CHARACTER_TOKENIZATION_SCRIPTS)


def uses_substring_term_matching(language: str) -> bool:
    """Whether glossary terms may attach without a Unicode word boundary."""

    scripts = scripts_for_language(language)
    return bool(scripts and scripts & SUBSTRING_MATCH_SCRIPTS)


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
            scripts = scripts_for_language(str(raw))
            if scripts is None:
                raise ValueError(
                    f"unknown script or language {raw!r}; "
                    f"scripts={known_scripts()} languages={known_languages()}"
                )
            resolved.update(scripts)
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
    "CHARACTER_TOKENIZATION_SCRIPTS",
    "LANGUAGE_SCRIPTS",
    "SCRIPT_RANGES",
    "SCRIPT_SUBTAG_SCRIPTS",
    "SPACELESS_SCRIPTS",
    "SUBSTRING_MATCH_SCRIPTS",
    "canonicalize_script_policy_name",
    "collapse_spurious_spaces",
    "foreign_scripts",
    "has_foreign_script",
    "is_monolingual",
    "is_spaceless",
    "known_languages",
    "known_scripts",
    "primary_language",
    "resolve_scripts",
    "script_of",
    "script_letter_count",
    "scripts_for_language",
    "scripts_in",
    "spurious_space_count",
    "uses_character_tokenization",
    "uses_substring_term_matching",
]
