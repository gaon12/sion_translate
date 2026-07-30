"""Preservation checks for the defects chrF and digit-F1 do not see.

chrF rewards character n-gram overlap, so a translation that changes one value
loses almost nothing. ``number_preservation`` in :mod:`sion_translate.evaluation`
covers the value itself, which catches ``250mg -> 1200mg``. Three classes of
defect survive both:

**Sign.** ``±0.05mm -> 0.05mm`` keeps the digits and drops the polarity. The
number pattern does not treat ``±`` or U+2212 as part of a value, so neither
metric registers the loss, and a dropped sign inverts the meaning of a tolerance
or a delta.

**Unit.** ``0.0037mg/L -> 0.0037mg 分の 1 L`` splits a compound unit into two
different ones while every digit survives, so chrF barely moves and digit-F1 is
perfect. Units are matched against a curated equivalence table
(:data:`UNIT_CLASSES`) rather than "whatever follows a digit". An earlier version
took any one to three CJK characters after a value, which flagged
``110-482-937561 입니다 -> ...です`` and ``1,286,400원 -> 1,286,400ウォン`` as
violations because it was reading grammar and correctly translated currency as
units. The table also means a spelled-out source number (``하루 두 번`` against
``1日2回``) does not register.

The table deliberately covers measurement only: mass, volume, length,
temperature, percentage, currency and data size, where translation is one to one
and substitution is dangerous. Time and counter words are excluded, so an
insertion like ``62.5kg 기준 -> 62.5kg時間基準`` is not a unit finding. That is an
addition, and belongs to an addition-rate check rather than here; claiming this
metric catches it would be wrong.

**Script purity.** A target that leaks the source script is wrong regardless of
its score. This matters for a code-mixed source variety such as 한본어, where the
input is mixed on purpose and the output must not be.

All three are measured against the source rather than a reference, so they work
without gold translations, which is what makes them usable as an acceptance gate
on machine-translated data. Every check is a multiset comparison so repeated
units are counted rather than deduplicated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
import re
import unicodedata

from sion_translate.evaluation import multiset_f1, numeric_tokens
from sion_translate.scripts_registry import foreign_scripts, resolve_scripts

# Sign markers that carry meaning when attached to a value. ASCII "-" and "+"
# are included, but only where the number pattern would treat them as a sign,
# so a hyphen inside 110-482-937561 does not count.
SIGN_CHARACTERS = "+-±−＋－"

_SIGNED_VALUE = re.compile(rf"(?<![A-Za-z0-9_])([{re.escape(SIGN_CHARACTERS)}])\s*(?=[0-9])")

# Measurement units, mapped to a canonical class so a legitimate translation
# (원 -> ウォン) compares equal while a substitution (mg/L -> mg + L) does not.
# Surfaces are matched after NFKC and casefolding. Time and counter words are
# excluded on purpose; see the module docstring.
UNIT_CLASSES: dict[str, str] = {}


def _register(canonical: str, *surfaces: str) -> None:
    for surface in surfaces:
        if surface in UNIT_CLASSES and UNIT_CLASSES[surface] != canonical:
            raise ValueError(f"unit surface {surface!r} already maps to {UNIT_CLASSES[surface]!r}")
        UNIT_CLASSES[surface] = canonical


_register("percent", "%", "퍼센트", "パーセント")
_register("permille", "‰", "퍼밀")
_register("celsius", "℃", "°c", "섭씨", "摂氏")
_register("fahrenheit", "℉", "°f", "화씨", "華氏")
_register("microgram", "mcg", "µg", "마이크로그램")
_register("milligram", "mg", "밀리그램", "ミリグラム")
_register("gram", "g", "그램", "グラム")
_register("kilogram", "kg", "킬로그램", "キログラム")
_register("ton", "톤", "トン")
_register("millilitre", "ml", "밀리리터", "ミリリットル")
_register("litre", "리터", "リットル")
_register("cubic_centimetre", "cc", "세제곱센티미터")
_register("milligram_per_litre", "mg/l")
_register("microgram_per_litre", "mcg/l", "µg/l")
_register("gram_per_litre", "g/l")
_register("milligram_per_kilogram", "mg/kg")
_register("millimetre", "mm", "밀리미터", "ミリメートル")
_register("centimetre", "cm", "센티미터", "センチメートル")
_register("metre", "미터", "メートル")
_register("kilometre", "km", "킬로미터", "キロメートル")
_register("kilobyte", "kb", "킬로바이트", "キロバイト")
_register("megabyte", "mb", "메가바이트", "メガバイト")
_register("gigabyte", "gb", "기가바이트", "ギガバイト")
_register("terabyte", "tb", "테라바이트", "テラバイト")
_register("won", "원", "ウォン", "krw", "₩")
_register("yen", "엔", "円", "jpy", "¥")
_register("dollar", "달러", "ドル", "usd")
_register("euro", "유로", "ユーロ", "eur", "€")
_register("kilometre_per_hour", "km/h", "kph")
_register("metre_per_second", "m/s")
_register("hertz", "hz", "헤르츠", "ヘルツ")

# Longest surface first so mg/l wins over mg and 킬로그램 over 그램.
_UNIT_PATTERN = re.compile(
    "|".join(re.escape(surface) for surface in sorted(UNIT_CLASSES, key=len, reverse=True))
)

# A Latin unit must not be a fragment of a longer word: the "l" in "level" is
# not a litre, and the "mm" in "comment" is not a millimetre. Single-letter
# Latin surfaces are excluded entirely for the same reason.
_LATIN_RUN = re.compile(r"[A-Za-z]")

_SEPARATOR = re.compile(r"[\s ]+")


def _normalize(text: str) -> str:
    """NFKC so fullwidth digits, units and signs compare with halfwidth ones."""

    return _SEPARATOR.sub(" ", unicodedata.normalize("NFKC", text))


def sign_markers(text: str) -> list[str]:
    """Signs attached to a value, normalized so ± and ＋ compare with + and -.

    U+2212 MINUS SIGN and fullwidth variants fold onto ASCII; ± stays distinct
    because losing it is a different error from flipping a sign.
    """

    folded = {"−": "-", "－": "-", "＋": "+"}
    return [
        folded.get(match.group(1), match.group(1))
        for match in _SIGNED_VALUE.finditer(_normalize(text))
    ]


def unit_tokens(text: str) -> list[str]:
    """Canonical unit classes attached to a value in ``text``.

    Surfaces come from :data:`UNIT_CLASSES`, so 원 and ウォン both yield "won"
    and a correctly translated currency is not a violation.

    Two constraints keep this from reading grammar as units. A match must follow
    a digit, optionally across spaces, because a unit qualifies a value: without
    it the 원 in 원조 counts as won and the 엔 in 초창기엔 counts as yen, which is
    what the pilot corpus produced. And a Latin surface must not sit inside a
    longer Latin word, so the g in "config" is not a gram.
    """

    normalized = _normalize(text).casefold()
    found: list[str] = []
    for match in _UNIT_PATTERN.finditer(normalized):
        surface = match.group(0)
        preceding = normalized[: match.start()].rstrip(" ")
        if not preceding or not preceding[-1].isdigit():
            continue
        if _LATIN_RUN.search(surface):
            after = normalized[match.end()] if match.end() < len(normalized) else ""
            if after and _LATIN_RUN.match(after):
                continue
        found.append(UNIT_CLASSES[surface])
    return found


@dataclass
class PreservationCounts:
    """How many sentences violated each check, and the mean per-sentence F1."""

    sentences: int = 0
    sign_f1: float = 0.0
    sign_violations: int = 0
    unit_f1: float = 0.0
    unit_violations: int = 0
    number_f1: float = 0.0
    number_violations: int = 0
    script_violations: int = 0
    foreign_script_names: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_pair(
    source: str,
    hypothesis: str,
    *,
    target_scripts: Sequence[str] = (),
) -> dict[str, object]:
    """Per-sentence preservation result for one source/hypothesis pair."""

    source_signs = sign_markers(source)
    hypothesis_signs = sign_markers(hypothesis)
    source_units = unit_tokens(source)
    hypothesis_units = unit_tokens(hypothesis)
    source_numbers = numeric_tokens(source)
    hypothesis_numbers = numeric_tokens(hypothesis)
    intruding = sorted(foreign_scripts(hypothesis, target_scripts))
    return {
        "sign_f1": multiset_f1(source_signs, hypothesis_signs),
        "sign_ok": Counter(source_signs) == Counter(hypothesis_signs),
        "unit_f1": multiset_f1(source_units, hypothesis_units),
        "unit_ok": Counter(source_units) == Counter(hypothesis_units),
        "number_f1": multiset_f1(source_numbers, hypothesis_numbers),
        "number_ok": Counter(source_numbers) == Counter(hypothesis_numbers),
        "foreign_scripts": intruding,
        "script_ok": not intruding,
    }


def check_corpus(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    *,
    target_scripts: Sequence[str] = (),
    examples: int = 5,
) -> PreservationCounts:
    """Aggregate preservation over a corpus, keeping a few failing examples."""

    if len(sources) != len(hypotheses):
        raise ValueError(f"{len(sources)} sources against {len(hypotheses)} hypotheses")
    if examples < 0:
        raise ValueError("examples must be non-negative")
    # Resolve eagerly so an unknown script name fails before the loop.
    resolve_scripts(target_scripts)

    counts = PreservationCounts()
    if not sources:
        return counts

    sign_scores: list[float] = []
    unit_scores: list[float] = []
    number_scores: list[float] = []
    intruding: set[str] = set()
    for source, hypothesis in zip(sources, hypotheses, strict=True):
        result = check_pair(source, hypothesis, target_scripts=target_scripts)
        counts.sentences += 1
        sign_scores.append(float(result["sign_f1"]))
        unit_scores.append(float(result["unit_f1"]))
        number_scores.append(float(result["number_f1"]))
        failures = [
            name
            for name, ok in (
                ("sign", result["sign_ok"]),
                ("unit", result["unit_ok"]),
                ("number", result["number_ok"]),
                ("script", result["script_ok"]),
            )
            if not ok
        ]
        if not result["sign_ok"]:
            counts.sign_violations += 1
        if not result["unit_ok"]:
            counts.unit_violations += 1
        if not result["number_ok"]:
            counts.number_violations += 1
        if not result["script_ok"]:
            counts.script_violations += 1
            intruding.update(result["foreign_scripts"])  # type: ignore[arg-type]
        if failures and len(counts.examples) < examples:
            counts.examples.append(
                {
                    "failed": ",".join(failures),
                    "source": source,
                    "hypothesis": hypothesis,
                }
            )

    counts.sign_f1 = 100.0 * sum(sign_scores) / len(sign_scores)
    counts.unit_f1 = 100.0 * sum(unit_scores) / len(unit_scores)
    counts.number_f1 = 100.0 * sum(number_scores) / len(number_scores)
    counts.foreign_script_names = sorted(intruding)
    return counts


def format_report(counts: PreservationCounts, *, title: str = "preservation") -> str:
    """A compact human-readable summary."""

    if not counts.sentences:
        return f"{title}: no sentences"
    total = counts.sentences

    def share(violations: int) -> str:
        return f"{violations:,}/{total:,} ({100.0 * violations / total:5.1f}%)"

    lines = [
        f"{title}: {total:,} sentences",
        f"  number  F1 {counts.number_f1:6.2f}  violations {share(counts.number_violations)}",
        f"  sign    F1 {counts.sign_f1:6.2f}  violations {share(counts.sign_violations)}",
        f"  unit    F1 {counts.unit_f1:6.2f}  violations {share(counts.unit_violations)}",
        f"  script  violations {share(counts.script_violations)}"
        + (f"  intruding {counts.foreign_script_names}" if counts.foreign_script_names else ""),
    ]
    for example in counts.examples:
        lines.append(f"  [{example['failed']}] {example['source']!r}")
        lines.append(f"      -> {example['hypothesis']!r}")
    return "\n".join(lines)


def iter_texts(rows: Iterable[dict[str, object]], *keys: str) -> list[str]:
    """Pull the first present key from each row, as text."""

    if not keys:
        raise ValueError("at least one key is required")
    values: list[str] = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str):
                values.append(value)
                break
        else:
            raise ValueError(f"row is missing all of {keys}: {row!r}")
    return values


__all__ = [
    "SIGN_CHARACTERS",
    "UNIT_CLASSES",
    "PreservationCounts",
    "check_corpus",
    "check_pair",
    "format_report",
    "iter_texts",
    "sign_markers",
    "unit_tokens",
]
