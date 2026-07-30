#!/usr/bin/env python3
"""Build a deterministic 한본어 (Korean-Japanese code-mixed) parallel corpus.

한본어 works because Korean and Japanese are both agglutinative with SOV order,
so a stem from one language takes an ending from the other and the result still
parses. Four registers occur, and they differ in what can be asserted about
their script, so every frame declares which one it produces:

    blend        morphology mixed inside one word
                 체고카요 = 최고 + かよ, やばいンデ = やばい + ~ㄴ데
    hangul_only  a Japanese clause transliterated into Hangul
                 닝겐노 유리와 튼튼데스네 = 人間の百合は丈夫ですね
    script       Hangul beside kana or kanji
                 그 スケジュール 어떻게 됐어
    kana_only    Korean vocabulary in katakana inside Japanese
                 チンチャそれな

The last three are uniform or mixed in ways a code-point check can verify;
``blend`` rows are where the interesting morphology lives and are drawn from a
hand-written lexicon rather than composed blindly, because the Japanese side is
often lexical: 대박 + い is やばい, not 大当たりい.

Every row is a triple, because the model has to read the mixture and answer in
exactly one language:

    {"kj": "엌ㅋㅋㅋ 닝겐노 유리와 튼튼데스넼ㅋㅋ",
     "ko": "헐ㅋㅋㅋ 인간의 백합은 튼튼하네요ㅋㅋ",
     "ja": "えっｗｗｗ 人間の百合は丈夫ですねｗｗ"}

The code-mixed key belongs in ``data.source_only_languages``, so mixed->Korean
and mixed->Japanese are trained and the reverse never is. The two monolingual
fields also yield a clean pair for free.

Generation is rule-based and seeded rather than sampled from a language model,
so it is reproducible and every Japanese surface is one a human put in the
lexicon. Semantic classes keep combinations meaningful and per-item and
per-frame caps keep the output from becoming the template restatement that
``audit_generated_shards.py`` rejects. The lexicon therefore bounds the corpus
size: ask for more rows than the caps allow and the builder stops early and
reports what it produced.

Korean particle selection uses hangulpy when it is installed, which covers 76
particle pairs including the 으로/로 exception. Without it a built-in fallback
handles 은/는, 이/가 and 을/를, which is all the frames here need.

Usage::

    python scripts/data/build_hanboneo.py --output data/synthetic_hanboneo.jsonl
    python scripts/data/build_hanboneo.py --output out.jsonl --report report.json
    python scripts/data/build_hanboneo.py --output out.jsonl \
        --mixed-key mixed --first-key korean --second-key japanese

Exit codes: 0 rows written, 2 nothing could be built.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import unicodedata

from hanboneo_lexicon import (
    BLENDS,
    FOOD,
    GENITIVE_HEADS,
    INTERJECTIONS,
    KOREAN_CONTENT_NOUNS,
    KOREAN_ENDINGS,
    KOREAN_FOOD_IN_KATAKANA,
    KOREAN_INTENSIFIERS_IN_KATAKANA,
    KOREAN_NOUNS_IN_KATAKANA,
    LOANWORDS,
    NOUNS,
    PERSON,
    PREDICATES,
    REACTIONS,
    Predicate,
)

from sion_translate.scripts_registry import scripts_in

try:  # Optional: gaon12/hangulpy, MIT, on PyPI as ``hangulpy``.
    from hangulpy import has_batchim as _hangulpy_has_batchim
    from hangulpy import josa as _hangulpy_josa

    HANGULPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the fallback test
    _hangulpy_has_batchim = None
    _hangulpy_josa = None
    HANGULPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hangul jamo arithmetic and particle selection.
# ---------------------------------------------------------------------------

_HANGUL_BASE = 0xAC00
_HANGUL_COUNT = 11172
_JONGSEONG_KIEUK = 24


def _syllable_offset(char: str) -> int | None:
    offset = ord(char) - _HANGUL_BASE
    return offset if 0 <= offset < _HANGUL_COUNT else None


def fuse_final_consonant(text: str, jongseong: int) -> str:
    """Attach a final consonant to the last syllable when it has none.

    Korean internet text fuses a trailing laugh consonant into the preceding
    open syllable: 네 + ㅋ -> 넼, 어 + ㅋ -> 엌.
    """

    if not 0 < jongseong < 28:
        raise ValueError("jongseong index must be in [1, 27]")
    if not text:
        return text
    offset = _syllable_offset(text[-1])
    if offset is None:
        return text
    onset, remainder = divmod(offset, 588)
    vowel, coda = divmod(remainder, 28)
    if coda:
        return text
    return text[:-1] + chr(_HANGUL_BASE + onset * 588 + vowel * 28 + jongseong)


def has_final_consonant(word: str) -> bool:
    """True when the last Hangul syllable of ``word`` carries a jongseong."""

    if _hangulpy_has_batchim is not None:
        last = word[-1:] if word else ""
        return bool(last) and bool(_hangulpy_has_batchim(last))
    if not word:
        return False
    offset = _syllable_offset(word[-1])
    return offset is not None and offset % 28 != 0


def attach_particle(word: str, pair: str) -> str:
    """Attach a Korean particle, e.g. ``attach_particle("오빠", "은/는")``.

    hangulpy handles this properly when installed; the fallback covers the three
    pairs these frames use.
    """

    if _hangulpy_josa is not None:
        return str(_hangulpy_josa(word, pair))
    with_batchim, without_batchim = pair.split("/")
    return word + (with_batchim if has_final_consonant(word) else without_batchim)


def topic_particle(word: str) -> str:
    return attach_particle(word, "은/는")[len(word) :]


def subject_particle(word: str) -> str:
    return attach_particle(word, "이/가")[len(word) :]


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One generated triple.

    ``mixing`` records which register produced the row, because two of the four
    are uniform in script and cannot be verified from code points alone.
    """

    mixed: str
    first: str  # the Korean rendering
    second: str  # the Japanese rendering
    frame: str
    items: tuple[str, ...]
    mixing: str = "script"


def _accepting(kind: str, forms: frozenset[str] | None = None) -> tuple[Predicate, ...]:
    return tuple(
        predicate
        for predicate in PREDICATES
        if kind in predicate.accepts and (forms is None or predicate.form in forms)
    )


def _polite(predicate: Predicate) -> tuple[str, str, str]:
    if predicate.form == "verb":
        return f"{predicate.hangul}마스네", f"{predicate.kana}ますね", predicate.ko_polite
    return f"{predicate.hangul}데스네", f"{predicate.kana}ですね", predicate.ko_polite


def _plain(predicate: Predicate) -> tuple[str, str, str]:
    if predicate.form == "verb":
        # Appending る would give 飲みる for a godan verb.
        return predicate.hangul_plain, predicate.kana_plain, predicate.ko_plain
    if predicate.form == "na":
        return f"{predicate.hangul}다", f"{predicate.kana}だ", predicate.ko_plain
    return predicate.hangul, predicate.kana, predicate.ko_plain


def _negative(predicate: Predicate) -> tuple[str, str, str]:
    if predicate.form == "verb":
        return f"{predicate.hangul}마센", f"{predicate.kana}ません", predicate.ko_negative
    if predicate.form == "na":
        return (
            f"{predicate.hangul}자 아리마센",
            f"{predicate.kana}じゃありません",
            predicate.ko_negative,
        )
    return (
        f"{predicate.hangul[:-1]}쿠 나이데스",
        f"{predicate.kana[:-1]}くないです",
        predicate.ko_negative,
    )


LAUGHTER: tuple[tuple[str, str], ...] = (
    ("ㅋㅋ", "ｗｗ"),
    ("ㅋㅋㅋ", "ｗｗｗ"),
    ("ㅎㅎ", "ふふ"),
    ("ㅋ", "w"),
    ("", ""),
)

_TOPIC = ("와", "は")
_SUBJECT = ("가", "が")
_GENITIVE = ("노", "の")


def _laugh(rng: random.Random, *, fuse_into: str | None = None) -> tuple[str, str, str]:
    korean, japanese = rng.choice(LAUGHTER)
    word = fuse_into or ""
    if fuse_into is not None and korean.startswith("ㅋ"):
        word = fuse_final_consonant(fuse_into, _JONGSEONG_KIEUK)
    return korean, japanese, word


def register_of(text: str) -> str:
    """Which register a code-mixed string belongs to, from the scripts present.

    Blends are written out by hand and can land in any of the three: 체고카요 is
    all Hangul, やばいンデ is all Japanese script, and 마지 스케줄 has both. The
    frame cannot assume one, so it asks.
    """

    present = scripts_in(text)
    has_hangul = "hangul" in present
    has_japanese = bool(present & {"kana", "han"})
    if has_hangul and has_japanese:
        return "script"
    if has_hangul:
        return "hangul_only"
    if has_japanese:
        return "kana_only"
    raise ValueError(f"code-mixed text has neither writing system: {text!r}")


# ---------------------------------------------------------------------------
# Frames: morphological blending
# ---------------------------------------------------------------------------


def frame_blend(rng: random.Random) -> Row:
    """A hand-verified blend: 체고카요, 카와이하다, やばいンデ."""

    blend = rng.choice(BLENDS)
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=blend.kj)
    mixed = f"{fused}{laugh_ko}"

    return Row(
        mixed=mixed,
        first=blend.ko + laugh_ko,
        second=blend.ja + laugh_ja,
        frame="blend",
        items=(blend.kj,),
        mixing=register_of(mixed),
    )


def frame_blend_after_interjection(rng: random.Random) -> Row:
    """A blend opened by an interjection, so it is not always utterance-initial."""

    blend = rng.choice(BLENDS)
    interjection = rng.choice(INTERJECTIONS)
    laugh_ko, laugh_ja, fused_lead = _laugh(rng, fuse_into=interjection.hangul)
    mixed = f"{fused_lead}{laugh_ko} {blend.kj}"

    return Row(
        mixed=mixed,
        first=f"{interjection.ko}{laugh_ko} {blend.ko}",
        second=f"{interjection.kana}{laugh_ja} {blend.ja}",
        frame="blend_after_interjection",
        items=(blend.kj, interjection.kana),
        mixing=register_of(mixed),
    )


def frame_japanese_stem_korean_ending(rng: random.Random) -> Row:
    """``카와이하다``: a Japanese predicate stem taking a Korean ending."""

    ending = rng.choice(KOREAN_ENDINGS)
    predicates = [item for item in PREDICATES if item.form in ending.accepts]
    predicate = rng.choice(predicates)
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=predicate.hangul + ending.hangul)
    korean = ending.ko_template.format(
        ko=predicate.ko_plain, ko_plain=predicate.ko_plain, ko_polite=predicate.ko_polite
    )

    return Row(
        mixed=f"{fused}{laugh_ko}",
        first=korean + laugh_ko,
        second=ending.ja_template.format(ja=predicate.kana) + laugh_ja,
        frame="japanese_stem_korean_ending",
        items=(predicate.kana, ending.katakana),
        mixing="hangul_only",
    )


def frame_blend_in_a_sentence(rng: random.Random) -> Row:
    """A blend used inside a carrier sentence, so it is not always utterance-final."""

    blend = rng.choice(BLENDS)
    noun = rng.choice(KOREAN_NOUNS_IN_KATAKANA)
    laugh_ko, laugh_ja, _ = _laugh(rng)
    mixed = f"{noun.katakana} 그거 {blend.kj}{laugh_ko}"

    return Row(
        mixed=mixed,
        first=f"{noun.ko} 그거 {blend.ko}{laugh_ko}",
        second=f"{noun.ja}それ{blend.ja}{laugh_ja}",
        frame="blend_in_a_sentence",
        items=(blend.kj, noun.katakana),
        mixing=register_of(mixed),
    )


# ---------------------------------------------------------------------------
# Frames: transliterated Japanese clauses
# ---------------------------------------------------------------------------


def frame_transliterated_clause(rng: random.Random) -> Row:
    """``닝겐노 유리와 튼튼데스네``, with a genitive subject."""

    modifier = rng.choice([noun for noun in NOUNS if noun.kind in GENITIVE_HEADS])
    allowed = GENITIVE_HEADS[modifier.kind]
    heads = [
        noun
        for noun in NOUNS
        if noun.kind in allowed
        and noun is not modifier
        and _accepting(noun.kind, frozenset({"na", "i"}))
    ]
    if not heads:
        raise LookupError(modifier.kind)
    head = rng.choice(heads)
    predicate = rng.choice(_accepting(head.kind, frozenset({"na", "i"})))
    interjection = rng.choice(INTERJECTIONS)
    lead_ko, lead_ja, fused_lead = _laugh(rng, fuse_into=interjection.hangul)
    predicate_kj, predicate_ja, predicate_ko = _polite(predicate)
    tail_ko, tail_ja, fused_tail = _laugh(rng, fuse_into=predicate_kj)
    subject_ko = f"{modifier.ko}의 {head.ko}"

    return Row(
        mixed=(
            f"{fused_lead}{lead_ko} {modifier.hangul}{_GENITIVE[0]} "
            f"{head.hangul}{_TOPIC[0]} {fused_tail}{tail_ko}"
        ),
        first=(
            f"{interjection.ko}{lead_ko} {attach_particle(subject_ko, '은/는')} "
            f"{predicate_ko}{tail_ko}"
        ),
        second=(
            f"{interjection.kana}{lead_ja} {modifier.kana}{_GENITIVE[1]}"
            f"{head.kana}{_TOPIC[1]}{predicate_ja}{tail_ja}"
        ),
        frame="transliterated_clause",
        items=(head.kana, modifier.kana, predicate.kana, interjection.kana),
        mixing="hangul_only",
    )


def frame_transliterated_plain(rng: random.Random) -> Row:
    noun = rng.choice(NOUNS)
    candidates = _accepting(noun.kind)
    if not candidates:
        raise LookupError(noun.kind)
    predicate = rng.choice(candidates)
    predicate_kj, predicate_ja, predicate_ko = _plain(predicate)
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        mixed=f"{noun.hangul}{_SUBJECT[0]} {predicate_kj}{laugh_ko}",
        first=f"{attach_particle(noun.ko, '이/가')} {predicate_ko}{laugh_ko}",
        second=f"{noun.kana}{_SUBJECT[1]}{predicate_ja}{laugh_ja}",
        frame="transliterated_plain",
        items=(noun.kana, predicate.kana),
        mixing="hangul_only",
    )


def frame_negative_clause(rng: random.Random) -> Row:
    noun = rng.choice(NOUNS)
    candidates = _accepting(noun.kind)
    if not candidates:
        raise LookupError(noun.kind)
    predicate = rng.choice(candidates)
    tail_kj, tail_ja, tail_ko = _negative(predicate)

    return Row(
        mixed=f"{noun.hangul}{_TOPIC[0]} {tail_kj}",
        first=f"{attach_particle(noun.ko, '은/는')} {tail_ko}",
        second=f"{noun.kana}{_TOPIC[1]}{tail_ja}",
        frame="negative_clause",
        items=(noun.kana, predicate.kana),
        mixing="hangul_only",
    )


def frame_particle_carried_clause(rng: random.Random) -> Row:
    """Japanese grammar in Hangul carrying Korean content words.

    ``친구노 마음와 타이헨데스네``: the nouns stay Korean while the particles and
    the copula are transliterated Japanese. This is the register the longest
    examples in the brief use, and it is composed rather than listed because
    both sides are regular here - the Korean nouns are uninflected and the
    predicate paradigm is already in the lexicon.
    """

    modifier = rng.choice([noun for noun in KOREAN_CONTENT_NOUNS if noun.kind in GENITIVE_HEADS])
    allowed = GENITIVE_HEADS[modifier.kind]
    heads = [
        noun
        for noun in KOREAN_CONTENT_NOUNS
        if noun.kind in allowed
        and noun is not modifier
        and _accepting(noun.kind, frozenset({"na", "i"}))
    ]
    if not heads:
        raise LookupError(modifier.kind)
    head = rng.choice(heads)
    predicate = rng.choice(_accepting(head.kind, frozenset({"na", "i"})))
    predicate_kj, predicate_ja, predicate_ko = _polite(predicate)
    laugh_ko, laugh_ja, fused_tail = _laugh(rng, fuse_into=predicate_kj)
    subject_ko = f"{modifier.ko}의 {head.ko}"

    return Row(
        mixed=(f"{modifier.ko}{_GENITIVE[0]} {head.ko}{_TOPIC[0]} {fused_tail}{laugh_ko}"),
        first=f"{attach_particle(subject_ko, '은/는')} {predicate_ko}{laugh_ko}",
        second=(f"{modifier.ja}{_GENITIVE[1]}{head.ja}{_TOPIC[1]}{predicate_ja}{laugh_ja}"),
        frame="particle_carried_clause",
        items=(modifier.ko, head.ko, predicate.kana),
        mixing="hangul_only",
    )


# ---------------------------------------------------------------------------
# Frames: script mixtures and katakana Korean
# ---------------------------------------------------------------------------


def frame_kana_loanword(rng: random.Random) -> Row:
    loan = rng.choice([word for word in LOANWORDS if word.pos == "noun"])
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        mixed=f"오늘 {loan.kana} 얘기 좀 하자{laugh_ko}",
        first=f"오늘 {loan.ko} 얘기 좀 하자{laugh_ko}",
        second=f"今日は{loan.kana}の話を少ししよう{laugh_ja}",
        frame="kana_loanword",
        items=(loan.kana,),
    )


def frame_kana_loanword_question(rng: random.Random) -> Row:
    loan = rng.choice([word for word in LOANWORDS if word.pos == "noun"])
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        mixed=f"그 {loan.kana} 어떻게 됐어{laugh_ko}",
        first=f"그 {loan.ko} 어떻게 됐어{laugh_ko}",
        second=f"あの{loan.kana}はどうなった{laugh_ja}",
        frame="kana_loanword_question",
        items=(loan.kana,),
    )


def frame_hangul_loanword(rng: random.Random) -> Row:
    loan = rng.choice([word for word in LOANWORDS if word.pos in {"noun", "adjective"}])
    interjection = rng.choice(INTERJECTIONS)
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=interjection.hangul)

    return Row(
        mixed=f"{fused}{laugh_ko} 이거 {loan.hangul} 아니냐",
        first=f"{interjection.ko}{laugh_ko} 이거 {loan.ko} 아니냐",
        second=f"{interjection.kana}{laugh_ja} これ{loan.kana}じゃないか",
        frame="hangul_loanword",
        items=(loan.kana, interjection.kana),
        mixing="hangul_only",
    )


def frame_hangul_phrase(rng: random.Random) -> Row:
    loan = rng.choice([word for word in LOANWORDS if word.pos == "phrase"])
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=loan.hangul)

    return Row(
        mixed=f"{fused}{laugh_ko}",
        first=f"{loan.ko}{laugh_ko}",
        second=f"{loan.kana}{laugh_ja}",
        frame="hangul_phrase",
        items=(loan.kana,),
        mixing="hangul_only",
    )


def frame_katakana_reaction(rng: random.Random) -> Row:
    """``チンチャそれな``: a Korean intensifier in katakana plus a Japanese reaction."""

    intensifier = rng.choice(KOREAN_INTENSIFIERS_IN_KATAKANA)
    # Skip pairs whose Korean glosses collide, which would read as 대박 대박이야.
    reaction = rng.choice([item for item in REACTIONS if intensifier.ko not in item.ko])
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        mixed=f"{intensifier.katakana}{reaction.kana}{laugh_ja}",
        first=f"{intensifier.ko} {reaction.ko}{laugh_ko}",
        second=f"{intensifier.ja}{reaction.kana}{laugh_ja}",
        frame="katakana_reaction",
        items=(intensifier.katakana, reaction.kana),
        mixing="kana_only",
    )


def frame_korean_person_in_japanese(rng: random.Random) -> Row:
    word = rng.choice(KOREAN_NOUNS_IN_KATAKANA)
    predicate = rng.choice(_accepting(PERSON))
    _, predicate_ja, predicate_ko = _polite(predicate)
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        mixed=f"{word.katakana}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        first=f"{attach_particle(word.ko, '은/는')} {predicate_ko}{laugh_ko}",
        second=f"{word.ja}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        frame="korean_person_in_japanese",
        items=(word.katakana, predicate.kana),
        mixing="kana_only",
    )


def frame_korean_food_in_japanese(rng: random.Random) -> Row:
    food = rng.choice(KOREAN_FOOD_IN_KATAKANA)
    predicate = rng.choice(_accepting(FOOD, frozenset({"na", "i"})))
    _, predicate_ja, predicate_ko = _polite(predicate)
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        mixed=f"{food.katakana}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        first=f"{attach_particle(food.ko, '은/는')} {predicate_ko}{laugh_ko}",
        second=f"{food.ja}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        frame="korean_food_in_japanese",
        items=(food.katakana, predicate.kana),
        mixing="kana_only",
    )


def frame_mixed_reply(rng: random.Random) -> Row:
    word = rng.choice(KOREAN_NOUNS_IN_KATAKANA)
    predicate = rng.choice(_accepting(PERSON))
    predicate_kj, predicate_ja, predicate_ko = _polite(predicate)
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=predicate_kj)

    return Row(
        mixed=f"{word.katakana} 그거 {fused}{laugh_ko}",
        first=f"{word.ko} 그거 {predicate_ko}{laugh_ko}",
        second=f"{word.ja}それ{predicate_ja}{laugh_ja}",
        frame="mixed_reply",
        items=(word.katakana, predicate.kana),
    )


FRAMES = (
    frame_blend,
    frame_blend_after_interjection,
    frame_blend_in_a_sentence,
    frame_japanese_stem_korean_ending,
    frame_particle_carried_clause,
    frame_transliterated_clause,
    frame_transliterated_plain,
    frame_negative_clause,
    frame_kana_loanword,
    frame_kana_loanword_question,
    frame_hangul_loanword,
    frame_hangul_phrase,
    frame_katakana_reaction,
    frame_korean_person_in_japanese,
    frame_korean_food_in_japanese,
    frame_mixed_reply,
)

FIRST_SCRIPTS = frozenset({"hangul"})
SECOND_SCRIPTS = frozenset({"kana", "han", "latin"})
MIXED_REGISTERS = frozenset({"blend", "hangul_only", "script", "kana_only"})


def validate_row(row: Row) -> None:
    """Reject a row that breaks the invariant the corpus exists to teach.

    A code-mixed target is exactly the failure mode a source-only language was
    introduced to prevent, so a generator bug must fail the build rather than
    reach the training set.
    """

    first_scripts = scripts_in(row.first)
    if not first_scripts <= FIRST_SCRIPTS:
        raise ValueError(
            f"Korean side is not monolingual: {row.first!r} "
            f"({sorted(first_scripts - FIRST_SCRIPTS)}, frame {row.frame})"
        )
    second_scripts = scripts_in(row.second)
    if not second_scripts <= SECOND_SCRIPTS:
        raise ValueError(
            f"Japanese side is not monolingual: {row.second!r} "
            f"({sorted(second_scripts - SECOND_SCRIPTS)}, frame {row.frame})"
        )

    mixed_scripts = scripts_in(row.mixed)
    has_hangul = "hangul" in mixed_scripts
    has_japanese = bool(mixed_scripts & {"kana", "han"})
    if row.mixing == "script":
        if not (has_hangul and has_japanese):
            raise ValueError(
                f"script mixture needs Hangul and Japanese: {row.mixed!r} (frame {row.frame})"
            )
    elif row.mixing == "hangul_only":
        if not has_hangul or has_japanese:
            raise ValueError(
                f"transliterated mixture must be all Hangul: {row.mixed!r} (frame {row.frame})"
            )
    elif row.mixing == "kana_only":
        if has_hangul or not has_japanese:
            raise ValueError(
                f"katakana mixture must be all Japanese script: {row.mixed!r} (frame {row.frame})"
            )
    else:
        raise ValueError(f"unknown mixing mode {row.mixing!r} (frame {row.frame})")

    for name, value in (("mixed", row.mixed), ("first", row.first), ("second", row.second)):
        if not value.strip():
            raise ValueError(f"empty {name} field in frame {row.frame}")
        if any(unicodedata.category(char) == "Cc" for char in value):
            raise ValueError(f"control character in {name} field, frame {row.frame}")


def build(
    *,
    max_rows: int,
    seed: int,
    max_per_item: int,
    max_per_frame: int,
    attempts_per_row: int = 200,
) -> tuple[list[Row], dict[str, object]]:
    """Generate rows under per-item and per-frame caps."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if max_per_item < 1:
        raise ValueError("max_per_item must be positive")
    if max_per_frame < 1:
        raise ValueError("max_per_frame must be positive")

    rng = random.Random(seed)
    rows: list[Row] = []
    seen: set[str] = set()
    item_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    duplicates = 0
    capped = 0

    while len(rows) < max_rows:
        placed = False
        for _ in range(attempts_per_row):
            frame = rng.choice(FRAMES)
            if frame_counts[frame.__name__] >= max_per_frame:
                continue
            row = frame(rng)
            validate_row(row)
            if row.mixed in seen:
                duplicates += 1
                continue
            if any(item_counts[item] >= max_per_item for item in row.items):
                capped += 1
                continue
            seen.add(row.mixed)
            rows.append(row)
            frame_counts[frame.__name__] += 1
            for item in row.items:
                item_counts[item] += 1
            placed = True
            break
        if not placed:
            break

    report: dict[str, object] = {
        "rows": len(rows),
        "requested_rows": max_rows,
        "seed": seed,
        "max_per_item": max_per_item,
        "max_per_frame": max_per_frame,
        "hangulpy": HANGULPY_AVAILABLE,
        "rejected_duplicates": duplicates,
        "rejected_by_caps": capped,
        "frames": dict(sorted(frame_counts.items())),
        "registers": dict(sorted(Counter(row.mixing for row in rows).items())),
        "distinct_lexical_items_used": len(item_counts),
        "most_used_lexical_items": item_counts.most_common(5),
        "lexicon_sizes": {
            "nouns": len(NOUNS),
            "predicates": len(PREDICATES),
            "interjections": len(INTERJECTIONS),
            "loanwords": len(LOANWORDS),
            "korean_content_nouns": len(KOREAN_CONTENT_NOUNS),
            "korean_endings": len(KOREAN_ENDINGS),
            "blends": len(BLENDS),
            "korean_intensifiers": len(KOREAN_INTENSIFIERS_IN_KATAKANA),
            "korean_nouns": len(KOREAN_NOUNS_IN_KATAKANA),
            "korean_food": len(KOREAN_FOOD_IN_KATAKANA),
            "reactions": len(REACTIONS),
        },
    }
    return rows, report


def write_rows(
    rows: list[Row],
    output: Path,
    *,
    mixed_key: str,
    first_key: str,
    second_key: str,
) -> None:
    """Write atomically so an interrupted build never leaves a partial shard."""

    if len({mixed_key, first_key, second_key}) != 3:
        raise ValueError("mixed, first and second keys must be distinct")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        mixed_key: row.mixed,
                        first_key: row.first,
                        second_key: row.second,
                        "synthetic": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic 한본어 parallel corpus.")
    parser.add_argument("--output", required=True, help="destination JSONL path")
    parser.add_argument("--report", help="write the build report JSON here")
    parser.add_argument("--max-rows", type=int, default=40_000, help="upper bound on rows")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--max-per-item",
        type=int,
        default=200,
        help="cap on how often one lexical entry may appear",
    )
    parser.add_argument(
        "--max-per-frame",
        type=int,
        default=6_000,
        help="cap on how many rows one sentence frame may produce",
    )
    parser.add_argument(
        "--mixed-key", default="kj", help="JSON key for the code-mixed side (default kj)"
    )
    parser.add_argument("--first-key", default="ko", help="JSON key for the Korean rendering")
    parser.add_argument("--second-key", default="ja", help="JSON key for the Japanese rendering")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows, report = build(
            max_rows=args.max_rows,
            seed=args.seed,
            max_per_item=args.max_per_item,
            max_per_frame=args.max_per_frame,
        )
    except (ValueError, LookupError) as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 2
    if not rows:
        print("build produced no rows", file=sys.stderr)
        return 2

    output = Path(args.output)
    try:
        write_rows(
            rows,
            output,
            mixed_key=args.mixed_key,
            first_key=args.first_key,
            second_key=args.second_key,
        )
    except (OSError, ValueError) as error:
        print(f"write failed: {error}", file=sys.stderr)
        return 2
    report["output"] = str(output)
    report["keys"] = [args.mixed_key, args.first_key, args.second_key]
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"{output}: {report['rows']:,} rows (requested {report['requested_rows']:,}), "
        f"{report['distinct_lexical_items_used']} lexical entries, "
        f"hangulpy={'yes' if HANGULPY_AVAILABLE else 'no'}, "
        f"registers={report['registers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
