"""Sentence-ending rewrites that turn a standard sentence into a regional variety.

The diagnostic scores 방언 at chrF 9.89, the worst of every category, against a
corpus that holds 3,409 dialect rows. That is a coverage problem, not a modelling
one, and the fix is more dialect data. Regional endings are systematic enough to
generate deterministically, the same way ``hanboneo_lexicon`` generates 한본어.

Scope is deliberately narrow, because a wrong rule teaches the model wrong
grammar and is worse than no rule:

* **Endings only.** A sentence-final ending is a closed class and rewriting it
  cannot change who did what. Free lexical substitution can, so the vocabulary
  tables hold only words with no competing reading.
* **Longest suffix wins**, so ``했습니다`` is matched before ``습니다``.
* **A rule may refuse.** ``blocked`` lists endings that look like the target but
  are not: Japanese ``まだ`` is not a copula, and ``少ない`` is an adjective, not a
  verb negative. Without those guards the rewrite produces ``まや`` and ``少へん``.
* **No match means no row.** A sentence whose ending no rule covers is skipped
  rather than emitted unchanged, so the corpus contains only real transformations.

Korean interrogatives get special handling for the 동남 varieties, where a
yes/no question ends in ``-나`` and a wh-question in ``-노``: 밥 먹었나? against
뭐 먹었노?. That contrast is the single most recognisable feature of the variety,
so it is modelled rather than flattened.

Every table is keyed by language, and a language with no table simply produces
nothing. Nothing here is imported by the translation stack itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class DialectProfile:
    """One regional variety: how its sentence endings differ from the standard."""

    code: str
    label: str
    language: str
    # (standard ending, dialect ending), tried longest-first regardless of order.
    endings: tuple[tuple[str, str], ...]
    # Applied instead of ``endings`` when the sentence contains an interrogative.
    wh_endings: tuple[tuple[str, str], ...] = ()
    # Whole-word substitutions. Only words with no competing reading.
    vocabulary: tuple[tuple[str, str], ...] = ()
    # Endings that must never be rewritten even though a rule appears to match.
    blocked: frozenset[str] = field(default_factory=frozenset)

    def sorted_endings(self, *, interrogative: bool) -> tuple[tuple[str, str], ...]:
        table = self.wh_endings if interrogative and self.wh_endings else self.endings
        return tuple(sorted(table, key=lambda pair: len(pair[0]), reverse=True))


# Korean interrogative words. Their presence selects the ``-노`` ending in the
# 동남 varieties and is otherwise unused.
KOREAN_INTERROGATIVES: tuple[str, ...] = (
    "누가",
    "누구",
    "무엇",
    "무슨",
    "뭐",
    "뭘",
    "어디",
    "언제",
    "why",
    "왜",
    "어떻게",
    "어떤",
    "얼마",
    "몇",
    "어느",
)

# Korean endings that look like the interrogative ``-니`` but are not: ``늘리다니``
# is an exclamative, and rewriting it produces ``늘리다냐``.
_NOT_KOREAN_QUESTION = frozenset(
    {
        "다니",
        "라니",
        "더니",
        "느니",
        "자니",
        "냐니",
        "으니",
        "니니",
        "이니",
    }
)

# Endings shared by the 동남 varieties, which differ only in a few forms.
_GYEONGSANG_COMMON: tuple[tuple[str, str], ...] = (
    ("했습니다", "했습니더"),
    ("습니다", "습니더"),
    ("합니다", "합니더"),
    ("입니다", "입니더"),
    ("됩니다", "됩니더"),
    ("드립니다", "드립니더"),
    ("이에요", "이라예"),
    ("예요", "라예"),
    ("거예요", "기라예"),
    ("있어요", "있어예"),
    ("했어요", "했어예"),
    ("겠어요", "겠어예"),
    ("어요", "어예"),
    ("아요", "아예"),
    ("해요", "해예"),
    ("지요", "지예"),
    ("거든요", "거든예"),
    ("잖아요", "잖아예"),
    ("같아요", "같아예"),
    ("주세요", "주이소"),
    ("하세요", "하이소"),
    ("세요", "이소"),
    ("십시오", "이소"),
    ("거야", "기다"),
    ("거지", "기지"),
    ("이야", "이다"),
    ("아니야", "아이다"),
    ("있어", "있다"),
    ("같아", "같다"),
    ("보자", "보자"),
    ("요", "예"),
)

_GYEONGSANG_WH: tuple[tuple[str, str], ...] = (
    ("했습니까", "했습니꺼"),
    ("습니까", "습니꺼"),
    ("입니까", "입니꺼"),
    ("나요", "노"),
    ("가요", "가노"),
    ("을까요", "을꼬"),
    ("었니", "었노"),
    ("니", "노"),
    ("어", "노"),
    ("야", "고"),
)

_GYEONGSANG_YESNO: tuple[tuple[str, str], ...] = (
    ("했습니까", "했습니꺼"),
    ("습니까", "습니꺼"),
    ("입니까", "입니꺼"),
    ("나요", "나예"),
    ("었니", "었나"),
    ("니", "나"),
)

KO_PROFILES: tuple[DialectProfile, ...] = (
    DialectProfile(
        code="gyeongsang",
        label="경상 방언 (동남)",
        language="ko",
        endings=_GYEONGSANG_COMMON + _GYEONGSANG_YESNO,
        wh_endings=_GYEONGSANG_WH + _GYEONGSANG_COMMON,
        vocabulary=(("정말", "진짜"),),
        blocked=_NOT_KOREAN_QUESTION,
    ),
    DialectProfile(
        code="jeonnam",
        label="전남 방언 (서남)",
        language="ko",
        endings=(
            ("했습니다", "했습니다"),
            ("이에요", "이라우"),
            ("예요", "라우"),
            ("거예요", "것이라우"),
            ("있어요", "있어라우"),
            ("했어요", "했어라우"),
            ("어요", "어라우"),
            ("아요", "아라우"),
            ("해요", "해라우"),
            ("지요", "지라우"),
            ("거든요", "거든이라우"),
            ("주세요", "주시오"),
            ("하세요", "하시오"),
            ("세요", "시오"),
            ("거야", "것이여"),
            ("이야", "이여"),
            ("아니야", "아니여"),
            ("있어", "있어야"),
            ("같아", "같어야"),
            ("는데", "는디"),
            ("니", "냐"),
            ("가요", "가라우"),
            ("보세요", "보시오"),
        ),
        vocabulary=(("그렇지", "그라제"), ("정말", "참말로")),
        blocked=_NOT_KOREAN_QUESTION,
    ),
    DialectProfile(
        code="jeonbuk",
        label="전북 방언 (서남)",
        language="ko",
        endings=(
            ("이에요", "이지라"),
            ("예요", "지라"),
            ("있어요", "있지라"),
            ("했어요", "했지라"),
            ("어요", "어라"),
            ("아요", "아라"),
            ("해요", "해라"),
            ("지요", "지라"),
            ("하세요", "하시오"),
            ("세요", "시오"),
            ("거야", "것이여"),
            ("이야", "이여"),
            ("는데", "는디"),
            ("니", "냐"),
            ("가요", "가라"),
        ),
        vocabulary=(("그렇지", "그라지"),),
        blocked=_NOT_KOREAN_QUESTION,
    ),
    DialectProfile(
        code="chungcheong",
        label="충청 방언",
        language="ko",
        endings=(
            ("이에요", "이유"),
            ("예요", "유"),
            ("거예요", "거유"),
            ("있어요", "있어유"),
            ("했어요", "했어유"),
            ("겠어요", "겄어유"),
            ("어요", "어유"),
            ("아요", "아유"),
            ("해요", "해유"),
            ("지요", "지유"),
            ("거든요", "거든유"),
            ("잖아요", "잖어유"),
            ("주세요", "주세유"),
            ("하세요", "하세유"),
            ("세요", "세유"),
            ("거야", "거여"),
            ("이야", "이여"),
            ("아니야", "아니여"),
            ("겠어", "겄어"),
            ("는데", "는디"),
            ("해", "혀"),
            ("요", "유"),
        ),
        vocabulary=(("그렇지", "그려"),),
    ),
    DialectProfile(
        code="gangwon",
        label="강원 방언 (영동)",
        language="ko",
        endings=(
            ("했습니다", "했슴다"),
            ("습니다", "슴다"),
            ("합니다", "함다"),
            ("있었어요", "있었드래요"),
            ("했어요", "했드래요"),
            ("었어요", "었드래요"),
            ("어요", "어유"),
            ("아요", "아유"),
            ("거야", "거야"),
            ("니", "나"),
            ("요", "유"),
        ),
        vocabulary=(),
        blocked=_NOT_KOREAN_QUESTION,
    ),
    DialectProfile(
        code="jeju",
        label="제주 방언",
        language="ko",
        endings=(
            ("했습니까", "했수과"),
            ("습니까", "수과"),
            ("입니까", "이우꽈"),
            ("했습니다", "했수다"),
            ("습니다", "수다"),
            ("합니다", "합니다"),
            ("입니다", "이우다"),
            ("있어요", "있수다"),
            ("했어요", "했수다"),
            ("어요", "수다"),
            ("아요", "수다"),
            ("주세요", "주십서"),
            ("하세요", "하십서"),
            ("세요", "십서"),
        ),
        vocabulary=(("어머니", "어멍"), ("아버지", "아방"), ("빨리", "혼저")),
    ),
    DialectProfile(
        code="pyeongan",
        label="평안 방언 (서북)",
        language="ko",
        endings=(
            ("했습니까", "했습네까"),
            ("습니까", "습네까"),
            ("입니까", "입네까"),
            ("했습니다", "했습네다"),
            ("습니다", "습네다"),
            ("합니다", "합네다"),
            ("입니다", "입네다"),
            ("됩니다", "됩네다"),
            ("있어요", "있습네다"),
            ("어요", "습네다"),
        ),
        vocabulary=(),
    ),
    DialectProfile(
        code="hamgyeong",
        label="함경 방언 (동북)",
        language="ko",
        endings=(
            ("했습니까", "했슴둥"),
            ("습니까", "슴둥"),
            ("입니까", "임둥"),
            ("했습니다", "했슴메"),
            ("습니다", "슴메"),
            ("합니다", "함메"),
            ("입니다", "임메"),
            ("있어요", "있슴메"),
            ("어요", "슴메"),
        ),
        vocabulary=(),
    ),
)

# Japanese copula endings that are not a copula at all. Rewriting these is what
# turns まだ into まや.
_NOT_COPULA_DA = frozenset(
    {
        "まだ",
        "ただ",
        "からだ",
        "あいだ",
        "むだ",
        "はだ",
        "えだ",
        "ひだ",
        "しだ",
        "みだ",
        "そだ",
        "いまだ",
    }
)

# Contexts where ない is not a verb negative, so the ん / へん forms do not apply.
# Two kinds: adjectives that merely end in ない, and the existence negative after
# a particle. 一つもない becomes 一つもん and 写真がない becomes 写真がん without
# these, both of which are not Japanese.
_NOT_NEGATIVE_NAI = frozenset(
    {
        "がない",
        "もない",
        "はない",
        "にない",
        "とない",
        "でない",
        "じゃない",
        "ではない",
        "ゃない",
        "少ない",
        "危ない",
        "汚ない",
        "切ない",
        "幼ない",
        "すくない",
        "あぶない",
        "きたない",
        "せつない",
        "はかない",
        "おさない",
        "もったいない",
        "情けない",
        "なさけない",
    }
)

JA_PROFILES: tuple[DialectProfile, ...] = (
    DialectProfile(
        code="kansai",
        label="関西弁 (大阪)",
        language="ja",
        endings=(
            ("ていません", "とりまへん"),
            ("ていない", "とらへん"),
            ("ではない", "やない"),
            ("じゃない", "やない"),
            ("でしょう", "やろ"),
            ("だろう", "やろ"),
            ("なのだ", "やねん"),
            ("なんだ", "やねん"),
            ("たんだ", "たんや"),
            ("ています", "とります"),
            ("ている", "とる"),
            ("だから", "やから"),
            ("ですから", "やから"),
            ("ません", "まへん"),
            ("ましょう", "まひょ"),
            ("ですね", "ですなあ"),
            ("だめだ", "あかん"),
            ("だめ", "あかん"),
            ("だよ", "やで"),
            ("だね", "やね"),
            ("だ", "や"),
            ("ない", "へん"),
        ),
        vocabulary=(("とても", "めっちゃ"), ("本当", "ほんま")),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="kyoto",
        label="京都弁",
        language="ja",
        endings=(
            ("ていません", "てまへん"),
            ("ていない", "てへん"),
            ("ています", "てはります"),
            ("ている", "てはる"),
            ("でしょう", "やろ"),
            ("だろう", "やろ"),
            ("ですね", "どすなあ"),
            ("です", "どす"),
            ("ません", "まへん"),
            ("だから", "やから"),
            ("だよ", "やで"),
            ("だめだ", "あかん"),
            ("だめ", "あかん"),
            ("だ", "や"),
            ("ない", "へん"),
        ),
        vocabulary=(("とても", "えらい"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="hakata",
        label="博多弁 (福岡)",
        language="ja",
        endings=(
            ("ていません", "とりまっせん"),
            ("ていない", "とらん"),
            ("ではない", "やない"),
            ("じゃない", "やない"),
            ("でしょう", "やろ"),
            ("だろう", "やろ"),
            ("ています", "とります"),
            ("ている", "とる"),
            ("だから", "やけん"),
            ("ですから", "やけん"),
            ("ません", "まっせん"),
            ("だよ", "ばい"),
            ("だね", "ばいね"),
            ("だ", "ばい"),
            ("ない", "ん"),
        ),
        vocabulary=(("とても", "ばり"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="kumamoto",
        label="熊本弁",
        language="ja",
        endings=(
            ("ていません", "とりません"),
            ("ていない", "とらん"),
            ("でしょう", "だろ"),
            ("だろう", "だろ"),
            ("ています", "とります"),
            ("ている", "とる"),
            ("だから", "だけん"),
            ("ですから", "ですけん"),
            ("だよ", "だけん"),
            ("だ", "だ"),
            ("ない", "ん"),
        ),
        vocabulary=(),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="kagoshima",
        label="鹿児島弁 (薩摩)",
        language="ja",
        endings=(
            ("ていません", "とりません"),
            ("ていない", "とらん"),
            ("でしょう", "じゃろ"),
            ("だろう", "じゃろ"),
            ("ています", "とります"),
            ("ている", "とる"),
            ("だから", "じゃっで"),
            ("だよ", "じゃ"),
            ("だね", "じゃね"),
            ("だ", "じゃ"),
            ("ない", "ん"),
        ),
        vocabulary=(("とても", "わっぜ"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="hiroshima",
        label="広島弁",
        language="ja",
        endings=(
            ("ていません", "とりません"),
            ("ていない", "とらん"),
            ("でしょう", "じゃろう"),
            ("だろう", "じゃろう"),
            ("ています", "とります"),
            ("ている", "とる"),
            ("だから", "じゃけぇ"),
            ("ですから", "じゃけぇ"),
            ("だよ", "じゃ"),
            ("だね", "じゃのう"),
            ("だ", "じゃ"),
            ("ない", "ん"),
        ),
        vocabulary=(("とても", "ぶち"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="nagoya",
        label="名古屋弁",
        language="ja",
        endings=(
            ("ていません", "とりません"),
            ("ていない", "とらん"),
            ("でしょう", "だら"),
            ("だろう", "だら"),
            ("ています", "とります"),
            ("ている", "とる"),
            ("だから", "だから"),
            ("だよ", "だがや"),
            ("だね", "だがね"),
            ("だ", "だがや"),
            ("ない", "ん"),
        ),
        vocabulary=(("とても", "でら"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="tohoku",
        label="東北弁 (仙台)",
        language="ja",
        endings=(
            ("ていません", "でません"),
            ("ていない", "でねぇ"),
            ("でしょう", "だべ"),
            ("だろう", "だべ"),
            ("ています", "でます"),
            ("ている", "でる"),
            ("だよ", "だべ"),
            ("だね", "だべ"),
            ("ない", "ねぇ"),
        ),
        vocabulary=(),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="tsugaru",
        label="津軽弁 (青森)",
        language="ja",
        endings=(
            ("ていません", "でません"),
            ("ていない", "でね"),
            ("でしょう", "だべさ"),
            ("だろう", "だべ"),
            ("ています", "でます"),
            ("ている", "でる"),
            ("だよ", "だでば"),
            ("だね", "だべ"),
            ("ない", "ね"),
        ),
        vocabulary=(),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="hokkaido",
        label="北海道方言",
        language="ja",
        endings=(
            ("でしょう", "でしょ"),
            ("だろう", "だべ"),
            ("だよね", "だべさ"),
            ("だよ", "だべ"),
            ("ている", "てる"),
            ("ない", "ないっしょ"),
        ),
        vocabulary=(("とても", "なまら"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="okinawa",
        label="沖縄方言",
        language="ja",
        endings=(
            ("ていない", "とーらん"),
            ("でしょう", "やさ"),
            ("だろう", "やさ"),
            ("ている", "とーん"),
            ("だよ", "さー"),
            ("だね", "さー"),
            ("だ", "やっさ"),
        ),
        vocabulary=(("とても", "でーじ"),),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="tosa",
        label="土佐弁 (高知)",
        language="ja",
        endings=(
            ("ていません", "ちょりません"),
            ("ていない", "ちょらん"),
            ("でしょう", "じゃろ"),
            ("だろう", "じゃろ"),
            ("ています", "ちゅうます"),
            ("ている", "ちゅう"),
            ("だから", "じゃき"),
            ("だよ", "ぜ"),
            ("だ", "じゃ"),
            ("ない", "ん"),
        ),
        vocabulary=(),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="shizuoka",
        label="静岡方言",
        language="ja",
        endings=(
            ("でしょう", "ずら"),
            ("だろう", "だら"),
            ("ています", "てます"),
            ("ている", "てる"),
            ("だよ", "だに"),
            ("だね", "だねぇ"),
        ),
        vocabulary=(),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
    DialectProfile(
        code="niigata",
        label="新潟方言",
        language="ja",
        endings=(
            ("ていない", "てねぇ"),
            ("でしょう", "だろ"),
            ("だろう", "だろ"),
            ("だった", "だっけ"),
            ("ている", "てる"),
            ("だよ", "だでば"),
            ("ない", "ねぇ"),
        ),
        vocabulary=(),
        blocked=_NOT_COPULA_DA | _NOT_NEGATIVE_NAI,
    ),
)

PROFILES: dict[str, tuple[DialectProfile, ...]] = {
    "ko": KO_PROFILES,
    "ja": JA_PROFILES,
}

# Trailing characters that carry no grammar and must be put back untouched.
_TRAILING = re.compile(r"[\s.!?。、！？…‥~〜\"'”』」)\]]*$")


def known_languages() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


def profiles_for(language: str) -> tuple[DialectProfile, ...]:
    return PROFILES.get(str(language).strip().lower(), ())


def profile(language: str, code: str) -> DialectProfile | None:
    for candidate in profiles_for(language):
        if candidate.code == code:
            return candidate
    return None


def all_profiles() -> tuple[DialectProfile, ...]:
    return tuple(item for language in known_languages() for item in profiles_for(language))


def is_interrogative(text: str, language: str) -> bool:
    """True when the sentence contains an interrogative word.

    Only Korean uses this, to choose between the ``-나`` and ``-노`` endings.
    """

    if str(language).strip().lower() != "ko":
        return False
    return any(word in text for word in KOREAN_INTERROGATIVES)


def split_trailing(text: str) -> tuple[str, str]:
    """Separate a sentence from its trailing punctuation."""

    match = _TRAILING.search(text)
    if match is None or match.start() == len(text):
        return text, ""
    return text[: match.start()], text[match.start() :]


def apply_vocabulary(text: str, profile_: DialectProfile) -> tuple[str, int]:
    """Substitute the profile's vocabulary, counting how many fired."""

    replaced = 0
    for standard, dialect in profile_.vocabulary:
        if standard in text:
            text = text.replace(standard, dialect)
            replaced += 1
    return text, replaced


def to_dialect(text: str, profile_: DialectProfile) -> tuple[str, str] | None:
    """Rewrite ``text`` into ``profile_``'s variety.

    Returns the rewritten sentence and the standard ending that matched, or None
    when no ending rule applies. Refusing is the point: an unmatched sentence
    would otherwise enter the corpus labelled as dialect while being standard.
    """

    body, trailing = split_trailing(text)
    if not body:
        return None
    # Only an interrogative *word* selects the wh table. A question mark alone
    # must not, or 밥 먹었니? becomes 밥 먹었노? when the 동남 varieties use
    # 밥 먹었나? for yes/no and reserve -노 for wh-questions.
    interrogative = is_interrogative(body, profile_.language)
    for standard, dialect in profile_.sorted_endings(interrogative=interrogative):
        if not body.endswith(standard):
            continue
        if standard == dialect:
            # An ending the variety shares with the standard. Matching it stops a
            # shorter rule from firing on a form this variety does not change.
            return None
        # A blocked context refuses a rule only when it is *more specific* than
        # the rule that matched. Otherwise a profile could not have both
        # ("じゃない", "やない") and a bare ("ない", "へん"): blocking じゃない
        # outright would kill the correct longer rule too. With this test, the
        # bare rule refuses on じゃない / もない / 少ない while the explicit
        # じゃない rule still fires.
        if any(
            len(context) > len(standard) and body.endswith(context) for context in profile_.blocked
        ):
            return None
        stem = body[: len(body) - len(standard)]
        if not stem:
            return None
        rewritten, _ = apply_vocabulary(stem + dialect, profile_)
        return rewritten + trailing, standard
    return None


__all__ = [
    "JA_PROFILES",
    "KOREAN_INTERROGATIVES",
    "KO_PROFILES",
    "PROFILES",
    "DialectProfile",
    "all_profiles",
    "apply_vocabulary",
    "is_interrogative",
    "known_languages",
    "profile",
    "profiles_for",
    "split_trailing",
    "to_dialect",
]
