"""Paired internet-register rewrites for a translation pair.

The corpus holds no internet register at all, and 구어 scores chrF 30.45. The
corpus does hold 2.4M spoken rows, but transcribed speech is standard-language
and sentence-complete; ``ㅋㅋ``, ``ㄹㅇ``, ``w`` and ``草`` appear nowhere.

The critical difference from :mod:`dialect_lexicon`: a dialect row keeps one side
standard, because the goal is to *understand* dialect input. Internet register has
to change **both** sides. A pair of Korean net-speak against plain Japanese is a
register mismatch, and training on it teaches the model to throw the register
away - the exact failure the data is meant to fix. So every rule here is a
*paired* rule, and a row is emitted only when both sides actually changed.

Three rule kinds, each with a paired form on both sides:

``laughter``
    ``ㅋㅋ``/``ㅋㅋㅋ``/``ㅎㅎ``/``ㅠㅠ`` against ``w``/``www``/``草``/``(泣)``.
    Applies to any casual sentence, which is what makes a usable yield possible.

``substitutions``
    Word-level net forms that exist on both sides: 진짜/本当 becomes ㄹㅇ/マジ.
    Entries where only one language has a net form are left out, because using
    them would produce exactly the mismatch above.

``intensifiers``
    Korean 개/핵/존 against Japanese ガチ/鬼/マジ, attached to an adjective that
    is present in both sides of the pair.

Formal sentences are refused outright. Appending ``ㅋㅋ`` to ``~습니다`` is not a
register anyone writes, and the source pool has to be filtered rather than
patched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

# A pair must be casual on both sides before any net marker is added, and
# formality is checked over the *whole* sentence rather than just the ending.
# `너무 오랫동안 기다리게 해서 정말 죄송합니다` clashes badly with ㄹㅇ, and
# `可愛い。ガチいいですね` is the same clash on the Japanese side. An
# end-anchored test also missed `합니다` outright: the pattern `ㅂ니다` is a
# compatibility-jamo literal and never matches 합+니+다.
KO_FORMAL = re.compile(r"(니다|니까|십시오|세요|해요|어요|아요|지요|군요|네요|을까요|나요)")
JA_FORMAL = re.compile(r"(ます|ません|です|でした|ましょう|ください|ございま)")
# Narrative past (~았다/~었다/~했다) is excluded: those are reported prose, and
# appending ㅋㅋ to `감사 인사를 전했다` is not a register anyone writes. The
# copula and adjective ~다 stay, because `대박이다` and `좋다` are chat.
# Present-tense narrative ~는다 is caught too. The fused ~ㄴ다 (떠난다, 간다)
# cannot be: the jamo sits inside a composed syllable, so a regex over
# characters cannot see it, and telling 떠난다 from 좋다 needs to know the stem
# is a verb. A residue of present-tense narration therefore survives.
KO_NARRATIVE = re.compile(r"(았다|었다|였다|했다|하였다|이었다|는다|ㄴ다)[.!?…~\s]*$")
KO_CASUAL = re.compile(
    r"(아|어|야|지|네|다|냐|까|군|걸|잖아|는데|거야|해|워|음|자|래|봐|줘|료|든)[.!?…~\s]*$"
)
JA_CASUAL = re.compile(r"(だ|ね|よ|た|い|か|な|ぜ|わ|の|ん|ぞ|さ|う)[。.!?…~\s]*$")

# Trailing punctuation is lifted off before a marker is appended and put back
# after, so `대박!` becomes `대박ㅋㅋ!` rather than `대박!ㅋㅋ`.
_TRAILING = re.compile(r"[\s.!?。、！？…‥~〜]*$")


@dataclass(frozen=True)
class NetStyle:
    """One internet-register style. Every rule carries both languages' forms."""

    code: str
    label: str
    # (ko marker, ja marker) appended to the end of the sentence.
    laughter: tuple[tuple[str, str], ...] = ()
    # Crying markers, used only when a negative cue is present on both sides.
    lament: tuple[tuple[str, str], ...] = ()
    # (ko standard, ko net, ja standard, ja net). Fires only when both standards
    # are present, so the two sides can never drift apart in register.
    substitutions: tuple[tuple[str, str, str, str], ...] = ()
    # (ko standard, ko net, ja standard, ja net) for words that only carry the
    # net form in interjection position.
    interjections: tuple[tuple[str, str, str, str], ...] = ()
    # (ko adjective, ja adjective) that an intensifier may attach to.
    intensifier_targets: tuple[tuple[str, str], ...] = ()
    # (ko intensifier, ja intensifier).
    intensifiers: tuple[tuple[str, str], ...] = field(default_factory=tuple)


# Laughter and reaction markers. The Korean consonant runs and the Japanese
# w/草 family are the direct counterparts of each other.
LAUGHTER: tuple[tuple[str, str], ...] = (
    ("ㅋㅋ", "w"),
    ("ㅋㅋㅋ", "www"),
    ("ㅋㅋㅋㅋ", "草"),
    ("ㅋㅋ", "ｗ"),
    ("ㅎㅎ", "ふふ"),
)

# Crying markers carry sentiment, so they cannot be chosen by hash like laughter
# can: `오늘 날씨 좋다ㅠㅠ` reads as the opposite of what it says. These are gated
# on a negative cue appearing in both sides of the pair.
LAMENT: tuple[tuple[str, str], ...] = (
    ("ㅠㅠ", "(泣)"),
    ("ㅠㅠ", "ぴえん"),
    ("ㅜㅜ", "(泣)"),
)

# (ko cue, ja cue). Both must be present before a lament marker is added.
NEGATIVE_CUES: tuple[tuple[str, str], ...] = (
    ("어려워", "難しい"),
    ("힘들어", "つらい"),
    ("슬퍼", "悲しい"),
    ("아쉬워", "残念"),
    ("미안", "ごめん"),
    ("안 돼", "ダメ"),
    ("안돼", "ダメ"),
    ("못해", "できない"),
    ("무서워", "怖い"),
    ("피곤해", "疲れた"),
)

# Word-level net forms that exist in both languages. Kept short on purpose: an
# entry whose Japanese side does not actually change would create the register
# mismatch this module exists to avoid.
SUBSTITUTIONS: tuple[tuple[str, str, str, str], ...] = (
    # Longest Japanese standard first: 本当 inside 本当に would otherwise yield
    # マジに, which is not Japanese. apply_substitutions sorts on this.
    ("진짜", "ㄹㅇ", "本当に", "マジで"),
    ("정말", "ㄹㅇ", "本当に", "マジで"),
    ("진짜", "ㄹㅇ", "本当", "マジ"),
    ("정말", "ㄹㅇ", "本当", "マジ"),
    ("너무", "넘", "とても", "めっちゃ"),
    ("최고", "갓", "最高", "神"),
)

# Interjections. These only carry the net form when they stand as an
# interjection: 감사 인사를 전했다 is a noun phrase, and rewriting it gives
# ㄱㅅ 인사를 전했다. So a match must start the sentence or be the whole of it.
INTERJECTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("수고했어", "ㅅㄱ", "お疲れ", "乙"),
    ("수고", "ㅅㄱ", "お疲れ", "乙"),
    ("축하해", "ㅊㅋ", "おめでとう", "おめ"),
    ("축하", "ㅊㅋ", "おめでとう", "おめ"),
    ("감사합니다", "ㄱㅅ", "ありがとう", "あり"),
    ("고마워", "ㄱㅅ", "ありがとう", "あり"),
    ("미안해", "ㅈㅅ", "ごめん", "ごめ"),
    ("미안", "ㅈㅅ", "ごめん", "ごめ"),
    ("알았어", "ㅇㅋ", "わかった", "おk"),
    ("인정", "ㅇㅈ", "わかる", "それな"),
)

# Intensifier prefixes. Korean stacks 개/핵/존 onto an adjective and Japanese
# stacks ガチ/鬼/マジ, so the two behave the same way.
INTENSIFIERS: tuple[tuple[str, str], ...] = (
    ("개", "ガチ"),
    ("핵", "鬼"),
)

# Adjectives an intensifier may attach to, paired so both sides get one.
INTENSIFIER_TARGETS: tuple[tuple[str, str], ...] = (
    ("좋아", "いい"),
    ("좋다", "いい"),
    ("재밌어", "面白い"),
    ("재미있어", "面白い"),
    ("맛있어", "うまい"),
    ("예뻐", "かわいい"),
    ("귀여워", "かわいい"),
    ("빨라", "速い"),
    ("강해", "強い"),
    ("쉬워", "簡単"),
)

STYLES: tuple[NetStyle, ...] = (
    NetStyle(code="laughter", label="웃음 표기 (ㅋㅋ / w·草)", laughter=LAUGHTER),
    NetStyle(
        code="abbreviation",
        label="초성·약어 (ㄹㅇ / マジ)",
        substitutions=SUBSTITUTIONS,
        interjections=INTERJECTIONS,
    ),
    NetStyle(
        code="intensifier",
        label="강조 접두 (개·핵 / ガチ·鬼)",
        intensifier_targets=INTENSIFIER_TARGETS,
        intensifiers=INTENSIFIERS,
    ),
    NetStyle(
        code="abbreviation_laughter",
        label="약어 + 웃음",
        laughter=LAUGHTER,
        substitutions=SUBSTITUTIONS,
        interjections=INTERJECTIONS,
    ),
    NetStyle(
        code="lament",
        label="울음 표기 (ㅠㅠ / (泣)·ぴえん)",
        lament=LAMENT,
    ),
    NetStyle(
        code="intensifier_laughter",
        label="강조 + 웃음",
        laughter=LAUGHTER,
        intensifier_targets=INTENSIFIER_TARGETS,
        intensifiers=INTENSIFIERS,
    ),
)


def known_styles() -> tuple[str, ...]:
    return tuple(style.code for style in STYLES)


def style(code: str) -> NetStyle | None:
    for candidate in STYLES:
        if candidate.code == code:
            return candidate
    return None


def is_casual_pair(ko: str, ja: str) -> bool:
    """True when neither side is formal and both end in a casual form."""

    if KO_FORMAL.search(ko) or JA_FORMAL.search(ja):
        return False
    if KO_NARRATIVE.search(ko):
        return False
    return bool(KO_CASUAL.search(ko)) and bool(JA_CASUAL.search(ja))


def split_trailing(text: str) -> tuple[str, str]:
    match = _TRAILING.search(text)
    if match is None or match.start() == len(text):
        return text, ""
    return text[: match.start()], text[match.start() :]


def append_marker(text: str, marker: str) -> str:
    """Attach ``marker`` before any trailing punctuation."""

    body, trailing = split_trailing(text)
    if not body:
        return text
    return body + marker + trailing


# Characters that end the interjection outright.
_SENTENCE_BREAK = frozenset(".!?…~,、。！？ｗ")
# A Korean form ending in one of these is conjugated, so it cannot be a noun
# modifier and is safe with more words after it.
_CONJUGATED_ENDINGS = ("워", "해", "했어", "았어", "겠어", "네", "다")


def _is_interjection_position(text: str, word: str) -> bool:
    """True when ``word`` stands alone or opens the sentence.

    ``감사`` opening an utterance is the interjection; ``감사 인사를 전했다`` is a
    noun phrase, and abbreviating it gives ``ㄱㅅ 인사를 전했다``.
    """

    stripped = text.strip()
    if stripped == word:
        return True
    if not stripped.startswith(word):
        return False
    following = stripped[len(word) : len(word) + 1]
    if following in _SENTENCE_BREAK:
        return True
    if following != " ":
        # A particle straight after the word makes it a noun, not an interjection.
        return False
    # A space may open an interjection (`고마워 진짜`) or continue a noun phrase
    # (`감사 인사를 전했다`). A conjugated form cannot modify a noun, so only those
    # are allowed to be followed by more words; a bare noun like 감사 is not.
    return word.endswith(_CONJUGATED_ENDINGS)


def apply_substitutions(
    ko: str,
    ja: str,
    style_: NetStyle,
) -> tuple[str, str, int]:
    """Replace paired words, counting how many pairs fired.

    A substitution applies only when *both* standards are present, so the sides
    cannot end up in different registers. Entries are tried with the longest
    Japanese standard first, or ``本当`` would match inside ``本当に`` and leave
    ``マジに``.
    """

    applied = 0
    ordered = sorted(style_.substitutions, key=lambda entry: len(entry[2]), reverse=True)
    for ko_standard, ko_net, ja_standard, ja_net in ordered:
        if ko_standard in ko and ja_standard in ja:
            ko = ko.replace(ko_standard, ko_net)
            ja = ja.replace(ja_standard, ja_net)
            applied += 1
    return ko, ja, applied


def apply_interjections(
    ko: str,
    ja: str,
    style_: NetStyle,
) -> tuple[str, str, int]:
    """Replace paired interjections, but only in interjection position."""

    ordered = sorted(style_.interjections, key=lambda entry: len(entry[0]), reverse=True)
    for ko_standard, ko_net, ja_standard, ja_net in ordered:
        if not _is_interjection_position(ko, ko_standard):
            continue
        if ja_standard not in ja:
            continue
        return (
            ko.replace(ko_standard, ko_net, 1),
            ja.replace(ja_standard, ja_net, 1),
            1,
        )
    return ko, ja, 0


def apply_intensifier(
    ko: str,
    ja: str,
    style_: NetStyle,
    variant: int,
) -> tuple[str, str, int]:
    """Prefix a paired adjective with a matched intensifier."""

    if not style_.intensifiers or not style_.intensifier_targets:
        return ko, ja, 0
    ko_intensifier, ja_intensifier = style_.intensifiers[variant % len(style_.intensifiers)]
    for ko_adjective, ja_adjective in style_.intensifier_targets:
        if ko_adjective in ko and ja_adjective in ja:
            # Guard against stacking onto an already intensified form.
            if ko_intensifier + ko_adjective in ko or ja_intensifier + ja_adjective in ja:
                continue
            return (
                ko.replace(ko_adjective, ko_intensifier + ko_adjective, 1),
                ja.replace(ja_adjective, ja_intensifier + ja_adjective, 1),
                1,
            )
    return ko, ja, 0


def to_netspeak(
    ko: str,
    ja: str,
    style_: NetStyle,
    variant: int = 0,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Rewrite both sides into ``style_``.

    Returns the two rewritten sides and which rule kinds fired, or None when the
    pair is formal or when either side would come out unchanged. Refusing on a
    one-sided change is the point: that pair would teach register loss.
    """

    if not ko or not ja or not is_casual_pair(ko, ja):
        return None

    new_ko, new_ja = ko, ja
    fired: list[str] = []

    new_ko, new_ja, substituted = apply_substitutions(new_ko, new_ja, style_)
    if substituted:
        fired.append("substitution")

    new_ko, new_ja, interjected = apply_interjections(new_ko, new_ja, style_)
    if interjected:
        fired.append("interjection")

    new_ko, new_ja, intensified = apply_intensifier(new_ko, new_ja, style_, variant)
    if intensified:
        fired.append("intensifier")

    if style_.lament and not fired:
        for ko_cue, ja_cue in NEGATIVE_CUES:
            if ko_cue in new_ko and ja_cue in new_ja:
                ko_marker, ja_marker = style_.lament[variant % len(style_.lament)]
                new_ko = append_marker(new_ko, ko_marker)
                new_ja = append_marker(new_ja, ja_marker)
                fired.append("lament")
                break

    if style_.laughter:
        # Laughter is what makes any casual sentence usable, but a style that
        # *only* appends it would make every row end the same way, so it is only
        # added when the style has nothing else or when another rule already
        # fired - and the variant picks which marker.
        collapsed = len(new_ko.strip()) <= 3
        if (not fired or style_.code.endswith("_laughter")) and not collapsed:
            ko_marker, ja_marker = style_.laughter[variant % len(style_.laughter)]
            new_ko = append_marker(new_ko, ko_marker)
            new_ja = append_marker(new_ja, ja_marker)
            fired.append("laughter")

    if new_ko == ko or new_ja == ja:
        return None
    return new_ko, new_ja, tuple(fired)


__all__ = [
    "INTENSIFIERS",
    "INTENSIFIER_TARGETS",
    "INTERJECTIONS",
    "JA_CASUAL",
    "JA_FORMAL",
    "KO_CASUAL",
    "KO_FORMAL",
    "KO_NARRATIVE",
    "LAMENT",
    "LAUGHTER",
    "NEGATIVE_CUES",
    "STYLES",
    "SUBSTITUTIONS",
    "NetStyle",
    "append_marker",
    "apply_intensifier",
    "apply_interjections",
    "apply_substitutions",
    "is_casual_pair",
    "known_styles",
    "split_trailing",
    "style",
    "to_netspeak",
]
