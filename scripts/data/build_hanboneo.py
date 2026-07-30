#!/usr/bin/env python3
"""Build a deterministic 한본어 (Korean-Japanese code-mixed) parallel corpus.

한본어 is internet-register text that mixes Korean and Japanese. Three registers
occur, and they differ in what can be checked about their script:

1. ``script``      Hangul next to kana or kanji, e.g. ``오늘 スケジュール 어때``.
2. ``hangul_only`` a whole Japanese clause transliterated into Hangul, particles
                   and copula included: ``닝겐노 유리와 튼튼데스네``
                   = 人間の百合は丈夫ですね.
3. ``kana_only``   Korean vocabulary written in katakana inside otherwise
                   Japanese text: ``チンチャそれな``.

Registers 2 and 3 are uniform in script, so mixing cannot be detected from code
points and each frame declares which register it produces.

Every row is a triple, because the model has to read the mixture and answer in
exactly one language:

    {"kj": "엌ㅋㅋㅋ 닝겐노 유리와 튼튼데스넼ㅋㅋ",
     "ko": "헐ㅋㅋㅋ 인간의 백합은 튼튼하네요ㅋㅋ",
     "ja": "えっｗｗｗ 人間の百合は丈夫ですねｗｗ"}

``kj`` belongs in ``data.source_only_languages``, so kj->ko and kj->ja are
trained and ko->kj / ja->kj never are. The ``ko`` and ``ja`` fields also yield a
clean monolingual pair for free.

Generation is rule-based and seeded rather than sampled from a language model,
so it is reproducible and every Japanese surface is one a human put in the
lexicon below. Two things keep the output from degenerating:

- semantic classes. Nouns carry a class and predicates declare which classes
  they accept, so the builder does not emit ``방이 강하다`` or ``시간이 약하다``.
- caps. Each lexical entry and each frame has a ceiling, so the corpus cannot
  become the template restatement that ``audit_generated_shards.py`` rejects.

The lexicon size therefore bounds the corpus size. Ask for more rows than the
caps allow and the builder stops early and reports how many it produced.

Usage::

    python scripts/data/build_hanboneo.py --output data/synthetic_hanboneo.jsonl
    python scripts/data/build_hanboneo.py --output out.jsonl --report report.json

Exit codes: 0 rows written, 2 nothing could be built.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import sys
import unicodedata


# ---------------------------------------------------------------------------
# Hangul jamo arithmetic. Korean internet text fuses a trailing laugh consonant
# into the last syllable when it has no final consonant: 네 + ㅋ -> 넼.
# ---------------------------------------------------------------------------

_HANGUL_BASE = 0xAC00
_HANGUL_COUNT = 11172
_JONGSEONG_KIEUK = 24


def _syllable_offset(char: str) -> int | None:
    offset = ord(char) - _HANGUL_BASE
    return offset if 0 <= offset < _HANGUL_COUNT else None


def fuse_final_consonant(text: str, jongseong: int) -> str:
    """Attach a final consonant to the last syllable when it has none."""

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

    if not word:
        return False
    offset = _syllable_offset(word[-1])
    return offset is not None and offset % 28 != 0


def topic_particle(word: str) -> str:
    """Korean topic marker: 은 after a final consonant, 는 otherwise."""

    return "은" if has_final_consonant(word) else "는"


def subject_particle(word: str) -> str:
    """Korean subject marker: 이 after a final consonant, 가 otherwise."""

    return "이" if has_final_consonant(word) else "가"


# ---------------------------------------------------------------------------
# Lexicon. Each entry carries every surface the corpus needs, so both
# monolingual sides are correct by construction rather than translated at
# generation time. ``kind`` and ``accepts`` keep combinations meaningful.
# ---------------------------------------------------------------------------

# Semantic classes. They are finer than they look necessary because a coarse
# "abstract" bucket lets the builder emit 시간이 어렵다 and 책의 마을은 춥네요.
PERSON = "person"
ANIMAL = "animal"
PLANT = "plant"
CELESTIAL = "celestial"  # 별 달 태양
PRECIP = "precipitation"  # 비 눈
LANDSCAPE = "landscape"  # 바다 산 숲 강
WIND = "wind"
PLACE = "place"
ARTIFACT = "artifact"
MEDIA = "media"
FOOD = "food"
TASK = "task"
PERCEPT = "percept"  # 목소리 소리 색 맛
TIME = "time"
WEATHER = "weather"
MIND = "mind"
BODY = "body"

# Which modifier class may take which head class in a ``N1의 N2`` genitive.
# Without this the builder produces 책의 마을 and 비의 달.
GENITIVE_HEADS: dict[str, frozenset[str]] = {
    PERSON: frozenset({BODY, ANIMAL, ARTIFACT, MIND, PERCEPT, TASK, PLACE, FOOD, MEDIA, PLANT}),
    # No LANDSCAPE head: a shop does not have a sea.
    PLACE: frozenset({PERCEPT, ARTIFACT, TIME, FOOD, MEDIA, PLANT}),
    # No MIND head: 영화의 꿈 reads as nonsense.
    MEDIA: frozenset({PERCEPT}),
    LANDSCAPE: frozenset({PERCEPT, TIME, WIND, PLANT}),
    ANIMAL: frozenset({BODY, PERCEPT}),
}


@dataclass(frozen=True)
class Noun:
    hangul: str  # Japanese noun transliterated into Hangul
    kana: str  # the Japanese surface
    ko: str  # the Korean word
    kind: str  # semantic class


@dataclass(frozen=True)
class Predicate:
    hangul: str  # transliterated stem: 튼튼 / 츠요이 / 타베
    kana: str  # Japanese stem: 丈夫 / 強い / 食べ
    form: str  # "na", "i" or "verb"
    ko_polite: str
    ko_plain: str
    ko_negative: str  # polite, to match the polite Japanese negative
    accepts: frozenset[str] = field(default_factory=frozenset)
    # Dictionary form. Only verbs need it: 食べ->食べる is ichidan, but
    # 飲み->飲む and 聞き->聞く are godan and cannot be derived by appending る.
    kana_plain: str = ""
    hangul_plain: str = ""

    def __post_init__(self) -> None:
        if self.form == "verb" and not (self.kana_plain and self.hangul_plain):
            raise ValueError(f"verb {self.kana!r} must declare its dictionary form")
        if self.form not in {"na", "i", "verb"}:
            raise ValueError(f"unknown predicate form {self.form!r}")


@dataclass(frozen=True)
class Interjection:
    hangul: str
    kana: str
    ko: str


@dataclass(frozen=True)
class Loanword:
    """A Japanese word Koreans drop into Korean sentences."""

    kana: str
    hangul: str  # its Hangul transliteration
    ko: str  # the plain Korean equivalent
    pos: str  # "noun", "adjective" or "phrase"


@dataclass(frozen=True)
class KoreanInKatakana:
    """A Korean word Japanese speakers write in katakana."""

    katakana: str
    ko: str
    ja: str


@dataclass(frozen=True)
class Reaction:
    """A short Japanese reaction phrase, for chat-register one-liners."""

    kana: str
    ko: str


NOUNS: tuple[Noun, ...] = (
    Noun("닝겐", "人間", "인간", PERSON),
    Noun("센세", "先生", "선생님", PERSON),
    Noun("토모다치", "友達", "친구", PERSON),
    Noun("카조쿠", "家族", "가족", PERSON),
    Noun("코도모", "子供", "아이", PERSON),
    Noun("네코", "猫", "고양이", ANIMAL),
    Noun("이누", "犬", "개", ANIMAL),
    Noun("토리", "鳥", "새", ANIMAL),
    Noun("유리", "百合", "백합", PLANT),
    Noun("사쿠라", "桜", "벚꽃", PLANT),
    Noun("하나", "花", "꽃", PLANT),
    Noun("호시", "星", "별", CELESTIAL),
    Noun("츠키", "月", "달", CELESTIAL),
    Noun("타이요", "太陽", "태양", CELESTIAL),
    Noun("아메", "雨", "비", PRECIP),
    Noun("유키", "雪", "눈", PRECIP),
    Noun("우미", "海", "바다", LANDSCAPE),
    Noun("야마", "山", "산", LANDSCAPE),
    Noun("모리", "森", "숲", LANDSCAPE),
    Noun("카와", "川", "강", LANDSCAPE),
    Noun("카제", "風", "바람", WIND),
    Noun("마치", "町", "마을", PLACE),
    Noun("에키", "駅", "역", PLACE),
    Noun("가코", "学校", "학교", PLACE),
    Noun("헤야", "部屋", "방", PLACE),
    Noun("미세", "店", "가게", PLACE),
    Noun("혼", "本", "책", ARTIFACT),
    Noun("샤신", "写真", "사진", ARTIFACT),
    Noun("가멘", "画面", "화면", ARTIFACT),
    Noun("쿠루마", "車", "자동차", ARTIFACT),
    Noun("온가쿠", "音楽", "음악", MEDIA),
    Noun("에이가", "映画", "영화", MEDIA),
    Noun("우타", "歌", "노래", MEDIA),
    Noun("료리", "料理", "요리", FOOD),
    Noun("판", "パン", "빵", FOOD),
    Noun("시고토", "仕事", "일", TASK),
    Noun("슈쿠다이", "宿題", "숙제", TASK),
    Noun("코에", "声", "목소리", PERCEPT),
    Noun("오토", "音", "소리", PERCEPT),
    Noun("이로", "色", "색", PERCEPT),
    Noun("아지", "味", "맛", PERCEPT),
    Noun("니오이", "匂い", "냄새", PERCEPT),
    Noun("지칸", "時間", "시간", TIME),
    Noun("아사", "朝", "아침", TIME),
    Noun("요루", "夜", "밤", TIME),
    Noun("나츠", "夏", "여름", TIME),
    Noun("후유", "冬", "겨울", TIME),
    Noun("텐키", "天気", "날씨", WEATHER),
    Noun("유메", "夢", "꿈", MIND),
    Noun("키모치", "気持ち", "마음", MIND),
    Noun("오모이데", "思い出", "추억", MIND),
    Noun("테", "手", "손", BODY),
    Noun("메", "目", "눈", BODY),
    Noun("코코로", "心", "심장", BODY),
)

PREDICATES: tuple[Predicate, ...] = (
    Predicate(
        "튼튼",
        "丈夫",
        "na",
        "튼튼하네요",
        "튼튼하다",
        "튼튼하지 않아요",
        frozenset({ARTIFACT, PLACE, PERSON, ANIMAL, BODY}),
    ),
    Predicate(
        "키레이",
        "綺麗",
        "na",
        "예쁘네요",
        "예쁘다",
        "예쁘지 않아요",
        frozenset({PLANT, CELESTIAL, LANDSCAPE, PLACE, ARTIFACT, PERSON, PERCEPT}),
    ),
    Predicate(
        "겐키",
        "元気",
        "na",
        "활기차네요",
        "활기차다",
        "활기차지 않아요",
        frozenset({PERSON, ANIMAL}),
    ),
    Predicate(
        "시즈카",
        "静か",
        "na",
        "조용하네요",
        "조용하다",
        "조용하지 않아요",
        frozenset({PLACE, LANDSCAPE, PERSON, ANIMAL, TIME}),
    ),
    Predicate(
        "유메이",
        "有名",
        "na",
        "유명하네요",
        "유명하다",
        "유명하지 않아요",
        frozenset({PERSON, PLACE, MEDIA, FOOD}),
    ),
    Predicate(
        "타이헨",
        "大変",
        "na",
        "힘드네요",
        "힘들다",
        "힘들지 않아요",
        frozenset({TASK, MIND}),
    ),
    Predicate(
        "다이지",
        "大事",
        "na",
        "중요하네요",
        "중요하다",
        "중요하지 않아요",
        frozenset({TASK, MIND, PERSON, ARTIFACT, TIME}),
    ),
    Predicate(
        "라쿠",
        "楽",
        "na",
        "편하네요",
        "편하다",
        "편하지 않아요",
        frozenset({TASK, PLACE}),
    ),
    Predicate(
        "스테키",
        "素敵",
        "na",
        "멋지네요",
        "멋지다",
        "멋지지 않아요",
        frozenset({PERSON, MEDIA, ARTIFACT, PLACE, PLANT}),
    ),
    Predicate(
        "츠요이",
        "強い",
        "i",
        "강하네요",
        "강하다",
        "강하지 않아요",
        frozenset({PERSON, ANIMAL, WIND, BODY, PERCEPT}),
    ),
    Predicate(
        "요와이",
        "弱い",
        "i",
        "약하네요",
        "약하다",
        "약하지 않아요",
        frozenset({PERSON, ANIMAL, WIND, BODY, PERCEPT}),
    ),
    Predicate(
        "타카이",
        "高い",
        "i",
        "비싸네요",
        "비싸다",
        "비싸지 않아요",
        frozenset({ARTIFACT, FOOD}),
    ),
    Predicate(
        "야스이",
        "安い",
        "i",
        "싸네요",
        "싸다",
        "싸지 않아요",
        frozenset({ARTIFACT, FOOD}),
    ),
    Predicate(
        "우마이",
        "うまい",
        "i",
        "맛있네요",
        "맛있다",
        "맛있지 않아요",
        frozenset({FOOD}),
    ),
    Predicate(
        "아츠이",
        "暑い",
        "i",
        "덥네요",
        "덥다",
        "덥지 않아요",
        frozenset({WEATHER, TIME, PLACE}),
    ),
    Predicate(
        "사무이",
        "寒い",
        "i",
        "춥네요",
        "춥다",
        "춥지 않아요",
        frozenset({WEATHER, TIME, PLACE}),
    ),
    Predicate(
        "타노시이",
        "楽しい",
        "i",
        "즐겁네요",
        "즐겁다",
        "즐겁지 않아요",
        frozenset({TASK, MEDIA, MIND}),
    ),
    Predicate(
        "무즈카시이",
        "難しい",
        "i",
        "어렵네요",
        "어렵다",
        "어렵지 않아요",
        frozenset({TASK, MEDIA, ARTIFACT}),
    ),
    Predicate(
        "야사시이",
        "優しい",
        "i",
        "친절하네요",
        "친절하다",
        "친절하지 않아요",
        frozenset({PERSON}),
    ),
    Predicate(
        "우츠쿠시이",
        "美しい",
        "i",
        "아름답네요",
        "아름답다",
        "아름답지 않아요",
        frozenset({CELESTIAL, LANDSCAPE, PLANT, MEDIA, PLACE, PERCEPT}),
    ),
    Predicate(
        "하야이",
        "早い",
        "i",
        "빠르네요",
        "빠르다",
        "빠르지 않아요",
        frozenset({TIME, PERSON, ANIMAL}),
    ),
    Predicate(
        "오소이",
        "遅い",
        "i",
        "느리네요",
        "느리다",
        "느리지 않아요",
        frozenset({TIME, PERSON, ANIMAL}),
    ),
    Predicate(
        "나츠카시이",
        "懐かしい",
        "i",
        "그립네요",
        "그립다",
        "그립지 않아요",
        frozenset({MIND, MEDIA, PLACE, TIME}),
    ),
    Predicate(
        "타베",
        "食べ",
        "verb",
        "먹어요",
        "먹는다",
        "먹지 않아요",
        frozenset({PERSON, ANIMAL}),
        kana_plain="食べる",
        hangul_plain="타베루",
    ),
    Predicate(
        "노미",
        "飲み",
        "verb",
        "마셔요",
        "마신다",
        "마시지 않아요",
        frozenset({PERSON, ANIMAL}),
        kana_plain="飲む",
        hangul_plain="노무",
    ),
    Predicate(
        "미",
        "見",
        "verb",
        "봐요",
        "본다",
        "보지 않아요",
        frozenset({PERSON, ANIMAL}),
        kana_plain="見る",
        hangul_plain="미루",
    ),
    Predicate(
        "키키",
        "聞き",
        "verb",
        "들어요",
        "듣는다",
        "듣지 않아요",
        frozenset({PERSON, ANIMAL}),
        kana_plain="聞く",
        hangul_plain="키쿠",
    ),
    Predicate(
        "이키",
        "行き",
        "verb",
        "가요",
        "간다",
        "가지 않아요",
        frozenset({PERSON, ANIMAL}),
        kana_plain="行く",
        hangul_plain="이쿠",
    ),
    Predicate(
        "카에리",
        "帰り",
        "verb",
        "돌아가요",
        "돌아간다",
        "돌아가지 않아요",
        frozenset({PERSON, ANIMAL}),
        kana_plain="帰る",
        hangul_plain="카에루",
    ),
    Predicate(
        "와카리",
        "分かり",
        "verb",
        "알아요",
        "안다",
        "알지 않아요",
        frozenset({PERSON}),
        kana_plain="分かる",
        hangul_plain="와카루",
    ),
    Predicate(
        "와라이",
        "笑い",
        "verb",
        "웃어요",
        "웃는다",
        "웃지 않아요",
        frozenset({PERSON}),
        kana_plain="笑う",
        hangul_plain="와라우",
    ),
    Predicate(
        "히카리",
        "光り",
        "verb",
        "빛나요",
        "빛난다",
        "빛나지 않아요",
        frozenset({CELESTIAL, ARTIFACT}),
        kana_plain="光る",
        hangul_plain="히카루",
    ),
    Predicate(
        "후리",
        "降り",
        "verb",
        "내려요",
        "내린다",
        "내리지 않아요",
        frozenset({PRECIP}),
        kana_plain="降る",
        hangul_plain="후루",
    ),
)

INTERJECTIONS: tuple[Interjection, ...] = (
    Interjection("엌", "えっ", "헐"),
    Interjection("우와", "うわ", "우와"),
    Interjection("아", "あっ", "아"),
    Interjection("에", "えー", "에"),
    Interjection("오오", "おお", "오오"),
    Interjection("헤에", "へー", "허"),
    Interjection("야바", "やば", "대박"),
    Interjection("마지", "マジ", "진짜"),
)

LOANWORDS: tuple[Loanword, ...] = (
    Loanword("スケジュール", "스케줄", "일정", "noun"),
    Loanword("ランチ", "란치", "점심", "noun"),
    Loanword("ミーティング", "미팅구", "회의", "noun"),
    Loanword("オタク", "오타쿠", "덕후", "noun"),
    Loanword("ツンデレ", "츤데레", "츤데레", "noun"),
    Loanword("イベント", "이벤토", "행사", "noun"),
    Loanword("バイト", "바이토", "아르바이트", "noun"),
    Loanword("テンション", "텐션", "분위기", "noun"),
    Loanword("コンビニ", "콘비니", "편의점", "noun"),
    Loanword("アニメ", "아니메", "애니", "noun"),
    Loanword("マンガ", "만가", "만화", "noun"),
    Loanword("ゲーム", "게무", "게임", "noun"),
    Loanword("カラオケ", "카라오케", "노래방", "noun"),
    Loanword("キャラ", "캬라", "캐릭터", "noun"),
    Loanword("ドラマ", "도라마", "드라마", "noun"),
    Loanword("グッズ", "굿즈", "굿즈", "noun"),
    Loanword("やばい", "야바이", "위험한 거", "adjective"),
    Loanword("かわいい", "카와이", "귀여운 거", "adjective"),
    Loanword("すごい", "스고이", "대단한 거", "adjective"),
    Loanword("うるさい", "우루사이", "시끄러운 거", "adjective"),
    Loanword("おつかれ", "오츠카레", "수고했다는 말", "phrase"),
    Loanword("がんばれ", "간바레", "힘내라는 말", "phrase"),
    Loanword("なるほど", "나루호도", "그렇구나 하는 말", "phrase"),
    Loanword("それな", "소레나", "그거지 하는 말", "phrase"),
    Loanword("しかたない", "시카타나이", "어쩔 수 없다는 말", "phrase"),
    Loanword("だいすき", "다이스키", "정말 좋아한다는 말", "phrase"),
)

# Korean words that work as an intensifier or exclamation in front of a
# Japanese phrase, which is the ``チンチャそれな`` pattern.
KOREAN_INTENSIFIERS_IN_KATAKANA: tuple[KoreanInKatakana, ...] = (
    KoreanInKatakana("チンチャ", "진짜", "ほんとに"),
    KoreanInKatakana("テバク", "대박", "やばくて"),
    KoreanInKatakana("アイゴー", "아이고", "あーあ"),
    KoreanInKatakana("ケンチャナ", "괜찮아", "大丈夫、"),
    KoreanInKatakana("ワンジョン", "완전", "めっちゃ"),
    KoreanInKatakana("ノム", "너무", "すごく"),
)

# Korean nouns Japanese speakers borrow, used inside a Japanese sentence.
KOREAN_NOUNS_IN_KATAKANA: tuple[KoreanInKatakana, ...] = (
    KoreanInKatakana("オッパ", "오빠", "お兄さん"),
    KoreanInKatakana("オンニ", "언니", "お姉さん"),
    KoreanInKatakana("ヌナ", "누나", "お姉さん"),
    KoreanInKatakana("アジョシ", "아저씨", "おじさん"),
    KoreanInKatakana("チング", "친구", "友達"),
    KoreanInKatakana("ソンベ", "선배", "先輩"),
    KoreanInKatakana("フベ", "후배", "後輩"),
    KoreanInKatakana("サジャン", "사장", "社長"),
)

KOREAN_FOOD_IN_KATAKANA: tuple[KoreanInKatakana, ...] = (
    KoreanInKatakana("トッポギ", "떡볶이", "トッポギ"),
    KoreanInKatakana("キムチ", "김치", "キムチ"),
    KoreanInKatakana("マッコリ", "막걸리", "マッコリ"),
    KoreanInKatakana("サムギョプサル", "삼겹살", "サムギョプサル"),
    KoreanInKatakana("ビビンバ", "비빔밥", "ビビンバ"),
    KoreanInKatakana("チゲ", "찌개", "チゲ"),
    KoreanInKatakana("ソジュ", "소주", "ソジュ"),
    KoreanInKatakana("チャプチェ", "잡채", "チャプチェ"),
)

REACTIONS: tuple[Reaction, ...] = (
    Reaction("それな", "그거지"),
    Reaction("わかる", "인정"),
    Reaction("むり", "무리야"),
    Reaction("かわいい", "귀여워"),
    Reaction("すごい", "대단해"),
    Reaction("おつかれ", "수고했어"),
    Reaction("がんばれ", "힘내"),
    Reaction("なるほど", "그렇구나"),
    Reaction("たのしい", "재밌어"),
    Reaction("うれしい", "기뻐"),
    Reaction("ねむい", "졸려"),
    Reaction("さすが", "역시"),
    Reaction("ずるい", "치사해"),
    Reaction("いいね", "좋네"),
)

# (Korean laughter, Japanese laughter). Korean ㅋ fuses into the preceding
# syllable; Japanese uses ｗ or ふふ, which is a real localization difference and
# worth teaching explicitly.
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


def _accepting_predicates(kind: str, forms: frozenset[str] | None = None) -> tuple[Predicate, ...]:
    return tuple(
        predicate
        for predicate in PREDICATES
        if kind in predicate.accepts and (forms is None or predicate.form in forms)
    )


def _predicate_polite(predicate: Predicate) -> tuple[str, str, str]:
    """The polite ``~ですね`` / ``~ますね`` surfaces as (kj, ja, ko)."""

    if predicate.form == "verb":
        return f"{predicate.hangul}마스네", f"{predicate.kana}ますね", predicate.ko_polite
    # Both な- and い-adjectives take ですね; only the stem differs.
    return f"{predicate.hangul}데스네", f"{predicate.kana}ですね", predicate.ko_polite


def _predicate_plain(predicate: Predicate) -> tuple[str, str, str]:
    if predicate.form == "verb":
        # Appending る would give 飲みる for a godan verb, so use the declared
        # dictionary form.
        return predicate.hangul_plain, predicate.kana_plain, predicate.ko_plain
    if predicate.form == "na":
        return f"{predicate.hangul}다", f"{predicate.kana}だ", predicate.ko_plain
    return predicate.hangul, predicate.kana, predicate.ko_plain


def _predicate_negative(predicate: Predicate) -> tuple[str, str, str]:
    if predicate.form == "verb":
        return f"{predicate.hangul}마센", f"{predicate.kana}ません", predicate.ko_negative
    if predicate.form == "na":
        return (
            f"{predicate.hangul}자 아리마센",
            f"{predicate.kana}じゃありません",
            predicate.ko_negative,
        )
    # い-adjective: 強い -> 強くないです, 츠요이 -> 츠요쿠 나이데스
    return (
        f"{predicate.hangul[:-1]}쿠 나이데스",
        f"{predicate.kana[:-1]}くないです",
        predicate.ko_negative,
    )


@dataclass(frozen=True)
class Row:
    kj: str
    ko: str
    ja: str
    frame: str
    items: tuple[str, ...]
    mixing: str = "script"


def _laugh(rng: random.Random, *, fuse_into: str | None = None) -> tuple[str, str, str]:
    """Pick a laughter pair and optionally fuse the Korean ㅋ into a word.

    Returns ``(korean_laugh, japanese_laugh, fused_word)``.
    """

    korean, japanese = rng.choice(LAUGHTER)
    word = fuse_into or ""
    if fuse_into is not None and korean.startswith("ㅋ"):
        word = fuse_final_consonant(fuse_into, _JONGSEONG_KIEUK)
    return korean, japanese, word


def frame_transliterated_clause(rng: random.Random) -> Row:
    """Register 2 with a genitive subject: ``닝겐노 유리와 튼튼데스네``."""

    modifier = rng.choice([noun for noun in NOUNS if noun.kind in GENITIVE_HEADS])
    allowed = GENITIVE_HEADS[modifier.kind]
    heads = [
        noun
        for noun in NOUNS
        if noun.kind in allowed
        and noun is not modifier
        and _accepting_predicates(noun.kind, frozenset({"na", "i"}))
    ]
    if not heads:
        raise LookupError(modifier.kind)
    head = rng.choice(heads)
    predicate = rng.choice(_accepting_predicates(head.kind, frozenset({"na", "i"})))
    interjection = rng.choice(INTERJECTIONS)
    lead_ko, lead_ja, fused_lead = _laugh(rng, fuse_into=interjection.hangul)
    tail_ko, tail_ja, fused_tail = _laugh(rng, fuse_into=_predicate_polite(predicate)[0])
    predicate_kj, predicate_ja, predicate_ko = _predicate_polite(predicate)
    subject_ko = f"{modifier.ko}의 {head.ko}"

    return Row(
        kj=(
            f"{fused_lead}{lead_ko} {modifier.hangul}{_GENITIVE[0]} "
            f"{head.hangul}{_TOPIC[0]} {fused_tail}{tail_ko}"
        ),
        ko=(
            f"{interjection.ko}{lead_ko} {subject_ko}{topic_particle(subject_ko)} "
            f"{predicate_ko}{tail_ko}"
        ),
        ja=(
            f"{interjection.kana}{lead_ja} {modifier.kana}{_GENITIVE[1]}"
            f"{head.kana}{_TOPIC[1]}{predicate_ja}{tail_ja}"
        ),
        frame="transliterated_clause",
        items=(head.kana, modifier.kana, predicate.kana, interjection.kana),
        mixing="hangul_only",
    )


def frame_transliterated_plain(rng: random.Random) -> Row:
    """Register 2 in the plain register with a subject particle."""

    noun = rng.choice(NOUNS)
    candidates = _accepting_predicates(noun.kind)
    if not candidates:
        raise LookupError(noun.kind)
    predicate = rng.choice(candidates)
    predicate_kj, predicate_ja, predicate_ko = _predicate_plain(predicate)
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        kj=f"{noun.hangul}{_SUBJECT[0]} {predicate_kj}{laugh_ko}",
        ko=f"{noun.ko}{subject_particle(noun.ko)} {predicate_ko}{laugh_ko}",
        ja=f"{noun.kana}{_SUBJECT[1]}{predicate_ja}{laugh_ja}",
        frame="transliterated_plain",
        items=(noun.kana, predicate.kana),
        mixing="hangul_only",
    )


def frame_negative_clause(rng: random.Random) -> Row:
    """Register 2, negated, so the corpus is not uniformly affirmative."""

    noun = rng.choice(NOUNS)
    candidates = _accepting_predicates(noun.kind)
    if not candidates:
        raise LookupError(noun.kind)
    predicate = rng.choice(candidates)
    tail_kj, tail_ja, tail_ko = _predicate_negative(predicate)

    return Row(
        kj=f"{noun.hangul}{_TOPIC[0]} {tail_kj}",
        ko=f"{noun.ko}{topic_particle(noun.ko)} {tail_ko}",
        ja=f"{noun.kana}{_TOPIC[1]}{tail_ja}",
        frame="negative_clause",
        items=(noun.kana, predicate.kana),
        mixing="hangul_only",
    )


def frame_kana_loanword(rng: random.Random) -> Row:
    """Register 1: a Japanese noun left in kana inside a Korean sentence."""

    loan = rng.choice([word for word in LOANWORDS if word.pos == "noun"])
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        kj=f"오늘 {loan.kana} 얘기 좀 하자{laugh_ko}",
        ko=f"오늘 {loan.ko} 얘기 좀 하자{laugh_ko}",
        ja=f"今日は{loan.kana}の話を少ししよう{laugh_ja}",
        frame="kana_loanword",
        items=(loan.kana,),
    )


def frame_kana_loanword_question(rng: random.Random) -> Row:
    """Register 1 as a question, which is where the mixture actually shows up."""

    loan = rng.choice([word for word in LOANWORDS if word.pos == "noun"])
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        kj=f"그 {loan.kana} 어떻게 됐어{laugh_ko}",
        ko=f"그 {loan.ko} 어떻게 됐어{laugh_ko}",
        ja=f"あの{loan.kana}はどうなった{laugh_ja}",
        frame="kana_loanword_question",
        items=(loan.kana,),
    )


def frame_hangul_loanword(rng: random.Random) -> Row:
    """Register 2: a Japanese word transliterated into Hangul inside Korean."""

    loan = rng.choice([word for word in LOANWORDS if word.pos in {"noun", "adjective"}])
    interjection = rng.choice(INTERJECTIONS)
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=interjection.hangul)

    return Row(
        kj=f"{fused}{laugh_ko} 이거 {loan.hangul} 아니냐",
        ko=f"{interjection.ko}{laugh_ko} 이거 {loan.ko} 아니냐",
        ja=f"{interjection.kana}{laugh_ja} これ{loan.kana}じゃないか",
        frame="hangul_loanword",
        items=(loan.kana, interjection.kana),
        mixing="hangul_only",
    )


def frame_hangul_phrase(rng: random.Random) -> Row:
    """Register 2 with a set phrase, e.g. ``오츠카레``."""

    loan = rng.choice([word for word in LOANWORDS if word.pos == "phrase"])
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=loan.hangul)

    return Row(
        kj=f"{fused}{laugh_ko}",
        ko=f"{loan.ko}{laugh_ko}",
        ja=f"{loan.kana}{laugh_ja}",
        frame="hangul_phrase",
        items=(loan.kana,),
        mixing="hangul_only",
    )


def frame_katakana_reaction(rng: random.Random) -> Row:
    """Register 3: ``チンチャそれな``, a Korean intensifier plus a Japanese reaction."""

    intensifier = rng.choice(KOREAN_INTENSIFIERS_IN_KATAKANA)
    # Avoid pairs whose Korean glosses collide, which would read as 대박 대박이야.
    reaction = rng.choice([item for item in REACTIONS if intensifier.ko not in item.ko])
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        kj=f"{intensifier.katakana}{reaction.kana}{laugh_ja}",
        ko=f"{intensifier.ko} {reaction.ko}{laugh_ko}",
        ja=f"{intensifier.ja}{reaction.kana}{laugh_ja}",
        frame="katakana_reaction",
        items=(intensifier.katakana, reaction.kana),
        mixing="kana_only",
    )


def frame_korean_person_in_japanese(rng: random.Random) -> Row:
    """Register 3: a borrowed Korean kinship term inside a Japanese sentence."""

    word = rng.choice(KOREAN_NOUNS_IN_KATAKANA)
    predicate = rng.choice(_accepting_predicates(PERSON))
    _, predicate_ja, predicate_ko = _predicate_polite(predicate)
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        kj=f"{word.katakana}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        ko=f"{word.ko}{topic_particle(word.ko)} {predicate_ko}{laugh_ko}",
        ja=f"{word.ja}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        frame="korean_person_in_japanese",
        items=(word.katakana, predicate.kana),
        mixing="kana_only",
    )


def frame_korean_food_in_japanese(rng: random.Random) -> Row:
    """Register 3: a Korean dish name, which keeps its katakana form in Japanese."""

    food = rng.choice(KOREAN_FOOD_IN_KATAKANA)
    predicate = rng.choice(_accepting_predicates(FOOD, frozenset({"na", "i"})))
    _, predicate_ja, predicate_ko = _predicate_polite(predicate)
    laugh_ko, laugh_ja, _ = _laugh(rng)

    return Row(
        kj=f"{food.katakana}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        ko=f"{food.ko}{topic_particle(food.ko)} {predicate_ko}{laugh_ko}",
        ja=f"{food.ja}{_TOPIC[1]}{predicate_ja}{laugh_ja}",
        frame="korean_food_in_japanese",
        items=(food.katakana, predicate.kana),
        mixing="kana_only",
    )


def frame_mixed_reply(rng: random.Random) -> Row:
    """Register 1: a katakana Korean term plus a Hangul-transliterated predicate."""

    word = rng.choice(KOREAN_NOUNS_IN_KATAKANA)
    predicate = rng.choice(_accepting_predicates(PERSON))
    predicate_kj, predicate_ja, predicate_ko = _predicate_polite(predicate)
    laugh_ko, laugh_ja, fused = _laugh(rng, fuse_into=predicate_kj)

    return Row(
        kj=f"{word.katakana} 그거 {fused}{laugh_ko}",
        ko=f"{word.ko} 그거 {predicate_ko}{laugh_ko}",
        ja=f"{word.ja}それ{predicate_ja}{laugh_ja}",
        frame="mixed_reply",
        items=(word.katakana, predicate.kana),
    )


FRAMES = (
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


def _has_hangul(text: str) -> bool:
    return any("가" <= char <= "힣" or "ㄱ" <= char <= "ㅣ" for char in text)


def _has_kana(text: str) -> bool:
    return any("぀" <= char <= "ヿ" or "ｦ" <= char <= "ﾟ" for char in text)


def _has_han(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


def validate_row(row: Row) -> None:
    """Reject a row whose sides break the invariant the corpus exists to teach.

    A generator bug has to fail the build rather than reach the training set,
    because a code-mixed target is exactly the failure mode kj was introduced
    to prevent.
    """

    if _has_kana(row.ko) or _has_han(row.ko):
        raise ValueError(f"Korean side is not monolingual: {row.ko!r} (frame {row.frame})")
    if _has_hangul(row.ja):
        raise ValueError(f"Japanese side is not monolingual: {row.ja!r} (frame {row.frame})")

    has_hangul = _has_hangul(row.kj)
    has_japanese = _has_kana(row.kj) or _has_han(row.kj)
    if row.mixing == "script":
        if not (has_hangul and has_japanese):
            raise ValueError(
                f"script mixture needs Hangul and Japanese: {row.kj!r} (frame {row.frame})"
            )
    elif row.mixing == "hangul_only":
        if not has_hangul or has_japanese:
            raise ValueError(
                f"transliterated mixture must be all Hangul: {row.kj!r} (frame {row.frame})"
            )
    elif row.mixing == "kana_only":
        if has_hangul or not has_japanese:
            raise ValueError(
                f"katakana mixture must be all Japanese script: {row.kj!r} (frame {row.frame})"
            )
    else:
        raise ValueError(f"unknown mixing mode {row.mixing!r} (frame {row.frame})")

    for name, value in (("kj", row.kj), ("ko", row.ko), ("ja", row.ja)):
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
    """Generate rows under per-item and per-frame caps.

    The caps are what keep this from becoming template restatement, and they
    also bound the corpus: the lexicon, not ``max_rows``, decides how many
    distinct rows exist.
    """

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
            if row.kj in seen:
                duplicates += 1
                continue
            if any(item_counts[item] >= max_per_item for item in row.items):
                capped += 1
                continue
            seen.add(row.kj)
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
            "korean_intensifiers": len(KOREAN_INTENSIFIERS_IN_KATAKANA),
            "korean_nouns": len(KOREAN_NOUNS_IN_KATAKANA),
            "korean_food": len(KOREAN_FOOD_IN_KATAKANA),
            "reactions": len(REACTIONS),
        },
    }
    return rows, report


def write_rows(rows: list[Row], output: Path) -> None:
    """Write atomically so an interrupted build never leaves a partial shard."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"kj": row.kj, "ko": row.ko, "ja": row.ja, "synthetic": True},
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic 한본어 parallel corpus.")
    parser.add_argument("--output", required=True, help="destination JSONL path")
    parser.add_argument("--report", help="write the build report JSON here")
    parser.add_argument("--max-rows", type=int, default=20_000, help="upper bound on rows")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--max-per-item",
        type=int,
        default=120,
        help="cap on how often one lexical entry may appear",
    )
    parser.add_argument(
        "--max-per-frame",
        type=int,
        default=2_000,
        help="cap on how many rows one sentence frame may produce",
    )
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
    write_rows(rows, output)
    report["output"] = str(output)
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"{output}: {report['rows']:,} rows "
        f"(requested {report['requested_rows']:,}), "
        f"{report['distinct_lexical_items_used']} lexical entries, "
        f"registers={report['registers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
