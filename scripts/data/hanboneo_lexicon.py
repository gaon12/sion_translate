"""Lexicon for the 한본어 builder, separated so the tables stay reviewable.

한본어 works because Korean and Japanese are both agglutinative with SOV order,
so a stem from one language takes an ending from the other and the result still
parses. The productive patterns, confirmed against 나무위키 and the examples in
the project brief:

    Korean stem + Japanese ending
        아랏소데스  = 알겠습니다      알았어 + です
        체고카요    = 최고냐고        최고 + かよ
        테바이      = 대박임          대박 + い
        키요이      = 귀여워          귀요 + い
        친챠소레나  = 진짜 그럼       진짜 + それな

    Japanese stem + Korean ending
        카와이하다  = 귀엽다          かわいい + ~하다
        やばいンデ  = 굉장한데        やばい + ~ㄴ데 (in katakana)

    Japanese adverb + Korean predicate
        마지코마워  = 정말 고마워     まじ + 고마워

Korean words are also respelled the way a Japanese speaker would pronounce them
(진짜 -> 친챠, 최고 -> 체고, 대박 -> 테바), which is what makes the blend read as
한본어 rather than as a loanword.

Because the Japanese realization of a blend is frequently lexical rather than
compositional (대박 + い is やばい, not 大当たりい), irregular pairs are listed
explicitly and only the regular ones are composed by rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Semantic classes, so the builder does not emit 방이 강하다 or 시간이 어렵다.
# --------------------------------------------------------------------------

PERSON = "person"
ANIMAL = "animal"
PLANT = "plant"
CELESTIAL = "celestial"
PRECIP = "precipitation"
LANDSCAPE = "landscape"
WIND = "wind"
PLACE = "place"
ARTIFACT = "artifact"
MEDIA = "media"
FOOD = "food"
TASK = "task"
PERCEPT = "percept"
TIME = "time"
WEATHER = "weather"
MIND = "mind"
BODY = "body"

# Which modifier class may head which in a ``N1의 N2`` genitive.
GENITIVE_HEADS: dict[str, frozenset[str]] = {
    PERSON: frozenset({BODY, ANIMAL, ARTIFACT, MIND, PERCEPT, TASK, PLACE, FOOD, MEDIA, PLANT}),
    PLACE: frozenset({PERCEPT, ARTIFACT, TIME, FOOD, MEDIA, PLANT}),
    MEDIA: frozenset({PERCEPT}),
    LANDSCAPE: frozenset({PERCEPT, TIME, WIND, PLANT}),
    ANIMAL: frozenset({BODY, PERCEPT}),
}


@dataclass(frozen=True)
class Noun:
    """A Japanese noun with its Hangul transliteration and Korean equivalent."""

    hangul: str
    kana: str
    ko: str
    kind: str


@dataclass(frozen=True)
class Predicate:
    hangul: str  # transliterated stem: 튼튼 / 츠요이 / 타베
    kana: str  # Japanese stem: 丈夫 / 強い / 食べ
    form: str  # "na", "i" or "verb"
    ko_polite: str
    ko_plain: str
    ko_negative: str  # polite, matching the polite Japanese negative
    accepts: frozenset[str] = field(default_factory=frozenset)
    kana_plain: str = ""  # dictionary form; 飲み -> 飲む is not 飲み + る
    hangul_plain: str = ""

    def __post_init__(self) -> None:
        if self.form not in {"na", "i", "verb"}:
            raise ValueError(f"unknown predicate form {self.form!r}")
        if self.form == "verb" and not (self.kana_plain and self.hangul_plain):
            raise ValueError(f"verb {self.kana!r} must declare its dictionary form")


@dataclass(frozen=True)
class Interjection:
    hangul: str
    kana: str
    ko: str


@dataclass(frozen=True)
class Loanword:
    """A Japanese word Koreans drop into Korean sentences."""

    kana: str
    hangul: str
    ko: str
    pos: str  # "noun", "adjective" or "phrase"


@dataclass(frozen=True)
class KoreanInKatakana:
    """A Korean word Japanese speakers write in katakana."""

    katakana: str
    ko: str
    ja: str


@dataclass(frozen=True)
class Reaction:
    kana: str
    ko: str


# --------------------------------------------------------------------------
# Morphological blending
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KoreanContentNoun:
    """A Korean noun that keeps its Hangul spelling inside a Japanese frame."""

    ko: str  # 마음
    ja: str  # 気持ち
    kind: str  # semantic class


@dataclass(frozen=True)
class KoreanEnding:
    """A Korean ending that attaches to a Japanese stem."""

    hangul: str  # 하다
    katakana: str  # ハダ
    ko_template: str  # "{ko}다"
    ja_template: str  # "{ja}だ"
    accepts: frozenset[str]  # which Predicate.form values


@dataclass(frozen=True)
class Blend:
    """A hand-verified blend.

    Blends are listed rather than composed because both sides resist it. The
    Japanese realization is often lexical (대박 + い is やばい, not 大当たりい),
    and the Korean one needs a stem rather than an inflected surface, so
    appending an ending to 괜찮아 gives 괜찮아인데 instead of 괜찮은데.
    """

    kj: str
    ko: str
    ja: str
    note: str = ""


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
    Predicate("우마이", "うまい", "i", "맛있네요", "맛있다", "맛있지 않아요", frozenset({FOOD})),
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
        "야사시이", "優しい", "i", "친절하네요", "친절하다", "친절하지 않아요", frozenset({PERSON})
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
    Interjection("호라", "ほら", "봐"),
    Interjection("야메로", "やめろ", "그만해"),
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
    Loanword("だめ", "다메", "안 된다는 말", "phrase"),
    Loanword("むり", "무리", "무리라는 말", "phrase"),
)

KOREAN_INTENSIFIERS_IN_KATAKANA: tuple[KoreanInKatakana, ...] = (
    KoreanInKatakana("チンチャ", "진짜", "ほんとに"),
    KoreanInKatakana("テバク", "대박", "やばくて"),
    KoreanInKatakana("アイゴー", "아이고", "あーあ"),
    KoreanInKatakana("ケンチャナ", "괜찮아", "大丈夫、"),
    KoreanInKatakana("ワンジョン", "완전", "めっちゃ"),
    KoreanInKatakana("ノム", "너무", "すごく"),
)

KOREAN_NOUNS_IN_KATAKANA: tuple[KoreanInKatakana, ...] = (
    KoreanInKatakana("オッパ", "오빠", "お兄さん"),
    KoreanInKatakana("オンニ", "언니", "お姉さん"),
    KoreanInKatakana("ヌナ", "누나", "お姉さん"),
    KoreanInKatakana("アジョシ", "아저씨", "おじさん"),
    KoreanInKatakana("チング", "친구", "友達"),
    KoreanInKatakana("ソンベ", "선배", "先輩"),
    KoreanInKatakana("フベ", "후배", "後輩"),
    KoreanInKatakana("サジャン", "사장님", "社長"),
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

# Korean content nouns that keep their Hangul spelling inside a Japanese frame.
# Used with transliterated Japanese particles: 친구노 마음와 타이헨데스네.
KOREAN_CONTENT_NOUNS: tuple[KoreanContentNoun, ...] = (
    KoreanContentNoun("친구", "友達", PERSON),
    KoreanContentNoun("선배", "先輩", PERSON),
    KoreanContentNoun("후배", "後輩", PERSON),
    KoreanContentNoun("동생", "弟", PERSON),
    KoreanContentNoun("가족", "家族", PERSON),
    KoreanContentNoun("고양이", "猫", ANIMAL),
    KoreanContentNoun("강아지", "子犬", ANIMAL),
    KoreanContentNoun("마음", "気持ち", MIND),
    KoreanContentNoun("추억", "思い出", MIND),
    KoreanContentNoun("목소리", "声", PERCEPT),
    KoreanContentNoun("노래", "歌", MEDIA),
    KoreanContentNoun("영화", "映画", MEDIA),
    KoreanContentNoun("사진", "写真", ARTIFACT),
    KoreanContentNoun("숙제", "宿題", TASK),
    KoreanContentNoun("여름", "夏", TIME),
    KoreanContentNoun("겨울", "冬", TIME),
    KoreanContentNoun("아침", "朝", TIME),
    KoreanContentNoun("날씨", "天気", WEATHER),
    KoreanContentNoun("학교", "学校", PLACE),
    KoreanContentNoun("가게", "店", PLACE),
)

# Korean endings that attach to a transliterated Japanese predicate stem. Only
# the two whose Korean realization is a plain paradigm form are composed; the
# rest (~ㄴ데, ~함, ~하자) need the Korean stem rather than the dictionary form,
# so they live in BLENDS instead of being derived wrongly.
KOREAN_ENDINGS: tuple[KoreanEnding, ...] = (
    # ~하다 keeps the bare Japanese stem, which is a complete adjective only for
    # the い class: 早い is a sentence, 有名 is not (有名だ is).
    KoreanEnding("하다", "ハダ", "{ko_plain}", "{ja}", frozenset({"i"})),
    # ~하네 maps to ~ね, which drops the な of a な-adjective: 大変ね, 楽しいね.
    KoreanEnding("하네", "ハネ", "{ko_polite}", "{ja}ね", frozenset({"i", "na"})),
)

# Hand-verified blends. Composition is not attempted here for two reasons: the
# Japanese side is frequently lexical (대박 + い is やばい, not 大当たりい), and the
# Korean side needs a stem rather than an already-inflected surface, so
# concatenating an ending to 괜찮아 or 맛있어 produces 괜찮아인데 and 맛있어겠지.
# Every row below was written out and checked by hand. Entries marked (brief)
# came from the project brief verbatim.
BLENDS: tuple[Blend, ...] = (
    # --- Korean stem + Japanese ending, from the brief -------------------
    Blend("아랏소데스", "알겠습니다", "分かりました", "알았어 + です (brief)"),
    Blend("키요이", "귀여워", "かわいい", "귀요 + い (brief)"),
    Blend("마지코마워", "정말 고마워", "まじありがとう", "まじ + 고마워 (brief)"),
    Blend("체고카요", "최고냐고", "最高かよ", "최고 + かよ (brief)"),
    Blend("친챠소레나", "진짜 그거지", "ほんとそれな", "진짜 + それな (brief)"),
    Blend("테바이", "대박임", "やばい", "대박 + い (brief)"),
    Blend("やばいンデ", "굉장한데", "やばいけど", "やばい + ~ㄴ데 (brief)"),
    Blend("부라더 다메요", "형 안 돼", "ブラザーだめよ", "brother + だめよ (brief)"),
    Blend("아나타와 햄스터데스까", "당신은 햄스터입니까", "あなたはハムスターですか", "(brief)"),
    Blend(
        "호라 모 젠젠 멀쩡하자나", "봐 이제 전혀 멀쩡하잖아", "ほらもう全然無事じゃん", "(brief)"
    ),
    Blend(
        "야메로 이런 싸움은 모 야메룽다",
        "그만해 이런 싸움은 이제 그만하는 거다",
        "やめろこんな喧嘩はもうやめるんだ",
        "(brief)",
    ),
    # --- Korean noun + Japanese copula ----------------------------------
    Blend("체고데스", "최고입니다", "最高です", "최고 + です"),
    Blend("체고데스네", "최고네요", "最高ですね", "최고 + ですね"),
    Blend("체고쟝", "최고잖아", "最高じゃん", "최고 + じゃん"),
    Blend("체고자나이카", "최고 아니야", "最高じゃないか", "최고 + じゃないか"),
    Blend("체고다요", "최고야", "最高だよ", "최고 + だよ"),
    Blend("체고데쇼", "최고겠지", "最高でしょ", "최고 + でしょ"),
    Blend("테바데스네", "대박이네요", "やばいですね", "대박 + ですね"),
    Blend("테바쟝", "대박이잖아", "やばいじゃん", "대박 + じゃん"),
    Blend("테바다요", "대박이야", "やばいよ", "대박 + だよ"),
    Blend("친챠데스카", "진짜입니까", "ほんとですか", "진짜 + ですか"),
    Blend("친챠다요", "진짜야", "ほんとだよ", "진짜 + だよ"),
    Blend("친챠카요", "진짜냐고", "ほんとかよ", "진짜 + かよ"),
    # --- Korean predicate stem + Japanese ending -------------------------
    Blend("키요이데스네", "귀엽네요", "かわいいですね", "귀요 + いですね"),
    Blend("키요이쟝", "귀엽잖아", "かわいいじゃん", "귀요 + いじゃん"),
    Blend("키요이카요", "귀엽냐고", "かわいいかよ", "귀요 + いかよ"),
    Blend("마시쏘데스네", "맛있네요", "おいしいですね", "맛있어 + ですね"),
    Blend("마시쏘쟝", "맛있잖아", "おいしいじゃん", "맛있어 + じゃん"),
    Blend("머시쏘데스네", "멋있네요", "かっこいいですね", "멋있어 + ですね"),
    Blend("켄챤아데스", "괜찮습니다", "大丈夫です", "괜찮아 + です"),
    Blend("켄챤아데스카", "괜찮습니까", "大丈夫ですか", "괜찮아 + ですか"),
    Blend("피곤헤데스", "피곤합니다", "疲れました", "피곤해 + です"),
    Blend("페고파데스", "배고픕니다", "お腹すきました", "배고파 + です"),
    Blend("사랑헤요", "사랑해요", "愛してるよ", "사랑해 + よ"),
    Blend("미안헤데스", "미안합니다", "ごめんなさい", "미안해 + です"),
    Blend("코마워데스", "고맙습니다", "ありがとうございます", "고마워 + です"),
    # --- Japanese stem + Korean ending -----------------------------------
    Blend("카와이하다", "귀엽다", "かわいい", "かわいい + ~하다"),
    Blend("카와이함", "귀여움", "かわいさ", "かわいい + ~함"),
    Blend("카와이한데", "귀여운데", "かわいいけど", "かわいい + ~ㄴ데"),
    Blend("스고이하다", "대단하다", "すごい", "すごい + ~하다"),
    Blend("스고이함", "대단함", "すごさ", "すごい + ~함"),
    Blend("야바이함", "위험함", "やばさ", "やばい + ~함"),
    Blend("야바이한데", "위험한데", "やばいけど", "やばい + ~ㄴ데"),
    Blend("우마이하다", "맛있다", "うまい", "うまい + ~하다"),
    Blend("우루사이하다", "시끄럽다", "うるさい", "うるさい + ~하다"),
    Blend("젠젠 다메핫소", "전혀 안 됐어", "全然だめだった", "ぜんぜん + だめ + 했어"),
    Blend("무리핫소", "무리했어", "むりした", "むり + 했어"),
    Blend("간바레핫소", "힘냈어", "がんばった", "がんばれ + 했어"),
    # --- Japanese adverb + Korean predicate -------------------------------
    Blend("마지 미안헤", "정말 미안해", "まじごめん", "まじ + 미안해"),
    Blend("마지 체고", "정말 최고", "まじ最高", "まじ + 최고"),
    Blend("모 무리야", "이제 무리야", "もうむりだよ", "もう + 무리야"),
    Blend("젠젠 켄챤아", "전혀 괜찮아", "全然大丈夫", "ぜんぜん + 괜찮아"),
    Blend("얏파리 체고", "역시 최고", "やっぱり最高", "やっぱり + 최고"),
    Blend("솟치 나이데스", "그쪽 아닙니다", "そっちじゃないです", "そっち + 아니 + です"),
    # --- Set phrases -------------------------------------------------------
    Blend("오츠카레삼", "수고하셨어요", "おつかれさま", "おつかれ + ~삼"),
    Blend("스미마셍이요", "죄송해요", "すみません", "すみません + ~이요"),
    Blend("이타다키마스네", "잘 먹겠습니다", "いただきますね", "いただきます + ね"),
    Blend("다이죠부데스요", "괜찮아요", "大丈夫ですよ", "大丈夫 + ですよ"),
    Blend("소레나데스", "그거지요", "それなです", "それな + です"),
)

__all__ = [
    "ANIMAL",
    "ARTIFACT",
    "BODY",
    "CELESTIAL",
    "FOOD",
    "GENITIVE_HEADS",
    "INTERJECTIONS",
    "BLENDS",
    "KOREAN_CONTENT_NOUNS",
    "KOREAN_ENDINGS",
    "KOREAN_FOOD_IN_KATAKANA",
    "KOREAN_INTENSIFIERS_IN_KATAKANA",
    "KOREAN_NOUNS_IN_KATAKANA",
    "LANDSCAPE",
    "LOANWORDS",
    "MEDIA",
    "MIND",
    "NOUNS",
    "PERCEPT",
    "PERSON",
    "PLACE",
    "PLANT",
    "PRECIP",
    "PREDICATES",
    "REACTIONS",
    "TASK",
    "TIME",
    "WEATHER",
    "WIND",
    "Blend",
    "Interjection",
    "KoreanContentNoun",
    "KoreanEnding",
    "KoreanInKatakana",
    "Loanword",
    "Noun",
    "Predicate",
    "Reaction",
]
