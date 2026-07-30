"""Paired sentence frames for the domains the corpus has no rows for.

법률 has 100 rows, 외교 331, and 전자상거래 / IT 기술문서 / 행정 민원 /
관광·교통 have none at all. Those are also the domains whose real text is
genuinely formulaic: a terms-of-service clause, a shipping notice and a dosage
instruction are written to a pattern, so a template matches the real distribution
here in a way it never does for 문학 or 방언.

That licence is narrow and easy to abuse, so the design fights template collapse
directly:

* Every frame carries at least **two word slots**, not only numbers. The audit's
  skeleton blanks digits, so a frame varying only in digits collapses to a single
  skeleton no matter how many rows it produces - exactly the defect that cost
  data44/data45 90% of their rows.
* Slot values are **paired**: one table holds the Korean and Japanese together,
  so a filled frame is a translation rather than two independently generated
  sentences.
* Currency stays a unit, not a conversion. ``원`` maps to ``ウォン`` and never to
  ``円``: rewriting the currency would be a mistranslation, and the preservation
  check would flag it (data23 had exactly this defect, ``원`` rendered ``銭``).

Nothing here is imported by the translation stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

try:  # optional dependency, same pattern as build_hanboneo
    from hangulpy import josa as _hangulpy_josa
except Exception:  # pragma: no cover - exercised by the fallback tests
    _hangulpy_josa = None

# Paired slot values. Keyed by slot name; each entry is (Korean, Japanese).
SLOTS: dict[str, tuple[tuple[str, str], ...]] = {
    # --- legal / terms of service ---
    "party": (
        ("회사", "当社"),
        ("당사", "当社"),
        ("서비스 제공자", "サービス提供者"),
        ("운영자", "運営者"),
        ("사업자", "事業者"),
        ("서비스 운영사", "サービス運営会社"),
        ("플랫폼 제공자", "プラットフォーム提供者"),
    ),
    "counterparty": (
        ("이용자", "利用者"),
        ("회원", "会員"),
        ("고객", "お客様"),
        ("가입자", "加入者"),
        ("구매자", "購入者"),
        ("수신자", "受信者"),
    ),
    "legal_document": (
        ("약관", "規約"),
        ("이용약관", "利用規約"),
        ("개인정보처리방침", "プライバシーポリシー"),
        ("계약", "契約"),
        ("운영정책", "運営ポリシー"),
        ("서비스 규정", "サービス規定"),
        ("환불 정책", "返金ポリシー"),
    ),
    "legal_action": (
        ("해지", "解約"),
        ("변경", "変更"),
        ("정지", "停止"),
        ("삭제", "削除"),
        ("열람", "閲覧"),
        ("정정", "訂正"),
        ("이용 제한", "利用制限"),
        ("계정 복구", "アカウント復旧"),
    ),
    "legal_purpose": (
        ("서비스 제공", "サービス提供"),
        ("본인 확인", "本人確認"),
        ("요금 정산", "料金精算"),
        ("고객 상담", "顧客対応"),
        ("부정 이용 방지", "不正利用防止"),
        ("통계 분석", "統計分析"),
        ("배송 처리", "配送処理"),
    ),
    # --- e-commerce ---
    "item": (
        ("상품", "商品"),
        ("제품", "製品"),
        ("도서", "書籍"),
        ("의류", "衣類"),
        ("가전제품", "家電製品"),
        ("식품", "食品"),
        ("화장품", "化粧品"),
        ("생활용품", "生活用品"),
        ("완구", "玩具"),
    ),
    "fulfilment": (
        ("배송", "配送"),
        ("발송", "発送"),
        ("출고", "出荷"),
        ("반품", "返品"),
        ("교환", "交換"),
        ("환불", "返金"),
    ),
    "payment_method": (
        ("신용카드", "クレジットカード"),
        ("계좌이체", "銀行振込"),
        ("간편결제", "スマホ決済"),
        ("휴대폰 결제", "携帯決済"),
        ("포인트", "ポイント"),
        ("상품권", "商品券"),
        ("가상계좌", "バーチャル口座"),
    ),
    "order_status": (
        ("준비 중", "準備中"),
        ("완료", "完了"),
        ("지연", "遅延"),
        ("취소", "キャンセル"),
        ("보류", "保留"),
        ("검수 중", "検品中"),
        ("반송", "返送"),
    ),
    # --- IT technical documentation ---
    "component": (
        ("서버", "サーバー"),
        ("데이터베이스", "データベース"),
        ("캐시", "キャッシュ"),
        ("인증 토큰", "認証トークン"),
        ("세션", "セッション"),
        ("로그 파일", "ログファイル"),
        ("메시지 큐", "メッセージキュー"),
        ("부하 분산기", "ロードバランサー"),
        ("저장소", "ストレージ"),
    ),
    "it_error": (
        ("시간이 초과되었습니다", "タイムアウトしました"),
        ("찾을 수 없습니다", "見つかりません"),
        ("권한이 없습니다", "権限がありません"),
        ("형식이 올바르지 않습니다", "形式が正しくありません"),
        ("이미 사용 중입니다", "すでに使用中です"),
    ),
    "it_action": (
        ("재시도", "再試行"),
        ("초기화", "初期化"),
        ("갱신", "更新"),
        ("배포", "デプロイ"),
        ("롤백", "ロールバック"),
        ("마이그레이션", "マイグレーション"),
        ("재기동", "再起動"),
        ("동기화", "同期"),
    ),
    "it_object": (
        ("설정 파일", "設定ファイル"),
        ("환경 변수", "環境変数"),
        ("접근 권한", "アクセス権限"),
        ("인덱스", "インデックス"),
        ("스키마", "スキーマ"),
        ("인증서", "証明書"),
        ("접속 정보", "接続情報"),
    ),
    # --- medical dosage guidance ---
    "dose_form": (
        ("정", "錠"),
        ("캡슐", "カプセル"),
        ("포", "包"),
        ("방울", "滴"),
        ("매", "枚"),
    ),
    "dose_timing": (
        ("식전", "食前"),
        ("식후", "食後"),
        ("취침 전", "就寝前"),
        ("공복 시", "空腹時"),
        ("아침", "朝"),
        ("저녁", "夕方"),
    ),
    "symptom": (
        ("발열", "発熱"),
        ("두통", "頭痛"),
        ("어지럼증", "めまい"),
        ("발진", "発疹"),
        ("구토", "嘔吐"),
        ("설사", "下痢"),
    ),
    "clinician": (
        ("의사", "医師"),
        ("약사", "薬剤師"),
        ("담당 의사", "担当医"),
        ("간호사", "看護師"),
    ),
    # --- tourism and transport ---
    "transport": (
        ("지하철", "地下鉄"),
        ("시내버스", "路線バス"),
        ("열차", "列車"),
        ("공항버스", "リムジンバス"),
        ("택시", "タクシー"),
        ("페리", "フェリー"),
        ("셔틀버스", "シャトルバス"),
        ("고속버스", "高速バス"),
    ),
    "facility": (
        ("매표소", "切符売り場"),
        ("개찰구", "改札口"),
        ("대합실", "待合室"),
        ("관광안내소", "観光案内所"),
        ("물품보관함", "コインロッカー"),
        ("환전소", "両替所"),
        ("승강장", "のりば"),
    ),
    "attraction": (
        ("전망대", "展望台"),
        ("박물관", "博物館"),
        ("온천", "温泉"),
        ("수족관", "水族館"),
        ("전통시장", "伝統市場"),
        ("미술관", "美術館"),
        ("식물원", "植物園"),
        ("성곽", "城跡"),
    ),
    # --- public administration ---
    "admin_document": (
        ("주민등록등본", "住民票"),
        ("인감증명서", "印鑑証明書"),
        ("납세증명서", "納税証明書"),
        ("사업자등록증", "事業者登録証"),
        ("가족관계증명서", "戸籍謄本"),
        ("건강보험증", "健康保険証"),
        ("소득증명서", "所得証明書"),
    ),
    "admin_office": (
        ("시청", "市役所"),
        ("구청", "区役所"),
        ("주민센터", "住民センター"),
        ("세무서", "税務署"),
        ("행정복지센터", "行政福祉センター"),
        ("출입국관리소", "入国管理局"),
    ),
    # Only the facilities that actually take a booking. Free combination put
    # `단체 2명 이상은 개찰구에서 예약해야 합니다` in the corpus otherwise.
    "booking_facility": (
        ("관광안내소", "観光案内所"),
        ("매표소", "切符売り場"),
        ("안내데스크", "案内デスク"),
    ),
    "admin_channel": (
        ("방문", "窓口"),
        ("온라인", "オンライン"),
        ("우편", "郵送"),
        ("무인발급기", "自動交付機"),
        ("모바일 앱", "モバイルアプリ"),
    ),
    # --- numeric slots. Written identically on both sides so the preservation
    # check can compare them, which is the point of keeping them explicit.
    "count": (("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")),
    "group_size": (("10", "10"), ("15", "15"), ("20", "20"), ("30", "30")),
    "dose_count": (("1", "1"), ("2", "2"), ("3", "3")),
    "days": (("3", "3"), ("7", "7"), ("14", "14"), ("30", "30")),
    "percent": (("5", "5"), ("10", "10"), ("15", "15"), ("20", "20")),
    "minutes": (("5", "5"), ("10", "10"), ("15", "15"), ("20", "20")),
    # Split by half of the day: `오전 18시` is a contradiction, and a frame
    # that says 오전 must not be handed an evening hour.
    "hour_am": (("8", "8"), ("9", "9"), ("10", "10"), ("11", "11")),
    "hour_late": (("21", "21"), ("22", "22"), ("23", "23")),
    "price": (("3,000", "3,000"), ("5,000", "5,000"), ("12,000", "12,000")),
    # No B1: the Latin letter trips the target-script check, and 지하 1층 is
    # the form both languages actually use anyway.
    "floor": (("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")),
    "major": (("2", "2"), ("3", "3"), ("4", "4")),
    "minor": (("0", "0"), ("1", "1"), ("2", "2")),
}


# Korean particle alternations, written in frames as ``은/는``: the form used
# after a final consonant first. Filling a slot changes which form is correct, so
# the frames cannot hardcode one - `부정 이용 방지을` and `회원가` were both
# produced that way before this existed.
PARTICLE_ALTERNATIONS: tuple[str, ...] = (
    "은/는",
    "을/를",
    "이/가",
    "과/와",
    "으로/로",
    "이라/라",
    "이나/나",
)

_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3


def has_final_consonant(word: str) -> bool:
    """True when the last Hangul syllable of ``word`` carries a final consonant."""

    for char in reversed(word):
        codepoint = ord(char)
        if _HANGUL_START <= codepoint <= _HANGUL_END:
            return (codepoint - _HANGUL_START) % 28 != 0
    return False


def attach_particle(word: str, pair: str) -> str:
    """Attach the correct member of ``pair`` to ``word``."""

    if _hangulpy_josa is not None:
        try:
            return str(_hangulpy_josa(word, pair))
        except Exception:  # pragma: no cover - defer to the arithmetic fallback
            pass
    with_final, without_final = pair.split("/")
    return word + (with_final if has_final_consonant(word) else without_final)


def resolve_particles(text: str) -> str:
    """Pick the right member of every particle alternation left in ``text``."""

    for pair in PARTICLE_ALTERNATIONS:
        with_final, without_final = pair.split("/")
        pattern = re.compile(r"([가-힣])" + re.escape(pair))
        text = pattern.sub(
            lambda match, w=with_final, o=without_final: (
                match.group(1) + (w if has_final_consonant(match.group(1)) else o)
            ),
            text,
        )
    return text


@dataclass(frozen=True)
class Frame:
    """One paired sentence template. ``{slot}`` names index :data:`SLOTS`."""

    ko: str
    ja: str

    def slots(self) -> tuple[str, ...]:
        found = _PLACEHOLDER.findall(self.ko)
        return tuple(dict.fromkeys(found))

    def word_slots(self) -> tuple[str, ...]:
        return tuple(name for name in self.slots() if name not in NUMERIC_SLOTS)


NUMERIC_SLOTS = frozenset(
    {
        "count",
        "group_size",
        "dose_count",
        "days",
        "percent",
        "minutes",
        "hour_am",
        "hour_late",
        "price",
        "floor",
        "major",
        "minor",
    }
)

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
# A slot followed immediately by a bare particle, i.e. not an alternation.
_HARDCODED_PARTICLE = re.compile(r"\{([a-z_]+)\}([은는을를이가과와])(?!/)")


@dataclass(frozen=True)
class Domain:
    code: str
    label: str
    frames: tuple[Frame, ...]


LEGAL = Domain(
    code="legal",
    label="법률·약관",
    frames=(
        Frame(
            "{party}이/가 정한 본 {legal_document}은/는 {days}일간의 공고 기간을 거쳐 시행됩니다.",
            "{party}が定めた本{legal_document}は、{days}日間の公告期間を経て施行されます。",
        ),
        Frame(
            "{party}은/는 {legal_action}에 관한 사항을 {days}일 전까지 통지합니다.",
            "{party}は{legal_action}に関する事項を{days}日前までに通知します。",
        ),
        Frame(
            "{counterparty}이/가 {legal_document}에 동의하지 않는 경우 {legal_action}을/를 요청할 수 있습니다.",
            "{counterparty}が{legal_document}に同意しない場合、{legal_action}を請求することができます。",
        ),
        Frame(
            "{party}은/는 {legal_purpose} 목적으로만 개인정보를 이용하며, 목적 달성 후 지체 없이 파기합니다.",
            "{party}は{legal_purpose}の目的でのみ個人情報を利用し、目的達成後は遅滞なく破棄します。",
        ),
        Frame(
            "{party}의 고의 또는 중대한 과실이 없는 경우 {counterparty}에 대한 손해배상 책임을 지지 않습니다.",
            "{party}の故意または重大な過失がない場合、{counterparty}に対する損害賠償責任を負いません。",
        ),
        Frame(
            "{counterparty}은/는 {legal_document}에 따라 {legal_action}을/를 신청할 수 있으며, 처리 결과는 {days}일 이내에 통보됩니다.",
            "{counterparty}は{legal_document}に基づき{legal_action}を申請でき、処理結果は{days}日以内に通知されます。",
        ),
        Frame(
            "{party}은/는 {legal_purpose}을/를 위하여 수집한 정보를 제삼자에게 제공하지 않습니다.",
            "{party}は{legal_purpose}のために収集した情報を第三者に提供しません。",
        ),
        Frame(
            "{party}이/가 제시한 본 {legal_document}의 일부 조항이 무효로 판단되더라도 나머지 조항의 효력은 유지됩니다.",
            "{party}が提示した本{legal_document}の一部の条項が無効と判断されても、残りの条項の効力は維持されます。",
        ),
        Frame(
            "{counterparty}의 귀책사유로 {legal_action}이/가 이루어진 경우 이미 납부한 요금은 반환되지 않습니다.",
            "{counterparty}の責めに帰すべき事由により{legal_action}が行われた場合、既に納付した料金は返還されません。",
        ),
        Frame(
            "{party}과/와 {counterparty} 간 분쟁은 관할 법원의 판결에 따라 해결합니다.",
            "{party}と{counterparty}との紛争は、管轄裁判所の判決に従って解決します。",
        ),
    ),
)

COMMERCE = Domain(
    code="commerce",
    label="전자상거래",
    frames=(
        Frame(
            "주문하신 {item}은/는 결제 확인 후 {days}일 이내에 {fulfilment}됩니다.",
            "ご注文の{item}は決済確認後、{days}日以内に{fulfilment}されます。",
        ),
        Frame(
            "{item}을/를 {payment_method}로 결제하시면 {percent}% 할인이 적용됩니다.",
            "{item}を{payment_method}でお支払いいただくと、{percent}%の割引が適用されます。",
        ),
        Frame(
            "{item}의 {fulfilment} 신청은 수령 후 {days}일 이내에만 가능합니다.",
            "{item}の{fulfilment}のお申し込みは、受け取り後{days}日以内のみ可能です。",
        ),
        Frame(
            "재고가 부족하여 {item}의 {fulfilment}이/가 {days}일 정도 지연될 수 있습니다.",
            "在庫が不足しているため、{item}の{fulfilment}が{days}日ほど遅れる場合があります。",
        ),
        Frame(
            "{item}에 하자가 있는 경우 {fulfilment} 비용은 판매자가 부담합니다.",
            "{item}に欠陥がある場合、{fulfilment}にかかる費用は販売者が負担します。",
        ),
        Frame(
            "현재 주문 상태는 {order_status}이/가며, {payment_method} 결제가 확인되었습니다.",
            "現在のご注文状況は{order_status}で、{payment_method}での決済が確認されました。",
        ),
        Frame(
            "{item}을/를 {price}원 이상 구매하시면 {fulfilment} 비용이 무료입니다.",
            "{item}を{price}ウォン以上ご購入の場合、{fulfilment}費用が無料になります。",
        ),
        Frame(
            "{item} 주문이 {order_status} 상태로 변경되었습니다. 자세한 내용은 주문 내역에서 확인해 주십시오.",
            "{item}のご注文が{order_status}に変更されました。詳細はご注文履歴でご確認ください。",
        ),
        Frame(
            "{item} 구매 시 {payment_method} 결제는 최대 {count}회까지 무이자 할부가 가능합니다.",
            "{item}のご購入時、{payment_method}でのお支払いは最大{count}回まで無金利分割が可能です。",
        ),
        Frame(
            "{fulfilment} 처리가 완료되면 등록된 연락처로 {order_status} 안내를 발송합니다.",
            "{fulfilment}の処理が完了しましたら、ご登録の連絡先に{order_status}のご案内をお送りします。",
        ),
    ),
)

TECHNICAL = Domain(
    code="technical",
    label="IT 기술문서",
    frames=(
        Frame(
            "{component}에 연결하는 동안 {it_error}",
            "{component}への接続中に{it_error}",
        ),
        Frame(
            "{it_object}을/를 확인하십시오. {component}의 {it_error}",
            "{it_object}を確認してください。{component}の{it_error}",
        ),
        Frame(
            "{it_action}을/를 수행하기 전에 {it_object}의 백업을 생성하십시오.",
            "{it_action}を実行する前に、{it_object}のバックアップを作成してください。",
        ),
        Frame(
            "이 기능은 버전 {major}.{minor} 이상에서 지원되며, {it_object}을/를 수정한 뒤 {component}을/를 재시작해야 적용됩니다.",
            "この機能はバージョン{major}.{minor}以降でサポートされ、{it_object}を修正した後に{component}の再起動が必要です。",
        ),
        Frame(
            "{component}의 {it_action}이/가 {minutes}분 이내에 완료되지 않으면 자동으로 중단됩니다.",
            "{component}の{it_action}が{minutes}分以内に完了しない場合、自動的に中断されます。",
        ),
        Frame(
            "{it_object}에 잘못된 값이 있어 {it_action}을/를 진행할 수 없습니다.",
            "{it_object}に不正な値があるため、{it_action}を続行できません。",
        ),
        Frame(
            "요청이 실패했습니다. {component}의 상태를 확인한 뒤 {it_action}을/를 다시 시도하십시오.",
            "リクエストが失敗しました。{component}の状態を確認してから{it_action}を再度お試しください。",
        ),
        Frame(
            "{component}의 {it_action} 로그는 {days}일간 보관되며, 이후 자동으로 삭제됩니다.",
            "{component}の{it_action}ログは{days}日間保管され、その後自動的に削除されます。",
        ),
        Frame(
            "{component}과/와 {it_object}의 버전이 일치하지 않습니다. {it_action}을/를 권장합니다.",
            "{component}と{it_object}のバージョンが一致しません。{it_action}を推奨します。",
        ),
        Frame(
            "하나의 {it_object}에 최대 {count}개의 {component}을/를 등록할 수 있습니다.",
            "一つの{it_object}に最大{count}台の{component}を登録できます。",
        ),
    ),
)

MEDICAL = Domain(
    code="medical",
    label="의료 복약지도",
    frames=(
        Frame(
            "이 약은 매일 {dose_timing}에 {dose_count}{dose_form} 복용하십시오.",
            "この薬は毎日{dose_timing}に{dose_count}{dose_form}服用してください。",
        ),
        Frame(
            "{symptom}이/가 나타나면 복용을 중단하고 {clinician}과/와 상담하십시오.",
            "{symptom}が現れた場合は服用を中止し、{clinician}に相談してください。",
        ),
        Frame(
            "다른 약과 함께 복용하면 {symptom}이/가 나타날 수 있으므로 반드시 {clinician}에게 알리십시오.",
            "他の薬と併用すると{symptom}が現れる場合があるため、必ず{clinician}にお知らせください。",
        ),
        Frame(
            "{dose_timing}에 복용하는 것이 어려우면 {clinician}과/와 복용 시간을 조정하십시오.",
            "{dose_timing}の服用が難しい場合は、{clinician}と服用時間を調整してください。",
        ),
        Frame(
            "복용 후 {symptom} 증상이 지속되면 즉시 {clinician}의 진료를 받으십시오.",
            "服用後に{symptom}の症状が続く場合は、直ちに{clinician}の診察を受けてください。",
        ),
        Frame(
            "이 약은 개봉 후 {days}일 이내에 사용하고, 남은 {dose_form}은/는 {clinician}에게 문의하십시오.",
            "この薬は開封後{days}日以内に使用し、残った{dose_form}は{clinician}にお問い合わせください。",
        ),
        Frame(
            "{count}{dose_form}을/를 초과하여 복용하지 마십시오. {symptom}의 위험이 있습니다.",
            "{count}{dose_form}を超えて服用しないでください。{symptom}の危険があります。",
        ),
        Frame(
            "{dose_timing}에 {dose_form}을/를 복용할 때는 물과 함께 삼키고, 씹거나 부수지 마십시오.",
            "{dose_timing}に{dose_form}を服用する際は、水と一緒に飲み込み、噛んだり砕いたりしないでください。",
        ),
    ),
)

TRAVEL = Domain(
    code="travel",
    label="관광·교통",
    frames=(
        Frame(
            "{transport}은/는 {minutes}분 간격으로 운행하며, 막차는 {facility} 앞에서 {hour_late}시에 출발합니다.",
            "{transport}は{minutes}分間隔で運行し、最終便は{facility}前を{hour_late}時に出発します。",
        ),
        Frame(
            "{facility}은/는 {attraction} 옆 {floor}층에 있으며, 이용 시간은 오전 {hour_am}시부터입니다.",
            "{facility}は{attraction}の隣の{floor}階にあり、ご利用時間は午前{hour_am}時からです。",
        ),
        Frame(
            "{attraction} 입장료는 성인 {price}원이며, {transport}로 약 {minutes}분 걸립니다.",
            "{attraction}の入場料は大人{price}ウォンで、{transport}で約{minutes}分かかります。",
        ),
        Frame(
            "{facility}에서 {attraction}까지 도보로 {minutes}분 거리입니다.",
            "{facility}から{attraction}まで徒歩{minutes}分の距離です。",
        ),
        Frame(
            "{transport} 승차권은 {facility}에서 구입하실 수 있습니다.",
            "{transport}の乗車券は{facility}でご購入いただけます。",
        ),
        Frame(
            "{attraction}은/는 매주 월요일 휴관하며, 단체 {group_size}명 이상은 {booking_facility}에서 예약해야 합니다.",
            "{attraction}は毎週月曜日が休館日で、団体{group_size}名以上は{booking_facility}での予約が必要です。",
        ),
        Frame(
            "{transport} 지연이 발생한 경우 {facility}의 직원에게 문의해 주십시오.",
            "{transport}の遅延が発生した場合は、{facility}の係員にお問い合わせください。",
        ),
        Frame(
            "{attraction} 관람 소요 시간은 약 {minutes}분이며, {facility} 이용료는 별도입니다.",
            "{attraction}の見学所要時間は約{minutes}分で、{facility}の利用料は別途です。",
        ),
    ),
)

ADMIN = Domain(
    code="administration",
    label="행정 민원",
    frames=(
        Frame(
            "{admin_document} 발급은 {admin_office}에서 {admin_channel}으로/로 신청할 수 있습니다.",
            "{admin_document}の発行は、{admin_office}にて{admin_channel}で申請できます。",
        ),
        Frame(
            "{admin_channel} 신청 시 {admin_document} 수수료는 {price}원입니다.",
            "{admin_channel}での申請の場合、{admin_document}の手数料は{price}ウォンです。",
        ),
        Frame(
            "{admin_document} 처리 기간은 접수일로부터 {days}일이며, {admin_office}에서 수령합니다.",
            "{admin_document}の処理期間は受付日から{days}日で、{admin_office}で受け取ります。",
        ),
        Frame(
            "{admin_office} 업무 시간은 오전 {hour_am}시부터이며, {admin_channel} 접수는 상시 가능합니다.",
            "{admin_office}の業務時間は午前{hour_am}時からで、{admin_channel}での受付は常時可能です。",
        ),
        Frame(
            "{admin_document}을/를 {admin_channel}으로/로 신청하는 경우 본인 확인 절차가 추가됩니다.",
            "{admin_document}を{admin_channel}で申請する場合、本人確認の手続きが追加されます。",
        ),
        Frame(
            "{admin_office}에 제출한 {admin_document}에 오류가 있으면 {days}일 이내에 보완을 요청합니다.",
            "{admin_office}に提出された{admin_document}に不備がある場合、{days}日以内に補正を求めます。",
        ),
        Frame(
            "{admin_channel}으로/로 제출하는 {admin_document}은/는 최근 {days}일 이내에 발급된 것만 유효합니다.",
            "{admin_channel}で提出する{admin_document}は、直近{days}日以内に発行されたものに限り有効です。",
        ),
        Frame(
            "{admin_channel} 신청 결과는 {admin_office}에 등록된 연락처로 안내됩니다.",
            "{admin_channel}での申請結果は、{admin_office}に登録された連絡先にご案内します。",
        ),
    ),
)

DOMAINS: tuple[Domain, ...] = (LEGAL, COMMERCE, TECHNICAL, MEDICAL, TRAVEL, ADMIN)


def known_domains() -> tuple[str, ...]:
    return tuple(domain.code for domain in DOMAINS)


def domain(code: str) -> Domain | None:
    for candidate in DOMAINS:
        if candidate.code == code:
            return candidate
    return None


def slot_values(name: str) -> tuple[tuple[str, str], ...]:
    return SLOTS.get(name, ())


def fill(frame: Frame, assignment: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Substitute a slot assignment into both sides of ``frame``."""

    ko, ja = frame.ko, frame.ja
    for name, (ko_value, ja_value) in assignment.items():
        ko = ko.replace("{" + name + "}", ko_value)
        ja = ja.replace("{" + name + "}", ja_value)
    # Which particle is correct depends on the value just substituted, so this
    # has to run after filling rather than being baked into the frame.
    return resolve_particles(ko), ja


def unresolved_slots(text: str) -> tuple[str, ...]:
    """Placeholders still present after filling. Any result means a bad frame."""

    return tuple(_PLACEHOLDER.findall(text))


def validate() -> tuple[str, ...]:
    """Structural problems in the tables. An empty result means they are sound."""

    problems: list[str] = []
    seen_codes: set[str] = set()
    for spec in DOMAINS:
        if spec.code in seen_codes:
            problems.append(f"{spec.code}: duplicate domain code")
        seen_codes.add(spec.code)
        if not spec.frames:
            problems.append(f"{spec.code}: no frames")
        for index, frame in enumerate(spec.frames):
            ko_slots = sorted(_PLACEHOLDER.findall(frame.ko))
            ja_slots = sorted(_PLACEHOLDER.findall(frame.ja))
            if ko_slots != ja_slots:
                problems.append(f"{spec.code}[{index}]: slot mismatch ko={ko_slots} ja={ja_slots}")
            for name in set(ko_slots):
                if name not in SLOTS:
                    problems.append(f"{spec.code}[{index}]: unknown slot {name!r}")
            # A particle glued straight to a slot, and not written as an
            # alternation, is wrong for at least some values of that slot. The
            # negative lookahead is what lets `{party}은/는` through.
            for match in _HARDCODED_PARTICLE.finditer(frame.ko):
                problems.append(
                    f"{spec.code}[{index}]: hardcoded particle "
                    f"{match.group(2)!r} after {{{match.group(1)}}}; "
                    "use an alternation like 은/는"
                )
            if len(frame.word_slots()) < 2:
                # A frame varying only in digits collapses to one skeleton.
                problems.append(
                    f"{spec.code}[{index}]: needs two word slots, has {frame.word_slots()}"
                )
    for name, values in SLOTS.items():
        if not values:
            problems.append(f"slot {name!r}: empty")
        for ko_value, ja_value in values:
            if not ko_value or not ja_value:
                problems.append(f"slot {name!r}: blank value")
    return tuple(problems)


__all__ = [
    "ADMIN",
    "COMMERCE",
    "DOMAINS",
    "LEGAL",
    "MEDICAL",
    "NUMERIC_SLOTS",
    "SLOTS",
    "TECHNICAL",
    "TRAVEL",
    "Domain",
    "Frame",
    "PARTICLE_ALTERNATIONS",
    "attach_particle",
    "domain",
    "fill",
    "has_final_consonant",
    "known_domains",
    "resolve_particles",
    "slot_values",
    "unresolved_slots",
    "validate",
]
