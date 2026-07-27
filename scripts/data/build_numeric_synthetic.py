#!/usr/bin/env python3
"""Build deterministic Korean-Japanese structured-value training pairs.

The output is deliberately named ``synthetic_numeric_data38.jsonl`` so the
training pipeline can force it into the train split and down-weight it. Every
row carries explicit synthetic/provenance metadata. Values are generated from
scratch; existing corpus sentences are never copied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Callable
import unicodedata


GENERATOR_VERSION = "sion-numeric-structured-v1"
DEFAULT_SEED = 20260727
DEFAULT_COUNT = 240_000
_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_PARENTHETICAL_PARTICLE_RE = re.compile(r"\([으이는를가을]\)(?:로|를|가|는)?")


def canonical_text(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def pair_key(ko: str, ja: str) -> str:
    return f"{canonical_text(ko)}\0{canonical_text(ja)}"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Generated:
    ko: str
    ja: str
    values: tuple[str, ...]
    template_id: str


Generator = Callable[[random.Random, int], Generated]


def _pick_template(
    rng: random.Random,
    templates: tuple[tuple[str, str], ...],
    values: dict[str, str],
    template_prefix: str,
) -> Generated:
    template_index = rng.randrange(len(templates))
    ko_template, ja_template = templates[template_index]
    ko = ko_template.format(**values)
    ja = ja_template.format(**values)
    shared_values = tuple(
        dict.fromkeys(value for value in values.values() if value and value in ko and value in ja)
    )
    return Generated(
        ko=ko,
        ja=ja,
        values=shared_values,
        template_id=f"{template_prefix}-{template_index:02d}",
    )


def _integer(rng: random.Random, minimum: int = 0, maximum: int = 999_999_999) -> int:
    return rng.randint(minimum, maximum)


def generate_cardinal(rng: random.Random, _: int) -> Generated:
    number = _integer(rng, 1, 999_999_999)
    value = f"{number:,}"
    count = f"{rng.randint(1, min(number, 99_999)):,}"
    templates = (
        ("총 수량은 {value}개입니다.", "合計数量は{value}個です。"),
        ("현재 누적 건수는 {value}건입니다.", "現在の累計件数は{value}件です。"),
        ("설정된 목표값은 {value}입니다.", "設定された目標値は{value}です。"),
        ("{value}명 중 {count}명이 응답했습니다.", "{value}人のうち{count}人が回答しました。"),
        ("재고가 {value}개 남아 있습니다.", "在庫は{value}個残っています。"),
        ("이번 회차 번호는 {value}입니다.", "今回の番号は{value}です。"),
        ("처리 대기 항목이 {value}개 있습니다.", "処理待ちの項目が{value}件あります。"),
        ("최대 {value}개까지 선택할 수 있습니다.", "最大{value}個まで選択できます。"),
        ("측정값이 기준값 {value}보다 큽니다.", "測定値が基準値{value}を超えました。"),
        ("{value}번째 기록을 불러왔습니다.", "{value}番目の記録を読み込みました。"),
    )
    return _pick_template(
        rng,
        templates,
        {"value": value, "count": count},
        "cardinal",
    )


def generate_percentage(rng: random.Random, _: int) -> Generated:
    whole = rng.randint(0, 99)
    decimal = rng.randint(0, 99)
    value = f"{whole}.{decimal:02d}%"
    previous = f"{rng.randint(0, 100)}.{rng.randint(0, 99):02d}%"
    templates = (
        ("진행률은 {value}입니다.", "進捗率は{value}です。"),
        ("성공률 변경: {previous}에서 {value}", "成功率の変更: {previous}から{value}"),
        ("할인율 {value}가 적용되었습니다.", "割引率{value}が適用されました。"),
        ("오차 범위는 ±{value}입니다.", "誤差範囲は±{value}です。"),
        ("배터리가 {value} 남았습니다.", "バッテリー残量は{value}です。"),
        ("전체의 {value}를 완료했습니다.", "全体の{value}が完了しました。"),
        (
            "점유율은 전월 {previous}, 이번 달 {value}입니다.",
            "占有率は前月{previous}、今月{value}です。",
        ),
        ("습도가 {value}로 측정되었습니다.", "湿度は{value}と測定されました。"),
    )
    return _pick_template(
        rng,
        templates,
        {"value": value, "previous": previous},
        "percentage",
    )


def generate_currency(rng: random.Random, _: int) -> Generated:
    currency = rng.choice(
        (
            ("원", "ウォン", "KRW"),
            ("엔", "円", "JPY"),
            ("달러", "ドル", "USD"),
            ("유로", "ユーロ", "EUR"),
        )
    )
    amount_number = rng.randint(1, 99_999_999)
    amount = f"{amount_number:,}"
    fee = f"{rng.randint(0, 999_999):,}"
    ko_unit, ja_unit, code = currency
    templates = (
        ("결제 금액은 {amount}{ko_unit}입니다.", "お支払い金額は{amount}{ja_unit}です。"),
        ("계좌 입금액은 {amount}{ko_unit}입니다.", "口座への入金額は{amount}{ja_unit}です。"),
        (
            "상품 가격은 {amount}{ko_unit}, 배송비는 {fee}{ko_unit}입니다.",
            "商品価格は{amount}{ja_unit}、送料は{fee}{ja_unit}です。",
        ),
        (
            "예산 한도를 {amount}{ko_unit}으로 설정했습니다.",
            "予算上限を{amount}{ja_unit}に設定しました。",
        ),
        ("환불 예정 금액은 {amount}{ko_unit}입니다.", "返金予定額は{amount}{ja_unit}です。"),
        ("{code} {amount} 결제를 승인하시겠습니까?", "{code} {amount}の決済を承認しますか。"),
        ("잔액은 {amount}{ko_unit}입니다.", "残高は{amount}{ja_unit}です。"),
        (
            "수수료는 {fee}{ko_unit}, 총액은 {amount}{ko_unit}입니다.",
            "手数料は{fee}{ja_unit}、合計は{amount}{ja_unit}です。",
        ),
    )
    return _pick_template(
        rng,
        templates,
        {
            "amount": amount,
            "fee": fee,
            "ko_unit": ko_unit,
            "ja_unit": ja_unit,
            "code": code,
        },
        "currency",
    )


def generate_date(rng: random.Random, _: int) -> Generated:
    start = date(2000, 1, 1) + timedelta(days=rng.randint(0, 14_975))
    end = start + timedelta(days=rng.randint(1, 365))
    ko_date = f"{start.year}년 {start.month}월 {start.day}일"
    ja_date = f"{start.year}年{start.month}月{start.day}日"
    ko_end = f"{end.year}년 {end.month}월 {end.day}일"
    ja_end = f"{end.year}年{end.month}月{end.day}日"
    iso = start.isoformat()
    templates = (
        ("예약일은 {ko_date}입니다.", "予約日は{ja_date}です。"),
        ("신청 마감일은 {ko_date}입니다.", "申請締切日は{ja_date}です。"),
        (
            "{ko_date}부터 {ko_end}까지 이용할 수 있습니다.",
            "{ja_date}から{ja_end}まで利用できます。",
        ),
        ("발행일: {ko_date}", "発行日: {ja_date}"),
        ("{iso} 기준 데이터를 불러왔습니다.", "{iso}時点のデータを読み込みました。"),
        ("변경된 배송 예정일은 {ko_date}입니다.", "変更後の配送予定日は{ja_date}です。"),
        (
            "유효 기간은 {ko_date}에서 {ko_end}까지입니다.",
            "有効期間は{ja_date}から{ja_end}までです。",
        ),
        ("다음 점검은 {ko_date}에 진행됩니다.", "次回の点検は{ja_date}に実施されます。"),
    )
    template_index = rng.randrange(len(templates))
    ko_template, ja_template = templates[template_index]
    ko = ko_template.format(ko_date=ko_date, ko_end=ko_end, iso=iso)
    ja = ja_template.format(ja_date=ja_date, ja_end=ja_end, iso=iso)
    return Generated(
        ko=ko,
        ja=ja,
        values=tuple(
            dict.fromkeys(value for value in (iso, end.isoformat()) if value in ko and value in ja)
        ),
        template_id=f"date-{template_index:02d}",
    )


def generate_time_duration(rng: random.Random, _: int) -> Generated:
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    time_value = f"{hour:02d}:{minute:02d}"
    precise = f"{hour:02d}:{minute:02d}:{second:02d}"
    duration = f"{rng.randint(1, 240)}분"
    ja_duration = duration.replace("분", "分")
    templates = (
        ("알람을 {time_value}에 설정했습니다.", "アラームを{time_value}に設定しました。"),
        ("회의는 {time_value}에 시작합니다.", "会議は{time_value}に始まります。"),
        ("처리 시간은 {ko_duration}입니다.", "処理時間は{ja_duration}です。"),
        (
            "{time_value}부터 {ko_duration} 동안 이용할 수 없습니다.",
            "{time_value}から{ja_duration}の間は利用できません。",
        ),
        ("마지막 동기화 시각은 {precise}입니다.", "最終同期時刻は{precise}です。"),
        ("변경된 재생 위치는 {precise}입니다.", "変更後の再生位置は{precise}です。"),
        ("예상 대기 시간은 {ko_duration}입니다.", "予想待ち時間は{ja_duration}です。"),
        ("타이머가 {ko_duration} 후에 울립니다.", "タイマーは{ja_duration}後に鳴ります。"),
    )
    template_index = rng.randrange(len(templates))
    ko_template, ja_template = templates[template_index]
    ko = ko_template.format(
        time_value=time_value,
        precise=precise,
        ko_duration=duration,
    )
    ja = ja_template.format(
        time_value=time_value,
        precise=precise,
        ja_duration=ja_duration,
    )
    return Generated(
        ko=ko,
        ja=ja,
        values=tuple(
            dict.fromkeys(
                value
                for value in (
                    time_value,
                    precise,
                    str(re.sub(r"\D", "", duration)),
                )
                if value in ko and value in ja
            )
        ),
        template_id=f"time-{template_index:02d}",
    )


def generate_measurement(rng: random.Random, _: int) -> Generated:
    unit = rng.choice(
        (
            ("길이", "長さ", "mm", 0, 100_000),
            ("길이", "長さ", "cm", 0, 10_000),
            ("거리", "距離", "km", 0, 100_000),
            ("무게", "重さ", "g", 0, 100_000),
            ("무게", "重さ", "kg", 0, 100_000),
            ("용량", "容量", "mL", 0, 100_000),
            ("용량", "容量", "L", 0, 100_000),
            ("속도", "速度", "km/h", 0, 1_500),
            ("온도", "温度", "°C", -100, 200),
            ("주파수", "周波数", "Hz", 0, 1_000_000_000),
            ("전압", "電圧", "V", 0, 100_000),
            ("전력", "電力", "W", 0, 10_000_000),
            ("저장 용량", "ストレージ容量", "GB", 0, 1_000_000),
            ("메모리", "メモリー", "MB", 0, 1_000_000),
        )
    )
    ko_name, ja_name, symbol, minimum, maximum = unit
    whole = rng.randint(minimum, maximum)
    decimal = rng.randint(0, 999)
    value = f"{whole}.{decimal:03d}{symbol}"
    lower_number = rng.randint(minimum, max(minimum, maximum - 1))
    upper_number = rng.randint(lower_number + 1, maximum)
    lower = f"{lower_number}{symbol}"
    upper = f"{upper_number}{symbol}"
    templates = (
        ("측정 결과 - {ko_name}: {value}", "測定結果 - {ja_name}: {value}"),
        ("설정 완료 - {ko_name}: {value}", "設定完了 - {ja_name}: {value}"),
        ("허용 범위는 {lower}에서 {upper}까지입니다.", "許容範囲は{lower}から{upper}までです。"),
        ("현재 {ko_name}: {value}", "現在の{ja_name}: {value}"),
        ("{value}를 초과하면 경고가 표시됩니다.", "{value}を超えると警告が表示されます。"),
        ("권장값 - {ko_name}: {value}", "推奨値 - {ja_name}: {value}"),
        ("센서가 {value}를 기록했습니다.", "センサーが{value}を記録しました。"),
        ("기준값 변경: {lower}에서 {value}", "基準値の変更: {lower}から{value}"),
    )
    return _pick_template(
        rng,
        templates,
        {
            "ko_name": ko_name,
            "ja_name": ja_name,
            "value": value,
            "lower": lower,
            "upper": upper,
        },
        "measurement",
    )


def generate_range(rng: random.Random, _: int) -> Generated:
    start = rng.randint(0, 900_000)
    end = start + rng.randint(1, 100_000)
    start_value, end_value = f"{start:,}", f"{end:,}"
    step = f"{rng.randint(1, 999):,}"
    templates = (
        ("유효 범위는 {start} 이상 {end} 이하입니다.", "有効範囲は{start}以上{end}以下です。"),
        (
            "{start}에서 {end} 사이의 값을 입력하세요.",
            "{start}から{end}までの値を入力してください。",
        ),
        ("지정된 검색 구간은 {start}~{end}입니다.", "指定された検索区間は{start}～{end}です。"),
        (
            "{start}부터 {end}까지 {step} 간격으로 증가합니다.",
            "{start}から{end}まで{step}刻みで増加します。",
        ),
        ("허용 오차는 {start}~{end}입니다.", "許容誤差は{start}～{end}です。"),
        ("설정된 축 범위는 [{start}, {end}]입니다.", "設定された軸の範囲は[{start}, {end}]です。"),
        (
            "페이지 {start}부터 {end}까지 내보냈습니다.",
            "{start}ページから{end}ページまでを書き出しました。",
        ),
        ("조회할 ID 범위는 {start}-{end}입니다.", "照会するID範囲は{start}-{end}です。"),
    )
    return _pick_template(
        rng,
        templates,
        {"start": start_value, "end": end_value, "step": step},
        "range",
    )


def _identifier(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return (
        f"{''.join(rng.choice(letters) for _ in range(3))}-"
        f"{rng.randint(0, 9999):04d}-"
        f"{''.join(rng.choice(letters) for _ in range(2))}"
    )


def generate_identifier(rng: random.Random, _: int) -> Generated:
    identifier = _identifier(rng)
    order = f"ORD-{rng.randint(0, 999999999):09d}"
    phone = f"+82-10-{rng.randint(0, 9999):04d}-{rng.randint(0, 9999):04d}"
    templates = (
        ("주문 번호 {order}의 상태를 확인해 주세요.", "注文番号{order}の状態を確認してください。"),
        ("인증 코드를 입력하세요: {identifier}", "認証コードを入力してください: {identifier}"),
        ("요청 ID는 {identifier}입니다.", "リクエストIDは{identifier}です。"),
        ("{phone} 번호로 확인 문자를 보냈습니다.", "{phone}宛てに確認メッセージを送信しました。"),
        ("찾을 수 없는 추적 번호: {order}", "見つからない追跡番号: {order}"),
        ("장치 {identifier}의 연결을 해제했습니다.", "デバイス{identifier}の接続を解除しました。"),
        ("생성된 티켓 번호는 {order}입니다.", "作成されたチケット番号は{order}です。"),
        ("이미 사용 중인 식별자: {identifier}", "既に使用されている識別子: {identifier}"),
    )
    return _pick_template(
        rng,
        templates,
        {"identifier": identifier, "order": order, "phone": phone},
        "identifier",
    )


def generate_technical(rng: random.Random, _: int) -> Generated:
    version = f"v{rng.randint(0, 99)}.{rng.randint(0, 99)}.{rng.randint(0, 999)}"
    port = str(rng.randint(1024, 65535))
    ip = ".".join(str(rng.randint(1 if index == 0 else 0, 254)) for index in range(4))
    width = rng.choice((1280, 1366, 1440, 1920, 2560, 3840))
    height = rng.choice((720, 768, 900, 1080, 1440, 2160))
    resolution = f"{width}×{height}"
    templates = (
        ("설치된 버전은 {version}입니다.", "インストールされたバージョンは{version}です。"),
        ("서버 {ip}:{port}에 연결했습니다.", "サーバー{ip}:{port}に接続しました。"),
        ("변경된 해상도는 {resolution}입니다.", "変更後の解像度は{resolution}です。"),
        ("이미 사용 중인 포트: {port}", "既に使用されているポート: {port}"),
        ("{version} 이상이 필요합니다.", "{version}以降が必要です。"),
        ("접속 대상 IP 주소는 {ip}입니다.", "接続先のIPアドレスは{ip}です。"),
        ("지원되는 최대 해상도는 {resolution}입니다.", "対応する最大解像度は{resolution}です。"),
        ("{ip}에서 열린 포트는 {port}입니다.", "{ip}で開かれたポートは{port}です。"),
    )
    return _pick_template(
        rng,
        templates,
        {
            "version": version,
            "port": port,
            "ip": ip,
            "resolution": resolution,
        },
        "technical",
    )


GENERATORS: tuple[tuple[str, int, Generator], ...] = (
    ("cardinal", 30_000, generate_cardinal),
    ("percentage", 20_000, generate_percentage),
    ("currency", 30_000, generate_currency),
    ("date", 30_000, generate_date),
    ("time_duration", 25_000, generate_time_duration),
    ("measurement", 35_000, generate_measurement),
    ("range_interval", 20_000, generate_range),
    ("identifier", 25_000, generate_identifier),
    ("technical", 25_000, generate_technical),
)


def _scaled_quotas(count: int) -> list[tuple[str, int, Generator]]:
    base_total = sum(quota for _, quota, _ in GENERATORS)
    assigned = 0
    output: list[tuple[str, int, Generator]] = []
    for index, (name, quota, generator) in enumerate(GENERATORS):
        if index == len(GENERATORS) - 1:
            scaled = count - assigned
        else:
            scaled = count * quota // base_total
            assigned += scaled
        output.append((name, scaled, generator))
    return output


def build_rows(count: int, seed: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    category_counts: Counter[str] = Counter()

    for category, quota, generator in _scaled_quotas(count):
        category_seed = seed + int.from_bytes(
            sha256(category.encode("utf-8")).digest()[:4],
            "big",
        )
        rng = random.Random(category_seed)
        attempts = 0
        while category_counts[category] < quota:
            attempts += 1
            if attempts > quota * 30:
                raise RuntimeError(f"Could not generate {quota} unique {category} rows")
            item = generator(rng, attempts)
            ko, ja = canonical_text(item.ko), canonical_text(item.ja)
            key = pair_key(ko, ja)
            if key in seen:
                continue
            if _PARENTHETICAL_PARTICLE_RE.search(ko):
                raise ValueError(f"Parenthetical particle in {item.template_id}: {ko!r}")
            if Counter(_NUMBER_RE.findall(ko)) != Counter(_NUMBER_RE.findall(ja)):
                raise ValueError(
                    f"Numeric signature mismatch in {item.template_id}: {ko!r} / {ja!r}"
                )
            for value in item.values:
                if value and (value not in ko or value not in ja):
                    raise ValueError(
                        f"Structure token {value!r} not preserved in {item.template_id}: "
                        f"{ko!r} / {ja!r}"
                    )
            seen.add(key)
            row_number = len(rows)
            original_direction = "ko_to_ja" if row_number % 2 == 0 else "ja_to_ko"
            rows.append(
                {
                    "ko": ko,
                    "ja": ja,
                    "synthetic": True,
                    "domain": "numeric_structured",
                    "subdomain": category,
                    "original_direction": original_direction,
                    "document_id": f"synthetic-numeric:{category}:{category_counts[category] // 1000}",
                    "family_id": f"synthetic-numeric:{category}:{item.template_id}:{row_number}",
                    "source_revision": GENERATOR_VERSION,
                    "generator": GENERATOR_VERSION,
                    "generator_seed": seed,
                    "template_id": item.template_id,
                    "structure_values": list(item.values),
                }
            )
            category_counts[category] += 1

    if len(rows) != count or len(seen) != count:
        raise AssertionError(f"Expected {count} unique rows, got {len(rows)}")
    return rows, dict(category_counts)


def write_output(
    output: Path,
    report: Path,
    *,
    count: int,
    seed: int,
) -> dict[str, object]:
    rows, category_counts = build_rows(count, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    digest = sha256_file(output)

    payload: dict[str, object] = {
        "schema": "sion-numeric-structured-synthetic-v1",
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "rows": count,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": digest,
        "category_counts": category_counts,
        "policies": {
            "train_only": True,
            "downweight_required": True,
            "existing_sentences_copied": False,
            "exact_pair_duplicates": 0,
            "shared_structure_tokens_checked": True,
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = report.with_suffix(report.suffix + ".part")
    report_temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_temporary.replace(report)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_numeric_data38.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/synthetic_numeric_data38.manifest.json"),
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    payload = write_output(
        args.output.resolve(),
        args.report.resolve(),
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
