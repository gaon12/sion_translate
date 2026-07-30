"""Template generation is licensed here only because these domains are formulaic.

That licence is easy to abuse, so the tables carry their own validator and these
tests pin what it must reject. Every case below was a real defect the validator or
the audit caught while building the shard.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "formulaic_lexicon.py"
SPEC = importlib.util.spec_from_file_location("formulaic_lexicon_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LEXICON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LEXICON
SPEC.loader.exec_module(LEXICON)


def test_the_shipped_tables_are_valid() -> None:
    assert LEXICON.validate() == ()


def test_every_zero_row_domain_is_covered() -> None:
    codes = set(LEXICON.known_domains())
    assert {"legal", "commerce", "technical", "medical", "travel", "administration"} <= codes


def test_every_frame_has_two_word_slots() -> None:
    # A frame varying only in digits collapses to one skeleton however many rows
    # it makes: exactly what cost data44/data45 90% of their rows.
    for domain in LEXICON.DOMAINS:
        for index, frame in enumerate(domain.frames):
            assert len(frame.word_slots()) >= 2, (domain.code, index, frame.word_slots())


def test_both_sides_of_a_frame_use_the_same_slots() -> None:
    for domain in LEXICON.DOMAINS:
        for index, frame in enumerate(domain.frames):
            ko = sorted(LEXICON._PLACEHOLDER.findall(frame.ko))
            ja = sorted(LEXICON._PLACEHOLDER.findall(frame.ja))
            assert ko == ja, (domain.code, index, ko, ja)


# --- Korean particles depend on the value substituted, so frames cannot fix them ---


def test_a_final_consonant_is_detected() -> None:
    assert LEXICON.has_final_consonant("회원")
    assert LEXICON.has_final_consonant("상품")
    assert not LEXICON.has_final_consonant("이용자")
    assert not LEXICON.has_final_consonant("도서")
    assert not LEXICON.has_final_consonant("")
    # A trailing non-Hangul character is skipped rather than guessed at.
    assert LEXICON.has_final_consonant("회원!")


def test_the_particle_matches_the_word_it_attaches_to() -> None:
    cases = [
        ("회원", "이/가", "회원이"),
        ("이용자", "이/가", "이용자가"),
        ("도서", "은/는", "도서는"),
        ("상품", "은/는", "상품은"),
        ("부정 이용 방지", "을/를", "부정 이용 방지를"),
        ("본인 확인", "을/를", "본인 확인을"),
        ("무인발급기", "으로/로", "무인발급기로"),
        ("방문", "으로/로", "방문으로"),
        ("인증 토큰", "과/와", "인증 토큰과"),
        ("캐시", "과/와", "캐시와"),
    ]
    for word, pair, expected in cases:
        assert LEXICON.attach_particle(word, pair) == expected, (word, pair)


def test_resolve_particles_rewrites_every_alternation() -> None:
    assert LEXICON.resolve_particles("회원이/가 신청했다") == "회원이 신청했다"
    assert LEXICON.resolve_particles("이용자이/가 신청했다") == "이용자가 신청했다"
    assert LEXICON.resolve_particles("도서은/는 배송된다") == "도서는 배송된다"
    assert LEXICON.resolve_particles("방지을/를 위하여") == "방지를 위하여"
    # Text with no alternation is untouched.
    assert LEXICON.resolve_particles("상품이 배송됩니다") == "상품이 배송됩니다"


def test_filling_resolves_the_particle_for_the_value_used() -> None:
    frame = LEXICON.Frame(
        "{counterparty}이/가 {legal_document}을/를 확인했습니다.",
        "{counterparty}が{legal_document}を確認しました。",
    )
    consonant = LEXICON.fill(
        frame, {"counterparty": ("회원", "会員"), "legal_document": ("약관", "規約")}
    )
    assert consonant[0] == "회원이 약관을 확인했습니다."
    vowel = LEXICON.fill(
        frame, {"counterparty": ("이용자", "利用者"), "legal_document": ("계약", "契約")}
    )
    assert vowel[0] == "이용자가 계약을 확인했습니다."


def test_the_validator_rejects_a_hardcoded_particle() -> None:
    # `{counterparty}가` is wrong for 회원 and 고객, which is how `회원가` and
    # `부정 이용 방지을` reached the shard before the alternation existed.
    bad = LEXICON.Domain(
        code="bad",
        label="bad",
        frames=(
            LEXICON.Frame(
                "{counterparty}가 {legal_document}을 봤다.",
                "{counterparty}が{legal_document}を見た。",
            ),
        ),
    )
    original = LEXICON.DOMAINS
    try:
        LEXICON.DOMAINS = (bad,)
        problems = LEXICON.validate()
    finally:
        LEXICON.DOMAINS = original
    assert any("hardcoded particle" in problem for problem in problems), problems


def test_the_validator_accepts_an_alternation() -> None:
    good = LEXICON.Domain(
        code="good",
        label="good",
        frames=(
            LEXICON.Frame(
                "{counterparty}이/가 {legal_document}을/를 봤다.",
                "{counterparty}が{legal_document}を見た。",
            ),
        ),
    )
    original = LEXICON.DOMAINS
    try:
        LEXICON.DOMAINS = (good,)
        problems = LEXICON.validate()
    finally:
        LEXICON.DOMAINS = original
    assert problems == ()


def test_the_validator_rejects_a_digit_only_frame() -> None:
    bad = LEXICON.Domain(
        code="bad",
        label="bad",
        frames=(LEXICON.Frame("{days}일 이내에 처리합니다.", "{days}日以内に処理します。"),),
    )
    original = LEXICON.DOMAINS
    try:
        LEXICON.DOMAINS = (bad,)
        problems = LEXICON.validate()
    finally:
        LEXICON.DOMAINS = original
    assert any("two word slots" in problem for problem in problems), problems


def test_the_validator_rejects_a_slot_mismatch() -> None:
    bad = LEXICON.Domain(
        code="bad",
        label="bad",
        frames=(LEXICON.Frame("{party} {item} {days}", "{party} {days}"),),
    )
    original = LEXICON.DOMAINS
    try:
        LEXICON.DOMAINS = (bad,)
        problems = LEXICON.validate()
    finally:
        LEXICON.DOMAINS = original
    assert any("slot mismatch" in problem for problem in problems), problems


def test_the_validator_rejects_an_unknown_slot() -> None:
    bad = LEXICON.Domain(
        code="bad",
        label="bad",
        frames=(LEXICON.Frame("{party} {nonexistent} {item}", "{party} {nonexistent} {item}"),),
    )
    original = LEXICON.DOMAINS
    try:
        LEXICON.DOMAINS = (bad,)
        problems = LEXICON.validate()
    finally:
        LEXICON.DOMAINS = original
    assert any("unknown slot" in problem for problem in problems), problems


# --- unit and semantic hazards found in the built shard ---


def test_currency_is_a_unit_and_never_converted() -> None:
    # 원 becomes ウォン. Rendering it 円 is a mistranslation, and data23 shipped
    # exactly that defect as 銭 before the preservation gate caught it.
    for domain in LEXICON.DOMAINS:
        for frame in domain.frames:
            if "원" in frame.ko and "{price}" in frame.ko:
                assert "ウォン" in frame.ja, (domain.code, frame.ja)
                assert "円" not in frame.ja, (domain.code, frame.ja)


def test_a_morning_frame_cannot_be_handed_an_evening_hour() -> None:
    # `오전 18시` is a contradiction, so the hour slot is split by half of day.
    morning = [value for value, _ in LEXICON.slot_values("hour_am")]
    assert morning and all(int(value) < 12 for value in morning)
    late = [value for value, _ in LEXICON.slot_values("hour_late")]
    assert late and all(int(value) >= 12 for value in late)
    for domain in LEXICON.DOMAINS:
        for frame in domain.frames:
            if "오전" in frame.ko:
                assert "{hour_late}" not in frame.ko, frame.ko


def test_no_slot_value_introduces_a_foreign_script() -> None:
    # `B1층` tripped the target-script check with a Latin letter.
    for name, values in LEXICON.SLOTS.items():
        for ko_value, _ in values:
            assert not any("A" <= char <= "z" for char in ko_value), (name, ko_value)


def test_a_group_booking_uses_a_place_that_takes_bookings() -> None:
    # `단체 2명 이상은 개찰구에서 예약` came from combining slots freely.
    booking = {value for value, _ in LEXICON.slot_values("booking_facility")}
    assert booking
    assert "개찰구" not in booking
    group = [int(value) for value, _ in LEXICON.slot_values("group_size")]
    assert group and min(group) >= 10


def test_unresolved_slots_are_reported() -> None:
    assert LEXICON.unresolved_slots("{party}가 왔다") == ("party",)
    assert LEXICON.unresolved_slots("회사가 왔다") == ()


def test_an_unknown_domain_is_none() -> None:
    assert LEXICON.domain("finance") is None
    assert LEXICON.domain("legal") is not None
