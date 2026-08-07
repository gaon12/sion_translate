"""Shared pytest fixtures for deployment-configurable package policies."""

from __future__ import annotations

import pytest

import sion_translate.content_screen as content_screen


@pytest.fixture
def configured_content_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a small deterministic policy for screening behavior tests.

    Production is allowed to ship with empty tables. Tests inject their own
    terms so they verify the matching engine instead of a deployment policy.
    """

    monkeypatch.setattr(
        content_screen,
        "CHILD_MARKERS",
        {
            "ko": ("초등학생", "초등학교", "유치원", "어린이"),
            "ja": ("小学生", "小学校", "小学"),
        },
    )
    monkeypatch.setattr(
        content_screen,
        "SEXUAL_MARKERS",
        {
            "ko": ("섹스", "성관계", "삽입", "알몸", "변태"),
            "ja": ("セックス", "性交", "挿入"),
        },
    )
    monkeypatch.setattr(
        content_screen,
        "_SPELLED_AGES",
        {
            "ko": {
                "여섯": 6,
                "열두": 12,
                "열네": 14,
                "열아홉": 19,
                "스무": 20,
                "스물": 20,
                "스물한": 21,
            },
            "ja": {
                "十二": 12,
                "十四": 14,
                "十九": 19,
                "二十": 20,
                "二十一": 21,
                "三十": 30,
            },
        },
    )
    monkeypatch.setattr(
        content_screen,
        "_SPELLED_AGE_PATTERNS",
        {
            language: pattern
            for language in content_screen._SPELLED_AGES
            if (pattern := content_screen._spelled_age_pattern(language)) is not None
        },
    )
