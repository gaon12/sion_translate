from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "fetch_todaii_kej.py"
SPEC = importlib.util.spec_from_file_location("fetch_todaii_kej_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
FETCH_TODAII = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FETCH_TODAII
SPEC.loader.exec_module(FETCH_TODAII)


def test_japanese_text_discards_ruby_reading_and_markup() -> None:
    value = '<ruby data-id="1"><rb>日本</rb><rt>にほん</rt></ruby>で<span>働く</span>。'

    assert FETCH_TODAII.japanese_text(value) == "日本で働く。"


def test_extract_article_rows_builds_exact_three_language_alignment() -> None:
    article = FETCH_TODAII.Article(
        article_id="abc123",
        date="2026-08-18",
        topic="society_life",
        level=3,
    )
    payload = {
        "result": {
            "content": [
                {
                    "text": "<ruby><rb>今日</rb><rt>きょう</rt></ruby>は晴れです。",
                    "translate": {
                        "translate_lang": "오늘은 맑습니다.",
                        "translate_en": "It is sunny today.",
                    },
                }
            ]
        }
    }

    rows = FETCH_TODAII.extract_article_rows(article, payload)

    assert len(rows) == 1
    assert list(rows[0])[:3] == ["ko", "en", "ja"]
    assert rows[0]["ko"] == "오늘은 맑습니다."
    assert rows[0]["en"] == "It is sunny today."
    assert rows[0]["ja"] == "今日は晴れです。"
    assert rows[0]["resource_id"] == "abc123:0000"
    assert rows[0]["document_id"] == "todaii/abc123"
    assert rows[0]["source_revision"] == "2026-08-18"
    assert rows[0]["jlpt_level"] == 3


def test_extract_article_rows_skips_incomplete_translation() -> None:
    article = FETCH_TODAII.Article("missing", "", "", None)
    payload = {
        "result": {
            "content": [
                {
                    "text": "日本語。",
                    "translate": {"translate_lang": "한국어.", "translate_en": ""},
                }
            ]
        }
    }

    assert FETCH_TODAII.extract_article_rows(article, payload) == []


def test_parse_news_sitemap_extracts_unique_article_ids() -> None:
    value = """<?xml version="1.0"?>
    <urlset>
      <url><loc>https://japanese.todaiinews.com/en/news/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</loc></url>
      <url><loc>https://japanese.todaiinews.com/en/news/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</loc></url>
      <url><loc>https://japanese.todaiinews.com/en/news/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb</loc></url>
    </urlset>
    """

    articles = FETCH_TODAII.parse_news_sitemap(value)

    assert [article.article_id for article in articles] == ["a" * 32, "b" * 32]


def test_fetch_catalogue_preserves_order_and_deduplicates_ids() -> None:
    pages = {
        1: {
            "data": {
                "data": [
                    {
                        "id": "new",
                        "time": "2026-08-18",
                        "value": {"topic": "health", "level": 4},
                    },
                    {"id": "shared", "time": "2026-08-17", "value": {}},
                ],
                "pagination": {"last_page": 2, "total": 3},
            }
        },
        2: {
            "data": {
                "data": [
                    {"id": "shared", "time": "2026-08-17", "value": {}},
                    {"id": "old", "time": "2026-08-16", "value": {}},
                ],
                "pagination": {"last_page": 2, "total": 3},
            }
        },
    }

    def fake_fetch(url: str) -> dict[str, object]:
        page = 2 if "page=2" in url else 1
        return pages[page]

    articles = FETCH_TODAII.fetch_catalogue(fake_fetch, page_size=2)

    assert [article.article_id for article in articles] == ["new", "shared", "old"]
    assert articles[0].topic == "health"
    assert articles[0].level == 4


def test_fetch_partitioned_catalogue_requires_advertised_coverage() -> None:
    original_topics = FETCH_TODAII.NEWS_TOPICS
    FETCH_TODAII.NEWS_TOPICS = ("health",)

    def fake_fetch(url: str) -> dict[str, object]:
        if "topic=health" not in url:
            return {
                "data": {
                    "data": [{"id": "summary", "time": "2026-08-19", "value": {}}],
                    "pagination": {"last_page": 1, "total": 2},
                }
            }
        level = next(value for value in range(1, 6) if f"level={value}" in url)
        rows = []
        if level == 1:
            rows = [{"id": "older", "time": "2026-08-18", "value": {"level": 1}}]
        elif level == 2:
            rows = [{"id": "newer", "time": "2026-08-19", "value": {"level": 2}}]
        return {
            "data": {
                "data": rows,
                "pagination": {"last_page": 1, "total": len(rows)},
            }
        }

    try:
        articles = FETCH_TODAII.fetch_partitioned_catalogue(fake_fetch, page_size=50, workers=2)
    finally:
        FETCH_TODAII.NEWS_TOPICS = original_topics

    assert [article.article_id for article in articles] == ["newer", "older"]


def test_sanitize_existing_jsonl_drops_whole_malformed_document(tmp_path: Path) -> None:
    path = tmp_path / "todaii.jsonl"
    rows = [
        {
            "ko": "정상 문장",
            "en": "Normal sentence",
            "ja": "正常な文",
            "resource_id": "good:0",
            "document_id": "todaii/good",
        },
        {
            "ko": "깨진 문장의 앞부분",
            "en": "Broken sentence prefix",
            "ja": '<span word="',
            "resource_id": "bad:0",
            "document_id": "todaii/bad",
        },
        {
            "ko": "나머지",
            "en": "Remainder",
            "ja": "残り",
            "resource_id": "bad:1",
            "document_id": "todaii/bad",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    stats = FETCH_TODAII.sanitize_existing_jsonl(path)
    remaining = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert stats["removed_document_count"] == 1
    assert stats["removed_row_count"] == 2
    assert [row["resource_id"] for row in remaining] == ["good:0"]
