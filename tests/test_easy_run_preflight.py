"""easy_run is the single entry point, so its preflight has to be the gate.

Both failures it guards against are silent: a shard whose keys match no language
pair yields zero sentences without a word, and a tokenizer that byte-falls-back on
content characters trains perfectly happily while wasting three tokens per
character. Neither shows up until someone counts, and by then the GPU hours are
spent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "easy_run.py"
SPEC = importlib.util.spec_from_file_location("easy_run_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EASY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EASY
SPEC.loader.exec_module(EASY)

FUSED = "넼"  # 네 + jongseong ㅋ, produced by the 한본어 generator


def write_corpus(directory: Path) -> Path:
    """A corpus where one content character is rare enough to be dropped."""

    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(400):
        rows.append(
            {
                "ko": f"오늘 날씨가 정말 좋네요 {index}",
                "ja": f"今日は天気がとても良いですね {index}",
            }
        )
    for index in range(40):
        rows.append({"ko": f"엌ㅋㅋ 튼튼데스{FUSED}ㅋㅋ {index}", "ja": f"やばい {index}"})
    path = directory / "mini.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def build_tokenizer(directory: Path, corpus: Path, *, reserve: str) -> Path:
    spm = pytest.importorskip("sentencepiece")
    text = directory / "corpus.txt"
    lines: list[str] = []
    with corpus.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lines.extend(str(value) for value in row.values())
    text.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prefix = directory / "sion"
    spm.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(prefix),
        vocab_size=600,
        model_type="unigram",
        character_coverage=0.98,
        byte_fallback=True,
        required_chars=reserve,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return prefix.with_suffix(".model")


def test_a_high_fallback_rate_stops_the_run(tmp_path: Path) -> None:
    data = tmp_path / "data"
    corpus = write_corpus(data)
    model = build_tokenizer(tmp_path, corpus, reserve="")
    # The gate is a rate, so force it low enough that any fallback trips it.
    with pytest.raises(SystemExit, match="상한"):
        EASY._verify_tokenizer(model, data, max_fallback_rate=0.0)


def test_reserving_the_character_removes_its_fallback(tmp_path: Path) -> None:
    data = tmp_path / "data"
    corpus = write_corpus(data)
    model = build_tokenizer(tmp_path, corpus, reserve=FUSED)
    spm = pytest.importorskip("sentencepiece")
    processor = spm.SentencePieceProcessor()
    processor.Load(str(model))
    assert not any(piece.startswith("<0x") for piece in processor.EncodeAsPieces(FUSED))


def test_a_tolerable_rate_passes(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data"
    corpus = write_corpus(data)
    model = build_tokenizer(tmp_path, corpus, reserve=FUSED)
    # Generous ceiling: the point is that a passing run does not raise.
    EASY._verify_tokenizer(model, data, max_fallback_rate=1.0)
    assert "허용 범위" in capsys.readouterr().out


def test_the_report_names_the_offending_characters(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data"
    corpus = write_corpus(data)
    model = build_tokenizer(tmp_path, corpus, reserve="")
    EASY._verify_tokenizer(model, data, max_fallback_rate=1.0)
    output = capsys.readouterr().out
    # A byte piece on its own says nothing about what to fix, so the character
    # and its codepoint have to be in the report.
    assert "U+" in output


def test_nothing_language_specific_is_hardcoded() -> None:
    # Fixed Korean probes would fail spuriously for an en-de corpus. The check
    # must read the corpus instead.
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    start = source.index("def _verify_tokenizer")
    end = source.index("def main(")
    body = source[start:end]
    for probe in ("닝겐노", "먹었나", "やばい", "대박이다"):
        assert probe not in body, probe


def test_a_missing_tokenizer_is_skipped_rather_than_crashing(tmp_path: Path, capsys) -> None:
    EASY._verify_tokenizer(tmp_path / "absent.model", tmp_path)
    assert "건너뜁니다" in capsys.readouterr().out


def test_a_missing_corpus_is_skipped_rather_than_crashing(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data"
    corpus = write_corpus(data)
    model = build_tokenizer(tmp_path, corpus, reserve=FUSED)
    empty = tmp_path / "empty"
    empty.mkdir()
    EASY._verify_tokenizer(model, empty)
    assert "코퍼스를 찾지 못해" in capsys.readouterr().out


def test_the_shard_key_check_stops_the_run(monkeypatch) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 1

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(EASY.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="조용히 빠집니다"):
        EASY._check_shard_keys({})
    assert calls, "the checker must actually be invoked"


def test_the_shard_key_check_passes_quietly(monkeypatch) -> None:
    class Result:
        returncode = 0

    monkeypatch.setattr(EASY.subprocess, "run", lambda command, **kwargs: Result())
    EASY._check_shard_keys({})  # must not raise


def test_a_missing_checker_does_not_block_the_run(monkeypatch, tmp_path: Path) -> None:
    # A trimmed deployment without scripts/ should still be able to train.
    monkeypatch.setattr(EASY, "ROOT", tmp_path)
    EASY._check_shard_keys({})  # must not raise


def test_the_foundation_corpus_is_reported_before_training(tmp_path, capsys) -> None:
    """건너뛰는 것이 정상 경로이므로, 조용히 건너뛰면 안 된다."""
    import easy_run

    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n  language_pair: [ko, ja]\n"
        f"foundation:\n  corpus_dir: {(tmp_path / 'corpus').as_posix()}\n",
        encoding="utf-8",
    )

    easy_run._report_foundation_corpus(config)

    output = capsys.readouterr().out
    assert "foundation 단계를 건너뜁니다" in output
    # 무엇을 어디에 두면 되는지가 출력에 있어야 한다.
    assert ".txt" in output and ".jsonl" in output


def test_a_usable_corpus_is_announced_with_its_languages(tmp_path, capsys) -> None:
    import easy_run

    corpus = tmp_path / "corpus"
    for language in ("ko", "ja"):
        (corpus / language).mkdir(parents=True)
        (corpus / language / "a.txt").write_text(
            "\n".join(f"{language} 문장 {index} 입니다" for index in range(10)) + "\n",
            encoding="utf-8",
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data:\n  language_pair: [ko, ja]\nfoundation:\n  corpus_dir: {corpus.as_posix()}\n",
        encoding="utf-8",
    )

    easy_run._report_foundation_corpus(config)

    output = capsys.readouterr().out
    assert "foundation 단계를 실행합니다" in output
    assert "ko" in output and "ja" in output
    # 산출물이 번역 모델과 분리된다는 사실을 미리 알린다.
    assert "runs/*/foundation/" in output


def test_a_skipped_directory_is_named_in_the_preflight(tmp_path, capsys) -> None:
    """`data/corpus/korean_tech_corpus_130m/` 같은 실제 사례."""
    import easy_run

    corpus = tmp_path / "corpus"
    (corpus / "ko").mkdir(parents=True)
    (corpus / "ko" / "a.txt").write_text("한국어 문장입니다\n", encoding="utf-8")
    (corpus / "korean_tech_corpus_130m").mkdir()
    (corpus / "korean_tech_corpus_130m" / "ui.txt").write_text("기술 문장\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data:\n  language_pair: [ko, ja]\nfoundation:\n  corpus_dir: {corpus.as_posix()}\n",
        encoding="utf-8",
    )

    easy_run._report_foundation_corpus(config)

    assert "korean_tech_corpus_130m" in capsys.readouterr().out
