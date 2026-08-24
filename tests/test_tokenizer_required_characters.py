"""Byte fallback on a content character is a regression, not a graceful degrade.

The shipped tokenizer renders 넼 as three ``<0x..>`` pieces because the 한본어
corpus did not exist when it was trained. Two things caused that to survive: the
corpus was sampled down to 22.2%, and nothing forced a frequent character into
the vocabulary. These tests pin the second fix and keep it language-generic.
"""

from __future__ import annotations

from collections import Counter

import pytest

from sion_translate.tokenizer import (
    SENTENCEPIECE_RESERVED_CHARACTERS,
    acceptable_required_characters,
    required_characters_from_counts,
)


def test_a_frequent_character_is_required() -> None:
    counts = Counter({"넼": 1233, "엌": 1178, "슥": 338})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert set(required) == {"넼", "엌", "슥"}


def test_an_incidental_character_is_left_to_byte_fallback() -> None:
    # Below the floor a character is genuinely rare, and byte fallback is the
    # right answer - that is what byte fallback exists for.
    counts = Counter({"넼": 1233, "콬": 5, "鸚": 12})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert required == ["넼"]


def test_whitespace_is_never_required() -> None:
    # SentencePiece handles whitespace through its own meta symbol; reserving it
    # as a required character would waste a slot and confuse the segmenter.
    counts = Counter({" ": 10**6, "\n": 10**6, "　": 10**6, "가": 100})
    assert required_characters_from_counts(counts, min_occurrences=25) == ["가"]


def test_the_sentencepiece_unknown_character_is_never_required() -> None:
    """실측 실패. ⁇ 하나가 코퍼스 전체를 읽은 뒤에 학습을 죽인다.

    U+2047 은 SentencePiece 가 unknown 조각을 찍을 때 쓰는 문자라
    ``required_chars`` 로 넘기면 트레이너가 assert 로 죽습니다
    (``[!port::ContainsKey(required_chars_, kUNKChar)]``). 단일어 웹 코퍼스에
    이 문자가 임계값을 넘겨 들어 있었고, 실패는 코퍼스 스캔이 끝난 뒤에야
    났습니다 — 빌린 CPU 시간을 다 쓰고 난 다음입니다.
    """

    counts = Counter({"⁇": 10**6, "가": 100})
    assert required_characters_from_counts(counts, min_occurrences=25) == ["가"]


def test_the_verifier_finds_a_character_sentencepiece_will_not_take() -> None:
    """규칙을 추측하지 않고 SentencePiece 에게 직접 묻는다.

    U+2585 는 실제 실패를 이분 탐색해서 찾은 문자입니다. ``kUNKChar`` 도 아니고
    공백 기호 U+2581 도 아니며, 이웃한 블록 문자는 전부 통과합니다. 설명되지
    않는 거부가 존재하는 이상 목록을 손으로 관리하는 것은 다음 코퍼스에서 또
    무너집니다.
    """

    accepted, refused = acceptable_required_characters(list("가나▅다"))

    assert refused == ["▅"]
    assert accepted == list("가나다")


def test_the_verifier_leaves_a_clean_set_alone() -> None:
    characters = list("가나다あいう漢")
    accepted, refused = acceptable_required_characters(characters)

    assert refused == []
    assert accepted == characters


def test_the_known_rejects_are_filtered_before_the_probe_runs() -> None:
    """확인된 것은 목록으로 빠르게 걸러 탐색 횟수를 줄인다."""

    counts = Counter({character: 10**6 for character in SENTENCEPIECE_RESERVED_CHARACTERS})
    counts["가"] = 100
    assert required_characters_from_counts(counts, min_occurrences=25) == ["가"]


def test_the_result_is_sorted_so_the_build_is_reproducible() -> None:
    counts = Counter({"힣": 100, "가": 100, "あ": 100, "漢": 100, "Z": 100})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert required == sorted(required)


def test_the_threshold_is_honoured_exactly() -> None:
    counts = Counter({"가": 25, "나": 24})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert required == ["가"]


def test_nothing_here_names_a_script() -> None:
    # A new language pair must get the same protection without a code change, so
    # the rule is frequency alone.
    counts = Counter({"Ж": 500, "ก": 500, "א": 500, "ñ": 500})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert set(required) == {"Ж", "ก", "א", "ñ"}


def test_an_empty_corpus_requires_nothing() -> None:
    assert required_characters_from_counts(Counter(), min_occurrences=25) == []


def test_a_non_positive_threshold_is_rejected() -> None:
    # Zero would require every character ever seen, filling the vocabulary with
    # noise. The caller disables the feature at the train_tokenizer level.
    with pytest.raises(ValueError, match="min_occurrences must be positive"):
        required_characters_from_counts(Counter({"가": 1}), min_occurrences=0)


def test_the_vocab_floor_is_reported_before_the_corpus_scan(tmp_path, monkeypatch) -> None:
    """SentencePiece only complains after reading the corpus; say it sooner.

    required_chars plus the control symbols plus 256 byte-fallback pieces have to
    fit inside vocab_size. Hitting that limit after a full pass over a nine
    million row corpus wastes the whole scan, and the SentencePiece message does
    not name the setting to change.
    """

    import json

    from sion_translate import tokenizer as tokenizer_module

    shard = tmp_path / "mini.jsonl"
    with shard.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(50):
            handle.write(
                json.dumps(
                    {"ko": f"문장 {index} 입니다", "ja": f"文 {index} です"}, ensure_ascii=False
                )
                + "\n"
            )

    trained: list[object] = []
    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        lambda **kwargs: trained.append(kwargs),
    )

    with pytest.raises(ValueError, match="required_character_min_occurrences"):
        tokenizer_module.train_tokenizer(
            [str(shard)],
            tmp_path / "out",
            vocab_size=300,
            required_character_min_occurrences=1,
            language_pair=("ko", "ja"),
        )
    assert not trained, "training must not start once the floor is known to fail"


def _tiny_shard(tmp_path):
    import json

    shard = tmp_path / "shard.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(50):
            handle.write(
                json.dumps(
                    {"ko": f"문장 {index} 입니다", "ja": f"文 {index} です"}, ensure_ascii=False
                )
                + "\n"
            )
    return shard


def test_the_vocab_floor_includes_all_four_sentencepiece_meta_pieces(
    tmp_path,
    monkeypatch,
) -> None:
    import sion_translate.tokenizer as tokenizer_module

    counts = tokenizer_module.CorpusCounts(
        characters=Counter({"Ж": 1}),
        sentences=2,
        sentences_per_language=Counter({"ko": 1, "ja": 1}),
    )
    monkeypatch.setattr(
        tokenizer_module,
        "corpus_character_counts",
        lambda *args, **kwargs: counts,
    )

    def fail_if_the_lower_bound_is_missed(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the vocabulary lower bound must fail before probing or training")

    monkeypatch.setattr(
        tokenizer_module,
        "acceptable_required_characters",
        fail_if_the_lower_bound_is_missed,
    )
    symbols = tokenizer_module.control_symbols(("ko", "ja")) + tokenizer_module.SLOT_SYMBOLS
    exact_consumed_slots = 1 + len(symbols) + 256 + tokenizer_module.SENTENCEPIECE_META_PIECE_COUNT

    with pytest.raises(ValueError, match="SentencePiece meta pieces"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=exact_consumed_slots,
            required_character_min_occurrences=1,
            language_pair=("ko", "ja"),
            num_workers=1,
            num_threads=1,
        )


def test_sentencepiece_022_multithreaded_regression_is_refused_before_scanning(
    tmp_path,
    monkeypatch,
) -> None:
    """0.2.2 dies after loading 20M rows, so version rejection must precede that pass."""
    import sion_translate.tokenizer as tokenizer_module

    scanned = False

    def fail_if_scanned(*args, **kwargs):
        nonlocal scanned
        scanned = True
        raise AssertionError("the corpus scan must not start")

    monkeypatch.setattr(tokenizer_module.spm, "__version__", "0.2.2")
    monkeypatch.setattr(tokenizer_module, "corpus_character_counts", fail_if_scanned)

    with pytest.raises(RuntimeError, match=r"0\.2\.2.*SIGSEGV"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=600,
            language_pair=("ko", "ja"),
            num_threads=4,
        )
    assert not scanned


def test_sentencepiece_022_single_thread_measured_workaround_is_allowed(monkeypatch) -> None:
    """The same 20,355,455-row corpus completed with 0.2.2 when threads=1."""
    import sion_translate.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module.spm, "__version__", "0.2.2")
    assert tokenizer_module.validate_sentencepiece_training_runtime(1) == "0.2.2"


def test_full_coverage_is_rejected_because_it_disables_the_other_two_mechanisms(
    tmp_path,
    monkeypatch,
) -> None:
    """``character_coverage=1.0`` 은 required_chars 와 byte fallback 을 동시에 무력화한다.

    코퍼스의 모든 문자를 어휘에 넣으므로 (a) required_chars 가 이미 포함된
    부분집합이 되고, (b) byte fallback 256 조각이 발화할 대상을 잃고,
    (c) GPU 시간 앞의 byte fallback 비율 관문이 정의상 통과합니다.
    실측: 8,978,338 레코드 코퍼스에서 distinct 문자 10,760개 중 4,275개가
    25회 미만입니다.
    """
    import sion_translate.tokenizer as tokenizer_module

    trained: list[object] = []
    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        lambda **kwargs: trained.append(kwargs),
    )

    with pytest.raises(ValueError, match="byte fallback unreachable"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=300,
            character_coverage=1.0,
            language_pair=("ko", "ja"),
        )
    assert not trained, "training must not start on a self-defeating coverage setting"


def test_full_coverage_is_allowed_when_the_frequency_floor_is_opted_out(
    tmp_path,
    monkeypatch,
) -> None:
    """두 장치 중 하나만 쓰겠다는 명시적 선택은 막지 않는다."""
    import sion_translate.tokenizer as tokenizer_module

    trained: list[dict] = []

    def observe_training_arguments(**kwargs) -> None:
        trained.append(kwargs)
        raise RuntimeError("stop after observing trainer arguments")

    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        observe_training_arguments,
    )

    with pytest.raises(RuntimeError, match="observing trainer arguments"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=300,
            character_coverage=1.0,
            required_character_min_occurrences=0,
            language_pair=("ko", "ja"),
        )
    assert trained and trained[0]["character_coverage"] == 1.0
    assert trained[0]["required_chars"] == ""


def test_the_default_coverage_leaves_a_tail_for_byte_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    import sion_translate.tokenizer as tokenizer_module

    trained: list[dict] = []

    def observe_training_arguments(**kwargs) -> None:
        trained.append(kwargs)
        raise RuntimeError("stop after observing trainer arguments")

    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        observe_training_arguments,
    )

    with pytest.raises(RuntimeError, match="observing trainer arguments"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=600,
            language_pair=("ko", "ja"),
        )
    assert trained
    assert trained[0]["character_coverage"] < 1.0
    assert trained[0]["byte_fallback"] is True


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5])
def test_coverage_outside_the_unit_interval_is_rejected(tmp_path, coverage) -> None:
    import sion_translate.tokenizer as tokenizer_module

    with pytest.raises(ValueError, match="character_coverage"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=300,
            character_coverage=coverage,
            language_pair=("ko", "ja"),
        )
