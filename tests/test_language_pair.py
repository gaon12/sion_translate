"""언어쌍 일반화(en-de 등)와 데이터 증강 안전장치 검증."""

from __future__ import annotations

import json
import random
from pathlib import Path

from kjx.cli.augment import synthetic_budget
from kjx.data import IndexedParallelDataset, KJBatchCollator
from kjx.data.prepare import prepare_dataset
from kjx.data.quality import assess_pair
from kjx.tokenizer import KJTokenizer, train_tokenizer


def test_generic_language_pair_passes_quality() -> None:
    # 라틴 문자 언어쌍은 문자 기반 언어 판별이 불가능하므로 script 검사를
    # 건너뛰고, 나머지 손상 검사(동일문 등)만 적용되어야 한다.
    ok = assess_pair("The weather is nice today.", "Das Wetter ist heute schön.", languages=("en", "de"))
    assert ok.accepted
    identical = assess_pair("Same text.", "Same text.", languages=("en", "de"))
    assert not identical.accepted


def write_en_de_jsonl(path: Path, count: int = 60) -> None:
    en_words = ["today", "tomorrow", "weather", "good", "bad", "school", "office", "train", "book", "friend"]
    de_words = ["heute", "morgen", "Wetter", "gut", "schlecht", "Schule", "Büro", "Zug", "Buch", "Freund"]
    rng = random.Random(0)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            picks = [rng.randrange(len(en_words)) for _ in range(5)]
            handle.write(
                json.dumps(
                    {
                        "en": " ".join(en_words[i] for i in picks) + f" number {index}.",
                        "de": " ".join(de_words[i] for i in picks) + f" Nummer {index}.",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_en_de_pipeline_end_to_end(tmp_path: Path) -> None:
    """설정의 언어쌍만 바꾸면 토크나이저→전처리→데이터셋→collator 가
    그대로 동작해야 한다 (ko-ja 하드코딩이 남아 있지 않은지 검증)."""
    source = tmp_path / "corpus.jsonl"
    write_en_de_jsonl(source)

    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pair=("en", "de"),
    )
    tokenizer = KJTokenizer(model_path)
    # 언어쌍이 vocab 의 <2xx> 태그에서 자동 인식되어야 한다.
    assert set(tokenizer.languages) == {"en", "de"}
    assert set(tokenizer.language_tags) == {"en", "de"}

    dataset_dir = tmp_path / "dataset"
    stats = prepare_dataset(
        [str(source)],
        model_path,
        dataset_dir,
        validation_fraction=0.1,
        test_fraction=0.1,
        dedup_backend="memory",
        language_pair=("en", "de"),
    )
    assert stats.valid_pairs > 0

    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)
    assert dataset.language_pair == ("en", "de")
    forward = dataset[0]
    reverse = dataset[1]
    assert forward["src_language"] == "en" and forward["target_language"] == "de"
    assert reverse["src_language"] == "de" and reverse["target_language"] == "en"

    collator = KJBatchCollator(tokenizer, max_source_length=64, max_target_length=64)
    batch = collator([forward, reverse])
    # 방향 태그: en→de 예시는 <2de>, de→en 예시는 <2en> 으로 시작해야 한다.
    assert batch["input_ids"][0, 0].item() == tokenizer.language_tags["de"]
    assert batch["input_ids"][1, 0].item() == tokenizer.language_tags["en"]


def test_synthetic_files_are_train_only(tmp_path: Path) -> None:
    """bt_* 합성 파일은 validation/test 에 절대 들어가면 안 된다."""
    real = tmp_path / "real.jsonl"
    synthetic = tmp_path / "bt_mono.jsonl"
    write_en_de_jsonl(real, count=60)
    write_en_de_jsonl(synthetic, count=40)
    # 실데이터와 중복되지 않도록 합성 파일 내용을 비틀어 준다.
    lines = synthetic.read_text(encoding="utf-8").splitlines()
    with synthetic.open("w", encoding="utf-8") as handle:
        for line in lines:
            row = json.loads(line)
            handle.write(
                json.dumps({k: "synthetic " + v for k, v in row.items()}, ensure_ascii=False) + "\n"
            )

    model_path = train_tokenizer(
        [str(real)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pair=("en", "de"),
    )
    dataset_dir = tmp_path / "dataset"
    prepare_dataset(
        [str(real), str(synthetic)],
        model_path,
        dataset_dir,
        validation_fraction=0.2,
        test_fraction=0.2,
        dedup_backend="memory",
        language_pair=("en", "de"),
        train_only_prefixes=("bt_",),
    )
    with (dataset_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_name = {source["name"]: source["stats"] for source in manifest["sources"]}
    assert by_name["bt_mono.jsonl"]["validation"] == 0
    assert by_name["bt_mono.jsonl"]["test"] == 0
    assert by_name["bt_mono.jsonl"]["train"] == by_name["bt_mono.jsonl"]["valid_pairs"] > 0
    # 실데이터는 정상적으로 세 split 에 나뉜다.
    assert by_name["real.jsonl"]["validation"] > 0


def test_synthetic_budget_caps_generation() -> None:
    # 실데이터 1000쌍, 비율 1.0 → 합성은 총 1000쌍까지만
    assert synthetic_budget(1000, 0, 1.0) == 1000
    assert synthetic_budget(1000, 800, 1.0) == 200
    assert synthetic_budget(1000, 1200, 1.0) == 0  # 이미 초과 → 추가 생성 금지
    assert synthetic_budget(1000, 0, 0.5) == 500


def test_source_token_dropout_keeps_slots_and_length(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    write_en_de_jsonl(source)
    model_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pair=("en", "de"),
    )
    tokenizer = KJTokenizer(model_path)
    # 보호 슬롯(<slot_n>) ID 영역(16~79)을 피해서 일반 토큰을 고른다.
    assert max(tokenizer.slot_ids) < 100
    item = {
        "src": list(range(100, 130)),  # 일반 토큰 30개
        "tgt": [100, 101, 102],
        "src_language": "en",
        "target_language": "de",
        "src_register": 0,
        "target_register": 0,
    }
    random.seed(3)
    clean = KJBatchCollator(
        tokenizer, max_source_length=64, max_target_length=64, source_token_dropout=0.0
    )([dict(item)])
    random.seed(3)
    dropped = KJBatchCollator(
        tokenizer, max_source_length=64, max_target_length=64, source_token_dropout=0.4
    )([dict(item)])
    clean_len = int(clean["attention_mask"][0].sum())
    dropped_len = int(dropped["attention_mask"][0].sum())
    assert dropped_len < clean_len  # 일부 토큰이 탈락했다
    assert dropped_len >= 3  # 태그 + 최소 1 토큰 + EOS 는 남는다
    # 목표(레이블)는 증강의 영향을 받지 않는다.
    assert dropped["labels"].tolist() == clean["labels"].tolist()
