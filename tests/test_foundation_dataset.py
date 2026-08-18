"""단일어 shard 준비와, 그것이 실제 학습 경로를 통과하는지.

이 파일의 핵심은 마지막 두 테스트입니다. shard 를 쓰는 것만으로는 아무것도
보장되지 않습니다 — 같은 shard 가 ``IndexedParallelDataset`` 을 거쳐
collator 까지 갔을 때 ``<denoise_xx>`` 배치가 나오고, 문장이 두 번 학습되지
않아야 이 단계가 의도대로 도는 것입니다.
"""

from __future__ import annotations

import json

import pytest

from sion_translate.data.collate import SionBatchCollator
from sion_translate.data.indexed import IndexedParallelDataset
from sion_translate.data.monolingual import discover_monolingual_sources
from sion_translate.data.prepare_foundation import (
    prepare_foundation_dataset,
    render_prepare_report,
)
from sion_translate.tokenizer import SionTokenizer, train_tokenizer


@pytest.fixture(scope="module")
def tokenizer_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tokenizer")
    shard = directory / "pairs.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(400):
            handle.write(
                json.dumps(
                    {
                        "ko": f"한국어 문장 {index} 입니다 그리고 조금 더 깁니다",
                        "ja": f"日本語の文 {index} です そしてもう少し長いです",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return train_tokenizer(
        [str(shard)],
        directory / "out",
        vocab_size=700,
        num_workers=1,
        language_pair=["ko", "ja"],
        reasoning_languages=["ja"],
    )


def _corpus(root, *, ko_lines=60, ja_lines=40):
    (root / "ko").mkdir(parents=True, exist_ok=True)
    (root / "ja").mkdir(parents=True, exist_ok=True)
    (root / "ko" / "wiki.txt").write_text(
        "\n".join(f"한국어 단일어 문장 {index} 입니다 조금 더 깁니다" for index in range(ko_lines))
        + "\n",
        encoding="utf-8",
    )
    (root / "ja" / "news.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"text": f"日本語の単言語文 {index} です もう少し長いです"}, ensure_ascii=False
            )
            for index in range(ja_lines)
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _prepare(tmp_path, tokenizer_model, **kwargs):
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    return discovery, prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "dataset",
        shard_size=32,
        **kwargs,
    )


def test_every_language_reaches_the_shards(tmp_path, tokenizer_model) -> None:
    _, stats = _prepare(tmp_path, tokenizer_model)
    assert stats.total_records == 100
    assert stats.languages["ko"].accepted == 60
    assert stats.languages["ja"].accepted == 40
    assert stats.validation_records >= 0
    assert stats.train_records > 0


def test_the_manifest_records_the_stage_identity(tmp_path, tokenizer_model) -> None:
    _prepare(tmp_path, tokenizer_model)
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "foundation"
    assert manifest["release_name"] == "sion"
    assert manifest["objective"] == "span-corruption-denoising"
    # 순서는 폴더 정렬 순서를 따른다. 각 언어가 자기 자신과 짝지어지는 것이 요점.
    assert sorted(manifest["language_pairs"]) == [["ja", "ja"], ["ko", "ko"]]
    assert all(pair[0] == pair[1] for pair in manifest["language_pairs"])
    assert manifest["source_only_languages"] == []
    assert set(manifest["language_sampling"]["weights"]) == {"ko", "ja"}


def test_the_manifest_carries_the_skipped_paths_forward(tmp_path, tokenizer_model) -> None:
    """왜 어떤 파일이 안 들어갔는지는 산출물에 남아야 한다."""
    root = _corpus(tmp_path / "corpus")
    (root / "ko" / "notes.md").write_text("마크다운\n", encoding="utf-8")
    discovery = discover_monolingual_sources(root, ["ko", "ja"])
    prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset", shard_size=32)

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    skipped = {entry["path"] for entry in manifest["skipped"]}
    assert any(path.endswith("notes.md") for path in skipped)


def test_short_and_long_lines_are_dropped_by_reason(tmp_path, tokenizer_model) -> None:
    root = tmp_path / "corpus"
    (root / "ko").mkdir(parents=True)
    (root / "ko" / "a.txt").write_text(
        "\n".join(["짧다", "충분히 긴 한국어 문장입니다", "가" * 5000]) + "\n",
        encoding="utf-8",
    )
    discovery = discover_monolingual_sources(root, ["ko"])
    stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "dataset",
        minimum_characters=8,
        maximum_characters=4000,
    )
    # 짧은 줄만 버립니다. 긴 문서는 버리지 않고 나눕니다 — e_gov 는 문자의
    # 97.3%, aozora 는 92.8% 가 "상한 초과" 한 줄이라 통째로 폐기됐었습니다.
    assert stats.languages["ko"].too_short == 1
    assert stats.languages["ko"].too_long == 0
    assert stats.languages["ko"].segmented_documents == 1
    assert stats.languages["ko"].accepted == 1 + stats.languages["ko"].segments - 1


def test_duplicates_are_removed_within_a_language(tmp_path, tokenizer_model) -> None:
    root = tmp_path / "corpus"
    (root / "ko").mkdir(parents=True)
    (root / "ko" / "a.txt").write_text(
        "\n".join(["같은 한국어 문장입니다"] * 5) + "\n", encoding="utf-8"
    )
    discovery = discover_monolingual_sources(root, ["ko"])
    stats = prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")
    assert stats.languages["ko"].accepted == 1
    assert stats.languages["ko"].duplicate == 4


def test_an_existing_non_empty_output_directory_is_refused(tmp_path, tokenizer_model) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "stale.bin").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")


def test_a_tokenizer_without_the_language_tags_is_refused(tmp_path, tokenizer_model) -> None:
    root = tmp_path / "corpus"
    (root / "en").mkdir(parents=True)
    (root / "en" / "a.txt").write_text("a reasonably long english sentence\n", encoding="utf-8")
    discovery = discover_monolingual_sources(root, ["en"])
    with pytest.raises(ValueError, match="missing denoise tags"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")


def test_an_empty_discovery_is_refused(tmp_path, tokenizer_model) -> None:
    discovery = discover_monolingual_sources(tmp_path / "absent", ["ko"])
    with pytest.raises(ValueError, match="학습 가능한 파일이 없습니다"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")


def test_the_report_names_the_drop_reasons(tmp_path, tokenizer_model) -> None:
    _, stats = _prepare(tmp_path, tokenizer_model)
    rendered = "\n".join(render_prepare_report(stats))
    assert "ko:" in rendered and "ja:" in rendered
    assert "train" in rendered


# ── 여기부터가 핵심: 실제 학습 경로를 통과하는가 ────────────────────────


def test_each_sentence_is_trained_once_not_twice(tmp_path, tokenizer_model) -> None:
    """복원 과제는 두 방향이 같은 예제다.

    ``forward_only`` 를 쓰지 않으면 양방향 확장이 모든 문장을 정확히 두 번
    학습시킵니다 — 조용히 epoch 이 두 배가 되고, 손실 곡선만 보면 알 수
    없습니다.
    """
    _, stats = _prepare(tmp_path, tokenizer_model)
    dataset = IndexedParallelDataset(tmp_path / "dataset", split="train", bidirectional=True)

    assert len(dataset) == stats.train_records
    assert dataset.forward_only_count == dataset.pair_count


def test_the_collator_produces_denoising_batches(tmp_path, tokenizer_model) -> None:
    """입력은 ``<denoise_xx>`` 로 시작하고, 정답은 손상되지 않은 원문이어야 한다."""
    _prepare(tmp_path, tokenizer_model)
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = IndexedParallelDataset(tmp_path / "dataset", split="train", bidirectional=True)
    collator = SionBatchCollator(
        tokenizer,
        max_source_length=64,
        max_target_length=64,
        denoise_probability=1.0,
        denoise_noise_density=0.15,
        denoise_mean_span=3.0,
    )

    batch = collator([dataset[index] for index in range(8)])

    denoise_ids = set(tokenizer.denoise_tags.values())
    first_tokens = batch["input_ids"][:, 0].tolist()
    assert all(token in denoise_ids for token in first_tokens), first_tokens
    # 손상은 입력에만 일어난다: 정답에는 <mask> 가 없어야 한다.
    assert not (batch["labels"] == tokenizer.mask_id).any()
    # 실제로 무언가 가려졌는지 확인한다. 그렇지 않으면 복원할 것이 없다.
    assert (batch["input_ids"] == tokenizer.mask_id).any()


def test_no_translation_direction_tag_appears_in_a_foundation_batch(
    tmp_path,
    tokenizer_model,
) -> None:
    """foundation 배치에 ``<2xx>`` 가 섞이면 번역을 미리 배우는 셈이 된다."""
    _prepare(tmp_path, tokenizer_model)
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = IndexedParallelDataset(tmp_path / "dataset", split="train", bidirectional=True)
    collator = SionBatchCollator(
        tokenizer,
        max_source_length=64,
        max_target_length=64,
        denoise_probability=1.0,
    )

    batch = collator([dataset[index] for index in range(min(32, len(dataset)))])

    translation_tags = set(tokenizer.language_tags.values())
    assert not translation_tags & set(batch["input_ids"][:, 0].tolist())


def test_reasoning_rows_bypass_forced_denoising_and_keep_trace_markers(
    tmp_path,
    tokenizer_model,
) -> None:
    root = _corpus(tmp_path / "corpus", ko_lines=20, ja_lines=20)
    reasoning_path = root / "ja" / "reasoning_math.jsonl"
    reasoning_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": f"{index} と 2 を足してください。",
                    "think": "二つの数を順番に確認してから加算する。",
                    "answer": f"答えは {index + 2} です。",
                    "language": "ja",
                    "category": "math",
                },
                ensure_ascii=False,
            )
            for index in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    discovery = discover_monolingual_sources(root, ["ko", "ja"])
    stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "dataset",
        max_tokens=62,
        max_target_tokens=63,
        validation_fraction=0.1,
    )
    tokenizer = SionTokenizer(tokenizer_model)
    dataset = IndexedParallelDataset(tmp_path / "dataset", split="train", bidirectional=True)
    reasoning_id = tokenizer.reasoning_tags["ja"]
    reasoning_items = [
        dataset[index]
        for index in range(len(dataset))
        if int(dataset[index]["src"][0]) == reasoning_id
    ]
    assert reasoning_items

    collator = SionBatchCollator(
        tokenizer,
        max_source_length=64,
        max_target_length=64,
        denoise_probability=1.0,
        denoise_noise_density=0.5,
    )
    batch = collator(reasoning_items[:4])

    assert batch["input_ids"][:, 0].tolist() == [reasoning_id] * min(4, len(reasoning_items))
    assert not (batch["input_ids"] == tokenizer.mask_id).any()
    assert (batch["labels"][:, 0] == tokenizer.reasoning_trace_ids["<think>"]).all()
    assert any(tokenizer.reasoning_trace_ids["</think>"] in row.tolist() for row in batch["labels"])
    assert any(tokenizer.reasoning_trace_ids["<answer>"] in row.tolist() for row in batch["labels"])
    assert any(
        tokenizer.reasoning_trace_ids["</answer>"] in row.tolist() for row in batch["labels"]
    )
    assert stats.languages["ja"].reasoning_records == 12

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["objective"] == "span-corruption-denoising+structured-reasoning"
    assert manifest["reasoning"]["records"] == 12
    assert manifest["reasoning"]["sample_share"] == 0.05
