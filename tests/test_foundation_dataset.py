"""단일어 shard 준비와, 그것이 실제 학습 경로를 통과하는지.

이 파일의 핵심은 마지막 두 테스트입니다. shard 를 쓰는 것만으로는 아무것도
보장되지 않습니다 — 같은 shard 가 ``IndexedParallelDataset`` 을 거쳐
collator 까지 갔을 때 ``<denoise_xx>`` 배치가 나오고, 문장이 두 번 학습되지
않아야 이 단계가 의도대로 도는 것입니다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

import sion_translate.data.prepare_foundation as foundation_prepare
from sion_translate.data.collate import SionBatchCollator
from sion_translate.data.indexed import IndexedParallelDataset
from sion_translate.data.monolingual import MonolingualDiscovery, discover_monolingual_sources
from sion_translate.data.prepare_foundation import (
    foundation_dataset_problem,
    prepare_foundation_dataset,
    render_prepare_report,
)
from sion_translate.token_audit import audit_indexed_token_exposure
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


def _dataset_problem(
    tmp_path,
    discovery,
    tokenizer_model,
    *,
    language_sampling_alpha=0.7,
    minimum_language_share=0.05,
) -> str | None:
    return foundation_dataset_problem(
        tmp_path / "dataset",
        discovery,
        tokenizer_model,
        minimum_characters=8,
        maximum_characters=4000,
        max_tokens=510,
        max_target_tokens=510,
        deduplicate=True,
        shard_size=32,
        validation_fraction=0.002,
        language_sampling_alpha=language_sampling_alpha,
        minimum_language_share=minimum_language_share,
        reasoning_sample_share=0.05,
        release_name="sion",
    )


def test_every_language_reaches_the_shards(tmp_path, tokenizer_model) -> None:
    _, stats = _prepare(tmp_path, tokenizer_model)
    assert stats.total_records == 100
    assert stats.languages["ko"].accepted == 60
    assert stats.languages["ja"].accepted == 40
    assert stats.validation_records >= 0
    assert stats.train_records > 0


def test_the_manifest_records_the_stage_identity(tmp_path, tokenizer_model) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "foundation"
    assert manifest["release_name"] == "sion"
    assert manifest["objective"] == "span-corruption-denoising"
    assert manifest["format"] == "sion-foundation-indexed-v3"
    assert manifest["target_storage"] == "row-shared-source-v1"
    # 순서는 폴더 정렬 순서를 따른다. 각 언어가 자기 자신과 짝지어지는 것이 요점.
    assert sorted(manifest["language_pairs"]) == [["ja", "ja"], ["ko", "ko"]]
    assert all(pair[0] == pair[1] for pair in manifest["language_pairs"])
    assert manifest["source_only_languages"] == []
    assert set(manifest["language_sampling"]["weights"]) == {"ko", "ja"}
    assert manifest["source_identity_schema"] == "corpus-relative-posix-sha256-v1"
    assert manifest["tokenizer_model"] == tokenizer_model.name
    assert manifest["tokenizer_identity"] == {
        "schema": "content-sha256-v1",
        "size_bytes": tokenizer_model.stat().st_size,
        "sha256": hashlib.sha256(tokenizer_model.read_bytes()).hexdigest(),
    }
    for source in manifest["sources"]:
        assert not PurePosixPath(source["logical_path"]).is_absolute()
        assert "\\" not in source["logical_path"]
        path = discovery.root.joinpath(*PurePosixPath(source["logical_path"]).parts)
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    source_bytes = 0
    target_bytes = 0
    for index_path in (tmp_path / "dataset").glob("*/*.idx.npy"):
        index = np.load(index_path, allow_pickle=False)
        assert (index["target_shared"] == 1).all()
        prefix = index_path.name.removesuffix(".idx.npy")
        source_bytes += index_path.with_name(f"{prefix}.src.bin").stat().st_size
        target_bytes += index_path.with_name(f"{prefix}.tgt.bin").stat().st_size
    assert source_bytes > 0
    assert target_bytes == 0

    audit = audit_indexed_token_exposure(
        tmp_path / "dataset",
        tokenizer_model,
        split="train",
    )
    assert audit["parameters"]["dataset_contract"] == "current-integrity-verified"
    assert audit["global_target_frequency"]["all_target_tokens"] > 0


def test_reader_rejects_shared_targets_without_the_v3_storage_marker(
    tmp_path,
    tokenizer_model,
) -> None:
    _prepare(tmp_path, tokenizer_model)
    manifest_path = tmp_path / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["target_storage"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated foundation manifest"):
        IndexedParallelDataset(
            tmp_path / "dataset",
            split="train",
            verify_integrity=False,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("target_shared", 0, "source tasks"),
        ("src_offset", 1, "stored token bytes"),
        ("tgt_offset", 1, "stored token bytes"),
        ("tgt_length", 1, "source alias contract"),
        ("tgt_register", 255, "source alias contract"),
        ("tgt_language_id", 255, "source alias contract"),
    ],
)
def test_reader_rejects_invalid_shared_target_aliases(
    tmp_path,
    tokenizer_model,
    field,
    replacement,
    message,
) -> None:
    _prepare(tmp_path, tokenizer_model)
    index_path = next(
        path
        for path in sorted((tmp_path / "dataset" / "train").glob("*.idx.npy"))
        if len(np.load(path, allow_pickle=False))
    )
    index = np.load(index_path, allow_pickle=False)
    index[field][0] = replacement
    np.save(index_path, index, allow_pickle=False)

    with pytest.raises(ValueError, match=message):
        IndexedParallelDataset(
            tmp_path / "dataset",
            split="train",
            verify_integrity=False,
        )


def test_manifest_source_identities_survive_corpus_relocation(
    tmp_path,
    tokenizer_model,
) -> None:
    original_root = _corpus(tmp_path / "windows-export" / "corpus")
    discovery = discover_monolingual_sources(original_root, ["ko", "ja"])
    prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "dataset",
        shard_size=32,
    )

    relocated_root = tmp_path / "linux-import" / "corpus"
    shutil.copytree(original_root, relocated_root)
    relocated = discover_monolingual_sources(relocated_root, ["ko", "ja"])

    assert _dataset_problem(tmp_path, relocated, tokenizer_model) is None


def test_absolute_path_manifest_requires_a_clear_portable_identity_migration(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    manifest_path = tmp_path / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["source_identity_schema"]
    manifest["sources"][0]["path"] = str(discovery.sources[0].path.resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    problem = _dataset_problem(tmp_path, discovery, tokenizer_model)

    assert problem is not None
    assert "obsolete" in problem
    assert "portable corpus-relative identities" in problem


def test_same_size_source_mutation_invalidates_the_prepared_dataset(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    assert _dataset_problem(tmp_path, discovery, tokenizer_model) is None

    source_path = discovery.sources[0].path
    original = source_path.read_bytes()
    mutated = original.replace(b"0", b"9", 1)
    assert mutated != original
    assert len(mutated) == len(original)
    source_path.write_bytes(mutated)

    problem = _dataset_problem(tmp_path, discovery, tokenizer_model)
    assert problem is not None
    assert "내용" in problem


def test_corrupt_source_id_invalidates_manifest_before_dataset_loading(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    manifest_path = tmp_path / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["sources"][0]["id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    problem = _dataset_problem(tmp_path, discovery, tokenizer_model)

    assert problem is not None
    assert "source id" in problem


def test_corrupt_sampling_weight_invalidates_manifest_before_training(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    manifest_path = tmp_path / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["language_sampling"]["weights"]["ko"] = -0.25
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    problem = _dataset_problem(tmp_path, discovery, tokenizer_model)

    assert problem is not None
    assert "sampling weight" in problem


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("language_sampling_alpha", 0.6, "sampling alpha"),
        ("minimum_language_share", 0.08, "minimum language share"),
    ],
)
def test_sampling_contract_changes_invalidate_the_prepared_dataset(
    tmp_path,
    tokenizer_model,
    option,
    value,
    message,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)

    problem = _dataset_problem(
        tmp_path,
        discovery,
        tokenizer_model,
        **{option: value},
    )

    assert problem is not None
    assert message in problem


def test_same_size_indexed_payload_mutation_invalidates_the_prepared_dataset(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    assert _dataset_problem(tmp_path, discovery, tokenizer_model) is None
    payload = tmp_path / "dataset" / "train" / "00000.src.bin"
    original = payload.read_bytes()
    assert original
    payload.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    problem = _dataset_problem(tmp_path, discovery, tokenizer_model)

    assert problem is not None
    assert "indexed payload" in problem
    assert "SHA-256 mismatch" in problem


def test_unreadable_source_reports_the_failing_path(tmp_path, tokenizer_model) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    source_path = discovery.sources[0].path
    source_path.unlink()

    problem = _dataset_problem(tmp_path, discovery, tokenizer_model)
    assert problem is not None
    assert "hash를 읽을 수 없습니다" in problem
    assert str(source_path) in problem


def test_the_manifest_carries_the_skipped_paths_forward(tmp_path, tokenizer_model) -> None:
    """왜 어떤 파일이 안 들어갔는지는 산출물에 남아야 한다."""
    root = _corpus(tmp_path / "corpus")
    (root / "ko" / "notes.md").write_text("마크다운\n", encoding="utf-8")
    discovery = discover_monolingual_sources(root, ["ko", "ja"])
    prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset", shard_size=32)

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    skipped = {entry["logical_path"] for entry in manifest["skipped"]}
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
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["preprocessing_options"]["deduplication_backend"] == "sqlite-blake2b-128-v1"
    assert not list((tmp_path / "dataset").glob(".foundation-dedup.sqlite3*"))


def test_disk_deduplication_uses_a_fixed_cache_and_removes_private_state(tmp_path) -> None:
    database = tmp_path / "dedup.sqlite3"
    index = foundation_prepare._DiskDigestIndex(database)

    assert index._connection.execute("PRAGMA cache_size").fetchone() == (-8192,)
    assert index.add(b"first") is True
    assert index.add(b"first") is False
    assert index.add(b"second") is True
    index.close()

    assert not list(tmp_path.glob("dedup.sqlite3*"))


def test_an_existing_non_empty_output_directory_is_refused(tmp_path, tokenizer_model) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "stale.bin").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")


def test_an_existing_empty_output_directory_is_also_refused(tmp_path, tokenizer_model) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(FileExistsError, match="must not exist"):
        prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    assert dataset.is_dir()
    assert not any(dataset.iterdir())


def test_source_mutation_aborts_publication_and_removes_staging(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    source_path = discovery.sources[0].path
    original_iterator = foundation_prepare.iter_monolingual_lines
    inventory_called = False

    def iter_then_mutate(path, *, stats=None):
        yield from original_iterator(path, stats=stats)
        if path == source_path:
            original = source_path.read_bytes()
            mutated = original.replace(b"0", b"9", 1)
            assert mutated != original
            assert len(mutated) == len(original)
            source_path.write_bytes(mutated)

    def record_inventory_call(_output_dir):
        nonlocal inventory_called
        inventory_called = True
        return []

    monkeypatch.setattr(foundation_prepare, "iter_monolingual_lines", iter_then_mutate)
    monkeypatch.setattr(
        foundation_prepare,
        "build_dataset_artifact_inventory",
        record_inventory_call,
    )

    with pytest.raises(RuntimeError, match="준비 중 변경"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert not inventory_called
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_tokenizer_mutation_aborts_publication_and_removes_staging(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    local_tokenizer = tmp_path / "tokenizer.model"
    local_tokenizer.write_bytes(tokenizer_model.read_bytes())
    source_path = discovery.sources[0].path
    original_iterator = foundation_prepare.iter_monolingual_lines
    mutated = False

    def iter_then_mutate_tokenizer(path, *, stats=None):
        nonlocal mutated
        yield from original_iterator(path, stats=stats)
        if path == source_path and not mutated:
            encoded = bytearray(local_tokenizer.read_bytes())
            encoded[-1] ^= 1
            local_tokenizer.write_bytes(encoded)
            mutated = True

    monkeypatch.setattr(
        foundation_prepare,
        "iter_monolingual_lines",
        iter_then_mutate_tokenizer,
    )

    with pytest.raises(RuntimeError, match="tokenizer가 준비 중 변경"):
        prepare_foundation_dataset(discovery, local_tokenizer, tmp_path / "dataset")

    assert mutated
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_new_source_added_during_build_aborts_publication(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    corpus = _corpus(tmp_path / "corpus")
    discovery = discover_monolingual_sources(corpus, ["ko", "ja"])
    source_path = discovery.sources[0].path
    original_iterator = foundation_prepare.iter_monolingual_lines
    added_source = corpus / "ko" / "added-during-build.txt"

    def iter_then_add_source(path, *, stats=None):
        yield from original_iterator(path, stats=stats)
        if path == source_path:
            added_source.write_text(
                "준비 도중 추가된 충분히 긴 한국어 문장입니다\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        foundation_prepare,
        "iter_monolingual_lines",
        iter_then_add_source,
    )

    with pytest.raises(RuntimeError, match="원천 파일 목록이 준비 중 변경"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert added_source.is_file()
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_source_mutation_during_inventory_is_caught_even_with_restored_metadata(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    source_path = discovery.sources[0].path
    original_inventory = foundation_prepare.build_dataset_artifact_inventory

    def inventory_then_mutate(output_dir):
        inventory = original_inventory(output_dir)
        source_stat = source_path.stat()
        original = source_path.read_bytes()
        mutated = original.replace(b"0", b"9", 1)
        assert mutated != original
        assert len(mutated) == len(original)
        source_path.write_bytes(mutated)
        os.utime(
            source_path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
        restored_stat = source_path.stat()
        assert restored_stat.st_size == source_stat.st_size
        assert restored_stat.st_mtime_ns == source_stat.st_mtime_ns
        return inventory

    monkeypatch.setattr(
        foundation_prepare,
        "build_dataset_artifact_inventory",
        inventory_then_mutate,
    )

    with pytest.raises(RuntimeError, match="준비 중 변경"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_manifest_failure_aborts_publication_and_removes_staging(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])

    def fail_inventory(_output_dir):
        raise RuntimeError("inventory failed")

    monkeypatch.setattr(
        foundation_prepare,
        "build_dataset_artifact_inventory",
        fail_inventory,
    )

    with pytest.raises(RuntimeError, match="inventory failed"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_publish_collision_preserves_destination_and_removes_staging(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    dataset = tmp_path / "dataset"
    original_publish = foundation_prepare._publish_staged_directory

    def publish_after_destination_appears(staging_dir, output_dir):
        dataset.mkdir()
        (dataset / "owner.txt").write_text("other publisher\n", encoding="utf-8")
        original_publish(staging_dir, output_dir)

    monkeypatch.setattr(
        foundation_prepare,
        "_publish_staged_directory",
        publish_after_destination_appears,
    )

    with pytest.raises(FileExistsError, match="must not exist"):
        prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    assert (dataset / "owner.txt").read_text(encoding="utf-8") == "other publisher\n"
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_incomplete_orphan_staging_is_removed_before_rebuild(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    orphan = tmp_path / ".dataset.staging-interrupted"
    (orphan / "train").mkdir(parents=True)
    (orphan / "train" / "partial.bin").write_bytes(b"partial")

    stats = prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert stats.total_records == 100
    assert (tmp_path / "dataset" / "manifest.json").is_file()
    assert not orphan.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_complete_exact_contract_orphan_staging_is_recovered(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovered = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    discovery = MonolingualDiscovery(
        root=discovered.root,
        sources=tuple(
            sorted(discovered.sources, key=lambda source: 0 if source.language == "ko" else 1)
        ),
        skipped=discovered.skipped,
        languages_without_data=discovered.languages_without_data,
        unconfigured_languages=discovered.unconfigured_languages,
    )
    assert discovery.languages == ("ko", "ja")
    dataset = tmp_path / "dataset"
    original_stats = prepare_foundation_dataset(discovery, tokenizer_model, dataset)
    orphan = tmp_path / ".dataset.staging-recoverable"
    dataset.rename(orphan)

    def forbid_rebuild(*_args, **_kwargs):
        raise AssertionError("a complete authenticated staging generation should be recovered")

    monkeypatch.setattr(
        foundation_prepare,
        "_prepare_foundation_dataset_in_staging",
        forbid_rebuild,
    )

    recovered_stats = prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    assert recovered_stats == original_stats
    assert (dataset / "manifest.json").is_file()
    assert not orphan.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_recovery_rejects_and_removes_uninventoried_deduplication_scratch(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    dataset = tmp_path / "dataset"
    prepare_foundation_dataset(discovery, tokenizer_model, dataset)
    orphan = tmp_path / ".dataset.staging-with-scratch"
    dataset.rename(orphan)
    (orphan / ".foundation-dedup.sqlite3").write_bytes(b"private scratch must not publish")

    rebuilt = prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    assert rebuilt.total_records == 100
    assert not (dataset / ".foundation-dedup.sqlite3").exists()
    assert not list(dataset.glob(".foundation-dedup.sqlite3*"))
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_publication_failure_preserves_a_complete_foundation_generation(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    dataset = tmp_path / "dataset"
    actual_publish = foundation_prepare._publish_staged_directory

    def fail_publication(_staging_dir, _output_dir):
        raise OSError("simulated foundation publication failure")

    monkeypatch.setattr(
        foundation_prepare,
        "_publish_staged_directory",
        fail_publication,
    )
    with pytest.raises(OSError, match="simulated foundation publication failure"):
        prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    staging = list(tmp_path.glob(".dataset.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "manifest.json").is_file()

    monkeypatch.setattr(
        foundation_prepare,
        "_publish_staged_directory",
        actual_publish,
    )

    def forbid_rebuild(*_args, **_kwargs):
        raise AssertionError("a complete foundation generation must resume without rebuilding")

    monkeypatch.setattr(
        foundation_prepare,
        "_prepare_foundation_dataset_in_staging",
        forbid_rebuild,
    )
    recovered = prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    assert recovered.total_records == 100
    assert (dataset / "manifest.json").is_file()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_unsafe_foundation_staging_root_is_preserved_and_refused(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    unsafe = tmp_path / ".dataset.staging-junction"
    unsafe.mkdir()
    (unsafe / "owner.txt").write_text("do not delete\n", encoding="utf-8")
    actual_lstat = foundation_prepare.os.lstat

    def mark_staging_as_reparse(path):
        value = actual_lstat(path)
        if Path(path) != unsafe:
            return value

        class ReparseStat:
            st_mode = value.st_mode
            st_size = value.st_size
            st_mtime_ns = value.st_mtime_ns
            st_ctime_ns = value.st_ctime_ns
            st_dev = value.st_dev
            st_ino = value.st_ino
            st_file_attributes = (
                getattr(value, "st_file_attributes", 0) | stat.FILE_ATTRIBUTE_REPARSE_POINT
            )

        return ReparseStat()

    monkeypatch.setattr(foundation_prepare.os, "lstat", mark_staging_as_reparse)

    with pytest.raises(RuntimeError, match="unsafe foundation staging path"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert (unsafe / "owner.txt").read_text(encoding="utf-8") == "do not delete\n"


def test_foundation_source_symlink_is_rejected_before_staging(
    tmp_path,
    tokenizer_model,
) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "ko").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("A sufficiently long sentence outside the corpus.\n", encoding="utf-8")
    linked = corpus / "ko" / "linked.txt"
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    discovery = discover_monolingual_sources(corpus, ["ko"])

    with pytest.raises(OSError, match="symlink or reparse point"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert outside.read_text(encoding="utf-8").startswith("A sufficiently long sentence")
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_publication_fsyncs_artifacts_and_namespace(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    synced_files = []
    synced_directories = []
    original_file_sync = foundation_prepare._fsync_file
    original_directory_sync = foundation_prepare._fsync_directory

    def record_file_sync(path):
        synced_files.append(path.name)
        original_file_sync(path)

    def record_directory_sync(path):
        synced_directories.append(path)
        original_directory_sync(path)

    monkeypatch.setattr(foundation_prepare, "_fsync_file", record_file_sync)
    monkeypatch.setattr(foundation_prepare, "_fsync_directory", record_directory_sync)

    prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert "manifest.json" in synced_files
    assert any(name.endswith(".src.bin") for name in synced_files)
    assert tmp_path in synced_directories


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

    shared_flags = []
    target_payload_bytes = 0
    for index_path in (tmp_path / "dataset").glob("*/*.idx.npy"):
        index = np.load(index_path, allow_pickle=False)
        shared_flags.extend(np.asarray(index["target_shared"], dtype=np.uint8).tolist())
        prefix = index_path.name.removesuffix(".idx.npy")
        target_payload_bytes += index_path.with_name(f"{prefix}.tgt.bin").stat().st_size
    assert 0 in shared_flags and 1 in shared_flags
    assert target_payload_bytes > 0

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["objective"] == "span-corruption-denoising+structured-reasoning"
    assert manifest["reasoning"]["records"] == 12
    assert manifest["reasoning"]["sample_share"] == 0.05
