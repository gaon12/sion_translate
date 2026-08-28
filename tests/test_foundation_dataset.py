"""Prepare monolingual shards and exercise them through the real training path.

The final tests are the critical checks. Writing a shard alone guarantees
nothing. After the same shard passes through ``IndexedParallelDataset`` and the
collator, it must produce a ``<denoise_xx>`` batch without training every
sentence twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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
    allow_offline_sources=False,
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
        allow_offline_sources=allow_offline_sources,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _resume_corpus(root: Path) -> Path:
    """Create mixed denoising/reasoning input with rejects and duplicates."""

    language_root = root / "ja"
    language_root.mkdir(parents=True)
    denoising_rows = [
        json.dumps({"text": "再開テスト用の十分に長い文 0 です。"}, ensure_ascii=False),
        json.dumps({"text": "再開テスト用の十分に長い文 1 です。"}, ensure_ascii=False),
        "",
        "{malformed",
        json.dumps({"text": "再開テスト用の十分に長い文 0 です。"}, ensure_ascii=False),
        json.dumps({"text": "再開テスト用の十分に長い文 2 です。"}, ensure_ascii=False),
        json.dumps({"text": "再開テスト用の十分に長い文 3 です。"}, ensure_ascii=False),
        json.dumps({"text": "再開テスト用の十分に長い文 4 です。"}, ensure_ascii=False),
    ]
    (language_root / "denoising.jsonl").write_text(
        "\n".join(denoising_rows) + "\n",
        encoding="utf-8",
    )
    reasoning_rows = [
        {
            "prompt": "三と四を加算してください。",
            "think": "二つの整数を確認してから合計する。",
            "answer": "答えは七です。",
            "language": "ja",
        },
        {
            "prompt": "五と六を加算してください。",
            "think": "二つの整数を確認してから合計する。",
            "answer": "答えは十一です。",
            "language": "ja",
        },
        {
            "prompt": "三と四を加算してください。",
            "think": "二つの整数を確認してから合計する。",
            "answer": "答えは七です。",
            "language": "ja",
        },
    ]
    (language_root / "reasoning_resume.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in reasoning_rows) + "\n",
        encoding="utf-8",
    )
    return root


def _leave_partial_foundation_generation(
    tmp_path: Path,
    tokenizer_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    deduplicate: bool = True,
) -> tuple[MonolingualDiscovery, Path, Path]:
    """Crash after one checkpoint plus two uncommitted physical lines."""

    discovery = discover_monolingual_sources(
        _resume_corpus(tmp_path / "corpus"),
        ["ja"],
    )
    assert discovery.sources[0].path.name == "denoising.jsonl"
    dataset = tmp_path / "resumed"
    original_iterator = foundation_prepare._iter_source_physical_lines
    interruption = {"armed": True}

    def interrupt_with_an_uncommitted_tail(path, *, skip_lines, strict_utf8):
        for physical_line, line in original_iterator(
            path,
            skip_lines=skip_lines,
            strict_utf8=strict_utf8,
        ):
            yield physical_line, line
            if interruption["armed"] and path == discovery.sources[0].path and physical_line == 6:
                raise RuntimeError("simulated recoverable interruption")

    monkeypatch.setattr(foundation_prepare, "_FOUNDATION_CHECKPOINT_INTERVAL", 4)
    monkeypatch.setattr(
        foundation_prepare,
        "_iter_source_physical_lines",
        interrupt_with_an_uncommitted_tail,
    )
    with pytest.raises(RuntimeError, match="simulated recoverable interruption"):
        prepare_foundation_dataset(
            discovery,
            tokenizer_model,
            dataset,
            deduplicate=deduplicate,
            shard_size=3,
            validation_fraction=0.1,
        )
    interruption["armed"] = False
    staging = list(tmp_path.glob(".resumed.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / ".foundation-resume.sqlite3").is_file()
    return discovery, dataset, staging[0]


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
    # Directory sorting sets the order; the important property is self-pairing.
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


def test_manifest_can_authenticate_prepared_shards_while_sources_are_offline(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery, _ = _prepare(tmp_path, tokenizer_model)
    offline = MonolingualDiscovery(
        root=discovery.root,
        languages_without_data=discovery.languages,
    )
    shutil.rmtree(discovery.root)

    assert (
        _dataset_problem(
            tmp_path,
            offline,
            tokenizer_model,
            allow_offline_sources=True,
        )
        is None
    )
    assert _dataset_problem(tmp_path, offline, tokenizer_model) is not None


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
    assert "content changed" in problem


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
    assert "Cannot read the foundation source hash" in problem
    assert str(source_path) in problem


def test_the_manifest_carries_the_skipped_paths_forward(tmp_path, tokenizer_model) -> None:
    """The artifact must retain the reason each skipped file was excluded."""
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
    # Drop only short lines and segment long documents. Earlier handling discarded
    # 97.3% of e_gov and 92.8% of aozora characters as single over-limit lines.
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


def test_disk_preflight_refuses_before_creating_staging(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    monkeypatch.setattr(
        foundation_prepare.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="Insufficient free disk space"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_disk_estimator_is_deterministic_and_uses_measured_token_layout(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    options = {
        "minimum_characters": 8,
        "maximum_characters": 4000,
        "max_tokens": 510,
        "max_target_tokens": 510,
        "deduplicate": True,
    }

    first = foundation_prepare._estimate_foundation_generation_bytes(
        discovery,
        tokenizer_model,
        **options,
    )
    second = foundation_prepare._estimate_foundation_generation_bytes(
        discovery,
        tokenizer_model,
        **options,
    )

    source_bytes = sum(source.size_bytes for source in discovery.sources)
    buffered_source_floor = sum((source.size_bytes * 3 + 1) // 2 for source in discovery.sources)
    assert first == second
    assert first >= buffered_source_floor
    # The estimator measures the actual shared-target token/index layout. It
    # must not regress to the former blanket 11x source-size reservation.
    assert first < source_bytes * 11


@pytest.mark.parametrize(
    ("estimated_bytes", "expected_reserve"),
    [
        (1 * 1024**3, 512 * 1024**2),
        (100 * 1024**3, 2 * 1024**3),
    ],
)
def test_disk_preflight_subtracts_authenticated_staging_and_preserves_the_larger_reserve(
    tmp_path,
    tokenizer_model,
    monkeypatch,
    estimated_bytes,
    expected_reserve,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    authenticated_staging = tmp_path / ".dataset.staging-authenticated"
    existing_bytes = estimated_bytes // 10
    required_free = estimated_bytes - existing_bytes + expected_reserve

    monkeypatch.setattr(
        foundation_prepare,
        "_estimate_foundation_generation_bytes",
        lambda *_args, **_kwargs: estimated_bytes,
    )

    def authenticated_size(path):
        assert path == authenticated_staging
        return existing_bytes

    monkeypatch.setattr(
        foundation_prepare,
        "_foundation_staging_size",
        authenticated_size,
    )
    free_bytes = {"value": required_free}
    monkeypatch.setattr(
        foundation_prepare.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=free_bytes["value"]),
    )
    options = {
        "existing_staging": authenticated_staging,
        "minimum_characters": 8,
        "maximum_characters": 4000,
        "max_tokens": 510,
        "max_target_tokens": 510,
        "deduplicate": True,
    }

    reserve = foundation_prepare._preflight_foundation_disk_space(
        discovery,
        tokenizer_model,
        tmp_path,
        **options,
    )
    assert reserve == expected_reserve

    free_bytes["value"] -= 1
    with pytest.raises(OSError, match=f"need at least {required_free:,} free bytes"):
        foundation_prepare._preflight_foundation_disk_space(
            discovery,
            tokenizer_model,
            tmp_path,
            **options,
        )


def test_source_mutation_aborts_publication_and_removes_staging(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    source_path = discovery.sources[0].path
    original_iterator = foundation_prepare._iter_source_physical_lines
    inventory_called = False

    def iter_then_mutate(path, *, skip_lines, strict_utf8):
        yield from original_iterator(
            path,
            skip_lines=skip_lines,
            strict_utf8=strict_utf8,
        )
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

    monkeypatch.setattr(
        foundation_prepare,
        "_iter_source_physical_lines",
        iter_then_mutate,
    )
    monkeypatch.setattr(
        foundation_prepare,
        "build_dataset_artifact_inventory",
        record_inventory_call,
    )

    with pytest.raises(RuntimeError, match="changed during preparation"):
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
    original_iterator = foundation_prepare._iter_source_physical_lines
    mutated = False

    def iter_then_mutate_tokenizer(path, *, skip_lines, strict_utf8):
        nonlocal mutated
        yield from original_iterator(
            path,
            skip_lines=skip_lines,
            strict_utf8=strict_utf8,
        )
        if path == source_path and not mutated:
            encoded = bytearray(local_tokenizer.read_bytes())
            encoded[-1] ^= 1
            local_tokenizer.write_bytes(encoded)
            mutated = True

    monkeypatch.setattr(
        foundation_prepare,
        "_iter_source_physical_lines",
        iter_then_mutate_tokenizer,
    )

    with pytest.raises(RuntimeError, match="Foundation tokenizer changed during preparation"):
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
    original_iterator = foundation_prepare._iter_source_physical_lines
    added_source = corpus / "ko" / "added-during-build.txt"

    def iter_then_add_source(path, *, skip_lines, strict_utf8):
        yield from original_iterator(
            path,
            skip_lines=skip_lines,
            strict_utf8=strict_utf8,
        )
        if path == source_path:
            added_source.write_text(
                "준비 도중 추가된 충분히 긴 한국어 문장입니다\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        foundation_prepare,
        "_iter_source_physical_lines",
        iter_then_add_source,
    )

    with pytest.raises(RuntimeError, match="Foundation source list changed during preparation"):
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

    with pytest.raises(RuntimeError, match="changed during preparation"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_manifest_failure_preserves_the_final_authenticated_checkpoint(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    original_inventory = foundation_prepare.build_dataset_artifact_inventory

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
    staging = list(tmp_path.glob(".dataset.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / ".foundation-resume.sqlite3").is_file()

    def checkpoint_sequence() -> int:
        connection = sqlite3.connect(staging[0] / ".foundation-resume.sqlite3")
        try:
            payload = json.loads(
                connection.execute("SELECT payload FROM state WHERE singleton = 1").fetchone()[0]
            )
        finally:
            connection.close()
        return int(payload["checkpoint_sequence"])

    final_sequence = checkpoint_sequence()

    def forbid_source_replay(*_args, **_kwargs):
        raise AssertionError("a final checkpoint must not replay input lines")

    monkeypatch.setattr(
        foundation_prepare,
        "_iter_source_physical_lines",
        forbid_source_replay,
    )

    # A second post-checkpoint failure must not manufacture another final
    # checkpoint. Otherwise its sequence becomes invalid and the next recovery
    # silently discards all completed work.
    with pytest.raises(RuntimeError, match="inventory failed"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")
    assert checkpoint_sequence() == final_sequence

    monkeypatch.setattr(
        foundation_prepare,
        "build_dataset_artifact_inventory",
        original_inventory,
    )
    recovered = prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert recovered.total_records == 100
    assert (tmp_path / "dataset" / "manifest.json").is_file()
    assert not list(tmp_path.glob(".dataset.staging-*"))


@pytest.mark.parametrize("deduplicate", [True, False])
def test_partial_resume_is_byte_identical_for_mixed_objectives(
    tmp_path,
    tokenizer_model,
    monkeypatch,
    deduplicate,
) -> None:
    discovery, resumed_path, staging = _leave_partial_foundation_generation(
        tmp_path,
        tokenizer_model,
        monkeypatch,
        deduplicate=deduplicate,
    )
    connection = sqlite3.connect(staging / ".foundation-resume.sqlite3")
    try:
        contract = json.loads(
            connection.execute("SELECT value FROM metadata WHERE key = 'contract'").fetchone()[0]
        )
        resume_state = json.loads(
            connection.execute("SELECT payload FROM state WHERE singleton = 1").fetchone()[0]
        )
        digest_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(digests)").fetchall()
        ]
    finally:
        connection.close()
    assert contract["schema"] == "sion-foundation-resume-v1"
    assert contract["preprocessing_schema"] == "foundation-mixed-objectives-v6"
    assert (
        contract["tokenizer_identity"]["sha256"]
        == hashlib.sha256(tokenizer_model.read_bytes()).hexdigest()
    )
    assert contract["options"]["deduplicate"] is deduplicate
    assert contract["options"]["checkpoint_interval_physical_lines"] == 4
    assert digest_columns == ["digest", "sequence"]
    assert len(resume_state["dedup_sequence_sha256"]) == 64
    assert [source["task"] for source in contract["sources"]] == [
        "denoising",
        "reasoning",
    ]

    resumed_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        resumed_path,
        deduplicate=deduplicate,
        shard_size=3,
        validation_fraction=0.1,
    )
    uninterrupted_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "uninterrupted",
        deduplicate=deduplicate,
        shard_size=3,
        validation_fraction=0.1,
    )

    assert resumed_stats == uninterrupted_stats
    assert resumed_stats.languages["ja"].reasoning_records == (2 if deduplicate else 3)
    assert resumed_stats.languages["ja"].duplicate == (2 if deduplicate else 0)
    assert _tree_bytes(resumed_path) == _tree_bytes(tmp_path / "uninterrupted")
    assert not list(tmp_path.glob(".resumed.staging-*"))
    assert not list(resumed_path.glob(".foundation-resume.sqlite3*"))


def test_disk_preflight_preserves_and_unlocks_an_authenticated_partial(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery, resumed_path, staging = _leave_partial_foundation_generation(
        tmp_path,
        tokenizer_model,
        monkeypatch,
    )
    actual_disk_usage = foundation_prepare.shutil.disk_usage
    monkeypatch.setattr(
        foundation_prepare.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="Insufficient free disk space"):
        prepare_foundation_dataset(
            discovery,
            tokenizer_model,
            resumed_path,
            shard_size=3,
            validation_fraction=0.1,
        )

    assert staging.is_dir()
    assert (staging / ".foundation-resume.sqlite3").is_file()

    monkeypatch.setattr(foundation_prepare.shutil, "disk_usage", actual_disk_usage)
    resumed = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        resumed_path,
        shard_size=3,
        validation_fraction=0.1,
    )
    assert resumed.total_records > 0
    assert not list(tmp_path.glob(".resumed.staging-*"))


def test_streaming_disk_reserve_stops_at_a_recoverable_checkpoint(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(
        _resume_corpus(tmp_path / "corpus"),
        ["ja"],
    )
    dataset = tmp_path / "resumed"
    actual_disk_usage = foundation_prepare.shutil.disk_usage
    probes = 0

    def exhaust_after_first_checkpoint(_path):
        nonlocal probes
        probes += 1
        # Initial estimate, then the first checkpoint, then exhaustion before
        # the second checkpoint can become authoritative.
        free = 1 << 60 if probes <= 2 else 0
        return SimpleNamespace(free=free)

    monkeypatch.setattr(foundation_prepare, "_FOUNDATION_CHECKPOINT_INTERVAL", 4)
    monkeypatch.setattr(
        foundation_prepare.shutil,
        "disk_usage",
        exhaust_after_first_checkpoint,
    )
    with pytest.raises(OSError, match="checkpoint disk reserve"):
        prepare_foundation_dataset(
            discovery,
            tokenizer_model,
            dataset,
            shard_size=3,
            validation_fraction=0.1,
        )

    staging = next(tmp_path.glob(".resumed.staging-*"))
    connection = sqlite3.connect(staging / ".foundation-resume.sqlite3")
    try:
        state = json.loads(
            connection.execute("SELECT payload FROM state WHERE singleton = 1").fetchone()[0]
        )
    finally:
        connection.close()
    assert state["cursor"]["total_physical_lines"] == 4

    monkeypatch.setattr(foundation_prepare.shutil, "disk_usage", actual_disk_usage)
    resumed_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        dataset,
        shard_size=3,
        validation_fraction=0.1,
    )
    clean_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "clean",
        shard_size=3,
        validation_fraction=0.1,
    )
    assert resumed_stats == clean_stats
    assert _tree_bytes(dataset) == _tree_bytes(tmp_path / "clean")


def test_streaming_checkpoint_gate_reuses_the_capacity_plan_reserve(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    planned_reserve = foundation_prepare._FOUNDATION_SPACE_RESERVE_BYTES + 123
    observed_reserves = []
    monkeypatch.setattr(
        foundation_prepare,
        "_preflight_foundation_disk_space",
        lambda *_args, **_kwargs: planned_reserve,
    )

    def record_checkpoint_gate(_path, minimum_free_bytes):
        observed_reserves.append(minimum_free_bytes)

    monkeypatch.setattr(
        foundation_prepare,
        "_preflight_foundation_checkpoint_space",
        record_checkpoint_gate,
    )

    prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")

    assert observed_reserves
    assert set(observed_reserves) == {planned_reserve}


def test_checkpoint_inventory_hashes_each_immutable_shard_only_once_before_final_audit(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    actual_file_sha256 = foundation_prepare.file_sha256
    shard_hashes: dict[str, int] = {}

    def count_shard_hashes(path):
        path = Path(path)
        if ".dataset.staging-" in path.as_posix() and path.parent.name in {
            "train",
            "validation",
        }:
            relative = "/".join(path.parts[-2:])
            shard_hashes[relative] = shard_hashes.get(relative, 0) + 1
        return actual_file_sha256(path)

    monkeypatch.setattr(foundation_prepare, "_FOUNDATION_CHECKPOINT_INTERVAL", 10)
    monkeypatch.setattr(foundation_prepare, "file_sha256", count_shard_hashes)
    prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "dataset",
        shard_size=3,
    )

    assert shard_hashes
    # One hash when a shard first enters the checkpoint inventory, then one
    # complete-prefix authentication before manifest publication.
    assert set(shard_hashes.values()) == {2}


@pytest.mark.parametrize(
    "corruption",
    ["database", "dedup_content", "shard_hash", "shard_semantics"],
)
def test_corrupt_partial_checkpoint_is_discarded_before_rebuild(
    tmp_path,
    tokenizer_model,
    monkeypatch,
    corruption,
) -> None:
    discovery, resumed_path, staging = _leave_partial_foundation_generation(
        tmp_path,
        tokenizer_model,
        monkeypatch,
    )
    database = staging / ".foundation-resume.sqlite3"
    if corruption == "database":
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE state SET payload_sha256 = ? WHERE singleton = 1",
                ("0" * 64,),
            )
            connection.commit()
        finally:
            connection.close()
    elif corruption == "dedup_content":
        connection = sqlite3.connect(database)
        try:
            sequence, raw_digest = connection.execute(
                "SELECT sequence, digest FROM digests ORDER BY sequence LIMIT 1"
            ).fetchone()
            changed = bytes([raw_digest[0] ^ 1]) + raw_digest[1:]
            connection.execute(
                "UPDATE digests SET digest = ? WHERE sequence = ?",
                (sqlite3.Binary(changed), sequence),
            )
            connection.commit()
        finally:
            connection.close()
    elif corruption == "shard_hash":
        payload = next(staging.glob("*/*.src.bin"))
        original = payload.read_bytes()
        assert original
        payload.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    else:
        connection = sqlite3.connect(database)
        try:
            raw_payload = connection.execute(
                "SELECT payload FROM state WHERE singleton = 1"
            ).fetchone()[0]
            state = json.loads(raw_payload)
            inventory_entry = next(
                entry for entry in state["artifact_inventory"] if entry["path"].endswith(".idx.npy")
            )
            relative = inventory_entry["path"]
            index_path = staging / relative
            index = np.load(index_path, allow_pickle=False)
            assert len(index)
            index["target_shared"][0] = 2
            np.save(index_path, index, allow_pickle=False)
            inventory_entry["size"] = index_path.stat().st_size
            inventory_entry["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            state_text = foundation_prepare._canonical_json(state)
            connection.execute(
                "UPDATE state SET payload = ?, payload_sha256 = ? WHERE singleton = 1",
                (state_text, hashlib.sha256(state_text.encode("utf-8")).hexdigest()),
            )
            connection.commit()
        finally:
            connection.close()

    rebuilt_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        resumed_path,
        shard_size=3,
        validation_fraction=0.1,
    )
    clean_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "clean",
        shard_size=3,
        validation_fraction=0.1,
    )

    assert rebuilt_stats == clean_stats
    assert _tree_bytes(resumed_path) == _tree_bytes(tmp_path / "clean")
    assert not list(tmp_path.glob(".resumed.staging-*"))


@pytest.mark.parametrize("drift", ["source", "options", "schema"])
def test_resume_contract_drift_discards_the_old_checkpoint(
    tmp_path,
    tokenizer_model,
    monkeypatch,
    drift,
) -> None:
    discovery, resumed_path, _staging = _leave_partial_foundation_generation(
        tmp_path,
        tokenizer_model,
        monkeypatch,
    )
    deduplicate = True
    if drift == "source":
        source_path = discovery.sources[0].path
        original = source_path.read_bytes()
        changed = original.replace(b" 0 ", b" 9 ", 1)
        assert changed != original and len(changed) == len(original)
        source_path.write_bytes(changed)
    elif drift == "options":
        deduplicate = False
    else:
        monkeypatch.setattr(
            foundation_prepare,
            "FOUNDATION_PREPROCESSING_SCHEMA",
            "foundation-mixed-objectives-test-drift",
        )

    rebuilt_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        resumed_path,
        deduplicate=deduplicate,
        shard_size=3,
        validation_fraction=0.1,
    )
    clean_stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        tmp_path / "clean",
        deduplicate=deduplicate,
        shard_size=3,
        validation_fraction=0.1,
    )

    assert rebuilt_stats == clean_stats
    assert _tree_bytes(resumed_path) == _tree_bytes(tmp_path / "clean")
    assert not list(tmp_path.glob(".resumed.staging-*"))


def test_locked_partial_checkpoint_is_preserved_for_its_owner(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    discovery, resumed_path, staging = _leave_partial_foundation_generation(
        tmp_path,
        tokenizer_model,
        monkeypatch,
    )
    database = staging / ".foundation-resume.sqlite3"
    owner = sqlite3.connect(database, timeout=0.0, isolation_level=None)
    owner.execute("PRAGMA locking_mode=EXCLUSIVE")
    owner.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(
            foundation_prepare._FoundationResumeBusy,
            match="still active",
        ):
            prepare_foundation_dataset(
                discovery,
                tokenizer_model,
                resumed_path,
                shard_size=3,
                validation_fraction=0.1,
            )
        assert staging.is_dir()
        assert database.is_file()
    finally:
        owner.rollback()
        owner.close()

    stats = prepare_foundation_dataset(
        discovery,
        tokenizer_model,
        resumed_path,
        shard_size=3,
        validation_fraction=0.1,
    )
    assert stats.total_records > 0
    assert not list(tmp_path.glob(".resumed.staging-*"))


def test_output_lock_refuses_a_concurrent_foundation_generation(
    tmp_path,
    tokenizer_model,
) -> None:
    discovery = discover_monolingual_sources(_corpus(tmp_path / "corpus"), ["ko", "ja"])
    dataset = tmp_path / "dataset"

    with foundation_prepare._foundation_output_lock(dataset):
        with pytest.raises(RuntimeError, match="output is locked by another process"):
            prepare_foundation_dataset(discovery, tokenizer_model, dataset)

    assert not dataset.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))
    stats = prepare_foundation_dataset(discovery, tokenizer_model, dataset)
    assert stats.total_records == 100


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

    def forbid_disk_preflight(_path):
        raise AssertionError("a complete generation needs only an atomic rename")

    monkeypatch.setattr(
        foundation_prepare.shutil,
        "disk_usage",
        forbid_disk_preflight,
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


def test_an_empty_discovery_is_refused(tmp_path, tokenizer_model, monkeypatch) -> None:
    discovery = discover_monolingual_sources(tmp_path / "absent", ["ko"])

    def forbid_disk_preflight(_path):
        raise AssertionError("an empty discovery must fail before disk inspection")

    monkeypatch.setattr(
        foundation_prepare.shutil,
        "disk_usage",
        forbid_disk_preflight,
    )
    with pytest.raises(ValueError, match="no usable training files"):
        prepare_foundation_dataset(discovery, tokenizer_model, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_the_report_names_the_drop_reasons(tmp_path, tokenizer_model) -> None:
    _, stats = _prepare(tmp_path, tokenizer_model)
    rendered = "\n".join(render_prepare_report(stats))
    assert "ko:" in rendered and "ja:" in rendered
    assert "train" in rendered


# ── Critical checks through the real training path ─────────────────────


def test_each_sentence_is_trained_once_not_twice(tmp_path, tokenizer_model) -> None:
    """Both directions of a reconstruction example are identical.

    Without ``forward_only``, bidirectional expansion trains every sentence
    exactly twice. This silently doubles the epoch and is invisible in the loss
    curve alone.
    """
    _, stats = _prepare(tmp_path, tokenizer_model)
    dataset = IndexedParallelDataset(tmp_path / "dataset", split="train", bidirectional=True)

    assert len(dataset) == stats.train_records
    assert dataset.forward_only_count == dataset.pair_count


def test_the_collator_produces_denoising_batches(tmp_path, tokenizer_model) -> None:
    """Input starts with ``<denoise_xx>`` and the target is the intact original."""
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
    # Corruption affects only input; the target must contain no <mask> token.
    assert not (batch["labels"] == tokenizer.mask_id).any()
    # Confirm that something was masked; otherwise there is nothing to reconstruct.
    assert (batch["input_ids"] == tokenizer.mask_id).any()


def test_no_translation_direction_tag_appears_in_a_foundation_batch(
    tmp_path,
    tokenizer_model,
) -> None:
    """A ``<2xx>`` tag in a foundation batch would teach translation prematurely."""
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
