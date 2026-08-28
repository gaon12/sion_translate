from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import package_gpu_bundle


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "gaon12")
    _git(root, "config", "user.email", "gokirito12@gmail.com")

    (root / "src").mkdir()
    (root / "src" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "README.md").write_text("training bundle\n", encoding="utf-8")
    (root / "sion_translate.yaml").write_text(
        """\
data:
  language_pair: [ko, ja]
foundation:
  languages: [ko]
""",
        encoding="utf-8",
    )
    (root / "data").mkdir()
    (root / "data" / ".gitkeep").write_text("", encoding="utf-8")
    _git(
        root,
        "add",
        "README.md",
        "sion_translate.yaml",
        "src/train.py",
        "data/.gitkeep",
    )
    _git(root, "commit", "-qm", "initial source")

    (root / "data" / "corpus.jsonl").write_text(
        '{"ko":"안녕","ja":"こんにちは"}\n',
        encoding="utf-8",
    )
    evaluation = root / "data" / "evaluation_only"
    evaluation.mkdir()
    (evaluation / "holdout.jsonl").write_text('{"id":1}\n', encoding="utf-8")

    excluded = root / "data" / "excluded"
    excluded.mkdir()
    (excluded / "secret.jsonl").write_text('{"secret":true}\n', encoding="utf-8")
    for directory in ("artifacts", "runs", ".venv", "translation_queue"):
        generated = root / directory
        generated.mkdir()
        (generated / "do-not-package.txt").write_text("stale\n", encoding="utf-8")
    (root / "untracked.txt").write_text("not allowlisted\n", encoding="utf-8")
    return root


def _manifest(archive_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        return json.loads(archive.read("sion_translate/PACKAGE_MANIFEST.json").decode("utf-8"))


def test_build_is_deterministic_allowlisted_and_verifiable(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = package_gpu_bundle.build_bundle(root, first)
    second_result = package_gpu_bundle.build_bundle(root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.archive_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_result.archive_sha256 == second_result.archive_sha256

    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert names == {
            "sion_translate/README.md",
            "sion_translate/sion_translate.yaml",
            "sion_translate/src/train.py",
            "sion_translate/data/.gitkeep",
            "sion_translate/data/corpus.jsonl",
            "sion_translate/data/evaluation_only/holdout.jsonl",
            "sion_translate/PACKAGE_MANIFEST.json",
            "sion_translate/SHA256SUMS",
        }
        assert all(name.startswith("sion_translate/") for name in names)
        checksums = archive.read("sion_translate/SHA256SUMS").decode("utf-8")
        assert "SHA256SUMS" not in checksums

    manifest = _manifest(first)
    assert manifest["git"] == {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }
    assert manifest["training_contract"] == {
        "schema": "sion-gpu-training-contract-v1",
        "config_path": "sion_translate.yaml",
        "config_sha256": _file_sha256(root / "sion_translate.yaml"),
        "raw_parallel_data_included": True,
        "language_pairs": [["ko", "ja"]],
        "translation_directions": [["ko", "ja"], ["ja", "ko"]],
        "source_only_languages": [],
        "foundation_enabled": True,
        "foundation_languages": ["ko"],
        "paths": {
            "raw_dir": "data",
            "tokenizer_model": "artifacts/tokenizer/sion.model",
            "tokenizer_features": "artifacts/tokenizer/token_features.npz",
            "translation_dataset": "artifacts/dataset",
            "foundation_dataset": "artifacts/foundation_dataset",
        },
    }
    origins = {entry["path"]: entry["origin"] for entry in manifest["files"]}
    assert origins["README.md"] == "git-index"
    assert origins["data/corpus.jsonl"] == "data-jsonl"
    assert origins["data/evaluation_only/holdout.jsonl"] == "evaluation-only"

    archive_result = package_gpu_bundle.verify_archive(first)
    assert archive_result.file_count == 6

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first) as archive:
        archive.extractall(extracted)
    tree_result = package_gpu_bundle.verify_tree(extracted)
    assert tree_result == archive_result


def test_tracked_data_keeps_its_semantic_origin_and_opt_in_boundary(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    monolingual = root / "data" / "corpus" / "ko" / "wiki.txt"
    monolingual.parent.mkdir(parents=True)
    monolingual.write_text("단일어 문장\n", encoding="utf-8")
    _git(
        root,
        "add",
        "data/corpus.jsonl",
        "data/evaluation_only/holdout.jsonl",
        "data/corpus/ko/wiki.txt",
    )
    _git(root, "commit", "-qm", "track every data role")

    default_bundle = tmp_path / "tracked-default.zip"
    package_gpu_bundle.build_bundle(root, default_bundle)
    default_manifest = _manifest(default_bundle)
    default_origins = {entry["path"]: entry["origin"] for entry in default_manifest["files"]}
    assert default_origins["data/corpus.jsonl"] == "data-jsonl"
    assert default_origins["data/evaluation_only/holdout.jsonl"] == "evaluation-only"
    assert "data/corpus/ko/wiki.txt" not in default_origins
    assert default_manifest["training_contract"]["raw_parallel_data_included"] is True

    corpus_bundle = tmp_path / "tracked-with-monolingual.zip"
    package_gpu_bundle.build_bundle(
        root,
        corpus_bundle,
        include_monolingual_corpus=True,
    )
    corpus_origins = {entry["path"]: entry["origin"] for entry in _manifest(corpus_bundle)["files"]}
    assert corpus_origins["data/corpus/ko/wiki.txt"] == "monolingual-corpus"


def _with_monolingual_corpus(root: Path) -> None:
    corpus = root / "data" / "corpus"
    for language, text in (
        ("ko", "단일어 문장\n"),
        ("ja", "単言語の文\n"),
        ("iw", "טקסט בשפה שהוגדרה.\n"),
        ("x-demo", "Private language corpus.\n"),
        ("de", "Nicht konfigurierte Sprache.\n"),
    ):
        directory = corpus / language
        directory.mkdir(parents=True)
        (directory / "wiki.txt").write_text(text, encoding="utf-8")
        # Not a readable monolingual format; must not be shipped.
        (directory / "notes.md").write_text("stray download\n", encoding="utf-8")
    config = root / "sion_translate.yaml"
    config.write_text(
        """\
data:
  language_pairs:
    - [ko, ja]
    - [he, ko]
    - [x-demo, ja]
    - [de, ko]
  source_only_languages: [de]
foundation:
  corpus_dir: data/corpus
  languages: []
  require_all_languages: true
""",
        encoding="utf-8",
    )
    _git(root, "add", config.name)
    _git(root, "commit", "-qm", "add bundle selection config")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_tokenizer_metadata(
    root: Path,
    *,
    monolingual_sources: list[dict[str, object]] | None = None,
) -> None:
    from sion_translate.config import load_config

    config = load_config(root / "sion_translate.yaml")
    translation_languages = [
        language for pair in config.data.configured_language_pairs() for language in pair
    ]
    denoise_languages = list(
        dict.fromkeys([*translation_languages, *config.foundation_languages()])
    )
    tokenizer = root / "artifacts" / "tokenizer"
    model = tokenizer / "sion.model"
    vocab = tokenizer / "sion.vocab"
    features = tokenizer / "token_features.npz"
    raw_source = root / "data" / "corpus.jsonl"
    contract_sources: list[dict[str, object]] = [
        {
            "role": "parallel",
            "path": raw_source.name,
            "size": raw_source.stat().st_size,
            "sha256": _file_sha256(raw_source),
        }
    ]
    for source in monolingual_sources or []:
        contract_sources.append(
            {
                "role": "monolingual",
                "path": source["logical_path"],
                "size": source["size_bytes"],
                "sha256": source["sha256"],
                "language": source["language"],
            }
        )
    reasoning_languages = list(
        dict.fromkeys(
            str(source["language"])
            for source in monolingual_sources or []
            if source["task"] == "reasoning"
        )
    )
    training_contract = {
        "schema": "sion-tokenizer-training-v4",
        "input_traversal_policy": "portable-input-order-v1",
        "sources": contract_sources,
        "language_pairs": [list(pair) for pair in config.data.configured_language_pairs()],
        "translation_directions": [
            list(direction) for direction in config.data.configured_translation_directions()
        ],
        "denoise_languages": denoise_languages,
        "reasoning_languages": reasoning_languages,
        "approximate_split": config.data.approximate_split,
        "source_only_languages": list(config.data.configured_source_only_languages()),
        "train_only_prefixes": list(config.data.configured_synthetic_prefixes()),
        "split_digits": True,
        "monolingual_sample_ratio": config.foundation.tokenizer_sample_ratio,
    }
    (tokenizer / "tokenizer_metadata.json").write_text(
        json.dumps(
            {
                "version": 2,
                "split_digits": True,
                "language_pair": list(config.data.configured_language_pairs()[0]),
                "language_pairs": [list(pair) for pair in config.data.configured_language_pairs()],
                "translation_directions": [
                    list(direction) for direction in config.data.configured_translation_directions()
                ],
                "denoise_languages": denoise_languages,
                "reasoning_languages": reasoning_languages,
                "model_file": model.name,
                "model_sha256": _file_sha256(model),
                "vocab_file": vocab.name,
                "vocab_sha256": _file_sha256(vocab),
                "token_features_file": features.name,
                "token_features_size": features.stat().st_size,
                "token_features_sha256": _file_sha256(features),
                "training_contract": training_contract,
                "training_contract_sha256": _canonical_json_sha256(training_contract),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _with_tokenizer(root: Path, *, complete: bool = True) -> None:
    tokenizer = root / "artifacts" / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "sion.model").write_bytes(b"model")
    if complete:
        (tokenizer / "sion.vocab").write_text("piece\t0\n", encoding="utf-8")
        (tokenizer / "token_features.npz").write_bytes(b"features")
        _rewrite_tokenizer_metadata(root)
    else:
        (tokenizer / "tokenizer_metadata.json").write_text(
            '{"version":2,"split_digits":true}\n',
            encoding="utf-8",
        )


def _artifact_inventory(dataset: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        split_root = dataset / split
        if not split_root.is_dir():
            continue
        for path in sorted(split_root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(dataset).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": _file_sha256(path),
                    }
                )
    return {"schema": "sion-indexed-artifact-inventory-v1", "files": files}


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_dataset(
    root: Path,
    *,
    tokenizer_sha256: str | None = None,
    manifest_tokenizer_sha256: str | None = None,
    include_raw_fingerprint: bool = True,
    include_manifest: bool = True,
    include_completion: bool = True,
) -> None:
    from sion_translate.config import load_config
    from sion_translate.data.prepare import prepare_preprocessing_options
    from sion_translate.fingerprint import PREPROCESSING_SCHEMA

    config = load_config(root / "sion_translate.yaml")
    language_pairs = [list(pair) for pair in config.data.configured_language_pairs()]
    translation_directions = [
        list(direction) for direction in config.data.configured_translation_directions()
    ]
    source_only_languages = list(config.data.configured_source_only_languages())
    preprocessing_options = prepare_preprocessing_options(
        approximate_split=config.data.approximate_split,
        source_only_languages=config.data.configured_source_only_languages(),
        translation_directions=config.data.configured_translation_directions(),
        train_only_prefixes=config.data.configured_synthetic_prefixes(),
        managed_augmentation_prefix=config.data.synthetic_prefix,
        synthetic_sampling_weight=config.data.synthetic_sampling_weight,
        language_pair_count=len(config.data.configured_language_pairs()),
    )
    dataset = root / "artifacts" / "dataset"
    dataset.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        (dataset / split).mkdir()
    (dataset / "train" / "train.00000.idx.npy").write_bytes(b"ids")
    digest = (
        tokenizer_sha256
        or hashlib.sha256(
            (root / "artifacts" / "tokenizer" / "sion.model").read_bytes()
        ).hexdigest()
    )
    fingerprint = {
        "schema": "sion-dataset-fingerprint-v2",
        "preprocessing_schema": PREPROCESSING_SCHEMA,
        "language_pairs": language_pairs,
        "tokenizer_sha256": digest,
        "preprocessing_options": preprocessing_options,
        "files": {
            "corpus.jsonl": {
                "size": (root / "data" / "corpus.jsonl").stat().st_size,
                "sha256": _file_sha256(root / "data" / "corpus.jsonl"),
            }
        },
    }
    raw_fingerprint = dataset / "raw_fingerprint.json"
    if include_raw_fingerprint:
        raw_fingerprint.write_text(json.dumps(fingerprint) + "\n", encoding="utf-8")
    manifest_path = dataset / "manifest.json"
    if include_manifest:
        manifest_path.write_text(
            json.dumps(
                {
                    "format": "sion-indexed-parallel-v6",
                    "language_pairs": language_pairs,
                    "translation_directions": translation_directions,
                    "source_only_languages": source_only_languages,
                    "preprocessing_schema": PREPROCESSING_SCHEMA,
                    "preprocessing_options": preprocessing_options,
                    "fingerprint": {
                        **fingerprint,
                        "tokenizer_sha256": manifest_tokenizer_sha256 or digest,
                    },
                    "artifact_inventory": _artifact_inventory(dataset),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    if include_completion and include_raw_fingerprint and include_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (dataset / ".sion-prepare-complete.json").write_text(
            json.dumps(
                {
                    "schema": "sion-prepare-completion-v1",
                    "manifest_sha256": _file_sha256(manifest_path),
                    "raw_fingerprint_sha256": _file_sha256(raw_fingerprint),
                    "artifact_inventory_sha256": _canonical_json_sha256(
                        manifest["artifact_inventory"]
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _with_foundation_dataset(
    root: Path,
    *,
    tokenizer_sha256: str | None = None,
    manifest_updates: dict[str, object] | None = None,
) -> None:
    from sion_translate.config import load_config

    config = load_config(root / "sion_translate.yaml")
    dataset = root / "artifacts" / "foundation_dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "validation").mkdir()
    index = np.asarray(
        [
            (0, 3, 0, 3, 0, 0, 0, 0, 0, 100, 0, 1, 1),
            (3, 2, 0, 4, 0, 0, 0, 0, 1, 100, 0, 1, 0),
        ],
        dtype=package_gpu_bundle.FOUNDATION_INDEX_DTYPE,
    )
    for split in ("train", "validation"):
        np.save(dataset / split / "00000.idx.npy", index, allow_pickle=False)
        (dataset / split / "00000.src.bin").write_bytes(
            np.asarray([1, 2, 3, 4, 5], dtype=np.uint32).tobytes()
        )
        (dataset / split / "00000.tgt.bin").write_bytes(
            np.asarray([6, 7, 8, 9], dtype=np.uint32).tobytes()
        )
    digest = tokenizer_sha256 or _file_sha256(root / "artifacts" / "tokenizer" / "sion.model")
    sources = [
        {
            "id": 0,
            "language": "ko",
            "logical_path": "ko/corpus.txt",
            "name": "corpus.txt",
            "records": 2,
            "sha256": hashlib.sha256(b"source corpus").hexdigest(),
            "size_bytes": len(b"source corpus"),
            "task": "denoising",
        },
        {
            "id": 1,
            "language": "ko",
            "logical_path": "ko/reasoning.jsonl",
            "name": "reasoning.jsonl",
            "records": 2,
            "sha256": hashlib.sha256(b"reasoning corpus").hexdigest(),
            "size_bytes": len(b"reasoning corpus"),
            "task": "reasoning",
        },
    ]
    _rewrite_tokenizer_metadata(root, monolingual_sources=sources)
    source_identities = [
        {
            "language": source["language"],
            "logical_path": source["logical_path"],
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
            "task": source["task"],
        }
        for source in sources
    ]
    manifest: dict[str, object] = {
        "format": "sion-foundation-indexed-v3",
        "stage": "foundation",
        "release_name": config.foundation.release_name,
        "objective": "span-corruption-denoising+structured-reasoning",
        "languages": list(config.foundation_languages()),
        "language_to_id": {"ko": 0},
        "language_pairs": [["ko", "ko"]],
        "source_only_languages": [],
        "storage_sides": ["src", "tgt"],
        "target_storage": "row-shared-source-v1",
        "index_dtype": json.loads(json.dumps(package_gpu_bundle.FOUNDATION_INDEX_DTYPE.descr)),
        "tokenizer_model": "sion.model",
        "tokenizer_sha256": digest,
        "fingerprint": {"tokenizer_sha256": digest},
        "tokenizer_identity": {
            "schema": "content-sha256-v1",
            "size_bytes": (root / "artifacts" / "tokenizer" / "sion.model").stat().st_size,
            "sha256": digest,
        },
        "preprocessing_schema": "foundation-mixed-objectives-v6",
        "preprocessing_options": {
            "deduplicate": config.foundation.deduplicate,
            "deduplication_backend": (
                "sqlite-blake2b-128-v1" if config.foundation.deduplicate else "disabled"
            ),
            "maximum_characters": config.foundation.maximum_characters,
            "max_tokens": config.data.max_source_length - 2,
            "max_target_tokens": config.data.max_target_length - 1,
            "minimum_characters": config.foundation.minimum_characters,
            "reasoning_sample_share": config.foundation.reasoning_sample_share,
            "shard_size": config.foundation.shard_size,
            "validation_fraction": config.foundation.validation_fraction,
        },
        "source_identity_schema": "corpus-relative-posix-sha256-v1",
        "sources_sha256": _canonical_json_sha256(source_identities),
        "sources": sources,
        "language_sampling": {
            "alpha": config.foundation.language_sampling_alpha,
            "minimum_share": config.foundation.minimum_language_share,
            "weights": {"ko": 1.0},
            "counts": {"ko": 4},
            "warnings": [],
        },
        "reasoning": {
            "contract": "prompt-to-delimited-trace-v1",
            "languages": ["ko"],
            "records": 2,
            "sample_share": 0.05,
            "trace_symbols": ["<think>", "</think>", "<answer>", "</answer>"],
        },
        "stats": {"train_records": 2, "validation_records": 2},
        "artifact_inventory": _artifact_inventory(dataset),
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    (dataset / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )


def test_the_monolingual_corpus_ships_only_when_asked_for(tmp_path: Path) -> None:
    """Only configured foundation languages should consume bundle space."""

    root = _repository(tmp_path)
    _with_monolingual_corpus(root)

    default_archive = tmp_path / "default.zip"
    package_gpu_bundle.build_bundle(root, default_archive)
    with zipfile.ZipFile(default_archive) as archive:
        # data/corpus.jsonl is a parallel shard. Only the data/corpus/ tree is opt-in.
        assert not any("data/corpus/" in name for name in archive.namelist())

    included = tmp_path / "with-corpus.zip"
    package_gpu_bundle.build_bundle(root, included, include_monolingual_corpus=True)
    with zipfile.ZipFile(included) as archive:
        names = set(archive.namelist())
    assert "sion_translate/data/corpus/ko/wiki.txt" in names
    assert "sion_translate/data/corpus/ja/wiki.txt" in names
    assert "sion_translate/data/corpus/iw/wiki.txt" in names
    assert "sion_translate/data/corpus/x-demo/wiki.txt" in names
    assert "sion_translate/data/corpus/de/wiki.txt" not in names
    # Unsupported formats must not consume upload space.
    assert not any(name.endswith("notes.md") for name in names)

    origins = {entry["path"]: entry["origin"] for entry in _manifest(included)["files"]}
    assert origins["data/corpus/ko/wiki.txt"] == "monolingual-corpus"
    package_gpu_bundle.verify_archive(included)


def test_the_tokenizer_ships_only_when_asked_for_and_only_if_complete(tmp_path: Path) -> None:
    """A partial tokenizer must fail before the GPU upload."""

    root = _repository(tmp_path)
    _with_tokenizer(root, complete=False)

    with pytest.raises(package_gpu_bundle.BundleError, match="sion.vocab"):
        package_gpu_bundle.build_bundle(root, tmp_path / "partial.zip", include_tokenizer=True)

    _with_tokenizer(root, complete=True)
    included = tmp_path / "with-tokenizer.zip"
    package_gpu_bundle.build_bundle(root, included, include_tokenizer=True)
    with zipfile.ZipFile(included) as archive:
        names = set(archive.namelist())
    assert "sion_translate/artifacts/tokenizer/sion.model" in names
    assert "sion_translate/artifacts/tokenizer/tokenizer_metadata.json" in names
    assert "sion_translate/artifacts/tokenizer/token_features.npz" in names

    origins = {entry["path"]: entry["origin"] for entry in _manifest(included)["files"]}
    assert origins["artifacts/tokenizer/sion.model"] == "tokenizer"
    package_gpu_bundle.verify_archive(included)


def test_tokenizer_sidecar_hashes_and_inventory_are_enforced(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    metadata_path = root / "artifacts" / "tokenizer" / "tokenizer_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["token_features_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="token_features_sha256"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "bad-tokenizer-sidecar.zip",
            include_tokenizer=True,
        )

    _rewrite_tokenizer_metadata(root)
    (root / "artifacts" / "tokenizer" / "stale.tmp").write_bytes(b"stale")
    with pytest.raises(package_gpu_bundle.BundleError, match="unexpected"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "unexpected-tokenizer-file.zip",
            include_tokenizer=True,
        )


def test_the_dataset_needs_the_tokenizer_that_produced_its_ids(tmp_path: Path) -> None:
    """Prepared token ids are reusable only with their authenticated tokenizer."""

    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root)

    with pytest.raises(package_gpu_bundle.BundleError, match="--with-tokenizer"):
        package_gpu_bundle.build_bundle(root, tmp_path / "dataset-only.zip", include_dataset=True)

    both = tmp_path / "both.zip"
    package_gpu_bundle.build_bundle(root, both, include_tokenizer=True, include_dataset=True)
    with zipfile.ZipFile(both) as archive:
        names = set(archive.namelist())
    assert "sion_translate/artifacts/dataset/train/train.00000.idx.npy" in names
    assert "sion_translate/artifacts/tokenizer/sion.model" in names

    origins = {entry["path"]: entry["origin"] for entry in _manifest(both)["files"]}
    assert origins["artifacts/dataset/train/train.00000.idx.npy"] == "dataset"
    archive_result = package_gpu_bundle.verify_archive(both)
    extracted = tmp_path / "both-extracted"
    with zipfile.ZipFile(both) as archive:
        archive.extractall(extracted)
    assert package_gpu_bundle.verify_tree(extracted) == archive_result


def test_the_dataset_refuses_a_different_tokenizer_fingerprint(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root, tokenizer_sha256="0" * 64)

    with pytest.raises(package_gpu_bundle.BundleError, match="dataset tokenizer mismatch"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "mismatch.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_the_dataset_requires_all_generation_sidecars(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root, include_raw_fingerprint=False)

    with pytest.raises(package_gpu_bundle.BundleError, match="missing required sidecars"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "missing-fingerprint.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_the_dataset_requires_a_completion_marker(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root, include_completion=False)

    with pytest.raises(package_gpu_bundle.BundleError, match="missing required sidecars"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "missing-completion.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_the_dataset_identity_sources_must_agree(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root, manifest_tokenizer_sha256="0" * 64)

    with pytest.raises(package_gpu_bundle.BundleError, match="sidecar disagrees"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "disagree.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_prepared_artifacts_must_match_the_selected_language_graph(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root)
    config = root / "sion_translate.yaml"
    config.write_text(
        "data:\n  language_pair: [de, fr]\nfoundation:\n  languages: [de]\n",
        encoding="utf-8",
    )
    _git(root, "add", config.name)
    _git(root, "commit", "-qm", "change the configured language graph")

    with pytest.raises(package_gpu_bundle.BundleError, match="language_pairs disagrees"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "wrong-language-graph.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_tokenizer_sources_must_match_the_current_parallel_corpus(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    (root / "data" / "corpus.jsonl").write_text(
        '{"ko":"토크나이저 이후 변경","ja":"変更後"}\n',
        encoding="utf-8",
    )

    with pytest.raises(package_gpu_bundle.BundleError, match="parallel-source provenance"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "stale-tokenizer.zip",
            include_tokenizer=True,
        )


def test_tokenizer_requires_an_authenticated_training_contract(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    metadata_path = root / "artifacts" / "tokenizer" / "tokenizer_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["training_contract"]
    del metadata["training_contract_sha256"]
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="training_contract"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "unauthenticated-tokenizer.zip",
            include_tokenizer=True,
        )


def test_tokenizer_bundle_requires_digit_splitting(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    metadata_path = root / "artifacts" / "tokenizer" / "tokenizer_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["split_digits"] = False
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="split_digits=true"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "merged-digit-tokenizer.zip",
            include_tokenizer=True,
        )


def test_included_raw_corpus_must_match_the_prepared_dataset(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root)
    (root / "data" / "corpus.jsonl").write_text(
        '{"ko":"바뀜","ja":"変更"}\n',
        encoding="utf-8",
    )

    with pytest.raises(package_gpu_bundle.BundleError, match="differs from the prepared dataset"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "stale-prepared-dataset.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_dataset_payload_must_match_the_manifest_inventory(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_dataset(root)
    (root / "artifacts" / "dataset" / "train" / "train.00000.idx.npy").write_bytes(b"changed ids")

    with pytest.raises(package_gpu_bundle.BundleError, match="artifact identity mismatch"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "changed-dataset.zip",
            include_tokenizer=True,
            include_dataset=True,
        )


def test_foundation_dataset_has_an_explicit_authenticated_opt_in(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(root)

    with pytest.raises(package_gpu_bundle.BundleError, match="requires --with-tokenizer"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "foundation-without-tokenizer.zip",
            include_foundation_dataset=True,
        )

    archive_path = tmp_path / "with-foundation-dataset.zip"
    package_gpu_bundle.build_bundle(
        root,
        archive_path,
        include_tokenizer=True,
        include_foundation_dataset=True,
    )
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "sion_translate/artifacts/foundation_dataset/train/00000.idx.npy" in names
    origins = {entry["path"]: entry["origin"] for entry in _manifest(archive_path)["files"]}
    assert origins["artifacts/foundation_dataset/train/00000.idx.npy"] == "foundation-dataset"
    package_gpu_bundle.verify_archive(archive_path)


def test_bundle_foundation_markers_match_the_current_preparer_contract() -> None:
    from sion_translate.artifacts import FOUNDATION_RELEASE_NAME
    from sion_translate.data.prepare import SHARED_TARGET_INDEX_DTYPE
    from sion_translate.data.prepare_foundation import (
        FOUNDATION_INDEX_FORMAT,
        FOUNDATION_PREPROCESSING_SCHEMA,
        FOUNDATION_SOURCE_IDENTITY_SCHEMA,
        FOUNDATION_TOKENIZER_IDENTITY_SCHEMA,
    )

    assert package_gpu_bundle.FOUNDATION_DATASET_FORMAT == FOUNDATION_INDEX_FORMAT
    assert package_gpu_bundle.FOUNDATION_RELEASE_NAME == FOUNDATION_RELEASE_NAME
    assert package_gpu_bundle.FOUNDATION_PREPROCESSING_SCHEMA == FOUNDATION_PREPROCESSING_SCHEMA
    assert package_gpu_bundle.FOUNDATION_SOURCE_IDENTITY_SCHEMA == FOUNDATION_SOURCE_IDENTITY_SCHEMA
    assert (
        package_gpu_bundle.FOUNDATION_TOKENIZER_IDENTITY_SCHEMA
        == FOUNDATION_TOKENIZER_IDENTITY_SCHEMA
    )
    assert package_gpu_bundle.FOUNDATION_INDEX_DTYPE == SHARED_TARGET_INDEX_DTYPE


def test_foundation_dataset_rejects_a_stale_tokenizer_identity(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(root, tokenizer_sha256="0" * 64)

    with pytest.raises(package_gpu_bundle.BundleError, match="tokenizer mismatch"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "stale-foundation.zip",
            include_tokenizer=True,
            include_foundation_dataset=True,
        )


def test_foundation_dataset_rejects_the_legacy_v2_generation(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(
        root,
        manifest_updates={"format": "sion-foundation-indexed-v2"},
    )

    with pytest.raises(package_gpu_bundle.BundleError, match="sion-foundation-indexed-v3"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "legacy-foundation.zip",
            include_tokenizer=True,
            include_foundation_dataset=True,
        )


@pytest.mark.parametrize(
    ("manifest_updates", "message"),
    [
        ({"stage": "translation"}, "stage marker"),
        ({"release_name": "translation"}, "release_name disagrees"),
        ({"preprocessing_schema": "foundation-mixed-objectives-v5"}, "mixed-objectives-v6"),
        ({"target_storage": "duplicated-target-v1"}, "target_storage marker"),
        ({"storage_sides": ["source", "target"]}, "storage_sides marker"),
        ({"index_dtype": [["src_offset", "<u8"]]}, "index_dtype marker"),
    ],
)
def test_foundation_dataset_requires_every_v3_contract_marker(
    tmp_path: Path,
    manifest_updates: dict[str, object],
    message: str,
) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(root, manifest_updates=manifest_updates)

    with pytest.raises(package_gpu_bundle.BundleError, match=message):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "invalid-foundation-marker.zip",
            include_tokenizer=True,
            include_foundation_dataset=True,
        )


def test_foundation_dataset_accepts_the_configured_release_name(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    config = root / "sion_translate.yaml"
    config.write_text(
        "data:\n  language_pair: [ko, ja]\nfoundation:\n  languages: [ko]\n"
        "  release_name: sion_base\n",
        encoding="utf-8",
    )
    _git(root, "add", config.name)
    _git(root, "commit", "-qm", "configure a custom foundation release")
    _with_tokenizer(root)
    _with_foundation_dataset(root)

    archive = tmp_path / "custom-foundation-release.zip"
    package_gpu_bundle.build_bundle(
        root,
        archive,
        include_tokenizer=True,
        include_foundation_dataset=True,
    )
    package_gpu_bundle.verify_archive(archive)


def test_foundation_sources_must_match_the_included_monolingual_corpus(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(root)
    corpus_root = root / "data" / "corpus" / "ko"
    corpus_root.mkdir(parents=True)
    (corpus_root / "corpus.txt").write_bytes(b"source corpus")
    reasoning = corpus_root / "reasoning.jsonl"
    reasoning.write_bytes(b"reasoning corpus")

    valid = tmp_path / "foundation-sources.zip"
    package_gpu_bundle.build_bundle(
        root,
        valid,
        include_monolingual_corpus=True,
        include_tokenizer=True,
        include_foundation_dataset=True,
    )
    reasoning.write_bytes(b"changed reasoning corpus")

    with pytest.raises(package_gpu_bundle.BundleError, match="monolingual corpus differs"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "stale-foundation-sources.zip",
            include_monolingual_corpus=True,
            include_tokenizer=True,
            include_foundation_dataset=True,
        )


def test_foundation_dataset_rejects_self_consistent_invalid_target_aliases(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(root)
    dataset = root / "artifacts" / "foundation_dataset"
    index_path = dataset / "train" / "00000.idx.npy"
    index = np.load(index_path, allow_pickle=False)
    index["target_shared"][0] = 0
    np.save(index_path, index, allow_pickle=False)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_inventory"] = _artifact_inventory(dataset)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="aliases disagree"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "invalid-foundation-alias.zip",
            include_tokenizer=True,
            include_foundation_dataset=True,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("language_sampling", "weights", {"ko": 0.5}, "weight is invalid"),
        ("language_sampling", "counts", {"ko": 3}, "counts disagree"),
        ("reasoning", "sample_share", 0.10, "reasoning policy contradicts"),
    ],
)
def test_foundation_dataset_rejects_malformed_gpu_sampling_policy(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root)
    _with_foundation_dataset(root)
    manifest_path = root / "artifacts" / "foundation_dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[section][field] = replacement
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match=message):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "invalid-foundation-sampling.zip",
            include_tokenizer=True,
            include_foundation_dataset=True,
        )


def test_archive_and_tree_verification_recheck_dataset_tokenizer_semantics(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    _with_dataset(root)
    sources = package_gpu_bundle._collect_sources(
        root,
        include_tokenizer=True,
        include_dataset=True,
    )

    # Emulate a tokenizer replacement after source selection. Rewriting the
    # tokenizer sidecar makes that artifact internally valid, while the dataset
    # remains bound to the previous model hash.
    (root / "artifacts" / "tokenizer" / "sion.model").write_bytes(b"replacement model")
    _rewrite_tokenizer_metadata(root)
    archive = tmp_path / "semantic-mismatch.zip"
    commit, tree = package_gpu_bundle._git_identity(root)
    package_gpu_bundle._write_archive(archive, sources, commit, tree)

    with pytest.raises(package_gpu_bundle.BundleError, match="dataset tokenizer mismatch"):
        package_gpu_bundle.verify_archive(archive)

    extracted = tmp_path / "semantic-mismatch"
    with zipfile.ZipFile(archive) as source:
        source.extractall(extracted)
    with pytest.raises(package_gpu_bundle.BundleError, match="dataset tokenizer mismatch"):
        package_gpu_bundle.verify_tree(extracted)


def test_the_default_bundle_still_refuses_stale_artifacts(tmp_path: Path) -> None:
    """One opt-in must never expose all stale files below artifacts/."""

    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    (root / "artifacts" / "dataset").mkdir(parents=True)
    (root / "artifacts" / "dataset" / "train.bin").write_bytes(b"stale")

    included = tmp_path / "tokenizer-only.zip"
    package_gpu_bundle.build_bundle(root, included, include_tokenizer=True)
    with zipfile.ZipFile(included) as archive:
        names = set(archive.namelist())

    assert "sion_translate/artifacts/tokenizer/sion.model" in names
    assert not any("artifacts/dataset" in name for name in names)
    assert not any("do-not-package" in name for name in names)


def test_requesting_an_absent_optional_tree_is_an_error(tmp_path: Path) -> None:
    """An explicitly requested missing input must fail before upload."""

    root = _repository(tmp_path)
    config = root / "sion_translate.yaml"
    config.write_text(
        "data:\n  language_pair: [ko, ja]\nfoundation:\n  corpus_dir: data/corpus\n",
        encoding="utf-8",
    )
    _git(root, "add", config.name)
    _git(root, "commit", "-qm", "add corpus config")
    with pytest.raises(package_gpu_bundle.BundleError, match="configured corpus directory"):
        package_gpu_bundle.build_bundle(
            root, tmp_path / "no-corpus.zip", include_monolingual_corpus=True
        )


def test_bundle_rejects_a_config_path_the_default_train_command_would_ignore(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    alternate = root / "configs" / "alternate.yaml"
    alternate.parent.mkdir()
    alternate.write_text(
        "data:\n  language_pair: [ko, ja]\nfoundation:\n  languages: [ko]\n",
        encoding="utf-8",
    )
    _git(root, "add", alternate.relative_to(root).as_posix())
    _git(root, "commit", "-qm", "add an alternate training config")

    with pytest.raises(package_gpu_bundle.BundleError, match="default sion-train command"):
        package_gpu_bundle.build_bundle(
            root,
            tmp_path / "alternate-config.zip",
            config_path="configs/alternate.yaml",
        )


def test_build_refuses_a_dirty_tracked_tree(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "README.md").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="tracked files are not clean"):
        package_gpu_bundle.build_bundle(root, tmp_path / "bundle.zip")


def test_build_requires_training_and_evaluation_corpora(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "data" / "corpus.jsonl").unlink()
    with pytest.raises(package_gpu_bundle.BundleError, match=r"data/\*\.jsonl"):
        package_gpu_bundle.build_bundle(root, tmp_path / "missing-training.zip")

    (root / "data" / "corpus.jsonl").write_text('{"ko":"가","ja":"あ"}\n', encoding="utf-8")
    (root / "data" / "evaluation_only" / "holdout.jsonl").unlink()
    with pytest.raises(package_gpu_bundle.BundleError, match="evaluation_only"):
        package_gpu_bundle.build_bundle(root, tmp_path / "missing-evaluation.zip")


def test_build_rejects_portable_metadata_name_collisions(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    collision = root / "package_manifest.json"
    collision.write_text("{}\n", encoding="utf-8")
    _git(root, "add", collision.name)
    _git(root, "commit", "-qm", "add colliding name")

    with pytest.raises(package_gpu_bundle.BundleError, match="portable path collision"):
        package_gpu_bundle.build_bundle(root, tmp_path / "bundle.zip")


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "bundle.zip"
    output.write_bytes(b"keep me")

    with pytest.raises(package_gpu_bundle.BundleError, match="--overwrite"):
        package_gpu_bundle.build_bundle(root, output)
    assert output.read_bytes() == b"keep me"

    result = package_gpu_bundle.build_bundle(root, output, overwrite=True)
    assert result.output_path == output
    package_gpu_bundle.verify_archive(output)


def test_non_overwrite_publication_refuses_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "bundle.zip"
    original_verify = package_gpu_bundle.verify_archive

    def create_destination_during_verification(
        archive_path: Path,
    ) -> package_gpu_bundle.VerificationResult:
        result = original_verify(archive_path)
        output.write_bytes(b"other process")
        return result

    monkeypatch.setattr(
        package_gpu_bundle, "verify_archive", create_destination_during_verification
    )
    with pytest.raises(package_gpu_bundle.BundleError, match="appeared while building"):
        package_gpu_bundle.build_bundle(root, output)

    assert output.read_bytes() == b"other process"


def test_archive_verification_rejects_portable_member_name_collisions(tmp_path: Path) -> None:
    archive_path = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        package_gpu_bundle._write_bytes(archive, "Payload.txt", b"first")
        package_gpu_bundle._write_bytes(archive, "payload.txt", b"second")

    with pytest.raises(package_gpu_bundle.BundleError, match="portable member-name collision"):
        package_gpu_bundle.verify_archive(archive_path)


@pytest.mark.parametrize(
    "path",
    ["AUX.txt", "data/name. ", "data/bad?.jsonl", "data/../escape.jsonl"],
)
def test_bundle_paths_reject_cross_platform_unsafe_names(path: str) -> None:
    with pytest.raises(package_gpu_bundle.BundleError):
        package_gpu_bundle._validated_relative_path(path)


def test_tree_verification_detects_payload_tampering(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    package_gpu_bundle.build_bundle(root, archive_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    payload = extracted / "sion_translate" / "README.md"
    payload.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="payload hash mismatch"):
        package_gpu_bundle.verify_tree(extracted)


def test_archive_verification_detects_payload_tampering(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    package_gpu_bundle.build_bundle(root, archive_path)
    tampered_path = tmp_path / "tampered.zip"

    with (
        zipfile.ZipFile(archive_path) as source,
        zipfile.ZipFile(
            tampered_path,
            mode="w",
        ) as destination,
    ):
        for info in source.infolist():
            content = source.read(info)
            if info.filename == "sion_translate/README.md":
                content = b"T" + content[1:]
            destination.writestr(info, content)

    with pytest.raises(package_gpu_bundle.BundleError, match="payload hash mismatch"):
        package_gpu_bundle.verify_archive(tampered_path)


def test_failed_verification_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "bundle.zip"
    output.write_bytes(b"previous bundle")

    def fail_verification(_path: Path) -> package_gpu_bundle.VerificationResult:
        raise package_gpu_bundle.BundleError("injected verification failure")

    monkeypatch.setattr(package_gpu_bundle, "verify_archive", fail_verification)
    with pytest.raises(package_gpu_bundle.BundleError, match="injected"):
        package_gpu_bundle.build_bundle(root, output, overwrite=True)

    assert output.read_bytes() == b"previous bundle"
    assert not list(tmp_path.glob(".bundle.zip.*.tmp"))


def test_build_refuses_insufficient_staging_space_before_creating_a_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr(
        package_gpu_bundle.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(package_gpu_bundle.BundleError, match="insufficient free disk space"):
        package_gpu_bundle.build_bundle(root, tmp_path / "no-space.zip")

    assert not list(tmp_path.glob(".no-space.zip.*.tmp"))


def test_build_detects_a_changed_archive_during_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "bundle.zip"
    original_publish = package_gpu_bundle._publish_archive

    def publish_then_corrupt(temporary_path: Path, destination: Path, *, overwrite: bool) -> None:
        original_publish(temporary_path, destination, overwrite=overwrite)
        destination.write_bytes(b"corrupted after publication")

    monkeypatch.setattr(package_gpu_bundle, "_publish_archive", publish_then_corrupt)
    with pytest.raises(package_gpu_bundle.BundleError, match="differ from the verified"):
        package_gpu_bundle.build_bundle(root, output)


def test_source_metadata_drift_is_detected_while_a_member_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source_path = root / "payload.bin"
    source_path.write_bytes(b"stable payload")
    entry = package_gpu_bundle.SourceEntry(
        relative_path=package_gpu_bundle.PurePosixPath("payload.bin"),
        source_path=source_path,
        origin="git-index",
        mode="100644",
    )
    original_copy = package_gpu_bundle._copy_and_hash

    def copy_then_touch(source, destination):
        result = original_copy(source, destination)
        metadata = source_path.stat()
        os.utime(
            source_path,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
        )
        return result

    monkeypatch.setattr(package_gpu_bundle, "_copy_and_hash", copy_then_touch)
    with (
        zipfile.ZipFile(tmp_path / "drift.zip", mode="w") as archive,
        pytest.raises(package_gpu_bundle.BundleError, match="changed while"),
    ):
        package_gpu_bundle._write_source(archive, entry)


def test_windows_reparse_attributes_are_treated_as_links() -> None:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    metadata = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=reparse_flag)

    assert package_gpu_bundle._metadata_is_link_like(metadata)


def test_archive_verification_rejects_nondeterministic_member_order(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    reordered_path = tmp_path / "reordered.zip"
    package_gpu_bundle.build_bundle(root, archive_path)

    with (
        zipfile.ZipFile(archive_path) as source,
        zipfile.ZipFile(reordered_path, mode="w") as destination,
    ):
        for info in reversed(source.infolist()):
            destination.writestr(info, source.read(info))

    with pytest.raises(package_gpu_bundle.BundleError, match="deterministic manifest order"):
        package_gpu_bundle.verify_archive(reordered_path)


def test_archive_verification_rejects_non_zip64_member_headers(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    rewritten_path = tmp_path / "rewritten.zip"
    package_gpu_bundle.build_bundle(root, archive_path)

    with (
        zipfile.ZipFile(archive_path) as source,
        zipfile.ZipFile(rewritten_path, mode="w") as destination,
    ):
        for source_info in source.infolist():
            rewritten_info = copy.copy(source_info)
            rewritten_info.create_version = 20
            rewritten_info.extract_version = 20
            destination.writestr(rewritten_info, source.read(source_info))

    with pytest.raises(package_gpu_bundle.BundleError, match="deterministic ZIP64 headers"):
        package_gpu_bundle.verify_archive(rewritten_path)
