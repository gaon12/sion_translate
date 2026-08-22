"""Direction-scoped, resumable backtranslation artifact contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from sion_translate.augmentation import (
    AUGMENT_ROW_SCHEMA,
    AugmentationIdentity,
    JobProgress,
    augmentation_identity_sha256,
    augmentation_shard_path,
    build_job_identity,
    count_prepared_direction_pairs,
    load_augmentation_registry,
    reconcile_job_identity,
    run_augmentation_job,
    snapshot_file,
    synthetic_budget,
    validate_prepared_raw_contract,
    write_job_progress,
)
from sion_translate.config import AppConfig, DataConfig
from sion_translate.data import IndexedParallelDataset
from sion_translate.data.prepare import prepare_dataset

_MODEL_IDENTITY = "1" * 64
_TOKENIZER_IDENTITY = "2" * 64


class StubTokenizer:
    languages = ("en", "de", "sw", "ar")

    def __init__(self, _model_path: str | Path):
        pass

    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(character) for character in text]


def _job_identity(
    mono_path: Path,
    *,
    pair: tuple[str, str] = ("en", "de"),
    mono_language: str = "de",
    model_identity: str = _MODEL_IDENTITY,
    synthetic_prefix: str = "bt_",
) -> AugmentationIdentity:
    return build_job_identity(
        synthetic_prefix=synthetic_prefix,
        pair=pair,
        mono_language=mono_language,
        input_snapshot=snapshot_file(mono_path),
        model_identity=model_identity,
        generator_tokenizer_sha256=_TOKENIZER_IDENTITY,
        num_beams=2,
        max_new_tokens=64,
    )


def _prepare_multigraph_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DataConfig, Path]:
    prepare_module = importlib.import_module("sion_translate.data.prepare")
    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.append(
            {
                "en": f"English real sentence number {index} for the large pair.",
                "de": f"Deutscher echter Satz Nummer {index} für das große Paar.",
            }
        )
    for index in range(2):
        rows.append(
            {
                "sw": f"sentensi halisi ya Kiswahili nambari {index}",
                "ar": f"هذه جملة عربية حقيقية رقم {index}",
            }
        )
    (raw_dir / "real.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (raw_dir / "mixed.jsonl").write_text(
        json.dumps(
            {
                "sw": "sentensi sintetiki ya chanzo pekee",
                "ar": "هذه جملة هدف حقيقية منفصلة",
                "synthetic": True,
                "training_direction": ["sw", "ar"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    pairs = (("en", "de"), ("sw", "ar"))
    directions = (("en", "de"), ("de", "en"), ("sw", "ar"), ("ar", "sw"))
    prepare_dataset(
        [str(raw_dir)],
        tokenizer_path,
        dataset_dir,
        language_pairs=pairs,
        translation_directions=directions,
        approximate_split=True,
        managed_augmentation_prefix="bt_",
        num_workers=1,
    )
    data = DataConfig(
        raw_dir=str(raw_dir),
        tokenizer_model=str(tokenizer_path),
        dataset_dir=str(dataset_dir),
        language_pairs=[list(pair) for pair in pairs],
        translation_directions=[list(direction) for direction in directions],
    )
    return data, raw_dir


def test_prepared_accounting_is_row_and_direction_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _ = _prepare_multigraph_fixture(tmp_path, monkeypatch)

    counts = count_prepared_direction_pairs(
        data.dataset_dir,
        (("sw", "ar"), ("ar", "sw")),
    )

    assert counts[("sw", "ar")].real == 2
    assert counts[("sw", "ar")].synthetic == 1
    assert counts[("ar", "sw")].real == 2
    assert counts[("ar", "sw")].synthetic == 0


def test_raw_contract_fails_closed_for_stale_graph_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, raw_dir = _prepare_multigraph_fixture(tmp_path, monkeypatch)

    fingerprint = validate_prepared_raw_contract(data, augment_prefix="bt_")
    assert {item.name for item in fingerprint.files} == {"mixed.jsonl", "real.jsonl"}

    stale_graph = DataConfig(**{**data.__dict__})
    stale_graph.translation_directions = [
        ["en", "de"],
        ["de", "en"],
        ["sw", "ar"],
    ]
    with pytest.raises(RuntimeError, match="언어 그래프"):
        validate_prepared_raw_contract(stale_graph, augment_prefix="bt_")

    (raw_dir / "new_real.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="아직 준비되지 않은"):
        validate_prepared_raw_contract(data, augment_prefix="bt_")
    (raw_dir / "new_real.jsonl").unlink()

    (raw_dir / "real.jsonl").unlink()
    with pytest.raises(RuntimeError, match="삭제"):
        validate_prepared_raw_contract(data, augment_prefix="bt_")


def test_registry_rejects_unowned_legacy_or_corrupt_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "bt_legacy.jsonl").write_text(
        json.dumps({"en": "generated", "de": "real", "synthetic": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ledger가 인증하지 않은"):
        load_augmentation_registry(data_dir, "bt_", ())


def test_training_raw_scan_rejects_unowned_bidirectional_synthetic_rows(
    tmp_path: Path,
) -> None:
    from sion_translate.cli.train import scan_configured_raw_data

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "real.jsonl").write_text(
        json.dumps({"en": "A real English sentence.", "de": "Ein echter deutscher Satz."}) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "bt_legacy.jsonl").write_text(
        json.dumps(
            {
                "en": "An unauthenticated pseudo source.",
                "de": "Ein echtes Ziel.",
                "synthetic": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"tokenizer")
    config = AppConfig(
        data=DataConfig(
            raw_dir=str(raw_dir),
            tokenizer_model=str(tokenizer_path),
            dataset_dir=str(tmp_path / "dataset"),
            language_pair=["en", "de"],
        )
    )

    with pytest.raises(ValueError, match="ledger가 인증하지 않은"):
        scan_configured_raw_data(config, raw_dir, tokenizer_path)


def test_training_raw_scan_recovers_and_authenticates_a_valid_orphan(tmp_path: Path) -> None:
    from sion_translate.cli.train import scan_configured_raw_data

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "real.jsonl").write_text(
        json.dumps({"en": "A real English sentence.", "de": "Ein echter deutscher Satz."}) + "\n",
        encoding="utf-8",
    )
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echtes deutsches Übersetzungsziel.\n", encoding="utf-8")
    identity = _job_identity(mono_path)

    class Translator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated English source sentence." for _ in texts]

    result = run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=raw_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    write_job_progress(raw_dir, JobProgress(identity))
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"tokenizer")
    config = AppConfig(
        data=DataConfig(
            raw_dir=str(raw_dir),
            tokenizer_model=str(tokenizer_path),
            dataset_dir=str(tmp_path / "dataset"),
            language_pair=["en", "de"],
        )
    )

    fingerprint = scan_configured_raw_data(config, raw_dir, tokenizer_path)
    recovered = load_augmentation_registry(raw_dir, "bt_", ())

    assert {file.name for file in fingerprint.files} == {
        "real.jsonl",
        result.progress.shards[0].name,
    }
    assert recovered.jobs[identity.job_id].accepted_rows == 1


def test_managed_orphan_prepares_only_its_authenticated_training_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_module = importlib.import_module("sion_translate.data.prepare")
    monkeypatch.setattr(prepare_module, "SionTokenizer", StubTokenizer)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "real.jsonl").write_text(
        json.dumps({"en": "A real English sentence.", "de": "Ein echter deutscher Satz."}) + "\n",
        encoding="utf-8",
    )
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echtes deutsches Übersetzungsziel.\n", encoding="utf-8")
    identity = _job_identity(mono_path)

    class Translator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated English source sentence." for _ in texts]

    run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=raw_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    # Recreate the rename→final-ledger crash window. prepare_dataset must run
    # registry recovery before it treats the public JSONL as training input.
    write_job_progress(raw_dir, JobProgress(identity))
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub tokenizer")
    dataset_dir = tmp_path / "dataset"

    stats = prepare_dataset(
        [str(raw_dir)],
        tokenizer_path,
        dataset_dir,
        language_pairs=(("en", "de"),),
        translation_directions=(("en", "de"), ("de", "en")),
        validation_fraction=0.0,
        test_fraction=0.0,
        filter_quality=False,
        dedup_backend="memory",
        managed_augmentation_prefix="bt_",
        num_workers=1,
    )
    dataset = IndexedParallelDataset(
        dataset_dir,
        "train",
        bidirectional=True,
        include_metadata=True,
    )
    synthetic_items = [
        dataset[index] for index in range(len(dataset)) if dataset[index]["synthetic"]
    ]

    assert stats.forward_only_pairs == 1
    assert len(synthetic_items) == 1
    assert (
        synthetic_items[0]["src_language"],
        synthetic_items[0]["target_language"],
    ) == ("en", "de")
    assert synthetic_items[0]["reverse_direction_trained"] is False
    assert synthetic_items[0]["training_direction"] == ["en", "de"]


def test_registry_rejects_a_wrong_direction_shard(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    identity = _job_identity(mono_path)
    progress = JobProgress(identity)
    write_job_progress(data_dir, progress)
    shard_path = augmentation_shard_path(data_dir, identity, 0)
    shard_path.write_text(
        json.dumps(
            {
                "en": "A generated source sentence.",
                "de": "Ein echter deutscher Zielsatz.",
                "synthetic": True,
                "training_direction": ["de", "en"],
                "_sion_augment": {
                    "schema": AUGMENT_ROW_SCHEMA,
                    "job_id": identity.job_id,
                    "identity_sha256": augmentation_identity_sha256(identity),
                    "input_sha256": identity.input.sha256,
                    "input_line": 0,
                    "mono_text_sha256": hashlib.sha256(
                        "Ein echter deutscher Zielsatz.".encode()
                    ).hexdigest(),
                    "synthetic_text_sha256": hashlib.sha256(
                        "A generated source sentence.".encode()
                    ).hexdigest(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="손상되거나"):
        load_augmentation_registry(data_dir, "bt_", ())


def test_streaming_reaches_valid_rows_after_leading_quality_rejection(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    first = "Dieser Satz wird als identische Übersetzung verworfen."
    second = "Dies ist ein gültiger echter Zielsatz."
    mono_path.write_text(f"{first}\n{second}\n", encoding="utf-8")
    identity = _job_identity(mono_path)

    class Translator:
        batches: list[list[str]] = []

        def translate(self, texts: list[str], **_kwargs: object) -> list[str]:
            self.batches.append(list(texts))
            return [
                text if text == first else "This is a valid generated source sentence."
                for text in texts
            ]

    translator = Translator()
    result = run_augmentation_job(
        translator,
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=2,
        seen_mono_hashes=set(),
    )

    assert result.written == 1
    assert result.quality_filtered == 1
    assert result.progress.accepted_rows == 1
    assert result.progress.cursor_line == 2
    assert max(map(len, translator.batches)) <= 2


def test_partial_job_resumes_when_direction_budget_increases(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text(
        "Erster echter Zielsatz.\nZweiter echter Zielsatz.\nDritter echter Zielsatz.\n",
        encoding="utf-8",
    )
    identity = _job_identity(mono_path)

    class Translator:
        def translate(self, texts: list[str], **_kwargs: object) -> list[str]:
            return [f"Generated source sentence number {index}." for index, _ in enumerate(texts)]

    seen: set[str] = set()
    first = run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=2,
        seen_mono_hashes=seen,
    )
    second = run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=first.progress,
        accepted_budget=2,
        batch_size=2,
        seen_mono_hashes=seen,
    )
    registry = load_augmentation_registry(data_dir, "bt_", ())
    recovered = registry.jobs[identity.job_id]

    assert first.progress.cursor_line == 1
    assert second.written == 2
    assert second.progress.accepted_rows == 3
    assert recovered.accepted_rows == 3
    assert len(recovered.shards) == 2


def test_published_orphan_shard_is_recovered_after_ledger_crash(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    identity = _job_identity(mono_path)

    class Translator:
        def translate(self, texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    result = run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    # Simulate power loss after shard publication but before the ledger update.
    write_job_progress(data_dir, JobProgress(identity))

    registry = load_augmentation_registry(data_dir, "bt_", ())
    recovered = registry.jobs[identity.job_id]

    assert result.written == 1
    assert recovered.accepted_rows == 1
    assert recovered.cursor_line == 1
    assert recovered.eof is False


def test_orphan_shard_rejects_synthetic_corruption_and_wrong_generator_identity(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    identity = _job_identity(mono_path)

    class Translator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    result = run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    shard_path = data_dir / result.progress.shards[0].name
    row = json.loads(shard_path.read_text(encoding="utf-8"))
    write_job_progress(data_dir, JobProgress(identity))

    row["en"] = "Silently corrupted synthetic source."
    shard_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="소유권이 불명확"):
        load_augmentation_registry(data_dir, "bt_", ())

    row["_sion_augment"]["synthetic_text_sha256"] = hashlib.sha256(row["en"].encode()).hexdigest()
    other_identity = _job_identity(mono_path, model_identity="9" * 64)
    row["_sion_augment"]["identity_sha256"] = augmentation_identity_sha256(other_identity)
    shard_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="소유권이 불명확"):
        load_augmentation_registry(data_dir, "bt_", ())


def test_input_change_during_translation_publishes_nothing(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    identity = _job_identity(mono_path)

    class MutatingTranslator:
        def translate(self, texts: list[str], **_kwargs: object) -> list[str]:
            with mono_path.open("a", encoding="utf-8") as handle:
                handle.write("Während der Übersetzung hinzugefügt.\n")
            return ["A valid generated source sentence." for _ in texts]

    seen: set[str] = set()
    with pytest.raises(RuntimeError, match="번역 중 단일어 입력이 변경"):
        run_augmentation_job(
            MutatingTranslator(),
            mono_path=mono_path,
            data_dir=data_dir,
            synthetic_prefix="bt_",
            progress=JobProgress(identity),
            accepted_budget=1,
            batch_size=1,
            seen_mono_hashes=seen,
        )

    assert seen == set()
    assert not list(data_dir.glob("bt_*.jsonl"))
    assert not list(data_dir.glob("*.partial"))

    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")

    class WorkingTranslator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    registry = load_augmentation_registry(data_dir, "bt_", ())
    retried = run_augmentation_job(
        WorkingTranslator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=reconcile_job_identity(registry, identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=seen,
    )
    assert retried.written == 1
    assert len(seen) == 1


def test_failed_pristine_reservation_can_retry_with_fixed_generation_settings(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    original = _job_identity(mono_path)

    class FailingTranslator:
        @staticmethod
        def translate(_texts: list[str], **_kwargs: object) -> list[str]:
            raise RuntimeError("invalid generation settings")

    with pytest.raises(RuntimeError, match="invalid generation settings"):
        run_augmentation_job(
            FailingTranslator(),
            mono_path=mono_path,
            data_dir=data_dir,
            synthetic_prefix="bt_",
            progress=JobProgress(original),
            accepted_budget=1,
            batch_size=1,
            seen_mono_hashes=set(),
        )

    registry = load_augmentation_registry(data_dir, "bt_", ())
    fixed = build_job_identity(
        synthetic_prefix="bt_",
        pair=original.pair,
        mono_language=original.mono_language,
        input_snapshot=original.input,
        model_identity=original.model_identity,
        generator_tokenizer_sha256=original.generator_tokenizer_sha256,
        num_beams=1,
        max_new_tokens=32,
    )
    retried = reconcile_job_identity(registry, fixed)
    assert fixed.job_id != original.job_id

    class WorkingTranslator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    result = run_augmentation_job(
        WorkingTranslator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=retried,
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )

    assert result.written == 1
    assert result.progress.identity == fixed
    assert not list(data_dir.glob("*.partial"))


def test_changed_input_creates_a_new_job_without_discarding_committed_history(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    original = _job_identity(mono_path)

    class Translator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(original),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    registry = load_augmentation_registry(data_dir, "bt_", ())

    mono_path.write_text("Ein geänderter deutscher Zielsatz.\n", encoding="utf-8")
    changed = _job_identity(mono_path)
    new_progress = reconcile_job_identity(registry, changed)

    assert changed.job_id != original.job_id
    assert new_progress == JobProgress(changed)
    assert registry.jobs[original.job_id].accepted_rows == 1


def test_completed_old_model_does_not_block_new_file_with_new_model(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    old_path = tmp_path / "news.de.txt"
    old_path.write_text("Ein alter echter deutscher Zielsatz.\n", encoding="utf-8")
    old_identity = _job_identity(old_path)

    class Translator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    run_augmentation_job(
        Translator(),
        mono_path=old_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(old_identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    registry = load_augmentation_registry(data_dir, "bt_", ())

    new_path = tmp_path / "corpus.de.txt"
    new_path.write_text("Ein neuer echter deutscher Zielsatz.\n", encoding="utf-8")
    upgraded_old = _job_identity(old_path, model_identity="8" * 64)
    upgraded_new = _job_identity(new_path, model_identity="8" * 64)

    upgraded_seen = registry.mono_hashes_by_direction()[upgraded_old.training_direction]
    suppressed = run_augmentation_job(
        Translator(),
        mono_path=old_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=reconcile_job_identity(registry, upgraded_old),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=upgraded_seen,
    )
    new_progress = reconcile_job_identity(registry, upgraded_new)
    result = run_augmentation_job(
        Translator(),
        mono_path=new_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=new_progress,
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=registry.mono_hashes_by_direction()[upgraded_new.training_direction],
    )

    assert upgraded_old.job_id != old_identity.job_id
    assert suppressed.written == 0
    assert result.written == 1
    assert load_augmentation_registry(data_dir, "bt_", ()).jobs.keys() >= {
        old_identity.job_id,
        upgraded_new.job_id,
    }


def test_new_model_can_retry_a_sentence_rejected_by_the_old_model(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_text = "Ein echter deutscher Zielsatz."
    mono_path.write_text(mono_text + "\n", encoding="utf-8")
    old_identity = _job_identity(mono_path)

    class CopyingTranslator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return list(texts)

    rejected = run_augmentation_job(
        CopyingTranslator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(old_identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    registry = load_augmentation_registry(data_dir, "bt_", ())
    upgraded = _job_identity(mono_path, model_identity="7" * 64)

    class ImprovedTranslator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated English source sentence." for _ in texts]

    retried = run_augmentation_job(
        ImprovedTranslator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=reconcile_job_identity(registry, upgraded),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=registry.mono_hashes_by_direction().get(
            upgraded.training_direction,
            set(),
        ),
    )

    assert rejected.written == 0
    assert rejected.quality_filtered == 1
    assert registry.jobs[old_identity.job_id].mono_text_hashes == frozenset()
    assert retried.written == 1


def test_registry_keeps_old_prefix_jobs_and_unrelated_partial_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    identity = _job_identity(mono_path, synthetic_prefix="bt_")

    class Translator:
        @staticmethod
        def translate(texts: list[str], **_kwargs: object) -> list[str]:
            return ["A valid generated source sentence." for _ in texts]

    run_augmentation_job(
        Translator(),
        mono_path=mono_path,
        data_dir=data_dir,
        synthetic_prefix="bt_",
        progress=JobProgress(identity),
        accepted_budget=1,
        batch_size=1,
        seen_mono_hashes=set(),
    )
    notes = data_dir / ".next_bt_notes.partial"
    notes.write_text("user-owned", encoding="utf-8")

    registry = load_augmentation_registry(data_dir, "next_bt_", ())

    assert identity.job_id in registry.jobs
    assert notes.read_text(encoding="utf-8") == "user-owned"


def test_synthetic_budget_accepts_huge_finite_ratios_without_overflow() -> None:
    assert synthetic_budget(2, 0, 1e308) > 10**300
    with pytest.raises(ValueError, match="finite"):
        synthetic_budget(2, 0, float("inf"))


def test_pair_qualified_shards_do_not_collide_or_escape(tmp_path: Path) -> None:
    mono_path = tmp_path / "news.en.txt"
    mono_path.write_text("A real English target sentence.\n", encoding="utf-8")
    en_de = _job_identity(mono_path, pair=("en", "de"), mono_language="en")
    en_fr = build_job_identity(
        synthetic_prefix="bt_",
        pair=("en", "fr"),
        mono_language="en",
        input_snapshot=snapshot_file(mono_path),
        model_identity=_MODEL_IDENTITY,
        generator_tokenizer_sha256=_TOKENIZER_IDENTITY,
        num_beams=2,
        max_new_tokens=64,
    )

    first = augmentation_shard_path(tmp_path, en_de, 0)
    second = augmentation_shard_path(tmp_path, en_fr, 0)

    assert first != second
    assert first.parent.resolve() == tmp_path.resolve()
    assert second.parent.resolve() == tmp_path.resolve()
