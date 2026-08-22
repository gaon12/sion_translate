"""Direction-scoped, resumable backtranslation artifact contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from sion_translate.augmentation import (
    AugmentationIdentity,
    JobProgress,
    augmentation_shard_path,
    build_job_identity,
    count_prepared_direction_pairs,
    load_augmentation_registry,
    reconcile_job_identity,
    run_augmentation_job,
    snapshot_file,
    validate_prepared_raw_contract,
    write_job_progress,
)
from sion_translate.config import DataConfig
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
) -> AugmentationIdentity:
    return build_job_identity(
        pair=pair,
        mono_language=mono_language,
        input_snapshot=snapshot_file(mono_path),
        model_identity=_MODEL_IDENTITY,
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


def test_registry_rejects_a_wrong_direction_shard(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    identity = _job_identity(mono_path)
    progress = JobProgress(identity)
    write_job_progress(data_dir, progress)
    shard_path = augmentation_shard_path(data_dir, "bt_", identity, 0)
    shard_path.write_text(
        json.dumps(
            {
                "en": "A generated source sentence.",
                "de": "Ein echter deutscher Zielsatz.",
                "synthetic": True,
                "training_direction": ["de", "en"],
                "_sion_augment": {
                    "schema": "sion-augment-row-v1",
                    "job_id": identity.job_id,
                    "input_sha256": identity.input.sha256,
                    "input_line": 0,
                    "mono_text_sha256": snapshot_file(mono_path).sha256,
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

    with pytest.raises(RuntimeError, match="번역 중 단일어 입력이 변경"):
        run_augmentation_job(
            MutatingTranslator(),
            mono_path=mono_path,
            data_dir=data_dir,
            synthetic_prefix="bt_",
            progress=JobProgress(identity),
            accepted_budget=1,
            batch_size=1,
            seen_mono_hashes=set(),
        )

    assert not list(data_dir.glob("bt_*.jsonl"))
    assert not list(data_dir.glob("*.partial"))


def test_existing_job_identity_cannot_be_reused_for_changed_input(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Zielsatz.\n", encoding="utf-8")
    original = _job_identity(mono_path)
    write_job_progress(data_dir, JobProgress(original))
    registry = load_augmentation_registry(data_dir, "bt_", ())

    mono_path.write_text("Ein geänderter deutscher Zielsatz.\n", encoding="utf-8")
    changed = _job_identity(mono_path)
    with pytest.raises(RuntimeError, match="입력·모델·토크나이저"):
        reconcile_job_identity(registry, changed)


def test_pair_qualified_shards_do_not_collide_or_escape(tmp_path: Path) -> None:
    mono_path = tmp_path / "news.en.txt"
    mono_path.write_text("A real English target sentence.\n", encoding="utf-8")
    en_de = _job_identity(mono_path, pair=("en", "de"), mono_language="en")
    en_fr = build_job_identity(
        pair=("en", "fr"),
        mono_language="en",
        input_snapshot=snapshot_file(mono_path),
        model_identity=_MODEL_IDENTITY,
        generator_tokenizer_sha256=_TOKENIZER_IDENTITY,
        num_beams=2,
        max_new_tokens=64,
    )

    first = augmentation_shard_path(tmp_path, "bt_", en_de, 0)
    second = augmentation_shard_path(tmp_path, "bt_", en_fr, 0)

    assert first != second
    assert first.parent.resolve() == tmp_path.resolve()
    assert second.parent.resolve() == tmp_path.resolve()
