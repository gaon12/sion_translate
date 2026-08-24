"""Test arbitrary language pairs and data-augmentation safeguards."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import sion_translate.cli.augment as augment_cli
from sion_translate.augmentation import (
    AugmentationRegistry,
    DirectionCount,
    FileSnapshot,
    JobProgress,
    JobRunResult,
    build_job_identity,
    snapshot_file,
    synthetic_budget,
)
from sion_translate.cli.augment import (
    generator_identity,
    preflight_backtranslation_directions,
    resolve_augmentation_destination,
    resolve_augmentation_pair,
)
from sion_translate.config import AppConfig, DataConfig
from sion_translate.data import IndexedParallelDataset, SionBatchCollator
from sion_translate.data.prepare import prepare_dataset
from sion_translate.data.quality import assess_pair
from sion_translate.tokenizer import SionTokenizer, train_tokenizer


def test_generic_language_pair_passes_quality() -> None:
    # Script alone cannot distinguish two Latin-script languages. Skip that
    # check while retaining other damage checks, such as identical text.
    ok = assess_pair(
        "The weather is nice today.", "Das Wetter ist heute schön.", languages=("en", "de")
    )
    assert ok.accepted
    identical = assess_pair("Same text.", "Same text.", languages=("en", "de"))
    assert not identical.accepted


def write_en_de_jsonl(path: Path, count: int = 60) -> None:
    en_words = [
        "today",
        "tomorrow",
        "weather",
        "good",
        "bad",
        "school",
        "office",
        "train",
        "book",
        "friend",
    ]
    de_words = [
        "heute",
        "morgen",
        "Wetter",
        "gut",
        "schlecht",
        "Schule",
        "Büro",
        "Zug",
        "Buch",
        "Freund",
    ]
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
    """Changing only the language pair keeps the complete pipeline operational.

    This tokenizer-to-collator check detects any remaining hard-coded ``ko-ja``
    assumptions.
    """
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
    tokenizer = SionTokenizer(model_path)
    # Discover the language pair automatically from the vocabulary's <2xx> tags.
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

    collator = SionBatchCollator(tokenizer, max_source_length=64, max_target_length=64)
    batch = collator([forward, reverse])
    # Direction tags: en->de starts with <2de>, while de->en starts with <2en>.
    assert batch["input_ids"][0, 0].item() == tokenizer.language_tags["de"]
    assert batch["input_ids"][1, 0].item() == tokenizer.language_tags["en"]


def test_synthetic_files_are_train_only(tmp_path: Path) -> None:
    """Synthetic files and records marked synthetic must remain training-only."""
    real = tmp_path / "real.jsonl"
    synthetic = tmp_path / "synthetic_numeric_data38.jsonl"
    write_en_de_jsonl(real, count=60)
    write_en_de_jsonl(synthetic, count=40)
    # Alter the synthetic file so that it does not duplicate real data.
    lines = synthetic.read_text(encoding="utf-8").splitlines()
    with synthetic.open("w", encoding="utf-8") as handle:
        for line in lines:
            row = json.loads(line)
            handle.write(
                json.dumps({k: "synthetic " + v for k, v in row.items()}, ensure_ascii=False) + "\n"
            )
    with real.open("a", encoding="utf-8") as handle:
        for index in range(10):
            handle.write(
                json.dumps(
                    {
                        "en": f"Explicit synthetic record source number {index + 1000}.",
                        "de": f"Expliziter synthetischer Zielsatz Nummer {index + 1000}.",
                        "synthetic": True,
                    }
                )
                + "\n"
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
    synthetic_stats = by_name["synthetic_numeric_data38.jsonl"]
    assert synthetic_stats["validation"] == 0
    assert synthetic_stats["test"] == 0
    assert synthetic_stats["train"] == synthetic_stats["valid_pairs"] > 0
    assert by_name["real.jsonl"]["synthetic_pairs"] == 10
    assert set(manifest["train_only_prefixes"]) >= {
        "bt_",
        "queue_bt_",
        "concat_",
        "revise_",
        "synthetic_",
    }
    assert manifest["fingerprint"]["language_pairs"] == [["en", "de"]]
    assert manifest["fingerprint"]["tokenizer_sha256"]
    # Real data still participates in all three splits.
    assert by_name["real.jsonl"]["validation"] > 0
    train_dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=False)
    assert train_dataset.pair_synthetic_flags.sum() == manifest["stats"]["synthetic_pairs"]


def test_synthetic_budget_caps_generation() -> None:
    # With 1,000 real pairs and a 1.0 ratio, allow at most 1,000 synthetic pairs.
    assert synthetic_budget(1000, 0, 1.0) == 1000
    assert synthetic_budget(1000, 800, 1.0) == 200
    assert synthetic_budget(1000, 1200, 1.0) == 0  # Already over budget: generate none.
    assert synthetic_budget(1000, 0, 0.5) == 500


def test_augmentation_pair_is_resolved_from_the_model_artifact() -> None:
    trained_pairs = (("de", "fr"), ("sw", "ar"))

    assert resolve_augmentation_pair(("ar", "sw"), trained_pairs) == ("sw", "ar")
    with pytest.raises(SystemExit, match="not present in the model"):
        resolve_augmentation_pair(("ko", "ja"), trained_pairs)
    with pytest.raises(SystemExit, match="--language-pair"):
        resolve_augmentation_pair(None, trained_pairs)

    assert resolve_augmentation_pair(
        ("PT-br", "zh-hant"),
        (("pt-BR", "zh-Hant"),),
    ) == ("pt-BR", "zh-Hant")


def test_augmentation_destination_requires_the_same_physical_pair() -> None:
    assert resolve_augmentation_destination(("ar", "sw"), (("de", "fr"), ("sw", "ar"))) == (
        "sw",
        "ar",
    )
    with pytest.raises(SystemExit, match="current training configuration"):
        resolve_augmentation_destination(("ar", "sw"), (("de", "fr"),))

    assert resolve_augmentation_destination(
        ("PT-br", "zh-hant"),
        (("zh-Hant", "pt-BR"),),
    ) == ("zh-Hant", "pt-BR")


def test_augmentation_discovers_canonical_language_alias_filenames(tmp_path: Path) -> None:
    mono_dir = tmp_path / "mono"
    mono_dir.mkdir()
    portuguese = mono_dir / "news.PT-br.txt"
    chinese = mono_dir / "news.zh-hant.txt"
    ignored = mono_dir / "news.en.txt"
    for path in (portuguese, chinese, ignored):
        path.write_text("one sentence\n", encoding="utf-8")

    jobs = augment_cli._discover_mono_files(  # noqa: SLF001 - CLI regression contract
        mono_dir,
        ("pt-BR", "zh-Hant"),
    )

    assert jobs == [(portuguese, "pt-BR"), (chinese, "zh-Hant")]


def test_augment_preflight_checks_generation_and_destination_edges(
    tmp_path: Path,
) -> None:
    mono_files = [(tmp_path / "news.ar.txt", "ar")]

    with pytest.raises(SystemExit, match="missing model generation directions: ar→sw"):
        preflight_backtranslation_directions(
            ("sw", "ar"),
            mono_files,
            (("sw", "ar"),),
            (("sw", "ar"),),
        )
    with pytest.raises(SystemExit, match="missing destination training directions: sw→ar"):
        preflight_backtranslation_directions(
            ("sw", "ar"),
            mono_files,
            (("ar", "sw"),),
            (("ar", "sw"),),
        )

    preflight_backtranslation_directions(
        ("sw", "ar"),
        mono_files,
        (("ar", "sw"),),
        (("sw", "ar"),),
    )

    preflight_backtranslation_directions(
        ("PT-br", "zh-hant"),
        [(tmp_path / "news.zh-hant.txt", "ZH-hant")],
        (("zh-Hant", "pt-BR"),),
        (("PT-br", "zh-hant"),),
    )


def test_augment_locks_before_loading_the_model_and_pair_mismatch_is_nonmutating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    config = AppConfig(
        data=DataConfig(
            raw_dir=str(raw_dir),
            dataset_dir=str(tmp_path / "artifacts" / "dataset"),
            tokenizer_model=str(tmp_path / "tokenizer.model"),
            language_pair=["de", "fr"],
            translation_directions=[["de", "fr"]],
        )
    )
    events: list[str] = []
    (tmp_path / "model.pt").write_bytes(b"model artifact")

    @contextmanager
    def fake_locks(roots):  # type: ignore[no-untyped-def]
        assert {Path(root).resolve() for root in roots} == {
            raw_dir.resolve(),
            (tmp_path / "artifacts").resolve(),
        }
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    class PairMismatchTranslator:
        language_pairs = (("sw", "ar"),)
        translation_directions = (("ar", "sw"),)

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            assert events == ["lock-enter"]
            events.append("model-load")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(augment_cli, "config_from_raw", lambda _raw: config)
    monkeypatch.setattr(
        augment_cli,
        "validate_prepared_raw_contract",
        lambda *_args, **_kwargs: type("Fingerprint", (), {"files": ()})(),
    )
    monkeypatch.setattr(
        augment_cli,
        "load_augmentation_registry",
        lambda *_args, **_kwargs: AugmentationRegistry({}, frozenset()),
    )
    monkeypatch.setattr(augment_cli, "artifact_locks", fake_locks)
    monkeypatch.setattr(augment_cli, "Translator", PairMismatchTranslator)
    monkeypatch.setattr(sys, "argv", ["sion-augment", "--model", "model.pt"])

    with pytest.raises(SystemExit, match="current training configuration"):
        augment_cli.main()

    assert events == ["lock-enter", "model-load", "lock-exit"]
    assert not raw_dir.exists()


def test_augment_rejects_excess_generation_length_before_creating_a_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    config = AppConfig(
        data=DataConfig(
            raw_dir=str(raw_dir),
            dataset_dir=str(tmp_path / "artifacts" / "dataset"),
            tokenizer_model=str(tmp_path / "tokenizer.model"),
            language_pair=["sw", "ar"],
            translation_directions=[["sw", "ar"]],
        )
    )
    (tmp_path / "model.pt").write_bytes(b"model artifact")

    class TranslatorWithShortContext:
        language_pairs = (("sw", "ar"),)
        translation_directions = (("ar", "sw"),)

        class model_config:
            max_seq_len = 64

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    @contextmanager
    def fake_locks(_roots):  # type: ignore[no-untyped-def]
        yield

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(augment_cli, "config_from_raw", lambda _raw: config)
    monkeypatch.setattr(
        augment_cli,
        "validate_prepared_raw_contract",
        lambda *_args, **_kwargs: type("Fingerprint", (), {"files": ()})(),
    )
    monkeypatch.setattr(
        augment_cli,
        "load_augmentation_registry",
        lambda *_args, **_kwargs: AugmentationRegistry({}, frozenset()),
    )
    monkeypatch.setattr(augment_cli, "artifact_locks", fake_locks)
    monkeypatch.setattr(augment_cli, "Translator", TranslatorWithShortContext)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sion-augment", "--model", "model.pt", "--max-new-tokens", "65"],
    )

    with pytest.raises(SystemExit, match="model maximum length"):
        augment_cli.main()

    assert not raw_dir.exists()


def test_generator_identity_binds_loaded_artifact_and_generation_contract() -> None:
    metadata = {
        "source": {"sha256": "1" * 64},
        "tokenizer": {"sha256": "2" * 64},
        "release_name": "sion_translate",
        "release_version": "1.5",
        "step": 42,
        "pipeline": {"stage": "translation"},
        "feature_flags": {"revision": True},
        "capabilities": {"translation": True},
        "quantization": {"format": "fp32"},
        "generation_defaults": {"reasoning_level": 0},
    }
    translator = SimpleNamespace(
        export_metadata=metadata,
        tokenizer_metadata=None,
        language_pairs=(("en", "de"),),
        translation_directions=(("de", "en"),),
    )
    artifact = FileSnapshot("model.pt", 100, "3" * 64)
    base, tokenizer_sha = generator_identity(translator, artifact)  # type: ignore[arg-type]

    quantized = SimpleNamespace(**translator.__dict__)
    quantized.export_metadata = {**metadata, "quantization": {"format": "int8"}}
    quantized_identity, _ = generator_identity(quantized, artifact)  # type: ignore[arg-type]
    other_artifact_identity, _ = generator_identity(  # type: ignore[arg-type]
        translator,
        FileSnapshot("renamed.pt", 101, "4" * 64),
    )

    assert tokenizer_sha == "2" * 64
    assert len({base, quantized_identity, other_artifact_identity}) == 3


def test_source_progress_rejects_impossible_cursor_and_eof_state(tmp_path: Path) -> None:
    mono_path = tmp_path / "news.de.txt"
    mono_path.write_text("Ein echter deutscher Satz.\n\n", encoding="utf-8")
    identity = build_job_identity(
        synthetic_prefix="bt_",
        pair=("en", "de"),
        mono_language="de",
        input_snapshot=snapshot_file(mono_path),
        model_identity="1" * 64,
        generator_tokenizer_sha256="2" * 64,
        num_beams=1,
        max_new_tokens=32,
    )

    with pytest.raises(ValueError, match="cursor"):
        augment_cli._source_has_remaining_text(  # noqa: SLF001 - CLI regression contract
            mono_path,
            JobProgress(identity, cursor_line=100),
        )
    with pytest.raises(ValueError, match="EOF"):
        augment_cli._source_has_remaining_text(  # noqa: SLF001 - CLI regression contract
            mono_path,
            JobProgress(identity, cursor_line=0, eof=True),
        )

    assert not augment_cli._source_has_remaining_text(  # noqa: SLF001 - CLI regression contract
        mono_path,
        JobProgress(identity, cursor_line=1),
    )


def test_model_upgrade_preflights_only_unseen_asymmetric_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mono_dir = tmp_path / "mono"
    mono_dir.mkdir()
    old_path = mono_dir / "old.ar.txt"
    old_text = "هذه جملة عربية منشورة من قبل."
    old_path.write_text(old_text + "\n", encoding="utf-8")
    new_path = mono_dir / "new.sw.txt"
    new_path.write_text("Hii ni sentensi mpya ya Kiswahili.\n", encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model-b")
    old_identity = build_job_identity(
        synthetic_prefix="bt_",
        pair=("sw", "ar"),
        mono_language="ar",
        input_snapshot=snapshot_file(old_path),
        model_identity="1" * 64,
        generator_tokenizer_sha256="2" * 64,
        num_beams=1,
        max_new_tokens=32,
    )
    old_hash = hashlib.sha256(old_text.encode("utf-8")).hexdigest()
    registry = AugmentationRegistry(
        {
            old_identity.job_id: JobProgress(
                old_identity,
                cursor_line=1,
                eof=True,
                mono_text_hashes=frozenset({old_hash}),
            )
        },
        frozenset(),
    )
    config = AppConfig(
        data=DataConfig(
            raw_dir=str(raw_dir),
            dataset_dir=str(tmp_path / "dataset"),
            tokenizer_model=str(tmp_path / "tokenizer.model"),
            language_pair=["sw", "ar"],
            translation_directions=[["ar", "sw"]],
        )
    )

    class FakeTokenizer:
        @staticmethod
        def encode(text: str) -> list[int]:
            return list(range(len(text)))

    class ModelBTranslator:
        language_pairs = (("sw", "ar"),)
        translation_directions = (("sw", "ar"),)
        export_metadata = {
            "source": {"sha256": "3" * 64},
            "tokenizer": {"sha256": "2" * 64},
            "release_name": "sion_translate",
            "release_version": "1.5",
            "step": 2,
        }
        tokenizer_metadata = None
        tokenizer = FakeTokenizer()

        class model_config:
            max_seq_len = 128

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    calls: list[str] = []

    def fake_run(_translator, *, mono_path: Path, progress: JobProgress, **_kwargs):
        calls.append(mono_path.name)
        written = int(mono_path == new_path)
        return JobRunResult(
            JobProgress(
                progress.identity,
                cursor_line=1,
                eof=True,
                mono_text_hashes=progress.mono_text_hashes,
            ),
            written,
            0,
            int(mono_path == old_path),
            0,
        )

    monkeypatch.setattr(
        augment_cli,
        "validate_prepared_raw_contract",
        lambda *_args, **_kwargs: type("Fingerprint", (), {"files": ()})(),
    )
    monkeypatch.setattr(augment_cli, "load_augmentation_registry", lambda *_args: registry)
    monkeypatch.setattr(augment_cli, "Translator", ModelBTranslator)
    monkeypatch.setattr(
        augment_cli,
        "count_prepared_direction_pairs",
        lambda _dataset, directions: {
            direction: DirectionCount(real=10) for direction in directions
        },
    )
    monkeypatch.setattr(augment_cli, "run_augmentation_job", fake_run)
    args = SimpleNamespace(
        model=str(model_path),
        tokenizer=None,
        language_pair=None,
        mono_dir=str(mono_dir),
        num_beams=1,
        max_new_tokens=32,
        max_ratio=1.0,
        batch_size=2,
    )

    augment_cli._run_locked(args, config)  # noqa: SLF001 - CLI integration regression

    assert calls == ["old.ar.txt", "new.sw.txt"]


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
    tokenizer = SionTokenizer(model_path)
    # Choose ordinary tokens outside the protected <slot_n> ID range (16-79).
    assert max(tokenizer.slot_ids) < 100
    item = {
        "src": list(range(100, 130)),  # 30 ordinary tokens
        "tgt": [100, 101, 102],
        "src_language": "en",
        "target_language": "de",
        "src_register": 0,
        "target_register": 0,
    }
    random.seed(3)
    clean = SionBatchCollator(
        tokenizer, max_source_length=64, max_target_length=64, source_token_dropout=0.0
    )([dict(item)])
    random.seed(3)
    dropped = SionBatchCollator(
        tokenizer, max_source_length=64, max_target_length=64, source_token_dropout=0.4
    )([dict(item)])
    clean_len = int(clean["attention_mask"][0].sum())
    dropped_len = int(dropped["attention_mask"][0].sum())
    assert dropped_len < clean_len  # Some tokens were dropped.
    assert dropped_len >= 3  # Keep the tag, at least one token, and EOS.
    # Augmentation does not change the target labels.
    assert dropped["labels"].tolist() == clean["labels"].tolist()


def test_source_token_dropout_preserves_revision_separator_and_both_segments(
    tmp_path: Path,
) -> None:
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
    tokenizer = SionTokenizer(model_path)
    assert tokenizer.draft_id is not None
    item = {
        "src": [100, 101, tokenizer.draft_id, 102, 103],
        "tgt": [104],
        "src_language": "en",
        "target_language": "de",
        "src_register": 0,
        "target_register": 0,
    }
    collated = SionBatchCollator(
        tokenizer,
        max_source_length=64,
        max_target_length=64,
        source_token_dropout=1.0,
        denoise_probability=1.0,
    )([item])
    content = collated["input_ids"][0][collated["attention_mask"][0].bool()].tolist()[1:-1]

    assert content.count(tokenizer.draft_id) == 1
    separator = content.index(tokenizer.draft_id)
    assert separator > 0
    assert separator < len(content) - 1
    assert collated["input_ids"][0, 0].item() == tokenizer.language_tags["de"]
    assert collated["labels"][0, :2].tolist() == [104, tokenizer.eos_id]
