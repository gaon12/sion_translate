from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import sion_translate.cli.train as train_module
import sion_translate.training.distributed as distributed_module
from sion_translate.cli.train import (
    build_collator_args,
    construct_training_model,
    dataloader_runtime_kwargs,
    export_final_model,
    find_existing_checkpoint,
    missing_final_export_dependencies,
    preflight_final_export_dependencies,
    release_stage_resources,
    requires_ddp_unused_parameter_detection,
    shutdown_dataloader,
    tokenizer_policy_problem,
    validate_training_capacity,
)
from sion_translate.config import (
    AppConfig,
    ExperimentalConfig,
    ModelConfig,
    config_from_raw,
)
from sion_translate.data.integrity import build_dataset_artifact_inventory
from sion_translate.fingerprint import file_sha256
from sion_translate.model.transformer import SionForConditionalGeneration
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.distributed import (
    distributed_precision_dtype,
    fsdp_reduce_dtype,
    initialize_distributed,
    parallelize_model,
    resolve_parallel_strategy,
)


class WorkerIterator:
    def __init__(self) -> None:
        self.stopped = False

    def _shutdown_workers(self) -> None:
        self.stopped = True


class LoaderStub:
    def __init__(self) -> None:
        self._iterator = WorkerIterator()


def test_embedded_gpu_bundle_requires_its_authenticated_config(tmp_path: Path) -> None:
    config = tmp_path / "sion_translate.yaml"
    config.write_text("data:\n  language_pair: [ko, ja]\n", encoding="utf-8")
    manifest = {
        "training_contract": {
            "config_path": "sion_translate.yaml",
            "config_sha256": file_sha256(config),
        }
    }
    (tmp_path / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )

    train_module.preflight_embedded_bundle_config(None, root=tmp_path)
    with pytest.raises(RuntimeError, match="refusing alternate config"):
        train_module.preflight_embedded_bundle_config(
            str(tmp_path / "alternate.yaml"),
            root=tmp_path,
        )

    config.write_text("data:\n  language_pair: [de, fr]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the GPU bundle"):
        train_module.preflight_embedded_bundle_config(None, root=tmp_path)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "1"],
        ["--max-steps", "1"],
        ["--posttrain-epochs", "1"],
        ["--posttrain-steps", "1"],
        ["--skip-posttraining"],
        ["--resume-from", "runs/manual/checkpoint"],
    ],
)
def test_embedded_gpu_bundle_rejects_every_config_mutating_cli_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    config = tmp_path / "sion_translate.yaml"
    config.write_text("data:\n  language_pair: [de, fr]\n", encoding="utf-8")
    (tmp_path / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(
            {
                "training_contract": {
                    "config_path": "sion_translate.yaml",
                    "config_sha256": file_sha256(config),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    args = train_module.build_parser().parse_args(arguments)

    with pytest.raises(RuntimeError, match="config-mutating command-line overrides"):
        train_module.resolve_config(args)


def test_embedded_gpu_bundle_allows_nonmutating_prepare_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "sion_translate.yaml"
    config.write_text("data:\n  language_pair: [de, fr]\n", encoding="utf-8")
    (tmp_path / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(
            {
                "training_contract": {
                    "config_path": "sion_translate.yaml",
                    "config_sha256": file_sha256(config),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    args = train_module.build_parser().parse_args(["--prepare-only"])

    resolved, _raw, source = train_module.resolve_config(args)

    assert resolved.data.configured_language_pairs() == (("de", "fr"),)
    assert source == "sion_translate.yaml"


def test_dataloader_runtime_settings_separate_training_and_validation() -> None:
    device = torch.device("cuda")
    training = dataloader_runtime_kwargs(12, device, training=True)
    validation = dataloader_runtime_kwargs(3, device, training=False)
    single_process = dataloader_runtime_kwargs(0, torch.device("cpu"), training=True)

    assert training == {
        "num_workers": 12,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    assert validation == {
        "num_workers": 3,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    assert single_process == {"num_workers": 0, "pin_memory": False}


def test_collator_pipeline_passes_configured_source_only_languages() -> None:
    config = config_from_raw(
        {
            "data": {
                "language_pairs": [
                    ["kj", "ko"],
                    ["kj", "ja"],
                    ["kd", "ko"],
                    ["jd", "ja"],
                    ["ko", "ja"],
                ],
                "source_only_languages": ["kj", "kd", "jd"],
            }
        }
    )
    tokenizer = SimpleNamespace()

    args = build_collator_args(config, tokenizer)  # type: ignore[arg-type]

    assert args["source_only_languages"] == ("kj", "kd", "jd")


def test_configured_raw_scan_uses_the_complete_prepare_contract(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "pairs.jsonl").write_text('{"ko":"가","ja":"あ"}\n', encoding="utf-8")
    tokenizer = tmp_path / "sion.model"
    tokenizer.write_bytes(b"tokenizer")
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.approximate_split = True
    config.data.source_only_languages = ["ko"]
    config.data.synthetic_prefixes = ["generated_"]
    config.data.synthetic_prefix = "legacy_"
    config.data.synthetic_sampling_weight = 0.125

    fingerprint = train_module.scan_configured_raw_data(config, raw_dir, tokenizer)
    expected_options = train_module.prepare_preprocessing_options(
        approximate_split=True,
        source_only_languages=("ko",),
        translation_directions=(("ko", "ja"),),
        train_only_prefixes=config.data.configured_synthetic_prefixes(),
        managed_augmentation_prefix=config.data.synthetic_prefix,
        synthetic_sampling_weight=0.125,
        language_pair_count=1,
    )

    assert fingerprint.preprocessing_options == expected_options


def test_preflight_rejects_a_prepared_direction_graph_from_another_run() -> None:
    config = AppConfig()
    config.data.language_pairs = [["de", "fr"], ["sw", "ar"]]
    config.data.translation_directions = [["de", "fr"], ["fr", "de"], ["sw", "ar"]]
    matching = SimpleNamespace(
        language_pairs=(("de", "fr"), ("sw", "ar")),
        translation_directions=(("de", "fr"), ("fr", "de"), ("sw", "ar")),
    )
    stale = SimpleNamespace(
        language_pairs=(("de", "fr"), ("sw", "ar")),
        translation_directions=(("de", "fr"), ("fr", "de"), ("sw", "ar"), ("ar", "sw")),
    )

    train_module.preflight_dataset_direction_contract(config, matching)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="differ from the training config"):
        train_module.preflight_dataset_direction_contract(config, stale)  # type: ignore[arg-type]

    incomplete = SimpleNamespace(
        language_pairs=(("de", "fr"), ("sw", "ar")),
        translation_directions=(("de", "fr"), ("fr", "de"), ("sw", "ar")),
        observed_language_pairs=(("de", "fr"),),
    )
    with pytest.raises(ValueError, match="no accepted rows"):
        train_module.preflight_dataset_direction_contract(  # type: ignore[arg-type]
            config,
            incomplete,
            require_all_pairs=True,
        )


def test_preflight_requires_validation_evidence_for_every_configured_direction() -> None:
    config = AppConfig()
    config.data.language_pairs = [["pt-BR", "zh-Hant"], ["sw", "ar"]]
    config.data.translation_directions = [
        ["pt-BR", "zh-Hant"],
        ["zh-Hant", "pt-BR"],
        ["sw", "ar"],
    ]
    validation = SimpleNamespace(
        language_pairs=(("pt-BR", "zh-Hant"), ("sw", "ar")),
        translation_directions=(
            ("pt-BR", "zh-Hant"),
            ("zh-Hant", "pt-BR"),
            ("sw", "ar"),
        ),
        pair_count=2,
        observed_translation_directions_for_physical_mask=lambda _mask: (
            ("pt-BR", "zh-Hant"),
            ("zh-Hant", "pt-BR"),
        ),
    )

    with pytest.raises(ValueError, match=r"validation split.*sw.*ar"):
        train_module.preflight_dataset_direction_contract(  # type: ignore[arg-type]
            config,
            validation,
            require_all_directions=True,
        )


def test_prepare_only_runs_training_contract_preflights_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local artifact preparation must expose failures before a GPU upload."""

    config = AppConfig()
    config.data.language_pair = ["de", "fr"]
    config.foundation.enabled = False
    config.training.output_dir = str(tmp_path / "run")
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    events: list[str] = []

    class TokenizerStub:
        draft_id = 17

        def __len__(self) -> int:
            return 128

    class DatasetStub:
        def __init__(self, split: str) -> None:
            self.split = split
            self.pair_count = 7
            self.physical_token_count = 113
            self.source_names = ("parallel.jsonl",)

        def __len__(self) -> int:
            return 11 if self.split == config.data.train_split else 3

    class SamplerStub:
        def __init__(self, dataset: DatasetStub, *_args: object, **_kwargs: object) -> None:
            self.dataset = dataset

        def positive_sampling_pair_mask(self) -> object:
            return object()

    plan = SimpleNamespace(enabled=False, discovery=SimpleNamespace(sources=()))
    monkeypatch.setattr(sys, "argv", ["sion-train", "--prepare-only"])
    monkeypatch.setattr(train_module, "configure_stdio", lambda: None)
    monkeypatch.setattr(train_module, "initialize_distributed", lambda: context)
    monkeypatch.setattr(
        train_module, "cleanup_distributed", lambda _context: events.append("cleanup")
    )
    monkeypatch.setattr(train_module, "probe_environment", lambda: SimpleNamespace())
    monkeypatch.setattr(train_module, "synchronize_environment", lambda env, _context: env)
    monkeypatch.setattr(train_module, "describe_environment", lambda _env: "local CPU")
    monkeypatch.setattr(train_module, "resolve_config", lambda _args: (config, {}, "test config"))
    monkeypatch.setattr(
        train_module,
        "coordinated_training_run_lock",
        lambda *_args, **_kwargs: nullcontext(None),
    )
    monkeypatch.setattr(
        train_module,
        "coordinated_artifact_run_locks",
        lambda *_args, **_kwargs: nullcontext(()),
    )
    monkeypatch.setattr(train_module, "plan_foundation_stage", lambda _config: plan)
    monkeypatch.setattr(
        train_module,
        "_configured_foundation_branch_plan",
        lambda _config, discovered: discovered,
    )
    monkeypatch.setattr(train_module, "_find_distributed_auto_resume", lambda **_kwargs: None)
    monkeypatch.setattr(
        train_module.AppConfig,
        "validate_training_supervision",
        lambda *_args, **_kwargs: events.append("training supervision"),
    )
    monkeypatch.setattr(
        train_module,
        "preflight_final_export_dependencies",
        lambda _formats: events.append("export dependencies"),
    )
    monkeypatch.setattr(
        train_module,
        "ensure_artifacts",
        lambda *_args, **_kwargs: events.append("artifacts"),
    )
    monkeypatch.setattr(train_module, "SionTokenizer", lambda _path: TokenizerStub())
    monkeypatch.setattr(
        train_module,
        "preflight_morphoscript_token_features",
        lambda *_args: events.append("token features"),
    )
    monkeypatch.setattr(
        train_module,
        "IndexedParallelDataset",
        lambda _root, split, **_kwargs: DatasetStub(split),
    )
    monkeypatch.setattr(
        train_module,
        "preflight_dataset_direction_contract",
        lambda *_args, **_kwargs: events.append("direction graph"),
    )
    monkeypatch.setattr(
        train_module,
        "apply_auto_data_settings",
        lambda *_args, **_kwargs: events.append("automatic data settings") or [],
    )
    monkeypatch.setattr(
        train_module,
        "apply_auto_settings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare-only must not apply local hardware settings")
        ),
    )
    monkeypatch.setattr(train_module, "DistributedBucketBatchSampler", SamplerStub)
    monkeypatch.setattr(
        train_module, "resolve_training_revision_directions", lambda *_args, **_kwargs: ()
    )
    monkeypatch.setattr(
        train_module,
        "preflight_effective_translation_training",
        lambda *_args, **_kwargs: events.append("effective sampling"),
    )
    monkeypatch.setattr(train_module, "announce", lambda *_args, **_kwargs: None)

    train_module.main()

    assert events == [
        "training supervision",
        "export dependencies",
        "artifacts",
        "token features",
        "direction graph",
        "direction graph",
        "automatic data settings",
        "effective sampling",
        "cleanup",
    ]


def test_automatic_resume_candidate_cannot_skip_raw_free_foundation_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale automatic checkpoint may fall back, so base shards must be safe first."""

    class ExpectedStop(RuntimeError):
        pass

    config = AppConfig()
    config.data.language_pair = ["de", "fr"]
    config.training.output_dir = str(tmp_path / "run")
    config.posttraining.enabled = False
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    plan = SimpleNamespace(enabled=True, discovery=SimpleNamespace(sources=()))
    observed_requirements: list[bool] = []

    monkeypatch.setattr(sys, "argv", ["sion-train"])
    monkeypatch.setattr(train_module, "configure_stdio", lambda: None)
    monkeypatch.setattr(train_module, "initialize_distributed", lambda: context)
    monkeypatch.setattr(train_module, "cleanup_distributed", lambda _context: None)
    monkeypatch.setattr(train_module, "probe_environment", lambda: SimpleNamespace())
    monkeypatch.setattr(train_module, "synchronize_environment", lambda env, _context: env)
    monkeypatch.setattr(train_module, "describe_environment", lambda _env: "local CPU")
    monkeypatch.setattr(train_module, "resolve_config", lambda _args: (config, {}, "test config"))
    monkeypatch.setattr(
        train_module,
        "coordinated_training_run_lock",
        lambda *_args, **_kwargs: nullcontext(None),
    )
    monkeypatch.setattr(
        train_module,
        "coordinated_artifact_run_locks",
        lambda *_args, **_kwargs: nullcontext(()),
    )
    monkeypatch.setattr(train_module, "plan_foundation_stage", lambda _config: plan)
    monkeypatch.setattr(
        train_module,
        "_configured_foundation_branch_plan",
        lambda _config, discovered: discovered,
    )
    monkeypatch.setattr(
        train_module,
        "_find_distributed_auto_resume",
        lambda **_kwargs: str(tmp_path / "automatic-checkpoint"),
    )
    monkeypatch.setattr(
        train_module.AppConfig,
        "validate_training_supervision",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(train_module, "preflight_final_export_dependencies", lambda _formats: None)
    monkeypatch.setattr(train_module, "announce", lambda *_args, **_kwargs: None)

    def stop_after_artifact_policy(
        *_args: object,
        require_offline_foundation: bool = False,
        **_kwargs: object,
    ) -> None:
        observed_requirements.append(require_offline_foundation)
        raise ExpectedStop

    monkeypatch.setattr(train_module, "ensure_artifacts", stop_after_artifact_policy)

    with pytest.raises(ExpectedStop):
        train_module.main()

    assert observed_requirements == [True]


def test_training_resolves_revision_rows_to_an_exact_asymmetric_subgraph() -> None:
    config = AppConfig()
    config.data.language_pairs = [["pt-BR", "zh-Hant"]]
    config.data.translation_directions = [["pt-BR", "zh-Hant"], ["zh-Hant", "pt-BR"]]
    dataset = SimpleNamespace(
        detect_revision_directions=lambda **_kwargs: (("pt-BR", "zh-Hant"),),
    )

    resolved = train_module.resolve_training_revision_directions(  # type: ignore[arg-type]
        config,
        dataset,
        draft_token_id=9,
        max_source_tokens=510,
    )

    assert resolved == (("pt-BR", "zh-Hant"),)
    assert config.data.revision_directions == [["pt-BR", "zh-Hant"]]
    assert config.data.revision_examples is True


def test_bare_revision_examples_flag_fails_without_direction_evidence() -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.data.revision_examples = True
    dataset = SimpleNamespace(detect_revision_directions=lambda **_kwargs: ())

    with pytest.raises(ValueError, match="no revision-marked indexed rows"):
        train_module.resolve_training_revision_directions(  # type: ignore[arg-type]
            config,
            dataset,
            draft_token_id=9,
            max_source_tokens=510,
        )


def test_explicit_revision_graph_cannot_hide_detected_revision_rows() -> None:
    config = AppConfig()
    config.data.language_pairs = [["pt-BR", "zh-Hant"]]
    config.data.translation_directions = [["pt-BR", "zh-Hant"], ["zh-Hant", "pt-BR"]]
    config.data.revision_directions = [["pt-BR", "zh-Hant"]]
    dataset = SimpleNamespace(
        detect_revision_directions=lambda **_kwargs: (("zh-Hant", "pt-BR"),),
    )

    with pytest.raises(ValueError, match="must exactly match revision-marked indexed rows"):
        train_module.resolve_training_revision_directions(  # type: ignore[arg-type]
            config,
            dataset,
            draft_token_id=9,
            max_source_tokens=510,
        )


def test_explicit_revision_graph_cannot_claim_unobserved_reverse_direction() -> None:
    config = AppConfig()
    config.data.language_pairs = [["pt-BR", "zh-Hant"]]
    config.data.translation_directions = [["pt-BR", "zh-Hant"], ["zh-Hant", "pt-BR"]]
    config.data.revision_directions = [["pt-BR", "zh-Hant"], ["zh-Hant", "pt-BR"]]
    dataset = SimpleNamespace(
        detect_revision_directions=lambda **_kwargs: (("pt-BR", "zh-Hant"),),
    )

    with pytest.raises(ValueError, match="unsupported=.*zh-Hant.*pt-BR"):
        train_module.resolve_training_revision_directions(  # type: ignore[arg-type]
            config,
            dataset,
            draft_token_id=9,
            max_source_tokens=510,
        )


def test_revision_capability_rejects_rows_with_zero_effective_sampling_mass() -> None:
    config = AppConfig()
    config.data.language_pair = ["pt-BR", "zh-Hant"]
    config.data.translation_directions = [["pt-BR", "zh-Hant"]]
    config.data.revision_directions = [["pt-BR", "zh-Hant"]]
    positive_mask = object()
    dataset = SimpleNamespace(
        detect_revision_directions=lambda **kwargs: (
            (("pt-BR", "zh-Hant"),) if kwargs.get("physical_mask") is None else ()
        ),
    )

    with pytest.raises(ValueError, match="unsupported=.*pt-BR.*zh-Hant"):
        train_module.resolve_training_revision_directions(  # type: ignore[arg-type]
            config,
            dataset,
            draft_token_id=9,
            max_source_tokens=510,
            physical_mask=positive_mask,  # type: ignore[arg-type]
        )


def test_artifact_preparation_locks_every_independent_mutation_root(tmp_path: Path) -> None:
    config = AppConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.tokenizer_model = str(tmp_path / "tokenizer-root" / "sion.model")
    config.data.dataset_dir = str(tmp_path / "translation-root" / "dataset")
    config.foundation.dataset_dir = str(tmp_path / "foundation-root" / "dataset")
    plan = SimpleNamespace(enabled=True)

    roots = train_module._artifact_mutation_roots(
        config,
        plan,
        prepare_foundation=True,
    )

    assert set(roots) == {
        (tmp_path / "raw").resolve(),
        (tmp_path / "tokenizer-root").resolve(),
        (tmp_path / "translation-root").resolve(),
        (tmp_path / "foundation-root").resolve(),
    }
    assert list(roots) == sorted(roots, key=lambda path: os.path.normcase(str(path)))
    assert train_module._artifact_mutation_roots(
        config,
        plan,
        prepare_foundation=False,
    ) == tuple(root for root in roots if root.name != "foundation-root")


def test_configured_foundation_root_is_leased_while_raw_corpus_is_offline(
    tmp_path: Path,
) -> None:
    config = AppConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.tokenizer_model = str(tmp_path / "tokenizer-root" / "sion.model")
    config.data.dataset_dir = str(tmp_path / "translation-root" / "dataset")
    config.foundation.dataset_dir = str(tmp_path / "foundation-root" / "dataset")
    offline_plan = SimpleNamespace(enabled=False)

    roots = train_module._artifact_mutation_roots(
        config,
        offline_plan,
        prepare_foundation=True,
    )

    assert (tmp_path / "foundation-root").resolve() in roots


def test_artifact_run_leases_remain_held_until_the_run_scope_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.tokenizer_model = str(tmp_path / "tokenizer" / "sion.model")
    config.data.dataset_dir = str(tmp_path / "translation" / "dataset")
    config.foundation.dataset_dir = str(tmp_path / "foundation" / "dataset")
    plan = SimpleNamespace(enabled=True)
    active: set[Path] = set()

    class Lease:
        def __init__(self, root: Path) -> None:
            self.root = root

        def __enter__(self) -> Path:
            active.add(self.root)
            return self.root / ".sion_artifacts.lock"

        def __exit__(self, *_args: object) -> None:
            active.remove(self.root)

    monkeypatch.setattr(train_module, "artifact_lock", lambda root: Lease(Path(root)))
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    expected = set(
        train_module._artifact_mutation_roots(
            config,
            plan,
            prepare_foundation=True,
        )
    )

    with train_module.coordinated_artifact_run_locks(config, plan, context) as roots:
        assert set(roots) == expected
        assert active == expected
    assert not active


def test_prepared_artifact_identity_binds_control_file_contents(tmp_path: Path) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    tokenizer = tmp_path / "tokenizer" / "sion.model"
    dataset = tmp_path / "translation" / "dataset"
    foundation_dataset = tmp_path / "foundation" / "dataset"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer-generation-a")
    (tokenizer.parent / "tokenizer_metadata.json").write_text("{}\n", encoding="utf-8")

    for root, format_name in (
        (dataset, "sion-indexed-parallel-v6"),
        (foundation_dataset, "sion-foundation-indexed-v3"),
    ):
        payload = root / "train" / "00000.bin"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"authenticated-indexed-payload")
        manifest = {
            "format": format_name,
            "generation": 1,
            "artifact_inventory": build_dataset_artifact_inventory(root),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (dataset / "raw_fingerprint.json").write_text('{"generation": 1}\n', encoding="utf-8")
    config.data.tokenizer_model = str(tokenizer)
    config.data.dataset_dir = str(dataset)
    config.foundation.dataset_dir = str(foundation_dataset)
    plan = SimpleNamespace(enabled=True)

    first = train_module._prepared_artifact_identity(
        config,
        plan,
        prepare_foundation=True,
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"] = 2
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    second = train_module._prepared_artifact_identity(
        config,
        plan,
        prepare_foundation=True,
    )

    assert first["translation_manifest"]["sha256"] != second["translation_manifest"]["sha256"]
    assert len(first["foundation_manifest"]["sha256"]) == 64
    with pytest.raises(FileNotFoundError, match="raw_fingerprint"):
        (dataset / "raw_fingerprint.json").unlink()
        train_module._prepared_artifact_identity(
            config,
            plan,
            prepare_foundation=True,
        )


def test_foundation_preparation_backs_up_a_file_at_the_dataset_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sion_translate.data.prepare_foundation as foundation_prepare

    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    tokenizer = tmp_path / "tokenizer" / "sion.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    dataset = tmp_path / "translation"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    foundation_dataset = tmp_path / "foundation"
    foundation_dataset.write_bytes(b"not-a-directory")
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.tokenizer_model = str(tokenizer)
    config.data.dataset_dir = str(dataset)
    config.foundation.dataset_dir = str(foundation_dataset)
    raw_identity = {"parallel.jsonl": 128}
    monolingual_source = tmp_path / "corpus" / "ko" / "wiki.txt"
    monolingual_source.parent.mkdir(parents=True)
    monolingual_source.write_text("충분히 긴 단일어 문장입니다\n", encoding="utf-8")
    plan = SimpleNamespace(
        enabled=True,
        discovery=SimpleNamespace(
            sources=(SimpleNamespace(language="ko", path=monolingual_source),)
        ),
        languages=("ko",),
        report=(),
        warnings=(),
    )
    prepared: list[Path] = []

    monkeypatch.setattr(train_module, "scan_configured_raw_data", lambda *_args: raw_identity)
    monkeypatch.setattr(train_module, "find_existing_checkpoint", lambda *_args: None)
    monkeypatch.setattr(
        train_module,
        "tokenizer_policy_problem",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(train_module, "stored_fingerprint", lambda *_args: raw_identity)
    monkeypatch.setattr(train_module, "dataset_artifact_problem", lambda *_args: None)
    monkeypatch.setattr(
        foundation_prepare,
        "foundation_dataset_problem",
        lambda *_args, **_kwargs: "foundation dataset path is not a directory",
    )

    def prepare(*_args: object, **_kwargs: object) -> object:
        prepared.append(foundation_dataset)
        foundation_dataset.mkdir()
        return SimpleNamespace()

    monkeypatch.setattr(foundation_prepare, "prepare_foundation_dataset", prepare)
    monkeypatch.setattr(foundation_prepare, "render_prepare_report", lambda _stats: ())

    train_module._ensure_artifacts_on_main(
        config,
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        plan,
        locks_held=True,
    )

    assert prepared == [foundation_dataset]
    assert foundation_dataset.is_dir()
    backups = list(tmp_path.glob("foundation.stale-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"not-a-directory"


def test_final_export_dependency_preflight_fails_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = {"torchao": False, "gguf": False}
    monkeypatch.setattr(
        train_module.importlib.util,
        "find_spec",
        lambda name: object() if available.get(name, True) else None,
    )

    assert missing_final_export_dependencies(["fp32", "transformers"]) == {}
    assert missing_final_export_dependencies(["int8", "gguf_q4_k_m"]) == {
        "int8": "torchao",
        "gguf_q4_k_m": "gguf-python",
    }
    with pytest.raises(RuntimeError, match=r"int8.*torchao.*gguf_q4_k_m.*gguf-python"):
        preflight_final_export_dependencies(["int8", "gguf_q4_k_m"])

    available["torchao"] = True
    available["gguf"] = True
    preflight_final_export_dependencies(["int8", "gguf_q4_k_m"])


def test_stage_release_stops_persistent_workers_on_cpu() -> None:
    loader = LoaderStub()
    iterator = loader._iterator
    shutdown_dataloader(loader)  # type: ignore[arg-type]
    assert iterator.stopped
    assert loader._iterator is None

    second = LoaderStub()
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        distributed=False,
    )
    assert release_stage_resources(context, second) == {}  # type: ignore[arg-type]
    assert second._iterator is None


def test_final_export_wires_all_formats_and_model_sidecars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = AppConfig()
    config.data.tokenizer_model = str(tmp_path / "sion.model")
    config.data.tokenizer_features = str(tmp_path / "token_features.npz")
    config.data.language_pairs = [["ko", "ja"], ["en", "ru"]]
    config.data.bidirectional = False
    config.data.revision_examples = True
    config.data.revision_directions = [["ko", "ja"]]
    config.model.experimental.morphoscript_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "sion_translate.cli.train.export_inference_models",
        capture,
    )
    destination = export_final_model(
        torch.nn.Linear(1, 1),
        config,
        context,
        tmp_path / "run",
        stage="posttrain",
        step=17,
        authenticated_revision_directions=(("ko", "ja"),),
        pipeline_identity={
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
        },
    )

    assert destination == tmp_path / "run" / "posttrain" / "exports" / "best"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == destination
    assert kwargs["formats"] == tuple(config.training.final_export_formats)
    assert kwargs["tokenizer_path"] == config.data.tokenizer_model
    assert kwargs["token_features_path"] == config.data.tokenizer_features
    assert kwargs["language_pairs"] == (("ko", "ja"), ("en", "ru"))
    assert kwargs["bidirectional"] is False
    assert kwargs["revision_trained"] is True
    assert kwargs["revision_directions"] == (("ko", "ja"),)
    assert kwargs["pipeline_identity"] == {
        "schema": "sion-translation-pipeline-v2",
        "branch": "translation-only",
    }
    assert kwargs["strict"] is True


def test_distributed_final_export_publishes_rank_zero_failure_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")

    def run_locally(_context, action, *, description: str):
        assert description == "publishing final export start state"
        return action()

    monkeypatch.setattr(train_module, "_run_rank_zero_action", run_locally)
    monkeypatch.setattr(
        train_module,
        "broadcast_text",
        lambda value, _context: value or "missing-invocation",
    )
    monkeypatch.setattr(
        train_module,
        "distributed_failure_scope",
        lambda _failed, _context: "none",
    )
    monkeypatch.setattr(
        train_module,
        "export_inference_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("injected strict export failure")
        ),
    )

    with pytest.raises(ValueError, match="injected strict export failure"):
        export_final_model(
            torch.nn.Linear(1, 1),
            config,
            context,
            tmp_path / "run",
            stage="foundation",
            step=11,
            release_name="sion",
            translation_capable=False,
        )

    status_path = tmp_path / "run" / "foundation" / "exports" / ".best.strict-export-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["step"] == 11
    assert status["error_type"] == "ValueError"
    assert status["message"] == "injected strict export failure"


def test_distributed_final_export_peer_surfaces_rank_zero_failure_without_collective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    context = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")
    run_root = tmp_path / "run"
    status_path = run_root / "foundation" / "exports" / ".best.strict-export-status.json"
    invocation = "peer-export-invocation"
    train_module._atomic_write_json(
        status_path,
        {
            "schema": train_module.FINAL_EXPORT_STATUS_SCHEMA,
            "state": "running",
            "invocation": invocation,
            "step": 11,
            "release_name": "sion",
        },
    )

    def peer_does_not_run(_context, _action, *, description: str):
        assert description == "publishing final export start state"
        return None

    monkeypatch.setattr(train_module, "_run_rank_zero_action", peer_does_not_run)
    monkeypatch.setattr(train_module, "broadcast_text", lambda _value, _context: invocation)
    monkeypatch.setattr(
        train_module,
        "distributed_failure_scope",
        lambda _failed, _context: "none",
    )

    def publish_failure(*_args, **_kwargs) -> None:
        train_module._atomic_write_json(
            status_path,
            {
                "schema": train_module.FINAL_EXPORT_STATUS_SCHEMA,
                "state": "failed",
                "invocation": invocation,
                "step": 11,
                "release_name": "sion",
                "error_type": "ValueError",
                "message": "injected strict export failure",
            },
        )

    monkeypatch.setattr(train_module, "export_inference_models", publish_failure)

    with pytest.raises(RuntimeError, match="rank 0 final export failed.*injected"):
        export_final_model(
            torch.nn.Linear(1, 1),
            config,
            context,
            run_root,
            stage="foundation",
            step=11,
            release_name="sion",
            translation_capable=False,
        )


@pytest.mark.parametrize("terminal_state", ("complete", "failed"))
def test_preallocated_backup_status_survives_one_terminal_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: str,
) -> None:
    status_path = tmp_path / "strict-export-status.json"
    invocation = f"terminal-{terminal_state}"
    running = {
        "schema": train_module.FINAL_EXPORT_STATUS_SCHEMA,
        "state": "running",
        "invocation": invocation,
        "step": 17,
        "release_name": "sion",
    }
    handles = train_module._initialize_control_status(status_path, running)
    original_overwrite = train_module._overwrite_control_status
    injected = False

    def fail_one_channel(handle, payload) -> None:
        nonlocal injected
        if payload.get("state") == terminal_state and not injected:
            injected = True
            raise PermissionError("injected primary status failure")
        original_overwrite(handle, payload)

    monkeypatch.setattr(train_module, "_overwrite_control_status", fail_one_channel)
    terminal = {
        **running,
        "state": terminal_state,
        "error_type": "ValueError",
        "message": "injected export failure",
    }
    try:
        train_module._publish_control_status(handles, terminal)
    finally:
        train_module._close_control_status(handles)

    if terminal_state == "complete":
        train_module._wait_for_final_export_status(
            status_path,
            step=17,
            release_name="sion",
            invocation=invocation,
        )
    else:
        with pytest.raises(RuntimeError, match="injected export failure"):
            train_module._wait_for_final_export_status(
                status_path,
                step=17,
                release_name="sion",
                invocation=invocation,
            )
    assert injected


def test_control_status_bounds_multibyte_and_control_character_diagnostics() -> None:
    diagnostic = ("오류🔥\x00\n" * 20_000) + "tail"
    bounded = train_module._bounded_status_text(diagnostic)
    encoded = train_module._encode_control_status(
        {
            "schema": train_module.FINAL_EXPORT_STATUS_SCHEMA,
            "state": "failed",
            "message": bounded,
        }
    )

    assert len(encoded) == train_module.RANK_ZERO_STATUS_FILE_BYTES
    assert len(bounded.encode("utf-8")) <= 6000
    assert "\x00" not in bounded
    assert "\n" not in bounded
    assert json.loads(encoded.decode("utf-8"))["message"].endswith("tail")


def test_long_rank_zero_action_rejects_a_stale_completion_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")
    status_path = tmp_path / "long-action.json"
    train_module._atomic_write_json(
        status_path,
        {
            "schema": train_module.RANK_ZERO_ACTION_STATUS_SCHEMA,
            "operation": "test validation",
            "state": "complete",
            "invocation": "old-invocation",
            "result": True,
        },
    )
    monkeypatch.setattr(
        train_module,
        "broadcast_text",
        lambda _value, _context: "new-invocation",
    )
    monkeypatch.setattr(
        train_module,
        "_run_rank_zero_action",
        lambda _context, _action, *, description: None,
    )
    monkeypatch.setattr(
        train_module,
        "distributed_failure_scope",
        lambda failed, _context: "partial" if failed else "none",
    )

    with pytest.raises(RuntimeError, match="not visible to every rank"):
        train_module._run_long_rank_zero_action(
            context,
            status_path,
            operation="test validation",
            action=lambda: (_ for _ in ()).throw(AssertionError("peer action must not run")),
        )


def test_long_rank_zero_action_terminal_status_cannot_be_overwritten_by_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    status_path = tmp_path / "long-action.json"
    monkeypatch.setattr(
        train_module,
        "broadcast_text",
        lambda value, _context: value or "a" * 32,
    )
    monkeypatch.setattr(
        train_module,
        "_run_rank_zero_action",
        lambda _context, action, *, description: action(),
    )
    monkeypatch.setattr(
        train_module,
        "distributed_failure_scope",
        lambda _failed, _context: "none",
    )
    acknowledged_states: list[set[str]] = []

    def acknowledge_terminal(*_args, **_kwargs) -> None:
        acknowledged_states.append(
            {str(status["state"]) for status in train_module._read_control_status(status_path)}
        )

    monkeypatch.setattr(
        train_module,
        "_wait_for_rank_zero_action_acknowledgements",
        acknowledge_terminal,
    )

    result = train_module._run_long_rank_zero_action(
        context,
        status_path,
        operation="heartbeat race test",
        action=lambda: (time.sleep(0.03), {"ok": True})[1],
        stale_timeout_seconds=0.2,
        heartbeat_interval_seconds=0.001,
    )

    assert result == {"ok": True}
    statuses = train_module._read_control_status(status_path)
    assert statuses
    assert {status["state"] for status in statuses} == {"complete"}
    assert acknowledged_states == [{"complete"}]


def test_long_rank_zero_action_peer_acknowledges_terminal_without_collective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")
    status_path = tmp_path / "long-action.json"
    invocation = "a" * 32
    monkeypatch.setattr(train_module, "broadcast_text", lambda _value, _context: invocation)
    monkeypatch.setattr(
        train_module,
        "_run_rank_zero_action",
        lambda _context, _action, *, description: None,
    )
    monkeypatch.setattr(
        train_module,
        "distributed_failure_scope",
        lambda _failed, _context: "none",
    )
    monkeypatch.setattr(train_module, "_control_status_is_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        train_module,
        "_wait_for_rank_zero_action",
        lambda *_args, **_kwargs: {"state": "complete", "result": {"ok": True}},
    )
    monkeypatch.setattr(
        train_module,
        "barrier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal acknowledgement must not use a collective")
        ),
    )

    result = train_module._run_long_rank_zero_action(
        context,
        status_path,
        operation="peer acknowledgement test",
        action=lambda: (_ for _ in ()).throw(AssertionError("peer action must not run")),
    )

    assert result == {"ok": True}
    ack_path = train_module._rank_zero_action_ack_path(
        status_path,
        invocation=invocation,
        rank=1,
    )
    ack = json.loads(ack_path.read_text(encoding="utf-8"))
    assert ack["state"] == "observed"
    assert ack["terminal_state"] == "complete"


def test_long_rank_zero_action_peer_reports_observer_error_without_collective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")
    status_path = tmp_path / "long-action.json"
    invocation = "b" * 32
    monkeypatch.setattr(train_module, "broadcast_text", lambda _value, _context: invocation)
    monkeypatch.setattr(
        train_module,
        "_run_rank_zero_action",
        lambda _context, _action, *, description: None,
    )
    monkeypatch.setattr(
        train_module,
        "distributed_failure_scope",
        lambda _failed, _context: "none",
    )
    monkeypatch.setattr(train_module, "_control_status_is_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        train_module,
        "_wait_for_rank_zero_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("lost heartbeat")),
    )
    monkeypatch.setattr(
        train_module,
        "barrier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("observer errors must not enter a collective")
        ),
    )

    with pytest.raises(TimeoutError, match="lost heartbeat"):
        train_module._run_long_rank_zero_action(
            context,
            status_path,
            operation="peer error acknowledgement test",
            action=lambda: (_ for _ in ()).throw(AssertionError("peer action must not run")),
        )

    ack_path = train_module._rank_zero_action_ack_path(
        status_path,
        invocation=invocation,
        rank=1,
    )
    ack = json.loads(ack_path.read_text(encoding="utf-8"))
    assert ack["state"] == "observer_error"
    assert "lost heartbeat" in ack["message"]


def test_rank_zero_action_rejects_acknowledgement_for_a_different_terminal_state(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "long-action.json"
    invocation = "c" * 32
    ack_path = train_module._rank_zero_action_ack_path(
        status_path,
        invocation=invocation,
        rank=1,
    )
    ack_path.write_text(
        json.dumps(
            {
                "schema": train_module.RANK_ZERO_ACTION_STATUS_SCHEMA,
                "operation": "terminal binding test",
                "invocation": invocation,
                "rank": 1,
                "state": "observed",
                "terminal_state": "failed",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="observed terminal state 'failed', expected 'complete'"):
        train_module._wait_for_rank_zero_action_acknowledgements(
            status_path,
            operation="terminal binding test",
            invocation=invocation,
            expected_terminal_state="complete",
            world_size=2,
            timeout_seconds=0.1,
        )


def test_two_gloo_ranks_reject_rank_local_resume_selection(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("distributed resume consensus test requires Gloo")

    script = tmp_path / "resume_consensus_worker.py"
    results = tmp_path / "results"
    results.mkdir()
    script.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            import sion_translate.cli.train as train_module
            from sion_translate.training.distributed import (
                cleanup_distributed,
                initialize_distributed,
            )

            result_dir = Path(sys.argv[1])
            context = initialize_distributed()
            errors = []
            try:
                train_module.checkpoint_path_exists = lambda _path: context.rank == 0
                try:
                    train_module._find_distributed_auto_resume(
                        explicit=None,
                        automatic=Path("auto-latest"),
                        context=context,
                        stage="SFT",
                    )
                except BaseException as error:
                    errors.append(f"{type(error).__name__}:{error}")
                else:
                    raise SystemExit("rank-local auto resume unexpectedly succeeded")

                try:
                    train_module._find_distributed_auto_resume(
                        explicit=f"rank-{context.rank}",
                        automatic=Path("unused"),
                        context=context,
                        stage="SFT",
                    )
                except BaseException as error:
                    errors.append(f"{type(error).__name__}:{error}")
                else:
                    raise SystemExit("rank-local explicit resume unexpectedly succeeded")

                (result_dir / f"rank-{context.rank}.json").write_text(
                    json.dumps({"errors": errors}, ensure_ascii=False),
                    encoding="utf-8",
                )
            finally:
                cleanup_distributed(context)
            """
        ),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUTF8": "1",
        "USE_LIBUV": "0",
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        rendezvous_port = int(port_probe.getsockname()[1])

    processes: list[subprocess.Popen[str]] = []
    for rank in range(2):
        worker_environment = {
            **environment,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(rendezvous_port),
            "WORLD_SIZE": "2",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
        processes.append(
            subprocess.Popen(
                [sys.executable, str(script), str(results)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=worker_environment,
            )
        )
    outputs: list[tuple[str, str]] = []
    try:
        for process in processes:
            outputs.append(process.communicate(timeout=60))
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        for process in processes:
            process.communicate()
        raise

    for process, (stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, stderr or stdout
    for rank in range(2):
        payload = json.loads((results / f"rank-{rank}.json").read_text(encoding="utf-8"))
        assert len(payload["errors"]) == 2
        assert "auto-resume checkpoint visibility differs" in payload["errors"][0]
        assert "explicit resume path differs" in payload["errors"][1]


def test_resume_preflight_retains_the_exact_generation_until_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    source = tmp_path / "checkpoints" / "latest"

    class GenerationLease:
        def __enter__(self) -> SimpleNamespace:
            nonlocal active
            active = True
            return SimpleNamespace(source=source, step=7, artifact_sha256="a" * 64)

        def __exit__(self, *_args: object) -> None:
            nonlocal active
            active = False

    monkeypatch.setattr(
        train_module,
        "verified_checkpoint_generation_lease",
        lambda *_args, **_kwargs: GenerationLease(),
    )
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")

    with pytest.raises(ValueError, match="lease scope"):
        train_module._coordinated_resume_preflight(source, {}, context, stage="SFT")

    scope = ExitStack()
    bound = train_module._coordinated_resume_preflight(
        source,
        {},
        context,
        stage="SFT",
        lease_scope=scope,
    )

    assert bound == str(source)
    assert active
    scope.close()
    assert not active


@pytest.mark.parametrize("rejected_phase", ["identity", "structure"])
def test_resume_candidate_selection_closes_rejected_generation_and_keeps_previous(
    monkeypatch: pytest.MonkeyPatch,
    rejected_phase: str,
) -> None:
    active: set[str] = set()
    closed: list[str] = []
    announcements: list[str] = []
    bindings = (
        SimpleNamespace(artifact_sha256="a" * 64),
        SimpleNamespace(artifact_sha256="b" * 64),
    )

    class Lease:
        def __init__(self, source: str) -> None:
            self.source = source

        def __enter__(self) -> None:
            active.add(self.source)

        def __exit__(self, *_args: object) -> None:
            active.remove(self.source)
            closed.append(self.source)

    def bind_candidate(
        _candidate: object,
        _identity: object,
        _context: object,
        *,
        lease_scope: ExitStack,
        expected_artifact_sha256: str,
        **_kwargs: object,
    ) -> str:
        source = "current" if expected_artifact_sha256 == "a" * 64 else "previous"
        lease_scope.enter_context(Lease(source))
        return source

    monkeypatch.setattr(
        train_module,
        "checkpoint_generation_bindings",
        lambda *_args: bindings,
    )
    monkeypatch.setattr(train_module, "_coordinated_resume_preflight", bind_candidate)
    monkeypatch.setattr(
        train_module,
        "_coordinated_checkpoint_pipeline_identity",
        lambda source, *_args: {"candidate": source},
    )
    monkeypatch.setattr(
        train_module,
        "build_training_checkpoint_identity",
        lambda _config, **kwargs: {"pipeline": kwargs["pipeline_identity"]},
    )

    def identity_preflight(source: str, *_args: object, **_kwargs: object) -> int:
        if source == "current" and rejected_phase == "identity":
            raise ValueError("current identity mismatch")
        return 5 if source == "current" else 4

    def structure_preflight(source: str, *_args: object, **_kwargs: object) -> int:
        if source == "current" and rejected_phase == "structure":
            raise ValueError("current structure mismatch")
        return 5 if source == "current" else 4

    monkeypatch.setattr(
        train_module,
        "_coordinated_exact_checkpoint_identity_preflight",
        identity_preflight,
    )
    monkeypatch.setattr(
        train_module,
        "_coordinated_checkpoint_load_structure",
        structure_preflight,
    )
    monkeypatch.setattr(
        train_module,
        "announce",
        lambda message, _context: announcements.append(message),
    )
    config = AppConfig()
    plan = SimpleNamespace(enabled=True)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with ExitStack() as winner_scope:
        source, pipeline = train_module._select_translation_resume_candidate(
            "logical/latest",
            config,
            plan,
            torch.nn.Linear(2, 2),
            SimpleNamespace(),
            context,
            stage="SFT",
            stage_name="pretrain/SFT",
            include_posttraining=False,
            lease_scope=winner_scope,
        )
        assert source == "previous"
        assert pipeline == {"candidate": "previous"}
        assert active == {"previous"}
        assert closed == ["current"]
        assert len(announcements) == 1
        assert announcements[0].startswith(
            "[warning] SFT: rejected a newer checkpoint before selecting authenticated "
            "generation 2 at previous."
        )
        assert f"generation 1 ({'a' * 12}...)" in announcements[0]
        assert f"current {rejected_phase} mismatch" in announcements[0]
    assert not active
    assert closed == ["current", "previous"]


def test_configured_foundation_branch_stays_enabled_when_raw_corpus_is_offline() -> None:
    config = AppConfig()
    offline = train_module.FoundationPlan(
        enabled=False,
        reason="offline",
        discovery=SimpleNamespace(),  # type: ignore[arg-type]
        languages=("ko", "ja"),
    )

    branch = train_module._configured_foundation_branch_plan(config, offline)

    assert branch.enabled
    assert branch.languages == ("ko", "ja")


def test_translation_only_resume_conflicts_with_configured_offline_foundation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    offline_plan = SimpleNamespace(enabled=False, languages=config.foundation_languages())
    monkeypatch.setattr(
        train_module,
        "inspect_checkpoint_identity",
        lambda *_args: {
            "pipeline": {
                "schema": "sion-translation-pipeline-v2",
                "branch": "translation-only",
            }
        },
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(ValueError, match="configured foundation-first"):
        train_module._checkpoint_pipeline_identity(
            "authenticated-sft",
            config,
            offline_plan,
            context,
        )


def test_sft_resume_uses_its_authenticated_lineage_without_base_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.foundation.release_name = "my_base"
    plan = SimpleNamespace(enabled=True, languages=("ko", "ja"))
    pipeline = {
        "schema": "sion-translation-pipeline-v2",
        "branch": "foundation-then-translation",
        "foundation": {
            "schema": "sion-foundation-lineage-v1",
            "release_name": "my_base",
            "release_version": "1.5",
            "languages": ["ko", "ja"],
            "selected_step": 7,
            "foundation_manifest_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "checkpoint_identity_sha256": "c" * 64,
            "checkpoint_artifact_sha256": "d" * 64,
        },
    }
    monkeypatch.setattr(
        train_module,
        "inspect_checkpoint_identity",
        lambda _source, _context: {
            "pipeline": pipeline,
            "tokenizer": {"model": {"sha256": "b" * 64}},
        },
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    recovered = train_module._checkpoint_pipeline_identity(
        "authenticated-sft",
        config,
        plan,
        context,
    )

    assert recovered == pipeline


def test_sft_resume_keeps_foundation_lineage_when_raw_base_corpus_is_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.foundation.release_name = "my_base"
    offline_plan = SimpleNamespace(enabled=False, languages=("ko", "ja"))
    pipeline = {
        "schema": "sion-translation-pipeline-v2",
        "branch": "foundation-then-translation",
        "foundation": {
            "schema": "sion-foundation-lineage-v1",
            "release_name": "my_base",
            "release_version": "1.5",
            "languages": list(config.foundation_languages()),
            "selected_step": 7,
            "foundation_manifest_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "checkpoint_identity_sha256": "c" * 64,
            "checkpoint_artifact_sha256": "d" * 64,
        },
    }
    monkeypatch.setattr(
        train_module,
        "inspect_checkpoint_identity",
        lambda _source, _context: {
            "pipeline": pipeline,
            "tokenizer": {"model": {"sha256": "b" * 64}},
        },
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    recovered = train_module._checkpoint_pipeline_identity(
        "authenticated-sft",
        config,
        offline_plan,
        context,
    )

    assert recovered == pipeline


def test_two_gloo_ranks_wait_for_one_artifact_preparation_and_verify_contents(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("distributed artifact preparation test requires Gloo")

    script = tmp_path / "artifact_worker.py"
    results = tmp_path / "artifact-results"
    artifacts = tmp_path / "artifacts"
    results.mkdir()
    script.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            import time
            from pathlib import Path
            from types import SimpleNamespace

            import sion_translate.cli.train as train_module
            from sion_translate.config import AppConfig
            from sion_translate.data.integrity import build_dataset_artifact_inventory
            from sion_translate.training.distributed import cleanup_distributed, initialize_distributed

            result_dir = Path(sys.argv[1])
            artifact_root = Path(sys.argv[2])
            context = initialize_distributed()
            config = AppConfig()
            config.data.tokenizer_model = str(artifact_root / "tokenizer" / "sion.model")
            config.data.dataset_dir = str(artifact_root / "translation")
            config.training.output_dir = str(artifact_root / "run")
            plan = SimpleNamespace(enabled=False)
            calls = 0

            def prepare_on_main(*_args, **_kwargs):
                global calls
                if not context.is_main:
                    raise AssertionError("peer executed rank-zero artifact preparation")
                calls += 1
                time.sleep(0.25)
                tokenizer = Path(config.data.tokenizer_model)
                tokenizer.parent.mkdir(parents=True, exist_ok=True)
                tokenizer.write_bytes(b"shared-tokenizer")
                dataset = Path(config.data.dataset_dir)
                train = dataset / "train"
                train.mkdir(parents=True, exist_ok=True)
                (train / "00000.src.bin").write_bytes(b"source")
                (train / "00000.tgt.bin").write_bytes(b"target")
                manifest = {
                    "format": "sion-indexed-parallel-v6",
                    "artifact_inventory": build_dataset_artifact_inventory(dataset),
                }
                (dataset / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                (dataset / "raw_fingerprint.json").write_text("{}", encoding="utf-8")

            train_module._ensure_artifacts_on_main = prepare_on_main
            try:
                train_module.ensure_artifacts(
                    config,
                    context,
                    plan,
                    prepare_foundation=False,
                )
                (result_dir / f"rank-{context.rank}.json").write_text(
                    json.dumps({"calls": calls}), encoding="utf-8"
                )
            finally:
                cleanup_distributed(context)
            """
        ),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUTF8": "1",
        "USE_LIBUV": "0",
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        rendezvous_port = int(port_probe.getsockname()[1])

    processes: list[subprocess.Popen[str]] = []
    for rank in range(2):
        worker_environment = {
            **environment,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(rendezvous_port),
            "WORLD_SIZE": "2",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
        processes.append(
            subprocess.Popen(
                [sys.executable, str(script), str(results), str(artifacts)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=worker_environment,
            )
        )
    outputs: list[tuple[str, str]] = []
    try:
        for process in processes:
            outputs.append(process.communicate(timeout=60))
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        for process in processes:
            process.communicate()
        raise

    for process, (stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, stderr or stdout
    assert json.loads((results / "rank-0.json").read_text(encoding="utf-8")) == {"calls": 1}
    assert json.loads((results / "rank-1.json").read_text(encoding="utf-8")) == {"calls": 0}


def test_final_export_advertises_only_the_eight_trained_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pairs = [
        ["kj", "ko"],
        ["kj", "ja"],
        ["kd", "ko"],
        ["kd", "ja"],
        ["jd", "ko"],
        ["jd", "ja"],
        ["ko", "ja"],
    ]
    config.data.source_only_languages = ["kj", "kd", "jd"]
    config.data.bidirectional = True
    captured: dict[str, object] = {}

    def capture(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(train_module, "export_inference_models", capture)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    export_final_model(
        torch.nn.Linear(1, 1),
        config,
        context,
        tmp_path / "run",
        stage="posttrain",
        step=23,
    )

    assert captured["translation_directions"] == (
        ("kj", "ko"),
        ("kj", "ja"),
        ("kd", "ko"),
        ("kd", "ja"),
        ("jd", "ko"),
        ("jd", "ja"),
        ("ko", "ja"),
        ("ja", "ko"),
    )


def test_parallel_strategy_prefers_ddp_and_supports_legacy_fsdp() -> None:
    distributed = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cuda"),
        distributed=True,
        backend="nccl",
    )
    single = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        distributed=False,
    )
    assert resolve_parallel_strategy("auto", distributed) == "ddp"
    assert resolve_parallel_strategy("auto", distributed, legacy_fsdp2=True) == "fsdp2"
    assert resolve_parallel_strategy("fsdp2", single) == "single"
    assert fsdp_reduce_dtype("auto", torch.bfloat16) == torch.bfloat16
    assert fsdp_reduce_dtype("auto", torch.float32) == torch.float32


def test_distributed_bf16_support_is_resolved_collectively(monkeypatch) -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cuda"),
        distributed=True,
        backend="nccl",
    )
    original_tensor = torch.tensor
    monkeypatch.setattr(
        distributed_module.torch,
        "tensor",
        lambda value, **kwargs: original_tensor(value, dtype=kwargs.get("dtype")),
    )
    monkeypatch.setattr(distributed_module.torch.cuda, "is_bf16_supported", lambda: True)

    monkeypatch.setattr(
        distributed_module.dist,
        "all_reduce",
        lambda value, **_kwargs: value.fill_(1),
    )
    assert distributed_precision_dtype("bf16", context) == torch.bfloat16

    monkeypatch.setattr(
        distributed_module.dist,
        "all_reduce",
        lambda value, **_kwargs: value.fill_(0),
    )
    with pytest.raises(RuntimeError, match="at least one distributed CUDA rank"):
        distributed_precision_dtype("bf16", context)


def test_parallel_strategy_config_rejects_ambiguous_legacy_override() -> None:
    config = config_from_raw(
        {
            "data": {"language_pair": ["ko", "ja"]},
            "training": {
                "parallel_strategy": "fsdp2",
                "fsdp_reduce_dtype": "bf16",
            },
        }
    )
    config.validate()
    assert config.training.parallel_strategy == "fsdp2"

    try:
        config_from_raw(
            {
                "training": {
                    "parallel_strategy": "ddp",
                    "fsdp2": True,
                }
            }
        )
    except ValueError as exc:
        assert "cannot both be set" in str(exc)
    else:
        raise AssertionError("ambiguous parallel settings must be rejected")


def test_cuda_multi_gpu_fails_before_process_group_without_nccl(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: False)
    initialized = False

    def record_initialization(**_kwargs: object) -> None:
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        record_initialization,
    )

    with pytest.raises(RuntimeError, match="requires the NCCL"):
        initialize_distributed()
    assert initialized is False


def test_fsdp2_registers_custom_generation_forward_methods(monkeypatch) -> None:
    from torch.distributed import fsdp as fsdp_api

    class Policy:
        def __init__(self, **_kwargs: object) -> None:
            pass

    model = SionForConditionalGeneration(
        ModelConfig(
            vocab_size=32,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=4,
            num_kv_heads=2,
            d_ff=32,
            max_seq_len=16,
            dropout=0.0,
        )
    )
    encoder_layer = model.encoder_layers[0]
    decoder_layer = model.decoder_layers[0]
    events: list[tuple[str, object]] = []

    def fully_shard(module: torch.nn.Module, **_kwargs: object) -> None:
        events.append(("shard", module))

    def register(module: torch.nn.Module, method_name: str) -> None:
        events.append((method_name, module))

    monkeypatch.setattr(fsdp_api, "MixedPrecisionPolicy", Policy)
    monkeypatch.setattr(fsdp_api, "fully_shard", fully_shard)
    monkeypatch.setattr(fsdp_api, "register_fsdp_forward_method", register)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        distributed=True,
        backend="gloo",
    )

    result = parallelize_model(
        model,
        context,
        strategy="fsdp2",
        precision="fp32",
        reshard_after_forward=True,
        materialize_meta=False,
    )

    assert result is model
    assert model._synchronize_generation_across_ranks is True
    assert events == [
        ("shard", encoder_layer),
        ("shard", decoder_layer),
        ("shard", model),
        ("project_cross_key_value", decoder_layer),
        ("forward_step", decoder_layer),
        ("generate", model),
        ("sample", model),
    ]


def test_fsdp2_cpu_gloo_forward_generate_and_sample_smoke(tmp_path: Path) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU FSDP2 smoke test requires torch.distributed with Gloo")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    rendezvous = (tmp_path / "fsdp2-rendezvous").resolve().as_uri()
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=0,
        world_size=1,
    )
    try:
        with torch.device("meta"):
            model = SionForConditionalGeneration(
                ModelConfig(
                    vocab_size=32,
                    d_model=16,
                    encoder_layers=1,
                    decoder_layers=1,
                    num_heads=4,
                    num_kv_heads=2,
                    d_ff=32,
                    max_seq_len=16,
                    dropout=0.0,
                    label_smoothing=0.0,
                    z_loss_weight=0.0,
                    experimental=ExperimentalConfig(
                        bats_enabled=True,
                        bats_coverage_weight=0.01,
                        core_enabled=True,
                        tetm_enabled=True,
                        morphoscript_enabled=True,
                        morphoscript_interval=1,
                        evidence_repair_enabled=True,
                        semantic_parity_enabled=True,
                    ),
                )
            )
        assert model.register_state is not None
        assert model.typed_memory is not None
        assert model.register_state.inject_gate.shape == (1,)
        assert model.typed_memory.gate.shape == (1,)

        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            distributed=True,
            backend="gloo",
        )
        model = parallelize_model(
            model,
            context,
            strategy="fsdp2",
            precision="fp32",
            reduce_dtype="fp32",
            reshard_after_forward=True,
            materialize_meta=True,
        )
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert torch.count_nonzero(model.morph_gates) == 0
        assert torch.count_nonzero(model.evidence_repair.repair_scale) == 0
        assert torch.count_nonzero(model.register_state.inject_gate) == 0
        assert torch.count_nonzero(model.typed_memory.gate) == 0
        assert torch.count_nonzero(model.alignment_head.null_source) == 0

        input_ids = torch.tensor([[4, 5, 3]])
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        decoder_input_ids = torch.tensor([[2, 6, 7]])
        labels = torch.tensor([[6, 7, 3]])
        memory_inputs = {
            "memory_token_ids": torch.tensor([[[8, 9], [10, 0]]]),
            "memory_mask": torch.tensor([[True, True]]),
            "memory_type_ids": torch.tensor([[1, 8]]),
            "memory_mode_ids": torch.tensor([[1, 4]]),
        }
        morphoscript_inputs = {
            "src_script_ids": torch.zeros_like(input_ids),
            "src_onset_ids": torch.zeros_like(input_ids),
            "src_vowel_ids": torch.zeros_like(input_ids),
            "src_coda_ids": torch.zeros_like(input_ids),
        }

        output = model(
            input_ids,
            attention_mask,
            decoder_input_ids,
            labels,
            register_labels=torch.tensor([1]),
            **memory_inputs,
            **morphoscript_inputs,
        )
        assert output.loss is not None
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert model.register_state.inject_gate.grad is not None
        assert model.typed_memory.gate.grad is not None

        generated = model.generate(
            input_ids,
            attention_mask,
            bos_id=2,
            eos_id=3,
            max_new_tokens=3,
            num_beams=1,
            min_new_tokens=2,
            **memory_inputs,
            **morphoscript_inputs,
        )
        sampled = model.sample(
            input_ids,
            attention_mask,
            bos_id=2,
            eos_id=3,
            num_samples=2,
            max_new_tokens=3,
            min_new_tokens=2,
            **memory_inputs,
            **morphoscript_inputs,
        )
        assert generated.shape == (1, 4)
        assert sampled.shape == (1, 2, 4)
    finally:
        torch.distributed.destroy_process_group()


def test_fsdp2_reports_missing_custom_forward_registration_api(
    monkeypatch,
) -> None:
    from torch.distributed import fsdp as fsdp_api

    monkeypatch.setattr(fsdp_api, "register_fsdp_forward_method", None)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        distributed=True,
        backend="gloo",
    )

    with pytest.raises(RuntimeError, match="register_fsdp_forward_method"):
        parallelize_model(
            torch.nn.Linear(2, 2),
            context,
            strategy="fsdp2",
            precision="fp32",
            reshard_after_forward=True,
            materialize_meta=False,
        )


@pytest.mark.parametrize("find_unused_parameters", [False, True])
def test_ddp_disables_static_graph_for_sion_outputs(
    monkeypatch,
    find_unused_parameters: bool,
) -> None:
    ddp_options: dict[str, object] = {}

    def capture_ddp(model: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        ddp_options.update(kwargs)
        return model

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        capture_ddp,
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        distributed=True,
        backend="gloo",
    )

    parallelize_model(
        SionForConditionalGeneration(
            ModelConfig(
                vocab_size=32,
                d_model=16,
                encoder_layers=1,
                decoder_layers=1,
                num_heads=4,
                num_kv_heads=2,
                d_ff=32,
                max_seq_len=16,
                dropout=0.0,
            )
        ),
        context,
        strategy="ddp",
        precision="fp32",
        reshard_after_forward=True,
        materialize_meta=False,
        find_unused_parameters=find_unused_parameters,
    )

    assert ddp_options["static_graph"] is False
    assert ddp_options["find_unused_parameters"] is find_unused_parameters


def test_ddp_unused_parameter_detection_covers_supervised_only_heads() -> None:
    config = AppConfig()
    config.model.experimental.bats_enabled = True
    config.model.experimental.bats_coverage_weight = 0.01
    config.posttraining.enabled = True
    assert requires_ddp_unused_parameter_detection(config) is True

    config.posttraining.enabled = False
    assert requires_ddp_unused_parameter_detection(config) is False

    config.model.experimental.bats_coverage_weight = 0.0
    assert requires_ddp_unused_parameter_detection(config) is True

    config.model.experimental.bats_enabled = False
    config.posttraining.enabled = True
    assert requires_ddp_unused_parameter_detection(config) is False

    config.model.experimental.semantic_parity_enabled = True
    assert requires_ddp_unused_parameter_detection(config) is True

    config.posttraining.enabled = False
    assert requires_ddp_unused_parameter_detection(config) is False


def test_h100_capacity_gate_requires_four_gpus_for_8b_and_sixteen_for_32b() -> None:
    def context(world_size: int) -> DistributedContext:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=world_size,
            device=torch.device("cuda"),
            distributed=True,
            backend="nccl",
        )

    with pytest.raises(RuntimeError, match="at least 4 GPUs"):
        validate_training_capacity(
            8_000_000_000,
            context(2),
            parallel_strategy="fsdp2",
            ema_enabled=True,
            per_gpu_vram_gib=80.0,
        )
    eight_billion = validate_training_capacity(
        8_000_000_000,
        context(4),
        parallel_strategy="fsdp2",
        ema_enabled=True,
        per_gpu_vram_gib=80.0,
    )
    assert eight_billion is not None
    assert eight_billion["per_rank_state_gib"] < eight_billion["state_budget_gib"]

    with pytest.raises(RuntimeError, match="at least 16 GPUs"):
        validate_training_capacity(
            32_083_082_800,
            context(8),
            parallel_strategy="fsdp2",
            ema_enabled=True,
            per_gpu_vram_gib=80.0,
        )
    thirty_two_billion = validate_training_capacity(
        32_083_082_800,
        context(16),
        parallel_strategy="fsdp2",
        ema_enabled=True,
        per_gpu_vram_gib=80.0,
    )
    assert thirty_two_billion is not None
    assert thirty_two_billion["minimum_world_size"] == 16


def test_single_gpu_capacity_gate_runs_before_parameter_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_devices: list[str] = []

    class OversizedModel(torch.nn.Module):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))
            construction_devices.append(self.weight.device.type)

        @staticmethod
        def parameter_count() -> int:
            return 32_083_082_800

    monkeypatch.setattr(train_module, "SionForConditionalGeneration", OversizedModel)
    monkeypatch.setattr(
        train_module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=80 * 2**30),
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cuda"),
        distributed=False,
    )
    with pytest.raises(RuntimeError, match="Switch to FSDP2"):
        construct_training_model(
            AppConfig(),
            context,
            pad_id=0,
            parallel_strategy="single",
        )
    assert construction_devices == ["meta"]


def test_existing_checkpoint_search_covers_stage_directories(tmp_path: Path) -> None:
    config = AppConfig()
    config.training.output_dir = str(tmp_path / "run")
    assert find_existing_checkpoint(config) is None

    checkpoint = tmp_path / "run" / "pretrain" / "checkpoints" / "latest"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"weights")
    assert find_existing_checkpoint(config) == checkpoint


def test_translation_initialization_log_states_the_no_corpus_release_contract() -> None:
    from sion_translate.foundation import FoundationOutcome

    message = train_module.translation_initialization_message(
        FoundationOutcome(ran=False, reason="missing corpus"),
        resume_from=None,
    )

    assert "fresh initialization" in message
    assert "foundation model (sion) will not be trained or exported" in message
    assert "SFT/MRT" in message
    assert "only sion_translate" in message


def test_validated_sft_resume_takes_priority_without_touching_foundation(monkeypatch) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validated SFT resume must skip foundation")

    monkeypatch.setattr(train_module, "run_foundation_stage", _fail)
    outcome = train_module.run_foundation_before_translation(
        AppConfig(),
        cast(Any, None),
        torch.nn.Linear(1, 1),
        cast(Any, None),
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        validated_pretrain_resume="runs/auto/pretrain/checkpoints/latest",
    )
    message = train_module.translation_initialization_message(
        outcome,
        resume_from="runs/auto/pretrain/checkpoints/latest",
        pipeline_branch="foundation-then-translation",
    )

    assert not outcome.ran
    assert "Resuming" in message and "first" in message
    assert "foundation stage" in message and "not be trained or loaded" in message


def test_tokenizer_policy_requires_digit_splitting_and_matching_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "sion.model"
    model_path.write_bytes(b"tokenizer")
    vocab_path = tmp_path / "sion.vocab"
    vocab_path.write_bytes(b"vocabulary")
    pairs = (("ko", "ja"),)

    class DigitTokenizer:
        splits_digits = True
        languages = ("ko", "ja")
        denoise_tags = {"ko": 10, "ja": 11}

    metadata = {
        "version": 2,
        "split_digits": True,
        "model_sha256": file_sha256(model_path),
        "vocab_sha256": file_sha256(vocab_path),
        "language_pairs": [["ko", "ja"]],
    }
    monkeypatch.setattr("sion_translate.cli.train.SionTokenizer", lambda _: DigitTokenizer())
    monkeypatch.setattr(
        "sion_translate.cli.train.load_tokenizer_metadata",
        lambda _: metadata,
    )
    monkeypatch.setattr(
        "sion_translate.cli.train.tokenizer_split_digits_policy",
        lambda _: True,
    )
    assert tokenizer_policy_problem(model_path, pairs) is None
    assert "explicit translation_directions" in str(
        tokenizer_policy_problem(
            model_path,
            pairs,
            translation_directions=(("ko", "ja"),),
            require_recorded_directions=True,
        )
    )
    metadata["translation_directions"] = "ko-ja"
    assert "invalid format" in str(tokenizer_policy_problem(model_path, pairs))
    metadata["translation_directions"] = [["ko", "ja"]]
    assert (
        tokenizer_policy_problem(
            model_path,
            pairs,
            translation_directions=(("ko", "ja"),),
            require_recorded_directions=True,
        )
        is None
    )
    # Foundation languages extend the tokenizer's translation-language
    # denoising controls; they do not replace that base set.
    assert tokenizer_policy_problem(model_path, pairs, ("ko",)) is None
    assert "denoising-tag language set" in str(
        tokenizer_policy_problem(model_path, pairs, ("ko", "ja", "en"))
    )
    assert "reasoning-tag language set" in str(
        tokenizer_policy_problem(model_path, pairs, ("ko", "ja"), ("ja",))
    )

    class MergedDigitTokenizer:
        splits_digits = False

    monkeypatch.setattr(
        "sion_translate.cli.train.SionTokenizer",
        lambda _: MergedDigitTokenizer(),
    )
    assert "split_digits=False" in str(tokenizer_policy_problem(model_path, pairs))

    class WrongLanguageTokenizer:
        splits_digits = True
        languages = ("ko",)

    monkeypatch.setattr(
        "sion_translate.cli.train.SionTokenizer",
        lambda _: WrongLanguageTokenizer(),
    )
    assert "language set" in str(tokenizer_policy_problem(model_path, pairs))


def test_raw_free_artifact_preflight_still_checks_tokenizer_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.data.raw_dir = str(tmp_path / "raw")
    tokenizer = tmp_path / "tokenizer" / "sion.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    config.data.tokenizer_model = str(tokenizer)
    config.data.dataset_dir = str(dataset)
    plan = SimpleNamespace(enabled=False)
    observed: list[tuple[str, ...]] = []

    monkeypatch.setattr(train_module, "scan_configured_raw_data", lambda *_args: {})
    monkeypatch.setattr(train_module, "find_existing_checkpoint", lambda *_args: None)
    monkeypatch.setattr(train_module, "dataset_artifact_problem", lambda *_args: None)

    def reject_policy(
        _path: Path,
        _pairs: tuple[tuple[str, str], ...],
        _foundation_languages: tuple[str, ...],
        reasoning_languages: tuple[str, ...],
        **_kwargs: object,
    ) -> str:
        observed.append(reasoning_languages)
        return "split_digits policy mismatch"

    monkeypatch.setattr(train_module, "tokenizer_policy_problem", reject_policy)

    with pytest.raises(RuntimeError, match="no source data.*split_digits policy mismatch"):
        train_module._ensure_artifacts_on_main(
            config,
            DistributedContext(0, 0, 1, torch.device("cpu"), False),
            plan,
            prepare_foundation=False,
            locks_held=True,
        )

    assert observed == [()]


def test_raw_free_fresh_run_preflights_foundation_before_tokenizer_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.data.raw_dir = str(tmp_path / "raw")
    tokenizer = tmp_path / "tokenizer" / "sion.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    config.data.tokenizer_model = str(tokenizer)
    config.data.dataset_dir = str(dataset)
    plan = SimpleNamespace(
        enabled=True,
        discovery=SimpleNamespace(sources=()),
        languages=("ko", "ja"),
    )
    events: list[object] = []

    monkeypatch.setattr(train_module, "scan_configured_raw_data", lambda *_args: {})
    monkeypatch.setattr(train_module, "find_existing_checkpoint", lambda *_args: None)
    monkeypatch.setattr(train_module, "dataset_artifact_problem", lambda *_args: None)
    monkeypatch.setattr(
        train_module,
        "_preflight_offline_foundation_dataset",
        lambda _config, _plan: events.append("foundation") or ("ja",),
    )

    def accept_policy(
        _path: Path,
        _pairs: tuple[tuple[str, str], ...],
        _foundation_languages: tuple[str, ...],
        reasoning_languages: tuple[str, ...],
        **_kwargs: object,
    ) -> None:
        events.append(reasoning_languages)
        return None

    monkeypatch.setattr(train_module, "tokenizer_policy_problem", accept_policy)

    train_module._ensure_artifacts_on_main(
        config,
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        plan,
        prepare_foundation=False,
        require_offline_foundation=True,
        locks_held=True,
    )

    assert events == ["foundation", ("ja",)]


def test_explicit_direction_graph_cannot_be_backfilled_onto_a_legacy_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.tokenizer_model = str(tmp_path / "tokenizer" / "sion.model")
    config.data.dataset_dir = str(tmp_path / "dataset")
    config.data.translation_directions = [["ja", "ko"]]
    tokenizer_path = Path(config.data.tokenizer_model)
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_bytes(b"legacy-tokenizer")
    plan = SimpleNamespace(enabled=False)
    metadata_writes: list[Path] = []

    monkeypatch.setattr(
        train_module,
        "scan_configured_raw_data",
        lambda *_args: {"parallel.jsonl": 128},
    )
    monkeypatch.setattr(train_module, "find_existing_checkpoint", lambda *_args: None)
    monkeypatch.setattr(
        train_module,
        "tokenizer_policy_problem",
        lambda *_args, **_kwargs: "명시적 translation_directions를 인증할 metadata가 없습니다",
    )
    monkeypatch.setattr(
        train_module,
        "SionTokenizer",
        lambda *_args: SimpleNamespace(splits_digits=True),
    )
    monkeypatch.setattr(train_module, "load_tokenizer_metadata", lambda *_args: None)
    monkeypatch.setattr(
        train_module,
        "write_tokenizer_metadata",
        lambda path, **_kwargs: metadata_writes.append(Path(path)),
    )

    with pytest.raises(RuntimeError, match="명시적 translation_directions"):
        train_module._ensure_artifacts_on_main(
            config,
            DistributedContext(0, 0, 1, torch.device("cpu"), False),
            plan,
            prepare_foundation=False,
            locks_held=True,
        )

    assert metadata_writes == []
