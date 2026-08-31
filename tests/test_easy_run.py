from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from bundle_contract_fixtures import rewrite_manifest, write_test_bundle
import easy_run
import pytest


def test_persistent_artifacts_use_the_stable_public_layout() -> None:
    artifact_root = easy_run.ROOT / "artifacts"

    assert easy_run.DEFAULT_ARTIFACT_ROOT == "artifacts"
    assert easy_run.PERSISTENT_ARTIFACTS == artifact_root
    assert easy_run._runtime_artifact_directory(None) == artifact_root


def test_ram_artifacts_use_the_same_stable_layout(tmp_path: Path) -> None:
    assert easy_run._runtime_artifact_directory(tmp_path) == tmp_path / "artifacts"


def test_ram_restore_excludes_historical_artifact_backups(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    runtime = tmp_path / "runtime"
    (persistent / "tokenizer").mkdir(parents=True)
    (persistent / "dataset").mkdir()
    (persistent / "foundation_dataset").mkdir()
    (persistent / "tokenizer.incompatible-old").mkdir()
    (persistent / "tokenizer" / "sion.model").write_bytes(b"active")

    easy_run._restore_active_artifacts(persistent, runtime)

    assert (runtime / "tokenizer" / "sion.model").read_bytes() == b"active"
    assert (runtime / "dataset").is_dir()
    assert (runtime / "foundation_dataset").is_dir()
    assert not (runtime / "tokenizer.incompatible-old").exists()


def test_atomic_sync_directory_replaces_complete_cache(tmp_path: Path) -> None:
    source = tmp_path / "ram" / "dataset"
    destination = tmp_path / "disk" / "dataset"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "manifest.json").write_text("new", encoding="utf-8")
    (destination / "manifest.json").write_text("old", encoding="utf-8")

    easy_run._atomic_sync_directory(source, destination)

    assert (destination / "manifest.json").read_text(encoding="utf-8") == "new"
    assert not list((tmp_path / "disk").glob(".*.sync-*"))
    assert not list((tmp_path / "disk").glob(".*.previous-*"))


def test_atomic_sync_directory_restores_previous_cache_when_publish_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "ram" / "dataset"
    destination = tmp_path / "disk" / "dataset"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "manifest.json").write_text("new", encoding="utf-8")
    (destination / "manifest.json").write_text("old", encoding="utf-8")
    original_replace = Path.replace

    def fail_temporary_publish(path: Path, target: Path) -> Path:
        if ".sync-" in path.name:
            raise OSError("injected publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_temporary_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        easy_run._atomic_sync_directory(source, destination)

    assert (destination / "manifest.json").read_text(encoding="utf-8") == "old"
    assert not list((tmp_path / "disk").glob(".*.sync-*"))
    assert not list((tmp_path / "disk").glob(".*.previous-*"))


def test_generated_config_points_only_data_artifacts_to_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "sion_translate.yaml").write_text(
        "training:\n  output_dir: runs/auto\n", encoding="utf-8"
    )
    monkeypatch.setattr(easy_run, "ROOT", root)

    config_path = easy_run._generated_config(tmp_path / "ram-data", tmp_path / "ram-artifacts")
    try:
        raw = easy_run.yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["data"]["raw_dir"] == str(tmp_path / "ram-data")
        assert raw["data"]["dataset_dir"] == str(tmp_path / "ram-artifacts" / "dataset")
        assert raw["foundation"]["dataset_dir"] == str(
            tmp_path / "ram-artifacts" / "foundation_dataset"
        )
        assert raw["data"]["source_sampling_alpha"] == 0.9
        assert raw["training"]["output_dir"] == "runs/auto"
    finally:
        config_path.unlink(missing_ok=True)


def test_generated_config_preserves_explicit_source_sampling_alpha(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "sion_translate.yaml").write_text(
        "data:\n  source_sampling_alpha: 0.75\n", encoding="utf-8"
    )
    monkeypatch.setattr(easy_run, "ROOT", root)

    config_path = easy_run._generated_config(tmp_path / "ram-data", tmp_path / "ram-artifacts")
    try:
        raw = easy_run.yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["data"]["source_sampling_alpha"] == 0.75
    finally:
        config_path.unlink(missing_ok=True)


def test_expressive_cultural_corpus_builder_targets_training_and_challenge_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    data_dir = root / "data"
    builder = root / "scripts" / "data" / "build_expressive_cultural_corpus.py"
    seed = root / "examples" / "expressive_cultural_seed_pairs.jsonl"
    builder.parent.mkdir(parents=True)
    seed.parent.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    builder.write_text("# test builder\n", encoding="utf-8")
    seed.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(easy_run, "ROOT", root)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], env: dict[str, str]) -> None:
        calls.append((command, env))
        output_index = command.index("--training-output") + 1
        Path(command[output_index]).write_text("generated\n", encoding="utf-8")

    monkeypatch.setattr(easy_run, "_run", fake_run)
    env = {"PYTHONPATH": "test-src"}

    output = easy_run._build_expressive_cultural_corpus(data_dir, env)

    assert output == data_dir / easy_run.EXPRESSIVE_CORPUS_NAME
    assert output.read_text(encoding="utf-8") == "generated\n"
    assert len(calls) == 1
    command, forwarded_env = calls[0]
    assert command == [
        easy_run.sys.executable,
        str(builder),
        "--seed",
        str(seed),
        "--training-output",
        str(output),
        "--challenge-output",
        str(root / "examples" / "expressive_cultural_cases.jsonl"),
    ]
    assert forwarded_env is env


def test_raw_discovery_builds_generated_corpus_before_listing(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    observed: list[Path] = []

    def build_first(path: Path, env: dict[str, str]) -> Path:
        assert not list(path.glob("*.jsonl"))
        output = path / easy_run.EXPRESSIVE_CORPUS_NAME
        output.write_text("generated\n", encoding="utf-8")
        observed.append(output)
        return output

    monkeypatch.setattr(easy_run, "_build_expressive_cultural_corpus", build_first)

    raw_files = easy_run._discover_raw_files(data_dir, {"PYTHONPATH": "test-src"})

    assert observed == [data_dir / easy_run.EXPRESSIVE_CORPUS_NAME]
    assert raw_files == observed


def _write_bundle_manifest(
    root: Path,
    *,
    raw_parallel_data_included: bool,
    monolingual_corpus_included: bool = False,
    foundation_enabled: bool = True,
) -> None:
    write_test_bundle(
        root,
        raw_files=(
            {"data/custom-language-graph.jsonl": b"{}\n"} if raw_parallel_data_included else None
        ),
        monolingual_files=(
            {"data/corpus/de/news.txt": b"Ein ausreichend langer Beispielsatz.\n"}
            if monolingual_corpus_included
            else None
        ),
        foundation_enabled=foundation_enabled,
    )


def test_embedded_bundle_policy_preserves_config_and_source_choices(tmp_path: Path) -> None:
    _write_bundle_manifest(
        tmp_path,
        raw_parallel_data_included=False,
        monolingual_corpus_included=False,
    )

    policy = easy_run._embedded_bundle_policy(tmp_path)

    assert policy == easy_run.EmbeddedBundlePolicy(
        config_path=tmp_path / "sion_translate.yaml",
        raw_parallel_data_included=False,
        monolingual_corpus_included=False,
        foundation_enabled=True,
    )
    assert easy_run._existing_bundle_raw_files(tmp_path / "data", policy) == []


def test_embedded_bundle_policy_rejects_an_inventory_disagreement(tmp_path: Path) -> None:
    manifest = write_test_bundle(
        tmp_path,
        raw_files={"data/corpus.jsonl": b"{}\n"},
    )
    manifest["training_contract"]["raw_parallel_data_included"] = False
    rewrite_manifest(tmp_path, manifest)

    with pytest.raises(SystemExit, match="raw data payload"):
        easy_run._embedded_bundle_policy(tmp_path)


def test_raw_bundle_lists_existing_payload_without_running_a_builder(tmp_path: Path) -> None:
    _write_bundle_manifest(tmp_path, raw_parallel_data_included=True)
    data_dir = tmp_path / "data"
    shard = data_dir / "custom-language-graph.jsonl"
    policy = easy_run._embedded_bundle_policy(tmp_path)
    assert policy is not None

    assert easy_run._existing_bundle_raw_files(data_dir, policy) == [shard]


def test_bundle_authentication_fails_before_gpu_runtime_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "sion_translate"
    _write_bundle_manifest(root, raw_parallel_data_included=False)
    (root / "sion_translate.yaml").write_text(
        "data:\n  language_pairs: [[es, it]]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(easy_run, "ROOT", root)
    monkeypatch.setattr(easy_run, "PERSISTENT_ARTIFACTS", root / "artifacts")
    monkeypatch.setattr(easy_run.sys, "argv", ["easy_run.py"])
    monkeypatch.setattr(
        easy_run,
        "_enter_tmux",
        lambda: (_ for _ in ()).throw(AssertionError("tmux must not start for an invalid bundle")),
    )
    monkeypatch.setattr(
        easy_run,
        "_validate_gpu_runtime",
        lambda _torch: (_ for _ in ()).throw(
            AssertionError("GPU preflight must not run for an invalid bundle")
        ),
    )

    with pytest.raises(SystemExit, match="failed runtime authentication"):
        easy_run.main()


def test_stripped_bundle_metadata_fails_before_tmux_or_gpu_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "sion_translate"
    _write_bundle_manifest(root, raw_parallel_data_included=False)
    (root / "PACKAGE_MANIFEST.json").unlink()
    (root / "SHA256SUMS").unlink()
    (root / "pyproject.toml").unlink()
    (root / ".git").mkdir()
    monkeypatch.setattr(easy_run, "ROOT", root)
    monkeypatch.setattr(easy_run, "PERSISTENT_ARTIFACTS", root / "artifacts")
    monkeypatch.setattr(easy_run.sys, "argv", ["easy_run.py", easy_run.LOCAL_CHECKOUT_FLAG])
    monkeypatch.setattr(
        easy_run,
        "_enter_tmux",
        lambda: (_ for _ in ()).throw(AssertionError("tmux must not start")),
    )
    monkeypatch.setattr(
        easy_run,
        "_validate_gpu_runtime",
        lambda _torch: (_ for _ in ()).throw(AssertionError("GPU preflight must not run")),
    )

    with pytest.raises(SystemExit, match="not a valid Git checkout"):
        easy_run.main()


def test_prepared_bundle_main_never_requires_or_mutates_raw_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "sion_translate"
    root.mkdir()
    (root / "data").mkdir()
    (root / "artifacts" / "tokenizer").mkdir(parents=True)
    _write_bundle_manifest(root, raw_parallel_data_included=False)
    monkeypatch.setattr(easy_run.sys, "argv", ["easy_run.py"])
    monkeypatch.setattr(easy_run, "ROOT", root)
    monkeypatch.setattr(easy_run, "PERSISTENT_ARTIFACTS", root / "artifacts")
    monkeypatch.setattr(easy_run, "_enter_tmux", lambda **_kwargs: None)
    monkeypatch.setattr(
        easy_run,
        "_validate_gpu_runtime",
        lambda _torch: (2, ("NVIDIA H100",)),
    )
    monkeypatch.setattr(
        easy_run,
        "_validate_installed_dependency_runtime",
        lambda _torch: {"torch": "locked-test-runtime"},
    )
    monkeypatch.setattr(easy_run, "_run_cuda_kernel_canaries", lambda _count, _env: [])
    for forbidden in (
        "_discover_raw_files",
        "_ram_workspace",
        "_generated_config",
        "_check_shard_keys",
        "_report_foundation_corpus",
        "_atomic_sync_directory",
    ):
        monkeypatch.setattr(
            easy_run,
            forbidden,
            lambda *args, _name=forbidden, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"prepared bundle must not call {_name}")
            ),
        )
    verified_tokenizers: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        easy_run,
        "_verify_tokenizer",
        lambda model, data: verified_tokenizers.append((model, data)),
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(easy_run, "_run", lambda command, env: commands.append(command))

    easy_run.main()

    config_path = root / "sion_translate.yaml"
    assert config_path.is_file()
    assert len(commands) == 2
    assert commands[0][-3:] == ["--config", str(config_path), "--prepare-only"]
    assert easy_run.LOCAL_CHECKOUT_FLAG not in commands[0]
    assert commands[0][commands[0].index("--config") + 1] == str(config_path)
    assert easy_run.LOCAL_CHECKOUT_FLAG not in commands[1]
    assert commands[1][commands[1].index("--config") + 1] == str(config_path)
    assert verified_tokenizers == [(root / "artifacts/tokenizer/sion.model", root / "data")]


def test_prepared_bundle_rejects_local_checkout_mode_before_tmux_or_gpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "sion_translate"
    _write_bundle_manifest(root, raw_parallel_data_included=False)
    monkeypatch.setattr(easy_run, "ROOT", root)
    monkeypatch.setattr(easy_run, "PERSISTENT_ARTIFACTS", root / "artifacts")
    monkeypatch.setattr(easy_run.sys, "argv", ["easy_run.py", easy_run.LOCAL_CHECKOUT_FLAG])
    monkeypatch.setattr(
        easy_run,
        "_enter_tmux",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("tmux must not start for a conflicting trust mode")
        ),
    )
    monkeypatch.setattr(
        easy_run,
        "_validate_gpu_runtime",
        lambda _torch: (_ for _ in ()).throw(
            AssertionError("GPU preflight must not run for a conflicting trust mode")
        ),
    )

    with pytest.raises(SystemExit, match="cannot be used with an authenticated GPU bundle"):
        easy_run.main()


def test_enter_tmux_is_noop_inside_existing_tmux(monkeypatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-session")
    monkeypatch.setattr(
        easy_run,
        "_install_tmux",
        lambda: (_ for _ in ()).throw(AssertionError("must not install recursively")),
    )
    easy_run._enter_tmux()


def test_install_tmux_skips_when_already_available(monkeypatch) -> None:
    monkeypatch.setattr(easy_run.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(
        easy_run.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run apt")),
    )
    assert easy_run._install_tmux() == "/usr/bin/tmux"


def test_enter_tmux_is_noop_for_noninteractive_launch(monkeypatch) -> None:
    monkeypatch.setattr(easy_run, "TMUX_SUPPORTED", True)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SION_TMUX_ACTIVE", raising=False)
    monkeypatch.setattr(easy_run.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        easy_run,
        "_install_tmux",
        lambda: (_ for _ in ()).throw(AssertionError("must not install in Slurm/nohup")),
    )
    easy_run._enter_tmux()


def test_enter_tmux_preserves_explicit_local_checkout_opt_in(monkeypatch) -> None:
    class ExpectedExec(RuntimeError):
        pass

    observed: list[object] = []
    monkeypatch.setattr(easy_run, "TMUX_SUPPORTED", True)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SION_TMUX_ACTIVE", raising=False)
    monkeypatch.delenv("SION_NO_TMUX", raising=False)
    monkeypatch.setattr(easy_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(easy_run.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(easy_run, "_install_tmux", lambda: "/usr/bin/tmux")
    monkeypatch.setattr(easy_run, "_tmux_session_name", lambda: "sion-test")
    monkeypatch.setattr(
        easy_run.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    def capture_execve(executable, arguments, environment):
        observed.extend((executable, arguments, environment))
        raise ExpectedExec

    monkeypatch.setattr(easy_run.os, "execve", capture_execve)

    with pytest.raises(ExpectedExec):
        easy_run._enter_tmux(allow_local_checkout=True)

    command = easy_run.shlex.split(observed[1][-1])
    assert command[-1] == easy_run.LOCAL_CHECKOUT_FLAG
    assert observed[2]["SION_TMUX_ACTIVE"] == "1"


def test_tmux_training_command_omits_unrequested_local_trust() -> None:
    command = easy_run._tmux_training_command()

    assert easy_run.LOCAL_CHECKOUT_FLAG not in command


def test_launcher_argument_parser_accepts_only_the_documented_option() -> None:
    assert easy_run._parse_allow_local_checkout([]) is False
    assert easy_run._parse_allow_local_checkout([easy_run.LOCAL_CHECKOUT_FLAG]) is True

    with pytest.raises(SystemExit):
        easy_run._parse_allow_local_checkout(["--unknown-option"])


def test_missing_tmux_continues_in_foreground(monkeypatch, capsys) -> None:
    monkeypatch.setattr(easy_run.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        easy_run.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run apt")),
    )
    assert easy_run._install_tmux() is None
    assert "continue in the current process" in capsys.readouterr().out


def _fake_torch(*, gpu_count: int, available: bool = True, nccl: bool = True):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: gpu_count,
            is_available=lambda: available,
            get_device_name=lambda index: ("NVIDIA A100", "NVIDIA H100")[index % 2],
        ),
        distributed=SimpleNamespace(is_nccl_available=lambda: nccl),
    )


def test_gpu_preflight_accepts_single_and_mixed_cuda_devices() -> None:
    assert easy_run._validate_gpu_runtime(_fake_torch(gpu_count=1)) == (
        1,
        ("NVIDIA A100",),
    )
    assert easy_run._validate_gpu_runtime(_fake_torch(gpu_count=2)) == (
        2,
        ("NVIDIA A100", "NVIDIA H100"),
    )


def test_gpu_preflight_rejects_cpu_only_pytorch() -> None:
    with pytest.raises(SystemExit, match="CUDA support"):
        easy_run._validate_gpu_runtime(_fake_torch(gpu_count=0, available=False))


def test_gpu_preflight_requires_nccl_only_for_multi_gpu() -> None:
    assert easy_run._validate_gpu_runtime(_fake_torch(gpu_count=1, nccl=False))[0] == 1
    with pytest.raises(SystemExit, match="NCCL"):
        easy_run._validate_gpu_runtime(_fake_torch(gpu_count=2, nccl=False))


def _locked_runtime_torch():
    return SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )


def test_dependency_runtime_matches_the_authenticated_gpu_lock() -> None:
    locked = {
        "numpy": "2.4.6",
        "sentencepiece": "0.2.1",
        "torchao": "0.17.0+cu128",
        "transformers": "5.16.1",
    }

    report = easy_run._validate_installed_dependency_runtime(
        _locked_runtime_torch(),
        python_version=(3, 11),
        operating_system="Linux",
        machine="x86_64",
        distribution_version=locked.__getitem__,
    )

    assert report == {**locked, "torch": "2.10.0+cu128", "cuda": "12.8"}


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("torch", "2.10.0+cu126", "torch"),
        ("cuda", "12.6", "compiled for CUDA"),
        ("numpy", "2.4.5", "numpy"),
        ("python", (3, 12), "Python 3.12"),
        ("system", "Windows", "operating system"),
        ("machine", "aarch64", "machine"),
    ),
)
def test_dependency_runtime_rejects_any_lock_or_platform_mismatch(
    field: str,
    replacement: object,
    message: str,
) -> None:
    torch_module = _locked_runtime_torch()
    locked = {
        "numpy": "2.4.6",
        "sentencepiece": "0.2.1",
        "torchao": "0.17.0+cu128",
        "transformers": "5.16.1",
    }
    python_version = (3, 11)
    operating_system = "Linux"
    machine = "x86_64"
    if field == "torch":
        torch_module.__version__ = replacement
    elif field == "cuda":
        torch_module.version.cuda = replacement
    elif field == "python":
        python_version = replacement  # type: ignore[assignment]
    elif field == "system":
        operating_system = str(replacement)
    elif field == "machine":
        machine = str(replacement)
    else:
        locked[field] = str(replacement)

    with pytest.raises(SystemExit, match=message):
        easy_run._validate_installed_dependency_runtime(
            torch_module,
            python_version=python_version,
            operating_system=operating_system,
            machine=machine,
            distribution_version=locked.__getitem__,
        )


def test_dependency_runtime_reports_missing_locked_packages() -> None:
    def missing_version(package: str) -> str:
        if package == "torchao":
            raise easy_run.importlib_metadata.PackageNotFoundError(package)
        return {
            "numpy": "2.4.6",
            "sentencepiece": "0.2.1",
            "transformers": "5.16.1",
        }[package]

    with pytest.raises(SystemExit, match="missing locked packages: torchao"):
        easy_run._validate_installed_dependency_runtime(
            _locked_runtime_torch(),
            python_version=(3, 11),
            operating_system="Linux",
            machine="x86_64",
            distribution_version=missing_version,
        )


def test_cuda_canary_runner_checks_every_device_with_a_hard_timeout(
    monkeypatch,
) -> None:
    commands: list[list[str]] = []
    options: list[dict[str, object]] = []

    def successful_run(command: list[str], **kwargs):
        commands.append(command)
        options.append(kwargs)
        device_index = int(command[-1])
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"schema":"sion-cuda-canary-v1","status":"passed",'
                '"device_name":"NVIDIA test GPU",'
                f'"device_index":{device_index},"elapsed_seconds":0.25,'
                '"peak_allocated_bytes":1048576}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(easy_run.subprocess, "run", successful_run)
    reports = easy_run._run_cuda_kernel_canaries(2, {"PYTHONPATH": "test"})

    assert [report["device_index"] for report in reports] == [0, 1]
    assert sorted(command[-1] for command in commands) == ["0", "1"]
    assert all("sion_translate.gpu_runtime" in command for command in commands)
    assert all(option["timeout"] == easy_run.CUDA_CANARY_TIMEOUT_SECONDS for option in options)
    assert all(option["cwd"] == easy_run.ROOT for option in options)
    assert all(option["env"] == {"PYTHONPATH": "test"} for option in options)


def test_cuda_canary_timeout_stops_before_training(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], easy_run.CUDA_CANARY_TIMEOUT_SECONDS)

    monkeypatch.setattr(easy_run.subprocess, "run", timeout)
    with pytest.raises(SystemExit, match="timed out.*Training was not started"):
        easy_run._run_one_cuda_kernel_canary(0, {})


def test_cuda_canary_failure_preserves_the_remote_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        easy_run.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="fused AdamW kernel is unsupported",
        ),
    )
    with pytest.raises(SystemExit, match="fused AdamW kernel is unsupported"):
        easy_run._run_one_cuda_kernel_canary(0, {})


def test_cuda_canary_rejects_a_success_report_for_the_wrong_device(monkeypatch) -> None:
    monkeypatch.setattr(
        easy_run.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=('{"schema":"sion-cuda-canary-v1","status":"passed","device_index":1}\n'),
            stderr="",
        ),
    )
    with pytest.raises(SystemExit, match="invalid success report"):
        easy_run._run_one_cuda_kernel_canary(0, {})
