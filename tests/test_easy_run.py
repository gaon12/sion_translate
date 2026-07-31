from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import easy_run
import pytest


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
    monkeypatch.setattr(easy_run.os, "name", "posix")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("SION_TMUX_ACTIVE", raising=False)
    monkeypatch.setattr(easy_run.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        easy_run,
        "_install_tmux",
        lambda: (_ for _ in ()).throw(AssertionError("must not install in Slurm/nohup")),
    )
    easy_run._enter_tmux()


def test_missing_tmux_continues_in_foreground(monkeypatch, capsys) -> None:
    monkeypatch.setattr(easy_run.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        easy_run.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run apt")),
    )
    assert easy_run._install_tmux() is None
    assert "현재 프로세스에서 계속" in capsys.readouterr().out


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
    with pytest.raises(SystemExit, match="CUDA 지원 PyTorch"):
        easy_run._validate_gpu_runtime(_fake_torch(gpu_count=0, available=False))


def test_gpu_preflight_requires_nccl_only_for_multi_gpu() -> None:
    assert easy_run._validate_gpu_runtime(_fake_torch(gpu_count=1, nccl=False))[0] == 1
    with pytest.raises(SystemExit, match="NCCL"):
        easy_run._validate_gpu_runtime(_fake_torch(gpu_count=2, nccl=False))
