from __future__ import annotations

from pathlib import Path

import easy_run


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
    easy_run._install_tmux()
