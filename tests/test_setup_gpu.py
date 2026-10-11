from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import setup_gpu


@pytest.fixture
def linux_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(setup_gpu, "ROOT", tmp_path)
    monkeypatch.setattr(setup_gpu.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_gpu.platform, "machine", lambda: "x86_64")
    return tmp_path


def test_invalid_bundle_stops_before_any_installation(
    linux_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def reject(command: list[str]) -> None:
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup_gpu, "_run", reject)
    with pytest.raises(subprocess.CalledProcessError):
        setup_gpu.main()
    assert len(commands) == 1
    assert commands[0][-3:] == ["scripts/package_gpu_bundle.py", "verify-tree", "."]
    assert not (linux_bundle / ".venv").exists()


def test_setup_reuses_existing_environment_without_starting_training(
    linux_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = linux_bundle / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"existing environment")
    commands: list[list[str]] = []
    monkeypatch.setattr(setup_gpu, "_run", commands.append)

    setup_gpu.main()

    assert python.read_bytes() == b"existing environment"
    assert not any("cpython@3.11" in command for command in commands)
    sync = next(command for command in commands if "sync" in command)
    assert "--require-hashes" in sync and "--strict" in sync and setup_gpu.LOCK in sync
    assert commands[-1][0] == str(python)
    assert "_validate_installed_dependency_runtime" in commands[-1][-1]
    assert not any(
        "easy_run.py" in command or "sion_translate.cli.train" in command for command in commands
    )


def test_failed_dependency_sync_does_not_continue_to_gpu_validation(
    linux_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def reject_sync(command: list[str]) -> None:
        commands.append(command)
        if "sync" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup_gpu, "_run", reject_sync)
    with pytest.raises(subprocess.CalledProcessError):
        setup_gpu.main()
    assert "cpython@3.11" in next(command for command in commands if "--managed-python" in command)
    assert "sync" in commands[-1]


def test_wrong_host_stops_before_bundle_or_package_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_gpu.platform, "system", lambda: "Windows")

    def unexpected(_command: list[str]) -> None:
        pytest.fail("An unsupported host must not install packages")

    monkeypatch.setattr(setup_gpu, "_run", unexpected)
    with pytest.raises(SystemExit, match="Linux x86_64"):
        setup_gpu.main()
