"""Install the verified GPU bundle's locked environment without starting training.

After extracting the prepared ZIP on Linux x86_64, run ``python3 setup_gpu.py``.
Internet access and Python's venv/ensurepip support are required. The temporary
uv bootstrap environment is outside the bundle and is removed after installation.
"""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
UV_VERSION = "0.12.3"
LOCK = "requirements/pylock.gpu-cp311-linux-x86_64-cu128.toml"


def _run(command: list[str]) -> None:
    print("[setup_gpu] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("This GPU bundle requires Linux x86_64 with an NVIDIA CUDA GPU.")

    # Refuse a damaged or edited archive before installing anything.
    _run([sys.executable, "scripts/package_gpu_bundle.py", "verify-tree", "."])
    environment = ROOT / ".venv"
    python = environment / "bin" / "python"
    if environment.exists() and not python.is_file():
        raise SystemExit("Existing .venv is incomplete; move it aside before retrying setup.")

    with tempfile.TemporaryDirectory(prefix="sion-gpu-setup-") as temporary:
        bootstrap = Path(temporary) / "bootstrap"
        _run([sys.executable, "-m", "venv", str(bootstrap)])
        bootstrap_python = str(bootstrap / "bin" / "python")
        _run([bootstrap_python, "-m", "pip", "install", f"uv=={UV_VERSION}"])
        uv = [bootstrap_python, "-m", "uv"]
        if not python.is_file():
            _run(
                [
                    *uv,
                    "venv",
                    "--no-config",
                    ".venv",
                    "--python",
                    "cpython@3.11",
                    "--managed-python",
                ]
            )
        else:
            _run(
                [
                    str(python),
                    "-c",
                    "import sys; assert sys.version_info[:2] == (3, 11), 'Expected Python 3.11'",
                ]
            )
        _run(
            [
                *uv,
                "pip",
                "sync",
                "--no-config",
                LOCK,
                "--python",
                str(python),
                "--require-hashes",
                "--strict",
                "--only-binary",
                ":all:",
            ]
        )
        _run(
            [
                *uv,
                "pip",
                "install",
                "--no-config",
                "--python",
                str(python),
                "--no-deps",
                "--no-build-isolation",
                "--editable",
                ".",
            ]
        )
    _run(
        [
            str(python),
            "-c",
            "import easy_run, torch; "
            "print(easy_run._validate_installed_dependency_runtime(torch)); "
            "print(easy_run._validate_gpu_runtime(torch))",
        ]
    )
    print("[setup_gpu] Setup complete. Start/resume training: .venv/bin/python easy_run.py")


if __name__ == "__main__":
    main()
