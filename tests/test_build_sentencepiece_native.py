from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from scripts import build_sentencepiece_native as native


ORIGINAL_PROXY = "# Version 4.3.0\n" + native._OLD_IMPORT + "\n    pass\n"
GENERATED_PROXY = "# Version 4.4.0\n" + native._NEW_IMPORT + "\n    pass\n"


def test_windows_ci_selects_a_compatible_compiler_and_preserves_source_bytes() -> None:
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    windows_job = workflow.split("  windows-native-binding:", 1)[1]
    assert "runs-on: windows-2022" in windows_job
    assert "CMAKE_GENERATOR: Visual Studio 17 2022" in windows_job
    assert "CMAKE_GENERATOR_PLATFORM: x64" in windows_job
    assert "git clone --config core.autocrlf=false" in windows_job


def _make_wheel(
    path: Path, *, proxy: bytes = GENERATED_PROXY.encode(), version: str = "0.2.1"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {
        "sentencepiece/__init__.py": proxy,
        "sentencepiece/_sentencepiece.cpython-test.so": b"test native extension bytes",
        f"sentencepiece-{version}.dist-info/METADATA": f"Name: sentencepiece\nVersion: {version}\n".encode(),
    }
    record_name = f"sentencepiece-{version}.dist-info/RECORD"
    record = io.StringIO()
    writer = csv.writer(record)
    for name, payload in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(payload)))
    writer.writerow((record_name, "", ""))
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
        archive.writestr(record_name, record.getvalue())


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    package = root / "python/src/sentencepiece"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(ORIGINAL_PROXY, encoding="utf-8")
    (package / "sentencepiece_wrap.cxx").write_text("original wrapper", encoding="utf-8")
    (package / "sentencepiece.i").write_text("original interface", encoding="utf-8")
    (root / "CMakeLists.txt").write_text("core contents", encoding="utf-8")
    native._capture(["git", "init", "--quiet"], cwd=root)
    native._capture(["git", "config", "core.autocrlf", "false"], cwd=root)
    native._capture(["git", "add", "."], cwd=root)
    native._capture(
        [
            "git",
            "-c",
            "user.name=Native test",
            "-c",
            "user.email=native@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Create isolated native build source fixture",
        ],
        cwd=root,
    )
    monkeypatch.setattr(
        native, "CORE_COMMIT", native._capture(["git", "rev-parse", "HEAD"], cwd=root)
    )
    monkeypatch.setattr(
        native, "CORE_TREE", native._capture(["git", "rev-parse", "HEAD^{tree}"], cwd=root)
    )
    return root


@pytest.mark.parametrize("change", ["tracked", "untracked", "wrong-commit", "hidden-tracked"])
def test_source_authentication_rejects_changes(
    source: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    if change == "wrong-commit":
        monkeypatch.setattr(native, "CORE_COMMIT", "0" * 40)
    elif change == "untracked":
        (source / "injected.cpp").write_text("injected", encoding="utf-8")
    else:
        if change == "hidden-tracked":
            native._capture(
                ["git", "update-index", "--assume-unchanged", "CMakeLists.txt"], cwd=source
            )
        (source / "CMakeLists.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(
        ValueError, match="exactly upstream|must be clean|differs from its Git identity"
    ):
        native._source_identity(source)


def test_proxy_accepts_only_reviewed_swig_delta() -> None:
    native._validate_proxy(ORIGINAL_PROXY, GENERATED_PROXY)
    with pytest.raises(ValueError, match="beyond the reviewed"):
        native._validate_proxy(ORIGINAL_PROXY, GENERATED_PROXY + "extra_behavior = True\n")
    with pytest.raises(ValueError, match="reviewed 0.2.1"):
        native._validate_proxy(ORIGINAL_PROXY.replace("4.3.0", "4.2.0"), GENERATED_PROXY)


@pytest.mark.parametrize("version", ["4.3.0", "4.4.1", "4.4.0-dev"])
def test_generator_version_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    executable = tmp_path / "swig"
    executable.write_bytes(b"generator")
    monkeypatch.setattr(native.shutil, "which", lambda name: str(executable))
    monkeypatch.setattr(native, "_capture", lambda *args, **kwargs: "SWIG Version " + version)
    with pytest.raises(ValueError, match="exactly SWIG 4.4.0"):
        native._generator_identity()


@pytest.fixture
def fake_build(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    generator = tmp_path / "swig"
    generator.write_bytes(b"exact generator")
    cmake = tmp_path / "cmake"
    cmake.write_bytes(b"cmake")
    compiler = tmp_path / "cxx"
    compiler.write_bytes(b"compiler")
    monkeypatch.setattr(
        native,
        "_generator_identity",
        lambda: {
            "version": "4.4.0",
            "executable": str(generator),
            "executable_sha256": native._sha256(generator),
            "library_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda module: object())
    monkeypatch.setattr(
        native.shutil, "which", lambda name: str(cmake) if name == "cmake" else None
    )
    real_capture = native._capture
    monkeypatch.setattr(
        native,
        "_capture",
        lambda command, **kwargs: (
            "cmake version test" if command[0] == str(cmake) else real_capture(command, **kwargs)
        ),
    )
    calls = []

    def run(command, *, output, label, env=None):
        calls.append((command, label, env))
        (output / f"{label}.log").write_text("retained build diagnostics\n", encoding="utf-8")
        package = source / "python/src/sentencepiece"
        if label == "01-swig":
            (package / "sentencepiece.py").write_text(GENERATED_PROXY, encoding="utf-8")
            (package / "sentencepiece_wrap.cxx").write_text("SWIG 4.4.0 wrapper", encoding="utf-8")
        elif label == "02-configure":
            directory = source / "build/CMakeFiles/test"
            directory.mkdir(parents=True)
            (directory / "CMakeCXXCompiler.cmake").write_text(
                f'set(CMAKE_CXX_COMPILER "{compiler.as_posix()}")\n'
                'set(CMAKE_CXX_COMPILER_ID "TestCompiler")\n'
                'set(CMAKE_CXX_COMPILER_VERSION "1.0")\n',
                encoding="utf-8",
            )
        elif label == "04-wheel":
            _make_wheel(
                output / "wheels/sentencepiece-0.2.1-test.whl",
                proxy=(package / "__init__.py").read_bytes(),
            )

    monkeypatch.setattr(native, "_run", run)
    return calls, run


def test_build_records_exact_core_native_hashes_and_keeps_version(
    source: Path, tmp_path: Path, fake_build
) -> None:
    calls, _ = fake_build
    manifest_path = native.build_native(source, tmp_path / "output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "built"
    assert manifest["core_version"] == "0.2.1"
    assert manifest["source"]["commit"] == native.CORE_COMMIT
    assert manifest["generator"]["version"] == "4.4.0"
    assert manifest["artifact_kind"] == "rebuilt-native-overlay-not-stock-lock-wheel"
    assert manifest["generated_python_sha256"] == manifest["wheel"]["members"]["proxy"]["sha256"]
    assert manifest["wheel"]["sha256"] == native._sha256(Path(manifest["wheel"]["path"]))
    assert not (source / "python/src/sentencepiece/sentencepiece.py").exists()
    assert [label for _, label, _ in calls] == [
        "01-swig",
        "02-configure",
        "03-static-core",
        "04-wheel",
    ]
    assert calls[2][0][-2:] == ["--parallel", "2"]
    assert "--no-isolation" in calls[3][0]
    assert all(env["SOURCE_DATE_EPOCH"].isdecimal() for _, _, env in calls)


def test_failed_build_preserves_manifest_and_never_continues(
    source: Path, tmp_path: Path, fake_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, run = fake_build

    def fail_configure(command, **kwargs):
        run(command, **kwargs)
        if kwargs["label"] == "02-configure":
            raise RuntimeError("compiler configuration failed")

    monkeypatch.setattr(native, "_run", fail_configure)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="configuration failed"):
        native.build_native(source, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "configuration failed" in manifest["error"]
    assert len(calls) == 2
    assert not (output / "wheels").exists()


def test_build_rejects_core_mutation_even_if_staged_during_build(
    source: Path, tmp_path: Path, fake_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run = fake_build

    def mutate_core(command, **kwargs):
        run(command, **kwargs)
        if kwargs["label"] == "04-wheel":
            (source / "CMakeLists.txt").write_text("staged malicious core change", encoding="utf-8")
            native._capture(["git", "add", "CMakeLists.txt"], cwd=source)

    monkeypatch.setattr(native, "_run", mutate_core)
    output = tmp_path / "staged-mutation-output"
    with pytest.raises(ValueError, match="differs from its Git identity"):
        native.build_native(source, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_build_rejects_existing_output_before_mutation(
    source: Path, tmp_path: Path, fake_build
) -> None:
    calls, _ = fake_build
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        native.build_native(source, output)
    assert calls == []
    assert native._source_identity(source)["commit"] == native.CORE_COMMIT


@pytest.mark.parametrize("version", ["0.2.2", "0.2.1.post1"])
def test_wheel_version_cannot_change(tmp_path: Path, version: str) -> None:
    wheel = tmp_path / "wrong.whl"
    _make_wheel(wheel, version=version)
    with pytest.raises(ValueError, match="preserve sentencepiece version"):
        native._wheel_identity(wheel)


def test_installed_verification_checks_native_bytes(
    source: Path, tmp_path: Path, fake_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = native.build_native(source, tmp_path / "build-output")
    manifest = json.loads(manifest_path.read_text())
    installed = tmp_path / "installed"
    with zipfile.ZipFile(manifest["wheel"]["path"]) as archive:
        archive.extractall(installed)
    distribution = SimpleNamespace(version="0.2.1", locate_file=lambda path: installed / path)
    monkeypatch.setattr(native.importlib.metadata, "distribution", lambda name: distribution)
    result = native.verify_installed(manifest_path)
    assert (
        result["installed_extension_sha256"] == manifest["wheel"]["members"]["extension"]["sha256"]
    )
    assert result["source_commit"] == native.CORE_COMMIT
    extension = installed / manifest["wheel"]["members"]["extension"]["relative_path"]
    extension.write_bytes(b"replaced native binary")
    with pytest.raises(ValueError, match="differs from the authenticated"):
        native.verify_installed(manifest_path)


def test_fresh_verification_rejects_replaced_wheel_before_install(
    source: Path, tmp_path: Path, fake_build
) -> None:
    manifest_path = native.build_native(source, tmp_path / "build-output")
    manifest = json.loads(manifest_path.read_text())
    _make_wheel(Path(manifest["wheel"]["path"]), proxy=b"replaced proxy")
    with pytest.raises(ValueError, match="wheel identity does not match"):
        native.verify_native(manifest_path, tmp_path / "verify-output")
    assert not (tmp_path / "verify-output").exists()


@pytest.mark.parametrize("tamper", ["proxy-member", "extension-member", "generated-proxy"])
def test_fresh_verification_rejects_forged_manifest_member_hashes(
    source: Path, tmp_path: Path, fake_build, tamper: str
) -> None:
    manifest_path = native.build_native(source, tmp_path / "build-output")
    manifest = json.loads(manifest_path.read_text())
    if tamper == "generated-proxy":
        manifest["generated_python_sha256"] = "f" * 64
    else:
        kind = tamper.removesuffix("-member")
        manifest["wheel"]["members"][kind]["sha256"] = "f" * 64
    native._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="wheel identity does not match|recorded generated source"):
        native.verify_native(manifest_path, tmp_path / "verify-output")
    assert not (tmp_path / "verify-output").exists()


def test_command_failure_retains_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failure(command, *, stdout, **kwargs):
        stdout.write(b"real compiler failure\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(native.subprocess, "run", failure)
    with pytest.raises(RuntimeError, match="failed"):
        native._run(["compiler"], output=tmp_path, label="compile")
    assert (tmp_path / "compile.log").read_bytes() == b"real compiler failure\n"
