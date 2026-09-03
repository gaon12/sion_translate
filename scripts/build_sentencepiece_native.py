"""Build the authenticated 0.2.1 core with fresh SWIG 4.4.0 Python bindings.

Install CMake, a C++ compiler, SWIG 4.4.0, and Python build/setuptools/wheel
before running this script. It never downloads source or build dependencies.
The source directory must be a fresh, clean checkout of CORE_COMMIT. The
recipe deliberately changes only its generated Python/C++ binding files and
creates its build artifacts. Do not use a checkout containing other work.

Build: python scripts/build_sentencepiece_native.py --source SOURCE --output NEW_DIR
Verify: python scripts/build_sentencepiece_native.py verify --manifest MANIFEST --output NEW_DIR

The resulting wheel is a locally rebuilt native overlay, not the stock wheel
authenticated by the GPU dependency lock. Its separate manifest records that
distinction, the exact upstream core, generator, compiler, and wheel hashes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any
import zipfile

CORE_COMMIT = "31646a467d2051eb904e0b45de3a73e91fe1c1e3"
CORE_TREE = "a256eb7f5d3e634041fa11aa2cbb4b1de065359b"
CORE_VERSION = "0.2.1"
SWIG_VERSION = "4.4.0"
BUILD_SCHEMA = "sion-sentencepiece-native-build-v1"
VERIFY_SCHEMA = "sion-sentencepiece-native-verification-v1"
_OLD_IMPORT = 'if __package__ or "." in __name__:'
_NEW_IMPORT = (
    'if getattr(globals().get("__spec__"), "parent", None) or __package__ or "." in __name__:'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture(command: list[str], *, cwd: Path | None = None, input_text: str | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, input=input_text, text=True, capture_output=True, check=False, timeout=60
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command!r}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def _tracked_source_hashes(source: Path, *, allow_generated: bool = False) -> dict[str, str]:
    """Verify bytes even if Git's stat cache/assume-unchanged flag hides edits."""
    if (
        _capture(["git", "rev-parse", "HEAD"], cwd=source) != CORE_COMMIT
        or _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=source) != CORE_TREE
    ):
        raise ValueError("source HEAD must still match the pinned upstream commit and tree")
    entries = _capture(["git", "ls-tree", "-r", "-z", CORE_COMMIT], cwd=source).split("\0")
    expected: dict[str, str] = {}
    for entry in entries:
        if not entry:
            continue
        header, relative = entry.split("\t", 1)
        mode, object_type, digest = header.split()
        path = source / relative
        if (
            mode not in {"100644", "100755"}
            or object_type != "blob"
            or "\n" in relative
            or "\r" in relative
            or not path.is_file()
            or path.is_symlink()
            or source not in path.resolve().parents
        ):
            raise ValueError("source has an unsupported, missing, or redirected tracked file")
        expected[relative] = digest
    if not expected:
        raise ValueError("source checkout has no authenticated tracked files")
    actual = _capture(
        ["git", "hash-object", "--stdin-paths"],
        cwd=source,
        input_text="".join(path + "\n" for path in expected),
    ).splitlines()
    if len(actual) != len(expected):
        raise ValueError("could not authenticate every tracked source file")
    generated = {
        "python/src/sentencepiece/__init__.py",
        "python/src/sentencepiece/sentencepiece_wrap.cxx",
    }
    for (relative, digest), current in zip(expected.items(), actual, strict=True):
        if current != digest and not (allow_generated and relative in generated):
            raise ValueError(f"tracked source content differs from its Git identity: {relative}")
    return dict(zip(expected, actual, strict=True))


def _source_identity(source: Path) -> dict[str, str]:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source must be an existing real Git checkout directory")
    top = Path(_capture(["git", "rev-parse", "--show-toplevel"], cwd=source)).resolve()
    if top != source:
        raise ValueError("source must name the Git checkout root")
    commit = _capture(["git", "rev-parse", "HEAD"], cwd=source)
    if commit != CORE_COMMIT:
        raise ValueError(f"source must be exactly upstream commit {CORE_COMMIT}")
    status = _capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"],
        cwd=source,
    )
    if status:
        raise ValueError(f"source must be clean, including untracked/ignored files:\n{status}")
    _tracked_source_hashes(source)
    tree = _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=source)
    timestamp = _capture(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=source)
    if tree != CORE_TREE or not timestamp.isdecimal():
        raise ValueError("invalid source tree or commit timestamp")
    return {"commit": commit, "tree": tree, "source_date_epoch": timestamp}


def _generator_identity() -> dict[str, str]:
    executable = shutil.which("swig")
    if executable is None:
        raise RuntimeError("SWIG 4.4.0 must already be installed")
    version = _capture([executable, "-version"])
    match = re.search(r"^SWIG Version ([^\s]+)$", version, flags=re.MULTILINE)
    if match is None or match.group(1) != SWIG_VERSION:
        raise ValueError("the binding generator must be exactly SWIG 4.4.0")
    if os.environ.get("SWIG_LIB"):
        raise ValueError("unset SWIG_LIB so the installed generator uses its own library")
    library = Path(_capture([executable, "-swiglib"])).resolve(strict=True)
    digest = hashlib.sha256()
    count = 0
    for path in sorted(library.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(library).as_posix().encode() + b"\0")
            digest.update(bytes.fromhex(_sha256(path)))
            count += 1
    if not count:
        raise ValueError("the SWIG support library is empty")
    return {
        "version": SWIG_VERSION,
        "executable": str(Path(executable).resolve()),
        "executable_sha256": _sha256(Path(executable)),
        "library": str(library),
        "library_sha256": digest.hexdigest(),
    }


def _validate_proxy(original: str, generated: str) -> None:
    """Allow exactly the independently reviewed generator header/import change."""
    if original.count("# Version 4.3.0\n") != 1 or original.count(_OLD_IMPORT) != 1:
        raise ValueError("upstream Python proxy does not match the reviewed 0.2.1 source")
    expected = original.replace("# Version 4.3.0\n", "# Version 4.4.0\n", 1).replace(
        _OLD_IMPORT, _NEW_IMPORT, 1
    )
    if generated != expected:
        raise ValueError("generated Python proxy has changes beyond the reviewed SWIG delta")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fresh_output(output: Path, *, source: Path | None = None) -> None:
    if source is not None and (output == source or source in output.parents):
        raise ValueError("output must be outside the authenticated source checkout")
    output.mkdir(parents=True, exist_ok=False)


def _run(
    command: list[str], *, output: Path, label: str, env: dict[str, str] | None = None
) -> None:
    """Keep every build diagnostic; any nonzero command stops the recipe."""
    with (output / f"{label}.log").open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {output / (label + '.log')}")


def _compiler_identity(build_directory: Path) -> dict[str, str]:
    candidates = list((build_directory / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(candidates) != 1:
        raise ValueError("CMake must record exactly one C++ compiler identity")
    text = candidates[0].read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for field in ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"):
        match = re.search(rf'set\({field} "([^"\n]+)"\)', text)
        if match is None:
            raise ValueError(f"CMake compiler identity is missing {field}")
        result[field] = match.group(1)
    executable = Path(result["CMAKE_CXX_COMPILER"])
    result["executable_sha256"] = _sha256(executable)
    result["cmake_metadata_sha256"] = _sha256(candidates[0])
    return result


def _wheel_identity(wheel: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must have exactly one distribution metadata file")
        metadata = archive.read(metadata_paths[0]).decode("utf-8")
        if not re.search(r"^Name: sentencepiece$", metadata, re.MULTILINE) or not re.search(
            r"^Version: 0\.2\.1$", metadata, re.MULTILINE
        ):
            raise ValueError("native wheel must preserve sentencepiece version 0.2.1")
        if "sentencepiece/sentencepiece.py" in archive.namelist():
            raise ValueError("wheel must not contain an extra generated proxy module")
        if archive.testzip() is not None:
            raise ValueError("wheel has a corrupt ZIP member")
        record_name = metadata_paths[0].removesuffix("METADATA") + "RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        records = {row[0]: row[1:] for row in rows if len(row) == 3}
        if len(records) != len(rows):
            raise ValueError("wheel RECORD contains duplicate or malformed entries")
        extensions = [
            name
            for name in archive.namelist()
            if name.startswith("sentencepiece/_sentencepiece") and name.endswith((".so", ".pyd"))
        ]
        if len(extensions) != 1:
            raise ValueError("wheel must contain exactly one native SentencePiece extension")
        members = {}
        for kind, name in (("proxy", "sentencepiece/__init__.py"), ("extension", extensions[0])):
            payload = archive.read(name)
            digest = hashlib.sha256(payload).digest()
            expected = [
                "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="),
                str(len(payload)),
            ]
            if records.get(name) != expected:
                raise ValueError(f"wheel RECORD does not authenticate {name}")
            members[kind] = {"relative_path": name, "sha256": digest.hex(), "size": len(payload)}
    return {
        "name": wheel.name,
        "path": str(wheel),
        "size": wheel.stat().st_size,
        "sha256": _sha256(wheel),
        "members": members,
    }


def build_native(source: Path, output: Path, *, jobs: int = 2) -> Path:
    if type(jobs) is not int or not 1 <= jobs <= 64:
        raise ValueError("jobs must be an integer between 1 and 64")
    source, output = source.resolve(strict=True), output.resolve()
    identity = _source_identity(source)
    generator = _generator_identity()
    for module in ("build", "setuptools", "wheel"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(
                f"install the {module} build prerequisite before running this recipe"
            )
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake must already be installed")
    _fresh_output(output, source=source)
    manifest: dict[str, Any] = {
        "schema": BUILD_SCHEMA,
        "status": "running",
        "artifact_kind": "rebuilt-native-overlay-not-stock-lock-wheel",
        "core_version": CORE_VERSION,
        "source": {"path": str(source), **identity},
        "generator": generator,
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "cmake_version": _capture([cmake, "--version"]),
        "jobs": jobs,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    try:
        package = source / "python" / "src" / "sentencepiece"
        proxy, generated_proxy = package / "__init__.py", package / "sentencepiece.py"
        original = proxy.read_text(encoding="utf-8")
        manifest["original_python_sha256"] = _sha256(proxy)
        environment = {**os.environ, "SOURCE_DATE_EPOCH": identity["source_date_epoch"]}
        _run(
            [
                generator["executable"],
                "-c++",
                "-python",
                f"-I{source / 'src'}",
                "-outdir",
                str(package),
                "-o",
                str(package / "sentencepiece_wrap.cxx"),
                str(package / "sentencepiece.i"),
            ],
            output=output,
            label="01-swig",
            env=environment,
        )
        _validate_proxy(original, generated_proxy.read_text(encoding="utf-8"))
        generated_proxy.replace(proxy)
        manifest["generated_cpp_sha256"] = _sha256(package / "sentencepiece_wrap.cxx")
        manifest["generated_python_sha256"] = _sha256(proxy)
        build_directory = source / "build"
        _run(
            [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build_directory),
                f"-DCMAKE_INSTALL_PREFIX={build_directory / 'root'}",
                "-DSPM_ENABLE_SHARED=OFF",
                "-DSPM_DISABLE_EMBEDDED_DATA=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            output=output,
            label="02-configure",
            env=environment,
        )
        _run(
            [
                cmake,
                "--build",
                str(build_directory),
                "--config",
                "Release",
                "--target",
                "install",
                "--parallel",
                str(jobs),
            ],
            output=output,
            label="03-static-core",
            env=environment,
        )
        manifest["compiler"] = _compiler_identity(build_directory)
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output / "wheels"),
                str(source / "python"),
            ],
            output=output,
            label="04-wheel",
            env=environment,
        )
        wheels = list((output / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("build must produce exactly one native wheel")
        manifest["wheel"] = _wheel_identity(wheels[0])
        if manifest["wheel"]["members"]["proxy"]["sha256"] != manifest["generated_python_sha256"]:
            raise ValueError(
                "built wheel does not contain the authenticated generated Python proxy"
            )
        _tracked_source_hashes(source, allow_generated=True)
        manifest["status"] = "built"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    return manifest_path


_TRAIN_VERIFY_CODE = r"""
from pathlib import Path
import sentencepiece as spm
import sys
root = Path(sys.argv[1])
corpus = root / "corpus.txt"
corpus.write_text(("A multilingual tokenizer learns stable pieces.\n"
                   "Ein Tokenizer lernt Texte. Bonjour le monde.\n") * 120, encoding="utf-8")
spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / "tiny"),
    vocab_size=64, model_type="unigram", character_coverage=1.0,
    hard_vocab_limit=False, num_threads=2, shuffle_input_sentence=False)
"""

_VERIFY_CODE = r"""
import importlib.metadata
import json
import pickle
import sentencepiece as spm
import sys

assert importlib.metadata.version("sentencepiece") == "0.2.1"
assert spm.__version__ == "0.2.1"
processor = spm.SentencePieceProcessor(model_file=sys.argv[1])
text = "Bonjour le monde."
tokens = processor.encode(text)
assert tokens and processor.decode(tokens) == text
restored = pickle.loads(pickle.dumps(processor))
assert restored.encode(text) == tokens
proto = processor.encode_as_immutable_proto(text)
assert proto.text == text and len(proto.pieces) == len(tokens)
assert processor.serialized_model_proto()
del proto, restored, processor
print(json.dumps({"version": spm.__version__, "multithread_training": True,
                  "pickle": True, "encode_decode": True, "immutable_proto": True}))
"""


def _validated_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != BUILD_SCHEMA
        or manifest.get("status") != "built"
        or manifest.get("source", {}).get("commit") != CORE_COMMIT
        or manifest.get("source", {}).get("tree") != CORE_TREE
        or manifest.get("core_version") != CORE_VERSION
        or manifest.get("generator", {}).get("version") != SWIG_VERSION
    ):
        raise ValueError("verification requires the exact completed native build manifest")
    for value in (manifest.get("source", {}).get("tree"),):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
            raise ValueError("build manifest has an invalid source tree")
    for value in (
        manifest.get("generated_cpp_sha256"),
        manifest.get("generated_python_sha256"),
        manifest.get("wheel", {}).get("sha256"),
        manifest.get("generator", {}).get("executable_sha256"),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("build manifest has an invalid provenance hash")
    return manifest


def verify_installed(manifest_path: Path) -> dict[str, Any]:
    """Authenticate installed native/proxy bytes without importing the extension.

    The manifest must itself be bound by the caller's runtime/image attestation.
    This function compares its wheel-RECORD-derived hashes with installed bytes;
    it does not claim an unsigned local JSON file is an upstream signature.
    """
    manifest = _validated_manifest(manifest_path)
    distribution = importlib.metadata.distribution("sentencepiece")
    if distribution.version != CORE_VERSION:
        raise ValueError("installed SentencePiece must remain version 0.2.1")
    members = manifest["wheel"].get("members", {})
    if set(members) != {"proxy", "extension"}:
        raise ValueError("native manifest must authenticate proxy and extension members")
    for kind, member in members.items():
        relative = member.get("relative_path", "")
        parts = relative.split("/")
        if len(parts) != 2 or parts[0] != "sentencepiece" or "\\" in relative:
            raise ValueError("native manifest contains an invalid installed member path")
        if kind == "proxy" and relative != "sentencepiece/__init__.py":
            raise ValueError("native manifest proxy path is invalid")
        if kind == "extension" and not (
            parts[1].startswith("_sentencepiece") and parts[1].endswith((".so", ".pyd"))
        ):
            raise ValueError("native manifest extension path is invalid")
        installed = Path(str(distribution.locate_file(relative)))
        if (
            not installed.is_file()
            or installed.stat().st_size != member.get("size")
            or _sha256(installed) != member.get("sha256")
        ):
            raise ValueError(
                f"installed SentencePiece {kind} differs from the authenticated native wheel"
            )
    if members["proxy"]["sha256"] != manifest["generated_python_sha256"]:
        raise ValueError("installed proxy provenance differs from the generated source")
    return {
        "source_commit": manifest["source"]["commit"],
        "source_tree": manifest["source"]["tree"],
        "swig_version": manifest["generator"]["version"],
        "wrapper_sha256": manifest["generated_cpp_sha256"],
        "proxy_sha256": manifest["generated_python_sha256"],
        "wheel_sha256": manifest["wheel"]["sha256"],
        "installed_extension_sha256": members["extension"]["sha256"],
        "manifest_sha256": _sha256(manifest_path),
    }


def verify_native(manifest_path: Path, output: Path, *, wheel: Path | None = None) -> Path:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _validated_manifest(manifest_path)
    wheel = (wheel or Path(manifest["wheel"]["path"])).resolve(strict=True)
    actual_wheel = _wheel_identity(wheel)
    if any(
        actual_wheel[key] != manifest["wheel"].get(key)
        for key in ("name", "size", "sha256", "members")
    ):
        raise ValueError("wheel identity does not match the native build manifest")
    if actual_wheel["members"]["proxy"]["sha256"] != manifest["generated_python_sha256"]:
        raise ValueError("wheel proxy does not match its recorded generated source")
    output = output.resolve()
    _fresh_output(output)
    result: dict[str, Any] = {
        "schema": VERIFY_SCHEMA,
        "status": "running",
        "build_manifest_sha256": _sha256(manifest_path),
        "wheel": actual_wheel,
        "python": sys.version,
        "platform": platform.platform(),
    }
    result_path = output / "verification.json"
    _write_json(result_path, result)
    try:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        _run(
            [sys.executable, "-m", "venv", str(output / "venv")],
            output=output,
            label="01-venv",
            env=environment,
        )
        python = output / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            output=output,
            label="02-install",
            env=environment,
        )
        with (
            (output / "training.stdout.log").open("wb") as stdout,
            (output / "training.stderr.log").open("wb") as stderr,
        ):
            training = subprocess.run(
                [str(python), "-I", "-W", "error", "-c", _TRAIN_VERIFY_CODE, str(output)],
                stdout=stdout,
                stderr=stderr,
                env=environment,
                check=False,
                timeout=120,
            )
        training_stderr = (output / "training.stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if training.returncode or re.search(r"warning|error|fatal", training_stderr, re.IGNORECASE):
            raise RuntimeError(
                "native multithread training verification failed; retained all trainer diagnostics"
            )
        with (
            (output / "runtime.stdout.log").open("wb") as stdout,
            (output / "runtime.stderr.log").open("wb") as stderr,
        ):
            run = subprocess.run(
                [str(python), "-I", "-W", "error", "-c", _VERIFY_CODE, str(output / "tiny.model")],
                stdout=stdout,
                stderr=stderr,
                env=environment,
                check=False,
                timeout=120,
            )
        if run.returncode or (output / "runtime.stderr.log").stat().st_size:
            raise RuntimeError(
                "native runtime verification failed; inspect the retained stdout/stderr logs"
            )
        result.update(status="passed", runtime_exit_code=run.returncode, runtime_stderr_bytes=0)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(result_path, result)
        raise
    _write_json(result_path, result)
    return result_path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "verify-installed":
        parser = argparse.ArgumentParser(
            description="Authenticate installed native SentencePiece bytes"
        )
        parser.add_argument("--manifest", type=Path, required=True)
        args = parser.parse_args(arguments[1:])
        print(json.dumps(verify_installed(args.manifest), indent=2, sort_keys=True))
        return 0
    verify = bool(arguments and arguments[0] == "verify")
    if verify:
        arguments.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    if verify:
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--wheel", type=Path)
    else:
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(arguments)
    path = (
        verify_native(args.manifest, args.output, wheel=args.wheel)
        if verify
        else build_native(args.source, args.output, jobs=args.jobs)
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
