"""Small, structurally complete GPU bundle fixtures for runtime-contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_payload(root: Path, relative_path: str, content: bytes) -> dict[str, Any]:
    path = root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "mode": "100644",
        "origin": "git-index",
        "path": relative_path,
        "sha256": _sha256(content),
        "size": len(content),
    }


def rewrite_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """Rewrite a self-consistent manifest and checksum list after a test mutation."""

    manifest_content = (
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (root / "PACKAGE_MANIFEST.json").write_bytes(manifest_content)
    records = manifest["files"]
    assert isinstance(records, list)
    lines = [f"{record['sha256']}  {record['path']}\n" for record in records]
    lines.append(f"{_sha256(manifest_content)}  PACKAGE_MANIFEST.json\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")


def write_test_bundle(
    root: Path,
    *,
    raw_files: Mapping[str, bytes] | None = None,
    monolingual_files: Mapping[str, bytes] | None = None,
    foundation_enabled: bool = False,
    language_pairs: Sequence[Sequence[str]] = (("de", "fr"),),
    translation_directions: Sequence[Sequence[str]] = (("de", "fr"), ("fr", "de")),
    source_only_languages: Sequence[str] = (),
    foundation_languages: Sequence[str] = ("de", "fr"),
) -> dict[str, Any]:
    """Create a valid format-2 tree without depending on production bundle helpers."""

    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    config_content = (
        "data:\n"
        f"  language_pairs: {json.dumps([list(pair) for pair in language_pairs])}\n"
        f"  translation_directions: "
        f"{json.dumps([list(direction) for direction in translation_directions])}\n"
        "foundation:\n"
        f"  enabled: {'true' if foundation_enabled else 'false'}\n"
    ).encode("utf-8")
    records.append(_write_payload(root, "sion_translate.yaml", config_content))

    dependency_payloads = {
        ".gitattributes": b"* text=auto\n",
        "pyproject.toml": b"[project]\nname='sion-translate'\n",
        "requirements/gpu-build.in": b"setuptools>=77\nwheel\n",
        "requirements/gpu-lock-provenance.json": b"{}\n",
        "requirements/pylock.gpu-cp311-linux-x86_64-cu128.toml": b"lock-version='1.0'\n",
        "src/sion_translate/bundle_contract.py": b'"""Test bundle verifier sentinel."""\n',
    }
    for path, content in dependency_payloads.items():
        records.append(_write_payload(root, path, content))

    raw_files = raw_files or {}
    for path, content in raw_files.items():
        record = _write_payload(root, path, content)
        record["origin"] = "data-jsonl"
        records.append(record)
    monolingual_files = monolingual_files or {}
    for path, content in monolingual_files.items():
        record = _write_payload(root, path, content)
        record["origin"] = "monolingual-corpus"
        records.append(record)

    if not raw_files:
        for path, origin, content in (
            ("artifacts/tokenizer/sion.model", "tokenizer", b"tokenizer"),
            ("artifacts/dataset/manifest.json", "dataset", b"{}\n"),
        ):
            record = _write_payload(root, path, content)
            record["origin"] = origin
            records.append(record)
        if foundation_enabled:
            record = _write_payload(
                root,
                "artifacts/foundation_dataset/manifest.json",
                b"{}\n",
            )
            record["origin"] = "foundation-dataset"
            records.append(record)

    records.sort(key=lambda record: record["path"])
    records_by_path = {record["path"]: record for record in records}

    def identity(path: str) -> dict[str, Any]:
        record = records_by_path[path]
        return {"path": path, "sha256": record["sha256"], "size": record["size"]}

    input_identities = {
        path: {"sha256": records_by_path[path]["sha256"], "size": records_by_path[path]["size"]}
        for path in ("pyproject.toml", "requirements/gpu-build.in")
    }
    dependency_environment = {
        "schema": "sion-gpu-dependency-environment-v1",
        "generator": {"name": "uv", "version": "0.12.3"},
        "target": {
            "machine": "x86_64",
            "manylinux": "2_28",
            "os": "linux",
            "python_implementation": "cpython",
            "python_version": "3.11",
            "torch_backend": "cu128",
        },
        "inputs": input_identities,
        "normalization": identity(".gitattributes"),
        "provenance": identity("requirements/gpu-lock-provenance.json"),
        "lock": {
            "format": "pep751",
            "lock_version": "1.0",
            "package_count": 17,
            **identity("requirements/pylock.gpu-cp311-linux-x86_64-cu128.toml"),
            "wheel_count": 42,
        },
        "venv_command": ["uv", "venv", ".venv"],
        "compile_command": ["uv", "pip", "compile"],
        "sync_command": ["uv", "pip", "sync"],
        "project_install_command": ["uv", "pip", "install", "--editable", "."],
        "resolved_runtime_versions": {
            "numpy": "2.4.6",
            "sentencepiece": "0.2.1",
            "torch": "2.10.0+cu128",
            "torchao": "0.17.0+cu128",
            "transformers": "5.16.1",
        },
    }
    config_record = records_by_path["sion_translate.yaml"]
    contract = {
        "schema": "sion-gpu-training-contract-v2",
        "config_path": "sion_translate.yaml",
        "config_sha256": config_record["sha256"],
        "raw_parallel_data_included": bool(raw_files),
        "language_pairs": [list(pair) for pair in language_pairs],
        "translation_directions": [list(direction) for direction in translation_directions],
        "source_only_languages": list(source_only_languages),
        "foundation_enabled": foundation_enabled,
        "foundation_languages": list(foundation_languages),
        "paths": {
            "raw_dir": "data",
            "tokenizer_model": "artifacts/tokenizer/sion.model",
            "tokenizer_features": "artifacts/tokenizer/token_features.npz",
            "translation_dataset": "artifacts/dataset",
            "foundation_dataset": "artifacts/foundation_dataset",
        },
        "dependency_environment": dependency_environment,
    }
    manifest = {
        "archive_root": "sion_translate",
        "files": records,
        "format_version": 2,
        "git": {"commit": "a" * 40, "tree": "b" * 40},
        "payload": {
            "file_count": len(records),
            "total_bytes": sum(record["size"] for record in records),
        },
        "training_contract": contract,
        "zip_metadata": {
            "compression": "deflate",
            "timestamp": "1980-01-01T00:00:00Z",
            "zip64": True,
        },
    }
    rewrite_manifest(root, manifest)
    return manifest
