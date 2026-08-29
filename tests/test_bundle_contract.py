from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bundle_contract_fixtures import rewrite_manifest, write_test_bundle
import sion_translate.bundle_contract as bundle_contract_module
from sion_translate.bundle_contract import (
    BundleContractError,
    load_embedded_training_contract,
    validate_monolingual_source_inventory,
    validate_parallel_source_inventory,
    validate_tokenizer_source_inventory,
    verify_embedded_bundle_payload,
)
from sion_translate.fingerprint import build_dataset_fingerprint


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_loads_a_prepared_contract_for_an_arbitrary_language_graph(tmp_path: Path) -> None:
    write_test_bundle(
        tmp_path,
        language_pairs=(("de", "fr"), ("fr", "zh-Hant")),
        translation_directions=(("de", "fr"), ("fr", "de"), ("fr", "zh-Hant")),
        foundation_languages=("de", "fr", "zh-Hant"),
    )

    contract = load_embedded_training_contract(tmp_path)

    assert contract is not None
    assert contract.raw_parallel_data_included is False
    assert contract.foundation_enabled is False
    assert contract.dependency_environment["target"]["torch_backend"] == "cu128"


def test_rejects_a_self_consistent_dependency_target_change(tmp_path: Path) -> None:
    manifest = write_test_bundle(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["training_contract"]["dependency_environment"]["target"]["python_version"] = "3.12"
    rewrite_manifest(tmp_path, changed)

    with pytest.raises(BundleContractError, match="CPython 3.11"):
        load_embedded_training_contract(tmp_path)


@pytest.mark.parametrize(
    ("forged_path", "message"),
    [
        ("data/e\u0301.jsonl", "NFC-normalized"),
        ("CON", "reserved Windows name"),
        ("data/bad\nname.jsonl", "canonical POSIX text"),
    ],
)
def test_runtime_rejects_paths_forbidden_by_the_authoritative_bundle_format(
    tmp_path: Path,
    forged_path: str,
    message: str,
) -> None:
    manifest = write_test_bundle(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["files"][0]["path"] = forged_path
    changed["files"].sort(key=lambda record: record["path"])
    rewrite_manifest(tmp_path, changed)

    with pytest.raises(BundleContractError, match=message):
        load_embedded_training_contract(tmp_path)


def test_rejects_a_config_change_even_when_the_manifest_is_unchanged(tmp_path: Path) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "sion_translate.yaml").write_text(
        "data:\n  language_pairs: [[es, it]]\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleContractError, match="differs from the GPU bundle"):
        load_embedded_training_contract(tmp_path)


def test_rejects_partial_bundle_metadata_instead_of_downgrading_to_a_checkout(
    tmp_path: Path,
) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()

    with pytest.raises(BundleContractError, match="incomplete integrity metadata"):
        load_embedded_training_contract(tmp_path)


def test_rejects_complete_bundle_metadata_removal_instead_of_failing_open(
    tmp_path: Path,
) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()

    with pytest.raises(BundleContractError, match="pass --allow-local-checkout"):
        load_embedded_training_contract(tmp_path)


def test_source_layout_git_checkout_without_bundle_metadata_remains_supported(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "sion_translate").mkdir(parents=True)
    (tmp_path / "src" / "sion_translate" / "bundle_contract.py").write_text(
        '"""Checkout sentinel."""\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "sion_translate.yaml").write_text("data: {}\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Bundle Contract Test")
    _git(tmp_path, "config", "user.email", "bundle-contract@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initialize checkout")

    with pytest.raises(BundleContractError, match="pass --allow-local-checkout"):
        load_embedded_training_contract(tmp_path)
    assert load_embedded_training_contract(tmp_path, allow_local_checkout=True) is None
    worktree = tmp_path.with_name(f"{tmp_path.name}-worktree")
    _git(
        tmp_path,
        "worktree",
        "add",
        "-q",
        "-b",
        "bundle-contract-worktree",
        str(worktree),
    )
    assert load_embedded_training_contract(worktree, allow_local_checkout=True) is None


def test_source_checkout_with_a_separate_git_directory_remains_supported(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    git_directory = tmp_path / "separate-git-directory"
    checkout.mkdir()
    sentinel = checkout / "src" / "sion_translate" / "bundle_contract.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('"""Checkout sentinel."""\n', encoding="utf-8")
    _git(
        tmp_path,
        "init",
        "-q",
        "--separate-git-dir",
        str(git_directory),
        str(checkout),
    )
    _git(checkout, "config", "user.name", "Bundle Contract Test")
    _git(checkout, "config", "user.email", "bundle-contract@example.invalid")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "initialize separate checkout")

    assert load_embedded_training_contract(checkout, allow_local_checkout=True) is None


def test_empty_git_directory_cannot_replace_removed_bundle_metadata(tmp_path: Path) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()
    (tmp_path / ".git").mkdir()

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(tmp_path, allow_local_checkout=True)


def test_index_only_verifier_cannot_turn_a_directory_into_a_checkout(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("unrelated repository\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Bundle Contract Test")
    _git(tmp_path, "config", "user.email", "bundle-contract@example.invalid")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "unrelated initial commit")
    sentinel = tmp_path / "src" / "sion_translate" / "bundle_contract.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('"""Staged only, not committed."""\n', encoding="utf-8")
    _git(tmp_path, "add", sentinel.relative_to(tmp_path).as_posix())

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(tmp_path, allow_local_checkout=True)


def test_git_environment_cannot_inject_a_decoy_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoy = tmp_path / "decoy"
    sentinel = decoy / "src" / "sion_translate" / "bundle_contract.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('"""Committed decoy verifier."""\n', encoding="utf-8")
    _git(decoy, "init", "-q")
    _git(decoy, "config", "user.name", "Bundle Contract Test")
    _git(decoy, "config", "user.email", "bundle-contract@example.invalid")
    _git(decoy, "add", ".")
    _git(decoy, "commit", "-qm", "decoy checkout")

    stripped = tmp_path / "stripped"
    write_test_bundle(stripped)
    (stripped / "PACKAGE_MANIFEST.json").unlink()
    (stripped / "SHA256SUMS").unlink()
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(stripped))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(stripped, allow_local_checkout=True)


def test_git_replacement_refs_cannot_forge_a_committed_verifier(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("original tree\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Bundle Contract Test")
    _git(tmp_path, "config", "user.email", "bundle-contract@example.invalid")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "original tree")
    original = _git(tmp_path, "rev-parse", "HEAD")

    sentinel = tmp_path / "src" / "sion_translate" / "bundle_contract.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('"""Replacement-only verifier."""\n', encoding="utf-8")
    _git(tmp_path, "add", sentinel.relative_to(tmp_path).as_posix())
    _git(tmp_path, "commit", "-qm", "replacement tree")
    replacement = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "reset", "--hard", original)
    _git(tmp_path, "replace", original, replacement)
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('"""Uncommitted source marker."""\n', encoding="utf-8")

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(
            tmp_path,
            require_project_identity=True,
            allow_local_checkout=True,
        )


def test_committed_symlink_cannot_stand_in_for_the_runtime_verifier(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("repository with a link entry\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Bundle Contract Test")
    _git(tmp_path, "config", "user.email", "bundle-contract@example.invalid")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "original tree")
    sentinel = tmp_path / "src" / "sion_translate" / "bundle_contract.py"
    sentinel.parent.mkdir(parents=True)
    link_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        input="outside-verifier.py",
    ).stdout.strip()
    _git(
        tmp_path,
        "update-index",
        "--add",
        "--cacheinfo",
        f"120000,{link_blob},{sentinel.relative_to(tmp_path).as_posix()}",
    )
    _git(tmp_path, "commit", "-qm", "commit verifier link")

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(
            tmp_path,
            require_project_identity=True,
            allow_local_checkout=True,
        )


def test_project_local_git_executable_cannot_forge_checkout_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "src" / "sion_translate" / "bundle_contract.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('"""Untrusted project verifier."""\n', encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Bundle Contract Test")
    _git(tmp_path, "config", "user.email", "bundle-contract@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initialize checkout")
    fake_git = tmp_path / "git.exe"
    fake_git.write_bytes(b"not a trusted Git executable")
    monkeypatch.setattr(bundle_contract_module.shutil, "which", lambda _name: str(fake_git))

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(
            tmp_path,
            require_project_identity=True,
            allow_local_checkout=True,
        )


def test_sibling_git_executable_cannot_replace_missing_git_administration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "stripped-project"
    project.mkdir()
    fake_git = tmp_path / "attacker-bin" / "git.exe"
    fake_git.parent.mkdir()
    fake_git.write_bytes(b"not a trusted Git executable")
    monkeypatch.setattr(bundle_contract_module.shutil, "which", lambda _name: str(fake_git))
    monkeypatch.setattr(
        bundle_contract_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Git must not run without real administrative state")
        ),
    )

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(
            project,
            require_project_identity=True,
            allow_local_checkout=True,
        )


def test_required_project_identity_rejects_a_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing-project"

    with pytest.raises(BundleContractError, match="required project root does not exist"):
        load_embedded_training_contract(missing, require_project_identity=True)


def test_removing_a_mutable_project_marker_cannot_hide_stripped_metadata(
    tmp_path: Path,
) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()
    (tmp_path / "pyproject.toml").unlink()
    (tmp_path / "src" / "sion_translate" / "bundle_contract.py").unlink()

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(tmp_path, allow_local_checkout=True)


def test_renaming_source_cannot_hide_a_stripped_bundle_from_an_installed_cli(
    tmp_path: Path,
) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()
    (tmp_path / "src").rename(tmp_path / "removed-source")

    with pytest.raises(BundleContractError, match="pass --allow-local-checkout"):
        load_embedded_training_contract(tmp_path)


def test_unrelated_directory_with_easy_run_filename_is_not_a_project_checkout(
    tmp_path: Path,
) -> None:
    (tmp_path / "easy_run.py").write_text("print('unrelated experiment')\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "unrelated-application"\n',
        encoding="utf-8",
    )

    assert load_embedded_training_contract(tmp_path) is None


def test_required_project_identity_rejects_a_fully_gutted_bundle_root(tmp_path: Path) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()
    (tmp_path / "src").rename(tmp_path / "removed-source")
    (tmp_path / "easy_run.py").unlink(missing_ok=True)

    with pytest.raises(BundleContractError, match="not a valid Git checkout"):
        load_embedded_training_contract(
            tmp_path,
            require_project_identity=True,
            allow_local_checkout=True,
        )


def test_dangling_integrity_metadata_is_not_treated_as_absent(tmp_path: Path) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / "PACKAGE_MANIFEST.json").unlink()
    try:
        (tmp_path / "PACKAGE_MANIFEST.json").symlink_to(tmp_path / "missing-manifest")
    except OSError:
        pytest.skip("this host does not permit file symlink creation")

    with pytest.raises(BundleContractError, match="not a regular file"):
        load_embedded_training_contract(tmp_path)


def test_cached_contract_rejects_a_self_consistent_manifest_replacement(
    tmp_path: Path,
) -> None:
    write_test_bundle(tmp_path)
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None
    write_test_bundle(
        tmp_path,
        language_pairs=(("es", "it"),),
        translation_directions=(("es", "it"),),
        foundation_languages=("es", "it"),
    )

    with pytest.raises(BundleContractError, match="manifest changed after"):
        verify_embedded_bundle_payload(contract)


def test_runtime_payload_verification_allows_only_known_generated_namespaces(
    tmp_path: Path,
) -> None:
    write_test_bundle(tmp_path)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_bytes(b"runtime")
    (tmp_path / "src" / "sion_translate.egg-info").mkdir(parents=True)
    (tmp_path / "src" / "sion_translate.egg-info" / "PKG-INFO").write_bytes(b"generated")
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None

    verify_embedded_bundle_payload(contract)

    rogue = tmp_path / "data" / "rogue.jsonl"
    rogue.parent.mkdir(exist_ok=True)
    rogue.write_text('{"de":"eins","fr":"un"}\n', encoding="utf-8")
    with pytest.raises(BundleContractError, match="extra=.*rogue.jsonl"):
        verify_embedded_bundle_payload(contract)
    rogue.unlink()

    (tmp_path / "sitecustomize.pyc").write_bytes(b"executable-bytecode")
    with pytest.raises(BundleContractError, match="extra=.*sitecustomize.pyc"):
        verify_embedded_bundle_payload(contract)
    (tmp_path / "sitecustomize.pyc").unlink()

    forged_metadata = tmp_path / "sion_translate-999.dist-info" / "METADATA"
    forged_metadata.parent.mkdir()
    forged_metadata.write_text("Version: 999\n", encoding="utf-8")
    with pytest.raises(BundleContractError, match="extra=.*dist-info/METADATA"):
        verify_embedded_bundle_payload(contract)


def test_runtime_payload_verification_allows_generated_artifacts_only_when_omitted(
    tmp_path: Path,
) -> None:
    content = b'{"de":"eins","fr":"un"}\n'
    write_test_bundle(tmp_path, raw_files={"data/training.jsonl": content})
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None
    for relative_path in (
        "artifacts/tokenizer/sion.model",
        "artifacts/dataset/manifest.json",
        "artifacts/foundation_dataset/manifest.json",
    ):
        path = tmp_path.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    (tmp_path / "data" / ".sion_artifacts.lock").write_text("runtime\n", encoding="utf-8")

    verify_embedded_bundle_payload(contract)

    prepared_root = tmp_path / "prepared"
    write_test_bundle(prepared_root)
    strict_contract = load_embedded_training_contract(prepared_root)
    assert strict_contract is not None
    (prepared_root / "artifacts" / "tokenizer" / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(BundleContractError, match="extra=.*unexpected.bin"):
        verify_embedded_bundle_payload(strict_contract)


def test_runtime_payload_verification_rejects_a_link_like_mutable_artifact_root(
    tmp_path: Path,
) -> None:
    content = b'{"de":"eins","fr":"un"}\n'
    write_test_bundle(tmp_path, raw_files={"data/training.jsonl": content})
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None
    external = tmp_path / "external-tokenizer"
    external.mkdir()
    artifact_root = tmp_path / "artifacts" / "tokenizer"
    artifact_root.parent.mkdir()
    try:
        artifact_root.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("this host does not permit directory symlink creation")

    with pytest.raises(BundleContractError, match="unsafe paths:.*artifacts/tokenizer"):
        verify_embedded_bundle_payload(contract)


def test_runtime_payload_verification_rejects_a_nested_mutable_artifact_link(
    tmp_path: Path,
) -> None:
    content = b'{"de":"eins","fr":"un"}\n'
    write_test_bundle(tmp_path, raw_files={"data/training.jsonl": content})
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None
    external = tmp_path / "external-generation"
    external.mkdir()
    artifact_root = tmp_path / "artifacts" / "dataset"
    artifact_root.mkdir(parents=True)
    nested = artifact_root / "train"
    try:
        nested.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("this host does not permit directory symlink creation")

    with pytest.raises(BundleContractError, match="unsafe paths:.*artifacts/dataset/train"):
        verify_embedded_bundle_payload(contract)


def test_generated_tokenizer_provenance_matches_arbitrary_authenticated_sources(
    tmp_path: Path,
) -> None:
    parallel = b'{"de":"eins","fr":"un"}\n'
    second_parallel = b'{"de":"zwei","fr":"deux"}\n'
    monolingual = "Une phrase francaise suffisamment longue.\n".encode()
    monolingual_path = tmp_path / "data" / "corpus" / "fr" / "news.txt"
    write_test_bundle(
        tmp_path,
        # Uppercase sorts before lowercase by code point, but the tokenizer's
        # portable traversal sorts by case-folded identity on every host.
        raw_files={"data/Zeta.jsonl": second_parallel, "data/alpha.jsonl": parallel},
        monolingual_files={"data/corpus/fr/news.txt": monolingual},
        foundation_enabled=True,
    )
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None
    fingerprint = build_dataset_fingerprint(
        [tmp_path / "data" / "Zeta.jsonl", tmp_path / "data" / "alpha.jsonl"]
    )
    sources = [
        SimpleNamespace(
            path=monolingual_path,
            language="fr",
            size_bytes=monolingual_path.stat().st_size,
        )
    ]
    tokenizer_contract = {
        "schema": "sion-tokenizer-training-v4",
        "input_traversal_policy": "portable-input-order-v1",
        "language_pairs": [["de", "fr"]],
        "approximate_split": False,
        "sources": [
            {
                "role": "parallel",
                "path": "alpha.jsonl",
                "size": len(parallel),
                "sha256": hashlib.sha256(parallel).hexdigest(),
            },
            {
                "role": "parallel",
                "path": "Zeta.jsonl",
                "size": len(second_parallel),
                "sha256": hashlib.sha256(second_parallel).hexdigest(),
            },
            {
                "role": "monolingual",
                "path": "fr/news.txt",
                "size": len(monolingual),
                "sha256": hashlib.sha256(monolingual).hexdigest(),
                "language": "fr",
            },
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            tokenizer_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "training_contract": tokenizer_contract,
        "training_contract_sha256": digest,
    }
    expected_policy = {
        "language_pairs": [["de", "fr"]],
        "approximate_split": False,
    }

    validate_tokenizer_source_inventory(
        contract,
        metadata,
        fingerprint,
        tmp_path / "data" / "corpus",
        sources,
        expected_policy=expected_policy,
    )

    wrong_policy = copy.deepcopy(tokenizer_contract)
    wrong_policy["approximate_split"] = True
    metadata["training_contract"] = wrong_policy
    metadata["training_contract_sha256"] = hashlib.sha256(
        json.dumps(
            wrong_policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(BundleContractError, match="policy approximate_split differs"):
        validate_tokenizer_source_inventory(
            contract,
            metadata,
            fingerprint,
            tmp_path / "data" / "corpus",
            sources,
            expected_policy=expected_policy,
        )

    stale_contract = copy.deepcopy(tokenizer_contract)
    stale_contract["sources"][0]["sha256"] = "0" * 64
    metadata["training_contract"] = stale_contract
    metadata["training_contract_sha256"] = hashlib.sha256(
        json.dumps(
            stale_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(BundleContractError, match="source provenance differs"):
        validate_tokenizer_source_inventory(
            contract,
            metadata,
            fingerprint,
            tmp_path / "data" / "corpus",
            sources,
            expected_policy=expected_policy,
        )


def test_prepared_bundle_rejects_a_rogue_parallel_source(tmp_path: Path) -> None:
    write_test_bundle(tmp_path)
    rogue = tmp_path / "data" / "rogue.jsonl"
    rogue.parent.mkdir(exist_ok=True)
    rogue.write_text('{"de":"eins","fr":"un"}\n', encoding="utf-8")
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None
    fingerprint = build_dataset_fingerprint([rogue])

    with pytest.raises(BundleContractError, match="extra=.*rogue.jsonl"):
        validate_parallel_source_inventory(contract, fingerprint)


def test_prepared_bundle_rejects_a_rogue_monolingual_source(tmp_path: Path) -> None:
    write_test_bundle(tmp_path, foundation_enabled=True)
    rogue = tmp_path / "data" / "corpus" / "zh-Hant" / "news.txt"
    rogue.parent.mkdir(parents=True)
    rogue.write_text("測試句子\n", encoding="utf-8")
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None

    with pytest.raises(BundleContractError, match="extra=.*news.txt"):
        validate_monolingual_source_inventory(
            contract,
            [SimpleNamespace(path=rogue, size_bytes=rogue.stat().st_size)],
        )


def test_raw_bundle_requires_the_exact_parallel_identity(tmp_path: Path) -> None:
    source = tmp_path / "data" / "training.jsonl"
    content = b'{"de":"eins","fr":"un"}\n'
    write_test_bundle(tmp_path, raw_files={"data/training.jsonl": content})
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None

    validate_parallel_source_inventory(contract, build_dataset_fingerprint([source]))

    source.write_bytes(content + b"{}\n")
    with pytest.raises(BundleContractError, match="changed=.*training.jsonl"):
        validate_parallel_source_inventory(contract, build_dataset_fingerprint([source]))


def test_raw_bundle_rejects_a_deleted_authenticated_shard(tmp_path: Path) -> None:
    first = b'{"de":"eins","fr":"un"}\n'
    second = b'{"de":"zwei","fr":"deux"}\n'
    write_test_bundle(
        tmp_path,
        raw_files={"data/first.jsonl": first, "data/second.jsonl": second},
    )
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None

    with pytest.raises(BundleContractError, match="missing=.*second.jsonl"):
        validate_parallel_source_inventory(
            contract,
            build_dataset_fingerprint([tmp_path / "data" / "first.jsonl"]),
        )


def test_raw_bundle_requires_the_exact_monolingual_identity(tmp_path: Path) -> None:
    parallel = b'{"de":"eins","fr":"un"}\n'
    monolingual = "Ein ausreichend langer deutscher Beispielsatz.\n".encode()
    source = tmp_path / "data" / "corpus" / "de" / "wiki.txt"
    write_test_bundle(
        tmp_path,
        raw_files={"data/training.jsonl": parallel},
        monolingual_files={"data/corpus/de/wiki.txt": monolingual},
        foundation_enabled=True,
    )
    contract = load_embedded_training_contract(tmp_path)
    assert contract is not None

    validate_monolingual_source_inventory(
        contract,
        [SimpleNamespace(path=source, size_bytes=source.stat().st_size)],
    )

    source.write_bytes(monolingual + b"changed\n")
    with pytest.raises(BundleContractError, match="changed=.*wiki.txt"):
        validate_monolingual_source_inventory(
            contract,
            [SimpleNamespace(path=source, size_bytes=source.stat().st_size)],
        )
    with pytest.raises(BundleContractError, match="missing=.*wiki.txt"):
        validate_monolingual_source_inventory(contract, [])
