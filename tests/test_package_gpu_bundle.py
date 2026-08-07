from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import package_gpu_bundle


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "gaon12")
    _git(root, "config", "user.email", "gokirito12@gmail.com")

    (root / "src").mkdir()
    (root / "src" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (root / "README.md").write_text("training bundle\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / ".gitkeep").write_text("", encoding="utf-8")
    _git(root, "add", "README.md", "src/train.py", "data/.gitkeep")
    _git(root, "commit", "-qm", "initial source")

    (root / "data" / "corpus.jsonl").write_text(
        '{"ko":"안녕","ja":"こんにちは"}\n',
        encoding="utf-8",
    )
    evaluation = root / "data" / "evaluation_only"
    evaluation.mkdir()
    (evaluation / "holdout.jsonl").write_text('{"id":1}\n', encoding="utf-8")

    excluded = root / "data" / "excluded"
    excluded.mkdir()
    (excluded / "secret.jsonl").write_text('{"secret":true}\n', encoding="utf-8")
    for directory in ("artifacts", "runs", ".venv", "translation_queue"):
        generated = root / directory
        generated.mkdir()
        (generated / "do-not-package.txt").write_text("stale\n", encoding="utf-8")
    (root / "untracked.txt").write_text("not allowlisted\n", encoding="utf-8")
    return root


def _manifest(archive_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        return json.loads(archive.read("sion_translate/PACKAGE_MANIFEST.json").decode("utf-8"))


def test_build_is_deterministic_allowlisted_and_verifiable(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = package_gpu_bundle.build_bundle(root, first)
    second_result = package_gpu_bundle.build_bundle(root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.archive_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_result.archive_sha256 == second_result.archive_sha256

    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert names == {
            "sion_translate/README.md",
            "sion_translate/src/train.py",
            "sion_translate/data/.gitkeep",
            "sion_translate/data/corpus.jsonl",
            "sion_translate/data/evaluation_only/holdout.jsonl",
            "sion_translate/PACKAGE_MANIFEST.json",
            "sion_translate/SHA256SUMS",
        }
        assert all(name.startswith("sion_translate/") for name in names)
        checksums = archive.read("sion_translate/SHA256SUMS").decode("utf-8")
        assert "SHA256SUMS" not in checksums

    manifest = _manifest(first)
    assert manifest["git"] == {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }
    origins = {entry["path"]: entry["origin"] for entry in manifest["files"]}
    assert origins["README.md"] == "git-index"
    assert origins["data/corpus.jsonl"] == "data-jsonl"
    assert origins["data/evaluation_only/holdout.jsonl"] == "evaluation-only"

    archive_result = package_gpu_bundle.verify_archive(first)
    assert archive_result.file_count == 5

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first) as archive:
        archive.extractall(extracted)
    tree_result = package_gpu_bundle.verify_tree(extracted)
    assert tree_result == archive_result


def _with_monolingual_corpus(root: Path) -> None:
    corpus = root / "data" / "corpus"
    for language, text in (("ko", "단일어 문장\n"), ("ja", "単言語の文\n")):
        directory = corpus / language
        directory.mkdir(parents=True)
        (directory / "wiki.txt").write_text(text, encoding="utf-8")
        # Not a readable monolingual format; must not be shipped.
        (directory / "notes.md").write_text("stray download\n", encoding="utf-8")


def _with_tokenizer(root: Path, *, complete: bool = True) -> None:
    tokenizer = root / "artifacts" / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "sion.model").write_bytes(b"model")
    (tokenizer / "tokenizer_metadata.json").write_text('{"split_digits":true}\n', encoding="utf-8")
    if complete:
        (tokenizer / "sion.vocab").write_text("piece\t0\n", encoding="utf-8")


def test_the_monolingual_corpus_ships_only_when_asked_for(tmp_path: Path) -> None:
    """foundation 입력이라 없으면 그 단계가 통째로 건너뛰어진다.

    기본에 넣지 않는 이유는 분량입니다. 실제 코퍼스는 13.86 GiB 이고, 그것을
    모르고 배포물에 넣는 것은 실수로 치르기에 비싼 비용입니다.
    """

    root = _repository(tmp_path)
    _with_monolingual_corpus(root)

    default_archive = tmp_path / "default.zip"
    package_gpu_bundle.build_bundle(root, default_archive)
    with zipfile.ZipFile(default_archive) as archive:
        # `data/corpus.jsonl` 은 병렬 shard 라 들어가야 한다. 빠져야 하는 것은
        # `data/corpus/` **폴더** 이므로 구분자까지 보고 판정한다.
        assert not any("data/corpus/" in name for name in archive.namelist())

    included = tmp_path / "with-corpus.zip"
    package_gpu_bundle.build_bundle(root, included, include_monolingual_corpus=True)
    with zipfile.ZipFile(included) as archive:
        names = set(archive.namelist())
    assert "sion_translate/data/corpus/ko/wiki.txt" in names
    assert "sion_translate/data/corpus/ja/wiki.txt" in names
    # 읽을 수 없는 형식은 빼야 한다. 넣으면 기가바이트를 조용히 낭비한다.
    assert not any(name.endswith("notes.md") for name in names)

    origins = {entry["path"]: entry["origin"] for entry in _manifest(included)["files"]}
    assert origins["data/corpus/ko/wiki.txt"] == "monolingual-corpus"
    package_gpu_bundle.verify_archive(included)


def test_the_tokenizer_ships_only_when_asked_for_and_only_if_complete(tmp_path: Path) -> None:
    """반쪽 토크나이저를 실으면 실패가 업로드 뒤 GPU 서버로 미뤄진다."""

    root = _repository(tmp_path)
    _with_tokenizer(root, complete=False)

    with pytest.raises(package_gpu_bundle.BundleError, match="sion.vocab"):
        package_gpu_bundle.build_bundle(root, tmp_path / "partial.zip", include_tokenizer=True)

    _with_tokenizer(root, complete=True)
    included = tmp_path / "with-tokenizer.zip"
    package_gpu_bundle.build_bundle(root, included, include_tokenizer=True)
    with zipfile.ZipFile(included) as archive:
        names = set(archive.namelist())
    assert "sion_translate/artifacts/tokenizer/sion.model" in names
    assert "sion_translate/artifacts/tokenizer/tokenizer_metadata.json" in names

    origins = {entry["path"]: entry["origin"] for entry in _manifest(included)["files"]}
    assert origins["artifacts/tokenizer/sion.model"] == "tokenizer"
    package_gpu_bundle.verify_archive(included)


def test_the_default_bundle_still_refuses_stale_artifacts(tmp_path: Path) -> None:
    """opt-in 이 artifacts/ 전체를 여는 것이 아니어야 한다."""

    root = _repository(tmp_path)
    _with_tokenizer(root, complete=True)
    (root / "artifacts" / "dataset").mkdir(parents=True)
    (root / "artifacts" / "dataset" / "train.bin").write_bytes(b"stale")

    included = tmp_path / "tokenizer-only.zip"
    package_gpu_bundle.build_bundle(root, included, include_tokenizer=True)
    with zipfile.ZipFile(included) as archive:
        names = set(archive.namelist())

    assert "sion_translate/artifacts/tokenizer/sion.model" in names
    assert not any("artifacts/dataset" in name for name in names)
    assert not any("do-not-package" in name for name in names)


def test_requesting_an_absent_optional_tree_is_an_error(tmp_path: Path) -> None:
    """조용히 빠지면 서버에서야 foundation 이 없다는 것을 알게 된다."""

    root = _repository(tmp_path)
    with pytest.raises(package_gpu_bundle.BundleError, match="data/corpus"):
        package_gpu_bundle.build_bundle(
            root, tmp_path / "no-corpus.zip", include_monolingual_corpus=True
        )


def test_build_refuses_a_dirty_tracked_tree(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "README.md").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="tracked files are not clean"):
        package_gpu_bundle.build_bundle(root, tmp_path / "bundle.zip")


def test_build_requires_training_and_evaluation_corpora(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "data" / "corpus.jsonl").unlink()
    with pytest.raises(package_gpu_bundle.BundleError, match=r"data/\*\.jsonl"):
        package_gpu_bundle.build_bundle(root, tmp_path / "missing-training.zip")

    (root / "data" / "corpus.jsonl").write_text('{"ko":"가","ja":"あ"}\n', encoding="utf-8")
    (root / "data" / "evaluation_only" / "holdout.jsonl").unlink()
    with pytest.raises(package_gpu_bundle.BundleError, match="evaluation_only"):
        package_gpu_bundle.build_bundle(root, tmp_path / "missing-evaluation.zip")


def test_build_rejects_portable_metadata_name_collisions(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    collision = root / "package_manifest.json"
    collision.write_text("{}\n", encoding="utf-8")
    _git(root, "add", collision.name)
    _git(root, "commit", "-qm", "add colliding name")

    with pytest.raises(package_gpu_bundle.BundleError, match="portable path collision"):
        package_gpu_bundle.build_bundle(root, tmp_path / "bundle.zip")


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "bundle.zip"
    output.write_bytes(b"keep me")

    with pytest.raises(package_gpu_bundle.BundleError, match="--overwrite"):
        package_gpu_bundle.build_bundle(root, output)
    assert output.read_bytes() == b"keep me"

    result = package_gpu_bundle.build_bundle(root, output, overwrite=True)
    assert result.output_path == output
    package_gpu_bundle.verify_archive(output)


def test_tree_verification_detects_payload_tampering(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    package_gpu_bundle.build_bundle(root, archive_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    payload = extracted / "sion_translate" / "README.md"
    payload.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(package_gpu_bundle.BundleError, match="payload hash mismatch"):
        package_gpu_bundle.verify_tree(extracted)


def test_archive_verification_detects_payload_tampering(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    package_gpu_bundle.build_bundle(root, archive_path)
    tampered_path = tmp_path / "tampered.zip"

    with (
        zipfile.ZipFile(archive_path) as source,
        zipfile.ZipFile(
            tampered_path,
            mode="w",
        ) as destination,
    ):
        for info in source.infolist():
            content = source.read(info)
            if info.filename == "sion_translate/README.md":
                content = b"T" + content[1:]
            destination.writestr(info, content)

    with pytest.raises(package_gpu_bundle.BundleError, match="payload hash mismatch"):
        package_gpu_bundle.verify_archive(tampered_path)


def test_failed_verification_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    output = tmp_path / "bundle.zip"
    output.write_bytes(b"previous bundle")

    def fail_verification(_path: Path) -> package_gpu_bundle.VerificationResult:
        raise package_gpu_bundle.BundleError("injected verification failure")

    monkeypatch.setattr(package_gpu_bundle, "verify_archive", fail_verification)
    with pytest.raises(package_gpu_bundle.BundleError, match="injected"):
        package_gpu_bundle.build_bundle(root, output, overwrite=True)

    assert output.read_bytes() == b"previous bundle"
    assert not list(tmp_path.glob(".bundle.zip.*.tmp"))
