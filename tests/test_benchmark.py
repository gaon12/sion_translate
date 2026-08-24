"""Tests for FLORES benchmark conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from sion_translate.benchmark import (
    flores_code,
    find_flores_file,
    pairs_from_hf_datasets,
    pairs_from_local_flores,
    write_jsonl,
)
from sion_translate.cli.prepare_benchmark import (
    parse_flores_code_overrides,
    resolve_benchmark_language_pair,
    resolve_flores_codes,
)
from sion_translate.evaluation import load_benchmark_pairs


def test_flores_code_lookup_and_override() -> None:
    assert flores_code("ko") == "kor_Hang"
    assert flores_code("ja") == "jpn_Jpan"
    assert flores_code("en") == "eng_Latn"
    # An explicit override takes precedence over the built-in map.
    assert flores_code("xx", "xxx_Yyyy") == "xxx_Yyyy"
    # An unknown language without an override reports an actionable error.
    with pytest.raises(ValueError, match="No default FLORES code"):
        flores_code("xx")


def _write_flores_split(root: Path, split: str) -> None:
    """Create language files that mimic a small FLORES distribution."""
    (root / split).mkdir(parents=True, exist_ok=True)
    ko = [f"한국어 문장 {i}" for i in range(5)]
    ja = [f"日本語の文 {i}" for i in range(5)]
    (root / split / f"kor_Hang.{split}").write_text("\n".join(ko) + "\n", encoding="utf-8")
    (root / split / f"jpn_Jpan.{split}").write_text("\n".join(ja) + "\n", encoding="utf-8")


def test_find_flores_file_locations(tmp_path: Path) -> None:
    _write_flores_split(tmp_path, "devtest")
    found = find_flores_file(tmp_path, "kor_Hang", "devtest")
    assert found.name == "kor_Hang.devtest"
    with pytest.raises(FileNotFoundError, match="Could not find FLORES file"):
        find_flores_file(tmp_path, "deu_Latn", "devtest")


@pytest.mark.parametrize(
    ("code", "split"),
    [
        ("../escaped", "devtest"),
        (r"..\escaped", "devtest"),
        ("kor_Hang", "../devtest"),
        ("kor_Hang", r"..\devtest"),
        ("kor_Hang", "."),
        ("CON", "devtest"),
        ("kor_Hang", "lPt9.test"),
    ],
)
def test_find_flores_file_rejects_traversal_and_reserved_components(
    tmp_path: Path,
    code: str,
    split: str,
) -> None:
    with pytest.raises(ValueError):
        find_flores_file(tmp_path, code, split)


def test_find_flores_file_never_reads_an_existing_parent_file(tmp_path: Path) -> None:
    root = tmp_path / "flores"
    root.mkdir()
    (tmp_path / "escaped.devtest").write_text("secret\n", encoding="utf-8")

    with pytest.raises(ValueError):
        find_flores_file(root, "../escaped", "devtest")


def test_find_flores_file_rejects_a_candidate_symlink_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "flores"
    root.mkdir()
    outside = tmp_path / "outside.devtest"
    outside.write_text("secret\n", encoding="utf-8")
    candidate = root / "kor_Hang.devtest"
    try:
        candidate.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="outside the configured root"):
        find_flores_file(root, "kor_Hang", "devtest")


def test_pairs_from_local_flores_end_to_end(tmp_path: Path) -> None:
    _write_flores_split(tmp_path, "devtest")
    pairs = pairs_from_local_flores(tmp_path, ("ko", "ja"), split="devtest")
    assert len(pairs) == 5
    assert pairs[0] == {"ko": "한국어 문장 0", "ja": "日本語の文 0"}

    # The evaluation loader must consume the converted JSONL without adaptation.
    output = tmp_path / "flores.jsonl"
    count = write_jsonl(pairs, output)
    assert count == 5
    loaded = load_benchmark_pairs([output], ("ko", "ja"), max_samples_per_direction=10)
    assert len(loaded[("ko", "ja")]) == 5
    assert loaded[("ja", "ko")][0] == ("日本語の文 0", "한국어 문장 0")


def test_line_count_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "kor_Hang.dev").write_text("한 줄\n두 줄\n", encoding="utf-8")
    (tmp_path / "dev" / "jpn_Jpan.dev").write_text("一行\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different sentence counts"):
        pairs_from_local_flores(tmp_path, ("ko", "ja"), split="dev")


def test_custom_language_pair_with_override(tmp_path: Path) -> None:
    (tmp_path / "devtest").mkdir()
    (tmp_path / "devtest" / "eng_Latn.devtest").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "devtest" / "deu_Latn.devtest").write_text("hallo\nWelt\n", encoding="utf-8")
    pairs = pairs_from_local_flores(
        tmp_path,
        ("en", "de"),
        split="devtest",
        code_overrides={"en": "eng_Latn", "de": "deu_Latn"},
    )
    assert pairs == [{"en": "hello", "de": "hallo"}, {"en": "world", "de": "Welt"}]


@pytest.mark.parametrize(
    ("split", "override"),
    [
        ("devtest", "../outside"),
        ("devtest", r"..\outside"),
        ("../devtest", "eng_Latn"),
        (r"..\devtest", "eng_Latn"),
        ("CON", "eng_Latn"),
        ("devtest", "aUx.txt"),
    ],
)
def test_local_flores_public_loader_rejects_traversal_and_reserved_components(
    tmp_path: Path,
    split: str,
    override: str,
) -> None:
    with pytest.raises(ValueError):
        pairs_from_local_flores(
            tmp_path,
            ("en", "de"),
            split=split,
            code_overrides={"en": override, "de": "deu_Latn"},
        )


def test_local_flores_public_loader_never_reads_existing_parent_files(tmp_path: Path) -> None:
    root = tmp_path / "flores"
    root.mkdir()
    (tmp_path / "escaped_en.devtest").write_text("secret en\n", encoding="utf-8")
    (tmp_path / "escaped_de.devtest").write_text("secret de\n", encoding="utf-8")

    with pytest.raises(ValueError):
        pairs_from_local_flores(
            root,
            ("en", "de"),
            split="devtest",
            code_overrides={"en": "../escaped_en", "de": "../escaped_de"},
        )


@pytest.mark.parametrize(
    ("split", "override"),
    [
        ("../devtest", "eng_Latn"),
        ("devtest", "NUL.txt"),
    ],
)
def test_hf_flores_public_loader_validates_before_optional_import_or_network(
    split: str,
    override: str,
) -> None:
    with pytest.raises(ValueError):
        pairs_from_hf_datasets(
            ("en", "de"),
            split=split,
            code_overrides={"en": override, "de": "deu_Latn"},
        )


def test_benchmark_cli_canonicalizes_requested_bcp47_pair() -> None:
    pair = resolve_benchmark_language_pair(
        ["ZH-hant", "X-ACME"],
        (("zh-Hant", "x-acme"),),
    )

    assert pair == ("zh-Hant", "x-acme")


def test_benchmark_cli_rejects_canonical_config_pair_collisions() -> None:
    with pytest.raises(SystemExit, match="duplicate"):
        resolve_benchmark_language_pair(
            None,
            (("zh-hant", "X-ACME"), ("x-acme", "zh-Hant")),
        )


def test_flores_override_keys_are_canonical_and_collision_safe() -> None:
    assert parse_flores_code_overrides(["ZH-hant=zho_Hant", "X-ACME=abc_Latn"]) == {
        "zh-Hant": "zho_Hant",
        "x-acme": "abc_Latn",
    }

    with pytest.raises(SystemExit, match="duplicate"):
        parse_flores_code_overrides(
            ["zh-hant=zho_Hant", "zh-Hant=zho_Hans"],
        )


def test_flores_codes_reject_same_identity_unsafe_names_and_unused_overrides() -> None:
    pair = ("zh-Hant", "x-acme")

    assert resolve_flores_codes(
        pair,
        {"zh-Hant": "zho_Hant", "x-acme": "abc_Latn"},
    ) == {"zh-Hant": "zho_Hant", "x-acme": "abc_Latn"}
    with pytest.raises(SystemExit, match="same FLORES code"):
        resolve_flores_codes(
            pair,
            {"zh-Hant": "zho_Hant", "x-acme": "zho_Hant"},
        )
    with pytest.raises(SystemExit, match="case-insensitive"):
        resolve_flores_codes(
            pair,
            {"zh-Hant": "zho_Hant", "x-acme": "ZHO_hANT"},
        )
    with pytest.raises(SystemExit, match="safe path component"):
        resolve_flores_codes(
            pair,
            {"zh-Hant": "../zho_Hant", "x-acme": "abc_Latn"},
        )
    with pytest.raises(SystemExit, match="outside the selected"):
        resolve_flores_codes(
            pair,
            {
                "zh-Hant": "zho_Hant",
                "x-acme": "abc_Latn",
                "de": "deu_Latn",
            },
        )


@pytest.mark.parametrize(
    "reserved_code",
    [
        "CON",
        "prn.devtest",
        "AuX.txt",
        "nul",
        *(f"cOm{index}.devtest" for index in range(1, 10)),
        *(f"LpT{index}" for index in range(1, 10)),
    ],
)
def test_flores_code_overrides_reject_windows_reserved_device_basenames(
    reserved_code: str,
) -> None:
    with pytest.raises(SystemExit, match="reserved Windows device name"):
        parse_flores_code_overrides([f"x-acme={reserved_code}"])
    with pytest.raises(SystemExit, match="reserved Windows device name"):
        resolve_flores_codes(
            ("zh-Hant", "x-acme"),
            {"zh-Hant": "zho_Hant", "x-acme": reserved_code},
        )


def test_flores_code_override_allows_non_reserved_device_prefixes() -> None:
    assert (
        resolve_flores_codes(
            ("zh-Hant", "x-acme"),
            {"zh-Hant": "zho_Hant", "x-acme": "COM10"},
        )["x-acme"]
        == "COM10"
    )


def test_local_flores_rejects_two_languages_bound_to_the_same_file(tmp_path: Path) -> None:
    split_dir = tmp_path / "devtest"
    split_dir.mkdir()
    (split_dir / "shared.devtest").write_text("one\ntwo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same physical file"):
        pairs_from_local_flores(
            tmp_path,
            ("zh-Hant", "x-acme"),
            split="devtest",
            code_overrides={"zh-Hant": "shared", "x-acme": "shared"},
        )
