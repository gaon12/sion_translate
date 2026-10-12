from __future__ import annotations

import pytest

from sion_translate.tokenizer import SLOT_SYMBOLS
from sion_translate.glossary import restore_targets
from sion_translate.numeric_units import legacy_unit_map, restore_legacy_units
from sion_translate.structured import mask_structured_spans


def mask(text):
    return mask_structured_spans(text, slot_symbols=SLOT_SYMBOLS, keep_number_units_visible=True)


@pytest.mark.parametrize(
    "source,values",
    [
        ("± 0.05mm", {"± 0.05"}),
        ("15mile와 40mile", {"15", "40"}),
        ("3.7e-5 및 3.7N", {"3.7e-5", "3.7"}),
        (".5m 및 − 2kg", {".5", "− 2"}),
        ("+82-10-1234-5678", {"+82-10-1234-5678"}),
        ("10-20mg", {"10-20"}),
        ("AI USAF v1.2 model_2.pt", {"v1.2", "model_2.pt"}),
    ],
)
def test_legacy_values_are_independent_of_known_suffixes(source, values):
    masked, surfaces = mask(source)
    assert set(surfaces.values()) == values
    assert restore_targets(masked, surfaces) == (source, [])


def test_same_value_with_distinct_units_uses_distinct_slots():
    masked, surfaces = mask("250mL와 250mg")
    assert len(surfaces) == 2
    assert legacy_unit_map(masked) == {"<slot_0>": "mL", "<slot_1>": "mg"}


@pytest.mark.parametrize(
    "decoded,unit,expected",
    [
        ("容量は<slot_0> mmです。", "mL", "容量は<slot_0>mLです。"),
        ("면적은 <slot_0> km mL입니다.", "km²", "면적은 <slot_0>km²입니다."),
        ("平方<slot_0>キロメートル", "km²", "<slot_0>km²"),
        ("<slot_0>ミリ", "mm", "<slot_0>mm"),
        ("<slot_0>mg/L 이하", "mg/L", "<slot_0>mg/L 이하"),
        ("<slot_0>キロ", "kg", "<slot_0>kg"),
    ],
)
def test_units_restore_only_adjacent_recognized_suffixes(decoded, unit, expected):
    assert restore_legacy_units(decoded, {"<slot_0>": unit}) == expected


@pytest.mark.parametrize("decoded", ["<slot_0>です。", "mg <slot_0>", "<slot_0>money"])
def test_missing_or_displaced_unit_is_rejected(decoded):
    with pytest.raises(ValueError, match="Cannot locate measurement unit"):
        restore_legacy_units(decoded, {"<slot_0>": "mg"})


def test_acronyms_stay_visible_and_opaque_structures_stay_protected():
    masked, surfaces = mask("AI https://host/15mile {count_30} 17時30分")
    assert masked.startswith("AI ")
    assert "https://" not in masked
    assert set(surfaces.values()) == {"https://host/15mile", "{count_30}", "17", "30"}
