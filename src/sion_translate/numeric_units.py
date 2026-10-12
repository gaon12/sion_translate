"""Conservative restoration of literal measurement units in legacy inference."""

from __future__ import annotations

import re


# Keep unit context visible during decoding. Only an adjacent, recognized unit
# is eligible for replacement; arbitrary translated words are never consumed.
_UNITS = (
    "KiB MiB GiB TiB KB MB GB TB B kg mg µg μg g km cm mm m "
    "ms µs μs ns s h sec min hrs hr kW MW GW W kWh mAh Ah "
    "Hz kHz MHz GHz Pa kPa MPa ml mL cl cL dl dL L mol N mile miles"
).split()
_UNIT = "(?:" + "|".join(re.escape(x) for x in sorted(_UNITS, key=len, reverse=True)) + ")"
_POWER = r"(?:[²³]|\^[23])?"
_SOURCE_UNIT = re.compile(
    r"\s*(?P<unit>" + _UNIT + _POWER + r"(?:\s*/\s*" + _UNIT + _POWER + r")?)(?![A-Za-z0-9_])"
)
_ALIASES = (
    "キロメートル ミリメートル センチメートル メートル キログラム ミリグラム グラム "
    "ミリリットル リットル マイル キロ ミリ センチ "
    "킬로미터 밀리미터 센티미터 미터 킬로그램 밀리그램 그램 밀리리터 리터 마일 킬로 밀리 센티"
).split()
_OUTPUT_ATOM = (
    "(?:"
    + "|".join(re.escape(x) for x in sorted([*_UNITS, *_ALIASES], key=len, reverse=True))
    + ")"
    + _POWER
)
_OUTPUT_UNIT = re.compile(
    r"\s*(?:平方|立方)?"
    + _OUTPUT_ATOM
    + r"(?:\s*/\s*"
    + _OUTPUT_ATOM
    + r")?(?![A-Za-z0-9_])"
    # An observed legacy failure emits two incompatible literal units: km mL.
    + r"(?:\s+(?:"
    + _OUTPUT_ATOM
    + r")(?![A-Za-z0-9_]))*"
)
_SLOT = re.compile(r"<slot_\d+>")


def source_unit_at(text: str, value_end: int) -> str | None:
    match = _SOURCE_UNIT.match(text, value_end)
    return match.group("unit") if match else None


def legacy_unit_map(masked_source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _SLOT.finditer(masked_source):
        unit = source_unit_at(masked_source, match.end())
        if unit is not None:
            slot = match.group()
            if slot in result and result[slot] != unit:
                raise ValueError("A numeric slot has ambiguous source units")
            result[slot] = unit
    return result


def restore_legacy_units(text: str, units: dict[str, str]) -> str:
    """Restore each recognized suffix while its numeric slot is still present.

    Missing or displaced units are rejected: inserting one into an unknown
    position could hide a semantic translation error.
    """
    for slot, unit in units.items():
        pattern = re.compile(re.escape(slot))

        # Include the matched suffix in each replacement, preserving all prose.
        spans = []
        for match in pattern.finditer(text):
            suffix = _OUTPUT_UNIT.match(text, match.end())
            if suffix is None:
                raise ValueError(f"Cannot locate measurement unit {unit!r} after {slot}")
            replacement = slot + unit
            start = match.start()
            if unit.endswith(("²", "^2")) and text[:start].endswith("平方"):
                start -= 2
            elif unit.endswith(("³", "^3")) and text[:start].endswith("立方"):
                start -= 2
            spans.append((start, suffix.end(), replacement))
        for start, end, replacement in reversed(spans):
            text = text[:start] + replacement + text[end:]
    return text
