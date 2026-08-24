"""BCP 47 language-tag validation and canonical casing.

The project uses language tags in configuration, JSONL records, directory names,
tokenizer control symbols, and exported model metadata.  Keeping the parser here
prevents those surfaces from accepting different language identities.

Syntax validation does not require IANA registry membership.  That is deliberate:
private or newly registered languages remain usable offline.  Identity
canonicalization does apply a pinned registry snapshot wherever IANA provides an
exact ``Preferred-Value``.  Without those mappings, aliases such as ``iw`` and
``he`` or ``en-BU`` and ``en-MM`` can name two tokenizer controls for the same
language and bypass pair-graph checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence, cast


_ALPHA = re.compile(r"^[A-Za-z]+$")
_ALNUM = re.compile(r"^[A-Za-z0-9]+$")
_DIGIT = re.compile(r"^[0-9]+$")

_GRANDFATHERED = (
    "art-lojban",
    "cel-gaulish",
    "en-GB-oed",
    "i-ami",
    "i-bnn",
    "i-default",
    "i-enochian",
    "i-hak",
    "i-klingon",
    "i-lux",
    "i-mingo",
    "i-navajo",
    "i-pwn",
    "i-tao",
    "i-tay",
    "i-tsu",
    "no-bok",
    "no-nyn",
    "sgn-BE-FR",
    "sgn-BE-NL",
    "sgn-CH-DE",
    "zh-guoyu",
    "zh-hakka",
    "zh-min",
    "zh-min-nan",
    "zh-xiang",
)
_GRANDFATHERED_BY_CASEFOLD = {tag.casefold(): tag for tag in _GRANDFATHERED}

# IANA Language Subtag Registry, File-Date 2026-08-08:
# https://www.iana.org/assignments/language-subtag-registry/
#
# This is intentionally *not* a bundled registry implementation.  The tables in
# this module contain only exact Preferred-Value relations needed to give graph
# nodes one stable identity.  Registry membership, Prefix validity (except for
# the extlang collapse defined by RFC 5646), Suppress-Script, and likely-subtag
# inference remain out of scope so offline/private tags stay usable.
_LANGUAGE_PREFERRED_VALUES: dict[str, str] = {
    "aam": "aas",
    "adp": "dz",
    "ajp": "apc",
    "ajt": "aeb",
    "asd": "snz",
    "aue": "ktz",
    "ayx": "nun",
    "bgm": "bcg",
    "bh": "bih",
    "bic": "bir",
    "bjd": "drl",
    "blg": "iba",
    "ccq": "rki",
    "cjr": "mom",
    "cka": "cmr",
    "cmk": "xch",
    "coy": "pij",
    "cqu": "quh",
    "dek": "sqm",
    "dit": "dif",
    "drh": "khk",
    "drr": "kzk",
    "drw": "prs",
    "gav": "dev",
    "gfx": "vaj",
    "ggn": "gvr",
    "gli": "kzk",
    "gti": "nyc",
    "guv": "duz",
    "hrr": "jal",
    "ibi": "opa",
    "ilw": "gal",
    "in": "id",
    "iw": "he",
    "jeg": "oyb",
    "ji": "yi",
    "jw": "jv",
    "kgc": "tdf",
    "kgh": "kml",
    "kgm": "plu",
    "koj": "kwv",
    "krm": "bmf",
    "ktr": "dtp",
    "kvs": "gdj",
    "kwq": "yam",
    "kxe": "tvd",
    "kxl": "kru",
    "kzj": "dtp",
    "kzt": "dtp",
    "lak": "ksp",
    "lii": "raq",
    "llo": "ngt",
    "lmm": "rmx",
    "meg": "cir",
    "mo": "ro",
    "mrd": "mgp",
    "mst": "mry",
    "mwj": "vaj",
    "myd": "aog",
    "myt": "mry",
    "nad": "xny",
    "ncp": "kdz",
    "nns": "nbr",
    "nnx": "ngv",
    "nom": "cbr",
    "nte": "eko",
    "nts": "pij",
    "nxu": "bpp",
    "oun": "vaj",
    "pat": "kxr",
    "pcr": "adx",
    "pmc": "huw",
    "pmk": "crr",
    "pmu": "phr",
    "ppa": "bfy",
    "ppr": "lcq",
    "prp": "gu",
    "pry": "prt",
    "puz": "pub",
    "sca": "hle",
    "shl": "mrh",
    "skk": "oyb",
    "smd": "kmb",
    "snb": "iba",
    "szd": "umi",
    "tdu": "dtp",
    "thc": "tpo",
    "thw": "ola",
    "thx": "oyb",
    "tie": "ras",
    "tkk": "twm",
    "tlw": "weo",
    "tmk": "tdg",
    "tmp": "tyj",
    "tne": "kak",
    "tnf": "prs",
    "tpw": "tpn",
    "tsf": "taj",
    "uok": "ema",
    "xba": "cax",
    "xia": "acn",
    "xkh": "waw",
    "xrq": "dmw",
    "xss": "zko",
    "ybd": "rki",
    "yma": "lrr",
    "ymt": "mtm",
    "yol": "enm",
    "yos": "zom",
    "yuu": "yug",
    "zir": "scv",
    "zkb": "kjh",
}

# The same registry snapshot's grandfathered records with Preferred-Value.
# Grandfathered tags are complete tags, so their replacements are applied only
# on an exact case-insensitive match; suffixes are never guessed or spliced in.
_GRANDFATHERED_PREFERRED_VALUES: dict[str, str] = {
    "art-lojban": "jbo",
    "en-GB-oed": "en-GB-oxendict",
    "i-ami": "ami",
    "i-bnn": "bnn",
    "i-hak": "hak",
    "i-klingon": "tlh",
    "i-lux": "lb",
    "i-navajo": "nv",
    "i-pwn": "pwn",
    "i-tao": "tao",
    "i-tay": "tay",
    "i-tsu": "tsu",
    "no-bok": "nb",
    "no-nyn": "nn",
    "sgn-BE-FR": "sfb",
    "sgn-BE-NL": "vgt",
    "sgn-CH-DE": "sgg",
    "zh-guoyu": "cmn",
    "zh-hakka": "hak",
    "zh-min-nan": "nan",
    "zh-xiang": "hsn",
}

_REDUNDANT_PREFERRED_VALUES: dict[str, str] = {
    "sgn-BR": "bzs",
    "sgn-CO": "csn",
    "sgn-DE": "gsg",
    "sgn-DK": "dsl",
    "sgn-ES": "ssp",
    "sgn-FR": "fsl",
    "sgn-GB": "bfi",
    "sgn-GR": "gss",
    "sgn-IE": "isg",
    "sgn-IT": "ise",
    "sgn-JP": "jsl",
    "sgn-MX": "mfs",
    "sgn-NI": "ncs",
    "sgn-NL": "dse",
    "sgn-NO": "nsl",
    "sgn-PT": "psr",
    "sgn-SE": "swl",
    "sgn-US": "ase",
    "sgn-ZA": "sfs",
    "zh-cmn": "cmn",
    "zh-cmn-Hans": "cmn-Hans",
    "zh-cmn-Hant": "cmn-Hant",
    "zh-gan": "gan",
    "zh-wuu": "wuu",
    "zh-yue": "yue",
}
_REDUNDANT_BY_CASEFOLD = {tag.casefold(): tag for tag in _REDUNDANT_PREFERRED_VALUES}

# All non-extlang redundant mappings in this registry snapshot have the shape
# ``sgn-REGION -> individual-sign-language``.  Applying that exact structural
# relation after syntax parsing preserves variants, extensions, and private use
# without accepting an invalid tag by blindly replacing a raw string prefix.
_REDUNDANT_LANGUAGE_REGION_PREFERRED_VALUES: dict[tuple[str, str], str] = {
    ("sgn", tag.split("-", 1)[1]): preferred
    for tag, preferred in _REDUNDANT_PREFERRED_VALUES.items()
    if tag.startswith("sgn-")
}

# Type=script currently has no Preferred-Value record in this snapshot.  Keep
# the table and resolver path explicit so a future registry update cannot be
# accidentally handled as casing-only canonicalization.
_SCRIPT_PREFERRED_VALUES: dict[str, str] = {}

_REGION_PREFERRED_VALUES: dict[str, str] = {
    "BU": "MM",
    "DD": "DE",
    "FX": "FR",
    "TP": "TL",
    "YD": "YE",
    "ZR": "CD",
}

_VARIANT_PREFERRED_VALUES: dict[str, str] = {
    "heploc": "alalc97",
}

# Every extlang Preferred-Value in the pinned registry equals its Subtag.  The
# material identity change is removal of its registered Prefix (for example,
# zh-cmn -> cmn).  Grouping the exact 258 records by Prefix keeps the static
# snapshot auditable without pretending to validate the rest of the registry.
_EXTLANG_SUBTAGS_BY_PREFIX: dict[str, str] = {
    "ar": "aao abh abv acm acq acw acx acy adf aeb aec afb ajp apc apd arb arq ars ary arz auz avl ayh ayl ayn ayp bbz pga shu ssh",
    "kok": "gom knn",
    "lv": "ltg lvs",
    "ms": "bjn btj bve bvu coa dup hji jak jax kvb kvr kxd lce lcf liw max meo mfa mfb min mqg msi mui orn ors pel pse tmw urk vkk vkt xmm zlm zmi zsm",
    "sgn": "ads aed aen afg ajs ase asf asp asq asw bfi bfk bog bqn bqy bvl bzs cds csc csd cse csf csg csl csn csq csr csx doq dse dsl dsz dyl ecs ehs esl esn eso eth fcs fse fsl fss gds gse gsg gsm gss gus hab haf hds hks hos hps hsh hsl icl iks ils inl ins ise isg isr jcs jhs jks jls jos jsl jus kgi kvk lbs lgs lls lsb lsc lsg lsl lsn lso lsp lst lsv lsw lsy lws mdl mfs mre msd msr mzc mzg mzy nbs ncs nsi nsl nsp nsr nzs okl pgz pks prl prz psc psd psg psl pso psp psr pys rib rms rnb rsi rsl rsm rsn sdl sfb sfs sgg sgx slf sls sqk sqs sqx ssp ssr svk swl syy szs tse tsm tsq tss tsy tza ugn ugy ukl uks vgt vsi vsl vsv wbs xki xml xms yds ygs yhs ysl ysm zhk zib zsl",
    "sw": "swc swh",
    "uz": "uzn uzs",
    "zh": "cdo cjy cmn cnp cpx csp czh czo gan hak hnm hsn luh lzh mnp nan sjc wuu yue",
}
_EXTLANG_PREFERRED_VALUES: dict[tuple[str, str], str] = {
    (prefix, subtag): subtag
    for prefix, subtags in _EXTLANG_SUBTAGS_BY_PREFIX.items()
    for subtag in subtags.split()
}


class LanguageTagError(ValueError):
    """A value is not a well-formed BCP 47 language tag."""


@dataclass(frozen=True, slots=True)
class LanguageTag:
    """Parsed language identity with RFC-recommended casing."""

    canonical: str
    language: str
    extlangs: tuple[str, ...] = ()
    script: str | None = None
    region: str | None = None
    variants: tuple[str, ...] = ()
    extensions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    private_use: tuple[str, ...] = ()
    grandfathered: bool = False


def _error(value: object, field: str, reason: str) -> LanguageTagError:
    return LanguageTagError(
        f"{field} must be a well-formed BCP 47 language tag; {reason}: {value!r}"
    )


def _is_alpha(value: str, length: int | range) -> bool:
    allowed = len(value) == length if isinstance(length, int) else len(value) in length
    return allowed and _ALPHA.fullmatch(value) is not None


def _is_alnum(value: str, length: range) -> bool:
    return len(value) in length and _ALNUM.fullmatch(value) is not None


def _resolve_preferred_value(
    value: str,
    preferred_values: Mapping[str, str],
    *,
    registry_type: str,
) -> str:
    """Resolve a static alias chain to a fixed point and reject table cycles."""

    current = value
    seen: set[str] = set()
    path: list[str] = []
    while current in preferred_values:
        if current in seen:
            cycle_start = path.index(current)
            chain = " -> ".join((*path[cycle_start:], current))
            raise RuntimeError(f"cyclic IANA {registry_type} Preferred-Value table: {chain}")
        seen.add(current)
        path.append(current)
        replacement = preferred_values[current]
        if replacement == current:
            return current
        current = replacement
    return current


def parse_language_tag(value: object, *, field: str = "language") -> LanguageTag:
    """Parse *value* as RFC 5646 syntax and return its canonical casing.

    Registry membership is intentionally not required.  For example, a new
    language subtag can be used before this package updates, and project-specific
    identities can use the standard ``x-...`` private-use form.
    """

    if not isinstance(value, str):
        raise _error(value, field, "expected a string")
    if not value or value != value.strip():
        raise _error(value, field, "empty or surrounding whitespace")
    if not value.isascii() or len(value) > 255:
        raise _error(value, field, "expected at most 255 ASCII characters")
    if "_" in value or "--" in value or value.startswith("-") or value.endswith("-"):
        raise _error(value, field, "subtags must be separated by single hyphens")

    grandfathered = _GRANDFATHERED_BY_CASEFOLD.get(value.casefold())
    if grandfathered is not None:
        preferred = _resolve_preferred_value(
            grandfathered,
            _GRANDFATHERED_PREFERRED_VALUES,
            registry_type="grandfathered",
        )
        if preferred != grandfathered:
            return parse_language_tag(preferred, field=field)
        primary = grandfathered.split("-", 1)[0].lower()
        return LanguageTag(
            canonical=grandfathered,
            language=primary,
            grandfathered=True,
        )

    redundant = _REDUNDANT_BY_CASEFOLD.get(value.casefold())
    if redundant is not None:
        preferred = _resolve_preferred_value(
            redundant,
            _REDUNDANT_PREFERRED_VALUES,
            registry_type="redundant",
        )
        if preferred != redundant:
            return parse_language_tag(preferred, field=field)

    raw = value.split("-")
    if any(not part or len(part) > 8 or _ALNUM.fullmatch(part) is None for part in raw):
        raise _error(value, field, "each subtag must contain 1-8 ASCII letters or digits")

    if raw[0].casefold() == "x":
        if len(raw) == 1:
            raise _error(value, field, "private-use prefix 'x' requires at least one subtag")
        private_use = tuple(part.lower() for part in raw[1:])
        canonical = "x-" + "-".join(private_use)
        return LanguageTag(canonical, "x", private_use=private_use)

    primary = raw[0]
    if not (
        _is_alpha(primary, range(2, 4)) or _is_alpha(primary, 4) or _is_alpha(primary, range(5, 9))
    ):
        raise _error(value, field, "primary language must contain 2-8 ASCII letters")
    language = _resolve_preferred_value(
        primary.lower(),
        _LANGUAGE_PREFERRED_VALUES,
        registry_type="language",
    )
    index = 1

    extlangs: list[str] = []
    if len(primary) in (2, 3):
        while index < len(raw) and len(extlangs) < 3 and _is_alpha(raw[index], 3):
            extlangs.append(raw[index].lower())
            index += 1

    script: str | None = None
    if index < len(raw) and _is_alpha(raw[index], 4):
        script = _resolve_preferred_value(
            raw[index].title(),
            _SCRIPT_PREFERRED_VALUES,
            registry_type="script",
        )
        index += 1

    region: str | None = None
    if index < len(raw):
        candidate = raw[index]
        if _is_alpha(candidate, 2):
            region = _resolve_preferred_value(
                candidate.upper(),
                _REGION_PREFERRED_VALUES,
                registry_type="region",
            )
            index += 1
        elif len(candidate) == 3 and _DIGIT.fullmatch(candidate) is not None:
            region = candidate
            index += 1

    variants: list[str] = []
    seen_variants: set[str] = set()
    while index < len(raw):
        candidate = raw[index]
        is_variant = _is_alnum(candidate, range(5, 9)) or (
            len(candidate) == 4
            and candidate[0].isdigit()
            and _ALNUM.fullmatch(candidate) is not None
        )
        if not is_variant:
            break
        normalized = _resolve_preferred_value(
            candidate.lower(),
            _VARIANT_PREFERRED_VALUES,
            registry_type="variant",
        )
        if normalized in seen_variants:
            raise _error(value, field, f"duplicate variant subtag {candidate!r}")
        seen_variants.add(normalized)
        variants.append(normalized)
        index += 1

    extensions: list[tuple[str, tuple[str, ...]]] = []
    seen_singletons: set[str] = set()
    while index < len(raw) and len(raw[index]) == 1 and raw[index].casefold() != "x":
        singleton = raw[index].lower()
        if not singleton.isalnum():
            raise _error(value, field, f"invalid extension singleton {raw[index]!r}")
        if singleton in seen_singletons:
            raise _error(value, field, f"duplicate extension singleton {singleton!r}")
        seen_singletons.add(singleton)
        index += 1
        extension_subtags: list[str] = []
        while index < len(raw) and _is_alnum(raw[index], range(2, 9)):
            extension_subtags.append(raw[index].lower())
            index += 1
        if not extension_subtags:
            raise _error(value, field, f"extension {singleton!r} requires a 2-8 character subtag")
        extensions.append((singleton, tuple(extension_subtags)))

    private_use: tuple[str, ...] = ()
    if index < len(raw) and raw[index].casefold() == "x":
        if index + 1 >= len(raw):
            raise _error(value, field, "private-use prefix 'x' requires at least one subtag")
        private_use = tuple(part.lower() for part in raw[index + 1 :])
        index = len(raw)

    if index != len(raw):
        raise _error(value, field, f"unexpected subtag {raw[index]!r}")

    # RFC 5646 canonicalization replaces a registered macrolanguage Prefix plus
    # its first extlang with the extlang's Preferred-Value.  Do not collapse an
    # extlang under an unregistered prefix; the syntax parser intentionally
    # permits such offline/private identities.
    if extlangs:
        extlang_preferred = _EXTLANG_PREFERRED_VALUES.get((language, extlangs[0]))
        if extlang_preferred is not None:
            language = _resolve_preferred_value(
                extlang_preferred,
                _LANGUAGE_PREFERRED_VALUES,
                registry_type="language",
            )
            del extlangs[0]

    if region is not None:
        redundant_preferred = _REDUNDANT_LANGUAGE_REGION_PREFERRED_VALUES.get((language, region))
        if redundant_preferred is not None:
            language = _resolve_preferred_value(
                redundant_preferred,
                _LANGUAGE_PREFERRED_VALUES,
                registry_type="language",
            )
            region = None

    # RFC 5646 canonical form orders extension sequences by singleton.  Without
    # this step, tags that differ only in extension order become two model
    # identities even though they denote the same language tag.
    extensions.sort(key=lambda item: item[0])
    canonical_parts = [language, *extlangs]
    if script is not None:
        canonical_parts.append(script)
    if region is not None:
        canonical_parts.append(region)
    canonical_parts.extend(variants)
    for singleton, extension_values in extensions:
        canonical_parts.extend((singleton, *extension_values))
    if private_use:
        canonical_parts.extend(("x", *private_use))
    canonical = "-".join(canonical_parts)
    return LanguageTag(
        canonical=canonical,
        language=language,
        extlangs=tuple(extlangs),
        script=script,
        region=region,
        variants=tuple(variants),
        extensions=tuple(extensions),
        private_use=private_use,
    )


def canonicalize_language_tag(value: object, *, field: str = "language") -> str:
    """Return one stable spelling for a well-formed language tag."""

    return parse_language_tag(value, field=field).canonical


def canonicalize_language_tags(
    values: Sequence[object],
    *,
    field: str = "languages",
    reject_duplicates: bool = True,
) -> tuple[str, ...]:
    """Canonicalize an ordered language list and handle canonical duplicates."""

    if isinstance(values, (str, bytes)):
        raise LanguageTagError(f"{field} must be a sequence of BCP 47 language tags")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        language = canonicalize_language_tag(value, field=f"{field}[{index}]")
        if language in seen:
            if reject_duplicates:
                raise LanguageTagError(
                    f"{field} must not contain duplicate language identities after "
                    f"canonicalization; duplicate={language!r}"
                )
            continue
        seen.add(language)
        normalized.append(language)
    return tuple(normalized)


def canonicalize_language_pair(
    value: object,
    *,
    field: str = "language pair",
) -> tuple[str, str]:
    """Canonicalize one ordered pair and require two distinct identities."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LanguageTagError(f"{field} must be a two-item language sequence of BCP 47 tags")
    items = cast(Sequence[object], value)
    if len(items) != 2:
        raise LanguageTagError(f"{field} must be a two-item language sequence of BCP 47 tags")
    languages = canonicalize_language_tags(
        items,
        field=field,
        reject_duplicates=True,
    )
    return languages[0], languages[1]


def is_well_formed_language_tag(value: object) -> bool:
    """Whether *value* has well-formed offline BCP 47 syntax."""

    try:
        parse_language_tag(value)
    except LanguageTagError:
        return False
    return True


__all__ = [
    "LanguageTag",
    "LanguageTagError",
    "canonicalize_language_pair",
    "canonicalize_language_tag",
    "canonicalize_language_tags",
    "is_well_formed_language_tag",
    "parse_language_tag",
]
