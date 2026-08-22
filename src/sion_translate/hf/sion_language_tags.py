"""Development-time source for the bundled Hub language-tag runtime.

Transformers discovers relative remote-code dependencies next to the Python
module that imports them.  Export replaces this facade with the self-contained
implementation from :mod:`sion_translate.language_tags`.
"""

from sion_translate.language_tags import (
    LanguageTag,
    LanguageTagError,
    canonicalize_language_pair,
    canonicalize_language_tag,
    canonicalize_language_tags,
    is_well_formed_language_tag,
    parse_language_tag,
)


__all__ = [
    "LanguageTag",
    "LanguageTagError",
    "canonicalize_language_pair",
    "canonicalize_language_tag",
    "canonicalize_language_tags",
    "is_well_formed_language_tag",
    "parse_language_tag",
]
