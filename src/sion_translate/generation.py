"""Shared deployment decoding defaults.

Training-time sequence objectives, native inference, and command-line inference must
evaluate the same decoding policy. Keeping these values in one dependency-free module
prevents validation from drifting to an easier, reference-informed generation task.
"""

from __future__ import annotations


DEFAULT_NUM_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_MIN_NEW_TOKENS = 1
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_MAX_OUTPUT_LENGTH_RATIO = 3.0
DEFAULT_MAX_OUTPUT_LENGTH_MARGIN = 16
