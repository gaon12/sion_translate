"""Make command-line standard streams reliably support multilingual UTF-8.

The default Windows console encoding can be a locale-specific code page such as
cp949 or cp932. Printing text outside that code page can fail even without a pipe
or redirection:

    UnicodeEncodeError: 'cp949' codec can't encode character '\\u4f1a'

Translation output may legitimately contain any Unicode script, so CLI entry
points reconfigure all standard streams as UTF-8. Standard input needs the same
protection when a pipeline such as ``cat input.txt | sion-translate`` supplies
text that the active Windows code page cannot decode.
"""

from __future__ import annotations

import os
import sys
from typing import IO, Any

_UTF8_ALIASES = {"utf8", "utf-8", "utf_8"}


def _is_utf8(stream: IO[Any]) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    return encoding.lower().replace("_", "-").replace("utf8", "utf-8") == "utf-8"


def configure_stdio() -> None:
    """Reconfigure standard streams as UTF-8 when the caller did not choose one.

    An explicit ``PYTHONIOENCODING`` takes precedence. Streams without a
    ``reconfigure`` method, including test doubles and some embedded runtimes,
    are left unchanged because an encoding convenience must not block the CLI.
    """
    if os.environ.get("PYTHONIOENCODING"):
        return

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is None or _is_utf8(stream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # One invalid byte must not terminate a whole translation job.
            # Input makes damage visible with replacement characters; output
            # uses backslash escapes so the original scalar remains recoverable.
            if stream is sys.stdin:
                reconfigure(encoding="utf-8", errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            continue
