"""CLI 표준 입출력을 UTF-8 로 고정한다.

Windows 콘솔의 기본 인코딩은 로케일에 따라 cp949/cp932 이므로, 한국어·일본어를
그대로 출력하면 파이프나 리다이렉션 없이도 다음처럼 죽습니다.

    UnicodeEncodeError: 'cp949' codec can't encode character '\\u4f1a'

번역 결과가 한자·가나를 포함하는 것이 정상인 프로젝트에서는 이 실패가 기본
경로에 놓여 있으므로, CLI 진입점에서 스트림을 UTF-8 로 다시 엽니다. 표준 입력도
같이 처리합니다 — ``cat input.txt | kjx-translate`` 로 일본어를 읽을 때 같은
이유로 UnicodeDecodeError 가 납니다.
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
    """표준 입출력이 UTF-8 이 아니면 UTF-8 로 다시 엽니다.

    호출자가 ``PYTHONIOENCODING`` 을 지정했으면 그 의도를 존중해 아무것도 하지
    않습니다. 스트림이 ``reconfigure`` 를 지원하지 않는 경우(테스트 더블, 일부
    임베딩 환경)에도 조용히 넘어갑니다 — 인코딩 편의 기능이 CLI 를 막아서는
    안 되기 때문입니다.
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
            # 잘못된 바이트 하나 때문에 번역 작업 전체가 죽지 않도록 대체 문자를
            # 씁니다. 입력은 손실을 눈에 보이게(replace), 출력은 원문 복원이
            # 가능하게(backslashreplace) 처리합니다.
            if stream is sys.stdin:
                reconfigure(encoding="utf-8", errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            continue
