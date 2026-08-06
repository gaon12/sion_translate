"""artifact 루트에 대한 프로세스 간 배타 락.

토크나이저와 데이터셋 생성은 "없으면 만든다" 규칙으로 돕니다. 두 작업이
동시에 시작하면 둘 다 "없다"고 판단하고 둘 다 만들기 시작하며, 각자 다른
세대의 산출물을 같은 경로에 씁니다. 결과는 실패가 아니라 **섞인 상태**입니다 —
한 세대의 토크나이저와 다른 세대의 데이터셋이 남고, 지문 검사는 그 조합을
처음 보는 것으로만 인식합니다.

락은 열린 파일 디스크립터에 겁니다. 파일의 **존재**로 잠그지 않는 이유는
그러면 크래시한 작업이 영영 풀리지 않는 락을 남기기 때문입니다. OS 수준
락은 프로세스가 어떻게 끝나든 커널이 놓아 줍니다.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_FILENAME = ".sion_artifacts.lock"

# 보유자 정보를 적는 영역과 겹치지 않도록, 락은 파일 내용 **바깥**의 고정
# 바이트에 겁니다. Windows 의 byte-range 락은 잠긴 구간에 대한 쓰기를 막기
# 때문에, 같은 바이트를 잠그고 거기에 쓰면 자기 자신이 막힙니다.
_LOCK_OFFSET = 1 << 30
# 보유자 한 줄을 고정 길이로 채워 truncate 없이 덮어씁니다.
_HOLDER_WIDTH = 128

if sys.platform == "win32":  # pragma: no cover - 플랫폼별 분기
    import msvcrt

    def _try_acquire(handle) -> bool:
        try:
            handle.seek(_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _release(handle) -> None:
        try:
            handle.seek(_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:  # pragma: no cover - 플랫폼별 분기
    import fcntl

    def _try_acquire(handle) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _release(handle) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def _describe_holder(path: Path) -> str:
    try:
        recorded = path.read_text(encoding="utf-8")[:_HOLDER_WIDTH].strip()
    except OSError:
        recorded = ""
    return recorded or "알 수 없는 프로세스"


@contextmanager
def artifact_lock(
    root: str | Path,
    *,
    timeout: float = 0.0,
    poll_interval: float = 1.0,
) -> Iterator[Path]:
    """``root`` 를 만들거나 바꾸는 동안 배타적으로 잠근다.

    ``timeout=0`` 이면 즉시 실패합니다. 기다리는 것이 기본이 아닌 이유는,
    다른 작업이 토크나이저를 만드는 중이라면 몇 시간이 걸릴 수 있고 그동안
    말없이 멈춰 있는 것보다 "누가 쥐고 있다"고 말하는 편이 낫기 때문입니다.

    락을 쥔 쪽의 host/pid 를 파일에 적어 두므로, 실패 메시지가 어느 프로세스를
    봐야 하는지 알려 줍니다.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_FILENAME
    deadline = time.monotonic() + max(0.0, timeout)
    handle = open(lock_path, "r+", encoding="utf-8") if lock_path.exists() else None
    if handle is None:
        lock_path.touch()
        handle = open(lock_path, "r+", encoding="utf-8")  # noqa: SIM115 - 컨텍스트가 소유
    try:
        while True:
            if _try_acquire(handle):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"artifact 루트가 다른 프로세스에 잠겨 있습니다: {root}\n"
                    f"  현재 보유자: {_describe_holder(lock_path)}\n"
                    "  같은 artifacts/ 를 쓰는 작업을 동시에 두 개 실행하면 서로 다른 "
                    "세대의 토크나이저와 데이터셋이 섞입니다. 앞선 작업이 끝나기를 "
                    "기다리거나, 이 작업에 별도의 artifact 경로를 주십시오."
                )
            time.sleep(poll_interval)
        handle.seek(0)
        handle.truncate()
        handle.write(f"host={socket.gethostname()} pid={os.getpid()} started={time.time():.0f}\n")
        handle.flush()
        yield lock_path
    finally:
        _release(handle)
        handle.close()


__all__ = ["LOCK_FILENAME", "artifact_lock"]
