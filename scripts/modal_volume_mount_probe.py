"""Inspect one read-only Modal Volume mount without reading bundle contents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import modal


APP_NAME = "sion-volume-mount-probe"
VOLUME_NAME = "sion-prepared-bundles"
MOUNT_PATH = Path("/mnt/sion-bundles")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=1)


def _metadata(path: Path) -> dict[str, object]:
    link_metadata = path.lstat()
    followed_metadata = path.stat()
    return {
        "path": str(path),
        "lexists": os.path.lexists(path),
        "is_symlink": path.is_symlink(),
        "readlink": os.readlink(path) if path.is_symlink() else None,
        "realpath": os.path.realpath(path),
        "lstat_mode_octal": oct(link_metadata.st_mode),
        "lstat_is_directory": stat.S_ISDIR(link_metadata.st_mode),
        "lstat_is_symlink": stat.S_ISLNK(link_metadata.st_mode),
        "stat_mode_octal": oct(followed_metadata.st_mode),
        "stat_is_directory": stat.S_ISDIR(followed_metadata.st_mode),
    }


@app.function(
    volumes={str(MOUNT_PATH): volume.with_mount_options(read_only=True)},
    cpu=0.125,
    memory=128,
    timeout=60,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
)
def probe() -> dict[str, object]:
    """Return link-aware metadata for the mount and its resolved target."""

    result = _metadata(MOUNT_PATH)
    resolved = Path(os.path.realpath(MOUNT_PATH))
    result["resolved_metadata"] = _metadata(resolved)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(probe.remote(), indent=2, sort_keys=True))
