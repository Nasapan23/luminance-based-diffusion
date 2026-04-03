from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f"{dst.name}.tmp.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_hardlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f"{dst.name}.tmp.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        try:
            os.link(src, tmp_path)
        except OSError:
            shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def as_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()

