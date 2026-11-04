"""Download a dataset archive, verify its md5, and extract it into data/raw/<dataset>/."""

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


def md5sum(path: Path, chunk_bytes: int) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def _safe_members(zf: zipfile.ZipFile, raw_dir: Path) -> list[str]:
    names = []
    for info in zf.infolist():
        p = PurePosixPath(info.filename)
        if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
            raise ValueError(f"unsafe path in archive: {info.filename!r}")
        if not info.is_dir():
            if (raw_dir / info.filename).exists():
                raise FileExistsError(f"refusing to overwrite {raw_dir / info.filename}")
            names.append(info.filename)
    return names


def fetch_files(cfg: Mapping) -> Path:
    """Download plain files listed in cfg["files"], each checked against its md5."""
    raw_dir = Path(cfg["raw_dir"])
    chunk = cfg["download"]["chunk_bytes"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for entry in cfg["files"]:
        target = raw_dir / entry["filename"]
        if target.exists():
            logger.info("already present: %s", target)
            continue
        part = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(entry["url"]) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out, chunk)
            got = md5sum(part, chunk)
            if got != entry["md5"]:
                raise ValueError(
                    f"md5 mismatch for {entry['filename']}: expected {entry['md5']}, got {got}"
                )
            part.rename(target)
        finally:
            part.unlink(missing_ok=True)
        logger.info("downloaded %s into %s", entry["filename"], raw_dir)
    return raw_dir


def fetch_dataset(cfg: Mapping) -> Path:
    if "files" in cfg:
        return fetch_files(cfg)
    raw_dir = Path(cfg["raw_dir"])
    if (raw_dir / cfg["log_file"]).exists() and (raw_dir / cfg["label_file"]).exists():
        logger.info("already present in %s, nothing to download", raw_dir)
        return raw_dir

    zen = cfg["zenodo"]
    chunk = cfg["download"]["chunk_bytes"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / zen["filename"]
    part = archive.with_name(archive.name + ".part")
    try:
        with urllib.request.urlopen(zen["url"]) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, chunk)
        got = md5sum(part, chunk)
        if got != zen["md5"]:
            raise ValueError(
                f"md5 mismatch for {zen['filename']}: expected {zen['md5']}, got {got}"
            )
        with zipfile.ZipFile(part) as zf:
            names = _safe_members(zf, raw_dir)
            zf.extractall(raw_dir, members=names)
    finally:
        part.unlink(missing_ok=True)
    logger.info("extracted %s into %s", zen["filename"], raw_dir)
    return raw_dir
