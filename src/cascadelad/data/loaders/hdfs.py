"""HDFS_v1 loader: one row per (log line, block id) pair."""

import logging
import re
from itertools import islice
from pathlib import Path

import pandas as pd

from cascadelad.data.loaders.base import STANDARD_COLUMNS, LogLoader, register_loader

logger = logging.getLogger(__name__)


@register_loader("hdfs")
class HDFSLoader(LogLoader):
    def _load(self) -> pd.DataFrame:
        cfg = self.cfg
        raw_dir = Path(cfg["raw_dir"])
        line_re = re.compile(cfg["line_regex"])
        block_re = re.compile(cfg["block_regex"])

        stamps: list[str] = []
        contents: list[str] = []
        blocks: list[str] = []
        no_block = 0
        with open(raw_dir / cfg["log_file"], encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(islice(f, cfg["max_lines"]), start=1):
                m = line_re.match(line.rstrip("\r\n"))
                if m is None:
                    raise ValueError(f"line {i}: does not match line_regex: {line[:100]!r}")
                ids = list(dict.fromkeys(block_re.findall(m["content"])))
                if not ids:
                    no_block += 1
                    continue
                for blk in ids:
                    stamps.append(f"{m['date']} {m['time']}")
                    contents.append(m["content"])
                    blocks.append(blk)
        if no_block:
            logger.info("dropped %d lines without a block id", no_block)

        labels = pd.read_csv(raw_dir / cfg["label_file"])
        label_of = dict(
            zip(
                labels[cfg["block_column"]],
                labels[cfg["label_column"]].map(cfg["label_map"]),
                strict=True,
            )
        )
        missing = sorted(set(blocks) - label_of.keys())
        if missing:
            raise ValueError(f"{len(missing)} blocks have no label, e.g. {missing[:3]}")

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(stamps, format=cfg["timestamp_format"]),
                "content": contents,
                "group_id": blocks,
                "label": [int(label_of[b]) for b in blocks],
            }
        )
        return df[list(STANDARD_COLUMNS)]
