"""Checks around the Drain fit_on modes: stream prefix stability (S2-3) and stream-vs-train
comparison of processed outputs (S2-6)."""

from collections.abc import Mapping, Sequence

from cascadelad.data.parsing import DrainParser


def prefix_mismatches(contents: Sequence[str], cfg: Mapping, train_ratio: float) -> int:
    """Count lines in the first int(n * train_ratio) whose stream template id differs between
    parsing that prefix alone and parsing the full stream (must be 0: ids never change)."""
    n_prefix = int(len(contents) * train_ratio)
    alone = DrainParser(cfg).add(contents[:n_prefix])
    full = DrainParser(cfg).add(contents)[:n_prefix]
    return sum(a[0] != f[0] for a, f in zip(alone, full, strict=True))
