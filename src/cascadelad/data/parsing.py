"""Drain log parsing: contents -> template ids.

fit() learns templates; transform() only matches and never learns, so lines whose
template was not seen in fit() map to unk_token (research invariant 2).
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig


class DrainParser:
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        tm_cfg = TemplateMinerConfig()
        tm_cfg.drain_depth = cfg["depth"]
        tm_cfg.drain_sim_th = cfg["sim_th"]
        tm_cfg.drain_max_children = cfg["max_children"]
        tm_cfg.drain_max_clusters = cfg["max_clusters"]
        tm_cfg.drain_extra_delimiters = list(cfg["extra_delimiters"])
        tm_cfg.parametrize_numeric_tokens = cfg["parametrize_numeric_tokens"]
        tm_cfg.masking_instructions = [
            MaskingInstruction(m["regex"], m["mask_with"]) for m in cfg["masking"]
        ]
        self._miner = TemplateMiner(config=tm_cfg)
        self._fitted = False

    def fit(self, contents: Iterable[str]) -> "DrainParser":
        for content in dict.fromkeys(contents):
            self._miner.add_log_message(content)
        self._fitted = True
        return self

    def transform(self, contents: Iterable[str]) -> list[str]:
        if not self._fitted:
            raise RuntimeError("transform() called before fit()")
        out = []
        for content in contents:
            cluster = self._miner.match(content, full_search_strategy=self.cfg["match_strategy"])
            out.append(self.cfg["unk_token"] if cluster is None else f"E{cluster.cluster_id}")
        return out

    def templates(self) -> dict[str, str]:
        return {f"E{c.cluster_id}": c.get_template() for c in self._miner.drain.clusters}


def grouping_accuracy(pred: Sequence[str], truth: Sequence[str]) -> float:
    """Loghub grouping accuracy: a line is correct iff the set of lines sharing its
    predicted id equals the set of lines sharing its true id."""
    if len(pred) != len(truth):
        raise ValueError(f"length mismatch: {len(pred)} predictions vs {len(truth)} labels")
    if not len(pred):
        raise ValueError("empty input")
    pred_groups: dict[str, list[int]] = defaultdict(list)
    true_groups: dict[str, list[int]] = defaultdict(list)
    for i, (p, t) in enumerate(zip(pred, truth, strict=True)):
        pred_groups[p].append(i)
        true_groups[t].append(i)
    correct = sum(len(idx) for idx in pred_groups.values() if idx == true_groups[truth[idx[0]]])
    return correct / len(pred)
