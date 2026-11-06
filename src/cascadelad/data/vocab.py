"""Template vocabulary, built from train sequences only (research invariant 2)."""

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


class Vocab:
    """Token <-> id map. id 0 = pad, id 1 = unk, then train tokens by (-count, token)."""

    def __init__(self, token_to_id: Mapping[str, int], pad_token: str, unk_token: str) -> None:
        self.token_to_id = dict(token_to_id)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = self.token_to_id[pad_token]
        self.unk_id = self.token_to_id[unk_token]

    @classmethod
    def build(cls, sequences: Iterable[Sequence[str]], cfg: Mapping) -> "Vocab":
        pad, unk = cfg["pad_token"], cfg["unk_token"]
        counts = Counter(tok for seq in sequences for tok in seq)
        kept = sorted(
            (t for t, c in counts.items() if c >= cfg["min_count"] and t not in (pad, unk)),
            key=lambda t: (-counts[t], t),
        )
        token_to_id = {pad: 0, unk: 1}
        token_to_id.update({t: i for i, t in enumerate(kept, start=2)})
        return cls(token_to_id, pad, unk)

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def unk_rate(self, sequences: Iterable[Sequence[str]]) -> float:
        """Share of events that encode to unk."""
        total = unk = 0
        for seq in sequences:
            ids = self.encode(seq)
            total += len(ids)
            unk += ids.count(self.unk_id)
        if not total:
            raise ValueError("empty input: no events to compute the unk rate on")
        return unk / total

    def to_json(self, path: Path) -> None:
        payload = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["token_to_id"], data["pad_token"], data["unk_token"])
