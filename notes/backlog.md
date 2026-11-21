# Backlog

Work that is decided or proposed but not built (D-4a1e). Tick a box when it is done and name the
commit; do not add a decision for that. A new rule found while doing an item goes to decisions.md.
Source IDs point to decisions-archive.md unless they are still in decisions.md.

## Before Stage 5
- [x] Merge `fix/stage-1-3-protocol` into `main`. (e8837b7)
- [x] `load_split(dir, name)` in `src/cascadelad/data/`: reads a processed parquet and turns `event_ids` into `list[int]`, with a test; every consumer of processed data uses it. Detector contract stays `list[int]`. (D-c9f0: `read_parquet` returns numpy arrays, `arr + [0, 0]` adds instead of padding.) (12b2bb5)

## G1: loader and prepare checks (D-d7e1)
- [ ] Test: a line naming the same block twice (`Deleting block blk_300 file /mnt/blk_300`) gives exactly 1 row. (D-bd28)
- [ ] `hdfs.py`: raise on duplicate BlockId (3 examples) and on labels outside `label_map`; drop the misleading `strict=True`; one test each. (D-3ee7)
- [ ] Keep `errors="replace"` in `hdfs.py` and `parser_check.py`, count lines with replaced bytes, print `decode errors: N` in `[data]`. CRLF fixture written with `write_bytes`, assert no content ends with `\r`. (D-eac4)
- [ ] Print `dropped lines (no group id): N` in `[data]`. (D-109c)
- [ ] `_hit_stats` divides by the number of test events: guard an empty test split. (D-e14d point 5)

## G2: shared hash and atomic cache-write helpers (D-d7e1)
- [ ] After zip extraction write `data/raw/<ds>/.manifest.json` (size + md5 per file); on "already present" check sizes; a missing manifest means an unfinished extraction: raise. (D-c333)
- [ ] A9, C: definitions are not in the repo (they were only in a chat). Re-audit or drop.

## G3: max_lines path (D-d7e1)
- [x] `max_lines` null by default, dev slice config, `[data]` prints it. (0695b41, D-7e94)
- [x] Write `data.max_lines` to `metrics.json` (when `metrics.json` exists, Stage 4/5). (D-346d) (e6796cc)
- [ ] Stage 7 aggregation script rejects runs with `max_lines != null`. (D-346d)
- [ ] Loader test with `max_lines: None` reading every fixture line. (D-cd49)
- [ ] Run the full-HDFS prepare once through `ex`, machine otherwise idle, record time and peak memory (estimate ~5.3 GB on 7.7 GB RAM: OOM risk). Only when the user says go. (D-cd49)
- [ ] B6: definition not in the repo. Re-audit or drop.

## G4, G5, G6 (D-d7e1): do with the stage that touches those files
- [ ] G4 run setup: A7, A8, B2, B3. G5 splitting: B4, D7. G6 Drain config: B5, D8, D9. Definitions are not in the repo. Re-audit or drop.

## Stage 4-6: decide at the stage, after explanation (D-cbb3)
- [x] Unit of evaluation: the group (D-6141). (Stage 5)
- [x] PCA input: per-group TF-IDF template count vector (D-58a7). (Stage 5)
- [ ] DeepLog session score (proposed: max over windows of -log p(actual next key)).

## Stage 6-7
- [ ] Contamination ablation flag (train on all train blocks, no label filter); whether to run it stays open; needs the user's yes (invariant 3). (D-bf45)
- [ ] On full HDFS, correlation between `<UNK>` in a test session and its label; if present report F1 with and without `<UNK>`. RQ4 needs real log evolution (BGL or synthetic changes). (D-149b)
- [ ] HDFS_2k: 24 of 26 test `<UNK>` are benign blk-list delete lines; consider a mask before Stage 7.
- [ ] BGL and Thunderbird loaders (`label_granularity: "line"`, `grouping.mode: fixed`). (D-cfbf)

## Small, not blocking (review 2026-11-19)
- [ ] docs/plan.md S2-5 says the key holds the "split boundary"; the code keys the fit mask (stricter). Reword.
- [ ] Purge drops 85 of 219 val blocks on HDFS_2k; check the purge rate on full HDFS.
- [ ] Guard: `grouping.mode: fixed` with a `label_granularity: "group"` loader gives meaningless window labels.
- [ ] docs/hdfs_setup_comparison.md "chọn g trên val": say it reads no labels (D-cbb3).

## Spec items with no stage yet
- [ ] HDFS run with sequences deduplicated between train and test (spec 5.3).
- [ ] Parameter normalization for the LLM cache and retrieval (spec 5.4 item 6).
