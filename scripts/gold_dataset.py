"""
load 1000 gold rows as dicts:
  gold_id 0-99   = common 100 (majority; 3-way tie uses tie_policies)
  gold_id 100-399 = annotator 1 unique 300
  gold_id 400-699 = annotator 2 unique 300
  gold_id 700-999 = annotator 3 unique 300
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from labels import NEG, NEU, POS, canon
from text_norm import norm_text
from tie_policies import TiePolicy, default_tie_policy

# filenames in this folder — tweak if you rename exports
COMMON_CSV = "common_100_majority_vote.csv"
ANNOTATOR_FILES: list[tuple[str, str]] = [
    ("annotator_1_unique", "annotator_1_annotations_2026-05-19T04-05-02.317Z (1).csv"),
    ("annotator_2_unique", "annotator_2_annotations_300.csv"),
    ("annotator_3_unique", "annotator_3_annotations_300_unique.csv"),
]


def _majority_or_tie(
    raw1: str,
    raw2: str,
    raw3: str,
    tie: TiePolicy,
) -> tuple[str, str]:
    a, b, c = canon(raw1), canon(raw2), canon(raw3)
    if a == b == c:
        return a, "unanimous"
    if a == b or a == c:
        return a, "majority"
    if b == c:
        return b, "majority"
    return canon(tie.pick(raw1, raw2, raw3)), "tie_break"


def _read_common100(root: Path, tie: TiePolicy) -> list[dict[str, Any]]:
    path = root / COMMON_CSV
    out: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g, how = _majority_or_tie(
                row["sentiment_a1"],
                row["sentiment_a2"],
                row["sentiment_a3"],
                tie,
            )
            text = row["aae_text"]
            out.append(
                {
                    "subset": "common_100",
                    "source_item_index": int(row["item_index"]),
                    "aae_text": text,
                    "text_key": norm_text(text),
                    "gold_label": g,
                    "resolution": how,
                }
            )
    out.sort(key=lambda r: r["source_item_index"])
    return out


def _read_annotator_block(
    tag: str,
    relpath: str,
    root: Path,
) -> list[dict[str, Any]]:
    path = root / relpath
    out: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["aae_text"]
            out.append(
                {
                    "subset": tag,
                    "source_item_index": int(row["item_index"]),
                    "aae_text": text,
                    "text_key": norm_text(text),
                    "gold_label": canon(row["sentiment"]),
                    "resolution": "single_annotator",
                }
            )
    out.sort(key=lambda r: r["source_item_index"])
    return out


def load_gold_rows(
    data_dir: str | Path | None = None,
    tie_policy: TiePolicy | None = None,
) -> list[dict[str, Any]]:
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parent
    tie = tie_policy if tie_policy is not None else default_tie_policy()

    rows: list[dict[str, Any]] = []
    rows.extend(_read_common100(root, tie))
    for tag, rel in ANNOTATOR_FILES:
        rows.extend(_read_annotator_block(tag, rel, root))

    for i, r in enumerate(rows):
        r["gold_id"] = i

    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 gold rows, got {len(rows)}")

    keys = [r["text_key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate text_key in gold set — join would be ambiguous")

    return rows


def gold_by_text_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["text_key"]: r for r in rows}


if __name__ == "__main__":
    g = load_gold_rows()
    ties = [r for r in g if r["resolution"] == "tie_break"]
    print("rows", len(g), "tie_break", len(ties))
    for r in ties:
        print(r["gold_id"], r["gold_label"], r["aae_text"][:60] + "...")
