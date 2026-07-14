#!/usr/bin/env python3
"""Build a deterministic, source-stratified SmolTalk JSONL mixture."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


SOURCES = {
    "smol-magpie-ultra": {"limit": None, "val": 1600},
    "smol-constraints": {"limit": None, "val": 140},
    "everyday-conversations": {"limit": None, "val": 10},
    "metamathqa-50k": {"limit": None, "val": 180},
    "smol-rewrite": {"limit": 20000, "val": 70},
}


def read_source(root: Path, source: str):
    rows = []
    for path in sorted((root / source).glob("train-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["messages"]):
            for record in batch.to_pylist():
                messages = record.get("messages")
                if isinstance(messages, list) and any(m.get("role") == "assistant" for m in messages):
                    rows.append({"source": source, "messages": messages})
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(466920)
    train, validation, counts = [], [], {}

    for source, settings in SOURCES.items():
        rows = read_source(root, source)
        rng.shuffle(rows)
        limit = settings["limit"]
        if limit is not None:
            rows = rows[:limit]
        val_count = settings["val"]
        validation.extend(rows[:val_count])
        train.extend(rows[val_count:])
        counts[source] = {"total": len(rows), "train": len(rows) - val_count, "validation": val_count}

    rng.shuffle(train)
    rng.shuffle(validation)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    summary = {"seed": 466920, "train": len(train), "validation": len(validation), "sources": counts}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
