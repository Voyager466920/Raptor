#!/usr/bin/env python3
"""Download a reproducible, stratified Nemotron-CC high-quality subset."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import os
import random
import urllib.request
from pathlib import Path


INDEX_URL = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
BASE_URL = "https://data.commoncrawl.org/"
ACTUAL = "quality=high/kind=actual/kind2=actual"
SYNTHETIC_QA = "quality=high/kind=synthetic/kind2=diverse_qa_pairs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-shards", type=int, default=96)
    parser.add_argument("--validation-shards", type=int, default=4)
    parser.add_argument("--synthetic-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=466920)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url)
    mode = "wb"
    if partial.exists() and partial.stat().st_size:
        offset = partial.stat().st_size
        request.add_header("Range", f"bytes={offset}-")
        mode = "ab"
    with urllib.request.urlopen(request, timeout=120) as response:
        if mode == "ab" and response.status != 206:
            mode = "wb"
        with partial.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    os.replace(partial, destination)
    return destination


def choose(paths, count, synthetic_fraction, rng):
    actual = [p for p in paths if ACTUAL in p]
    synthetic = [p for p in paths if SYNTHETIC_QA in p]
    rng.shuffle(actual)
    rng.shuffle(synthetic)
    synthetic_count = round(count * synthetic_fraction)
    actual_count = count - synthetic_count
    selected = actual[:actual_count] + synthetic[:synthetic_count]
    rng.shuffle(selected)
    return selected


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_bytes = urllib.request.urlopen(INDEX_URL, timeout=60).read()
    paths = gzip.decompress(index_bytes).decode().splitlines()
    rng = random.Random(args.seed)
    selected = choose(
        paths,
        args.train_shards + args.validation_shards,
        args.synthetic_fraction,
        rng,
    )
    validation = selected[: args.validation_shards]
    train = selected[args.validation_shards :]

    def fetch(remote_path: str):
        kind = "synthetic_qa" if SYNTHETIC_QA in remote_path else "actual"
        digest = hashlib.sha1(remote_path.encode()).hexdigest()[:10]
        local = output_dir / "shards" / f"{kind}__{digest}__{Path(remote_path).name}"
        print(f"downloading {remote_path} -> {local}", flush=True)
        return download(BASE_URL + remote_path, local)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        local_paths = list(pool.map(fetch, selected))
    mapping = dict(zip(selected, local_paths))

    for name, split in (("train", train), ("validation", validation)):
        manifest = output_dir / f"{name}.txt"
        manifest.write_text(
            "\n".join(str(mapping[path]) for path in split) + "\n", encoding="utf-8"
        )
        print(f"wrote {manifest} with {len(split)} shards")


if __name__ == "__main__":
    main()
