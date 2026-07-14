"""Streaming local Nemotron JSONL-Zstandard shards into packed token sequences."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List

import sentencepiece as spm
import torch
import zstandard as zstd
from torch.utils.data import IterableDataset, get_worker_info


def read_manifest(path: str | Path) -> List[str]:
    manifest = Path(path).resolve()
    files = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        shard = Path(line)
        if not shard.is_absolute():
            shard = manifest.parent / shard
        files.append(str(shard.resolve()))
    if not files:
        raise ValueError(f"manifest contains no shards: {manifest}")
    return files


def iter_zstd_jsonl(path: str) -> Iterator[dict]:
    with open(path, "rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as text_stream:
                for line in text_stream:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


class PackedNemotronDataset(IterableDataset):
    def __init__(
        self,
        manifest: str,
        tokenizer_path: str,
        sequence_length: int,
        seed: int = 1234,
        repeat: bool = True,
        document_shuffle_buffer: int = 512,
    ):
        super().__init__()
        self.files = read_manifest(manifest)
        self.tokenizer_path = str(Path(tokenizer_path).resolve())
        self.sequence_length = sequence_length
        self.seed = seed
        self.repeat = repeat
        self.document_shuffle_buffer = document_shuffle_buffer

    def _documents(self, files: Iterable[str], rng: random.Random) -> Iterator[str]:
        buffer: List[str] = []
        for path in files:
            for record in iter_zstd_jsonl(path):
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                buffer.append(text)
                if len(buffer) >= self.document_shuffle_buffer:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer.clear()
        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        eos_id = tokenizer.eos_id()
        cycle = 0
        token_buffer: List[int] = []

        while True:
            rng = random.Random(self.seed + cycle * 10_007)
            files = list(self.files)
            rng.shuffle(files)
            files = files[worker_id::num_workers]
            if not files:
                raise RuntimeError("more dataloader workers than dataset shards")

            for text in self._documents(files, rng):
                ids = tokenizer.encode(text, out_type=int)
                if not ids:
                    continue
                token_buffer.extend(ids)
                if eos_id >= 0:
                    token_buffer.append(eos_id)
                while len(token_buffer) >= self.sequence_length + 1:
                    window = token_buffer[: self.sequence_length + 1]
                    del token_buffer[: self.sequence_length]
                    yield {
                        "input_ids": torch.tensor(window[:-1], dtype=torch.long),
                        "labels": torch.tensor(window[1:], dtype=torch.long),
                    }

            cycle += 1
            if not self.repeat:
                break
