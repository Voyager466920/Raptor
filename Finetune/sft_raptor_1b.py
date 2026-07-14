#!/usr/bin/env python3
"""Full-parameter English instruction tuning for the Nemotron Raptor 1B model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Pretrain.raptor_model import RaptorConfig, RaptorForCausalLM


class AlpacaSFTDataset(Dataset):
    def __init__(self, records, tokenizer_path: str, max_length: int):
        self.records = records
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = None

    def __len__(self):
        return len(self.records)

    def _tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        return self.tokenizer

    def __getitem__(self, index):
        record = self.records[index]
        instruction = record["instruction"].strip()
        context = record.get("input", "").strip()
        response = record["output"].strip()
        prompt = f"### Instruction:\n{instruction}\n"
        if context:
            prompt += f"### Input:\n{context}\n"
        prompt += "### Response:\n"

        tokenizer = self._tokenizer()
        prompt_ids = tokenizer.encode(prompt, out_type=int)
        response_ids = tokenizer.encode(response, out_type=int)
        eos_id = tokenizer.eos_id()
        full_ids = (prompt_ids + response_ids + ([eos_id] if eos_id >= 0 else []))[: self.max_length + 1]
        if len(full_ids) < 2:
            full_ids = [tokenizer.bos_id(), eos_id]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        response_start = max(0, min(len(labels), len(prompt_ids) - 1))
        labels[:response_start] = [-100] * response_start
        return torch.tensor(input_ids), torch.tensor(labels)


def collate(batch, pad_id: int):
    max_len = max(item[0].numel() for item in batch)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, (input_ids, target_ids) in enumerate(batch):
        inputs[row, : input_ids.numel()] = input_ids
        labels[row, : target_ids.numel()] = target_ids
    return {"input_ids": inputs, "labels": labels}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=25):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp = nullcontext if device.type == "cpu" else lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with amp():
            logits, _ = model(inputs)
            loss_sum = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
        count = int((labels != -100).sum())
        total_loss += float(loss_sum)
        total_tokens += count
    model.train()
    return total_loss / max(1, total_tokens)


def atomic_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(466920)
    torch.manual_seed(466920)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source_step = int(source["step"])
    cfg = source["config"]
    tokenizer_path = cfg["data"]["tokenizer"]
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model_values = dict(cfg["model"])
    model_values["vocab_size"] = tokenizer.vocab_size()
    model = RaptorForCausalLM(RaptorConfig(**model_values))
    model.load_state_dict(source["model"])
    del source
    model.to(device)

    with open(args.data, encoding="utf-8") as handle:
        records = json.load(handle)
    random.Random(466920).shuffle(records)
    validation = records[: args.validation_size]
    training = records[args.validation_size :]
    train_dataset = AlpacaSFTDataset(training, tokenizer_path, args.max_length)
    val_dataset = AlpacaSFTDataset(validation, tokenizer_path, args.max_length)
    collate_fn = lambda batch: collate(batch, tokenizer.pad_id())
    train_loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.micro_batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    initial_val = evaluate(model, val_loader, device)
    print(json.dumps({"event": "start", "source_step": source_step, "train_examples": len(training),
                      "validation_examples": len(validation), "total_steps": total_steps,
                      "initial_validation_loss": initial_val, "initial_validation_perplexity": math.exp(initial_val)}), flush=True)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    running_micro_batches = 0
    running_tokens = 0
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        micro_batches_in_epoch = len(train_loader)
        final_group_size = micro_batches_in_epoch % args.gradient_accumulation
        for micro_step, batch in enumerate(train_loader, 1):
            inputs = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            group_size = (
                final_group_size
                if final_group_size and micro_step > micro_batches_in_epoch - final_group_size
                else args.gradient_accumulation
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux_loss = model(inputs)
                ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
                loss = (ce + aux_loss) / group_size
            loss.backward()
            running_loss += float(ce)
            running_micro_batches += 1
            running_tokens += int((labels != -100).sum())
            if micro_step % args.gradient_accumulation != 0 and micro_step != micro_batches_in_epoch:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            progress = global_step / max(1, total_steps)
            lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 20 == 0 or global_step == 1:
                elapsed = time.perf_counter() - started
                record = {"step": global_step, "epoch": epoch + 1, "loss": running_loss / max(1, running_micro_batches),
                          "learning_rate": lr, "response_tokens_seen": running_tokens,
                          "steps_per_second": global_step / elapsed,
                          "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
                print(json.dumps(record), flush=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                running_loss = 0.0
                running_micro_batches = 0

            if not args.no_save and global_step % args.save_interval == 0:
                atomic_save({"model": model.state_dict(), "config": cfg, "source_step": source_step,
                             "sft_step": global_step}, output_dir / "latest.pt")
                print(f"saved SFT checkpoint at step {global_step}", flush=True)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    final_val = evaluate(model, val_loader, device)
    summary = {"event": "complete", "source_step": source_step, "sft_steps": global_step,
               "validation_loss": final_val, "validation_perplexity": math.exp(final_val)}
    print(json.dumps(summary), flush=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_save:
        atomic_save({"model": model.state_dict(), "config": cfg, "source_step": source_step,
                     "sft_step": global_step}, output_dir / "final.pt")


if __name__ == "__main__":
    main()
