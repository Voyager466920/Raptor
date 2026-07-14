#!/usr/bin/env python3
"""Multi-turn, assistant-only SFT for the Raptor 1B causal LM."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Pretrain.raptor_model import RaptorConfig, RaptorForCausalLM


class JsonlChatDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_length: int):
        self.path = path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.offsets = []
        with open(path, "rb") as handle:
            while True:
                offset = handle.tell()
                if not handle.readline():
                    break
                self.offsets.append(offset)
        self.handle = None
        self.tokenizer = None

    def __len__(self):
        return len(self.offsets)

    def _resources(self):
        if self.handle is None:
            self.handle = open(self.path, "rb")
        if self.tokenizer is None:
            self.tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_path)
        return self.handle, self.tokenizer

    def _encode(self, index):
        handle, tokenizer = self._resources()
        handle.seek(self.offsets[index])
        record = json.loads(handle.readline())
        token_ids, targets = [], []
        for message in record["messages"]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            title = {"system": "System", "assistant": "Assistant"}.get(role, "User")
            header = tokenizer.encode(f"### {title}:\n", out_type=int)
            body = tokenizer.encode(content.strip() + "\n", out_type=int)
            token_ids.extend(header)
            targets.extend([False] * len(header))
            token_ids.extend(body)
            targets.extend([role == "assistant"] * len(body))
            if role == "assistant" and tokenizer.eos_id() >= 0:
                token_ids.append(tokenizer.eos_id())
                targets.append(True)

        token_ids = token_ids[: self.max_length + 1]
        targets = targets[: self.max_length + 1]
        if len(token_ids) < 2:
            return None
        inputs = token_ids[:-1]
        labels = [tok if target else -100 for tok, target in zip(token_ids[1:], targets[1:])]
        if not any(label != -100 for label in labels):
            return None
        return torch.tensor(inputs), torch.tensor(labels)

    def __getitem__(self, index):
        for attempt in range(16):
            item = self._encode((index + attempt) % len(self.offsets))
            if item is not None:
                return item
        raise RuntimeError("could not find an example with an assistant target")


def collate(batch, pad_id):
    length = max(inputs.numel() for inputs, _ in batch)
    input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    for row, (inputs, targets) in enumerate(batch):
        input_ids[row, : inputs.numel()] = inputs
        labels[row, : targets.numel()] = targets
    return {"input_ids": input_ids, "labels": labels}


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
        loss_sum += float(loss)
        token_count += int((labels != -100).sum())
    model.train()
    return loss_sum / max(1, token_count)


def atomic_save(payload, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = args_parser()
    random.seed(466920)
    torch.manual_seed(466920)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source_step = int(source.get("step", source.get("source_step", -1)))
    cfg = source["config"]
    tokenizer_path = cfg["data"]["tokenizer"]
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model_values = dict(cfg["model"])
    model_values["vocab_size"] = tokenizer.vocab_size()
    model = RaptorForCausalLM(RaptorConfig(**model_values))
    model.load_state_dict(source["model"])
    del source
    model.to(device)

    train_set = JsonlChatDataset(args.train_data, tokenizer_path, args.max_length)
    val_set = JsonlChatDataset(args.validation_data, tokenizer_path, args.max_length)
    collate_fn = lambda rows: collate(rows, tokenizer.pad_id())
    train_loader = DataLoader(train_set, batch_size=args.micro_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.micro_batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)
    groups = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = groups * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=True)
    metrics = output / "metrics.jsonl"
    initial_val = evaluate(model, val_loader, device, args.eval_batches)
    start_record = {"event": "start", "source_step": source_step, "train_examples": len(train_set),
                    "validation_examples": len(val_set), "total_steps": total_steps,
                    "initial_validation_loss": initial_val,
                    "initial_validation_perplexity": math.exp(initial_val)}
    print(json.dumps(start_record), flush=True)

    global_step = 0
    best_val = initial_val
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_micro = 0
    response_tokens = 0
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        micro_total = len(train_loader)
        remainder = micro_total % args.gradient_accumulation
        for micro_step, batch in enumerate(train_loader, 1):
            inputs = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            group_size = remainder if remainder and micro_step > micro_total - remainder else args.gradient_accumulation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(inputs)
                ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
                loss = (ce + aux) / group_size
            loss.backward()
            running_loss += float(ce)
            running_micro += 1
            response_tokens += int((labels != -100).sum())
            if micro_step % args.gradient_accumulation and micro_step != micro_total:
                continue

            global_step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if global_step <= args.warmup_steps:
                lr = args.learning_rate * global_step / max(1, args.warmup_steps)
            else:
                progress = (global_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
                lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step == 1 or global_step % 20 == 0:
                record = {"step": global_step, "epoch": epoch + 1,
                          "loss": running_loss / max(1, running_micro), "learning_rate": lr,
                          "response_tokens_seen": response_tokens,
                          "steps_per_second": global_step / (time.perf_counter() - started),
                          "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
                print(json.dumps(record), flush=True)
                with metrics.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                running_loss = 0.0
                running_micro = 0

            if global_step % args.eval_interval == 0 or global_step == total_steps:
                val_loss = evaluate(model, val_loader, device, args.eval_batches)
                val_record = {"step": global_step, "validation_loss": val_loss,
                              "validation_perplexity": math.exp(val_loss)}
                print(json.dumps(val_record), flush=True)
                with metrics.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(val_record) + "\n")
                if val_loss < best_val and not args.no_save:
                    best_val = val_loss
                    atomic_save({"model": model.state_dict(), "config": cfg, "source_step": source_step,
                                 "sft_step": global_step, "validation_loss": val_loss}, output / "best.pt")
                    print(f"saved best SFT checkpoint at step {global_step}", flush=True)

            if not args.no_save and global_step % args.save_interval == 0:
                atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
                             "source_step": source_step, "sft_step": global_step}, output / "latest.pt")
                print(f"saved resumable SFT checkpoint at step {global_step}", flush=True)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    summary = {"event": "complete", "source_step": source_step, "sft_steps": global_step,
               "best_validation_loss": best_val, "best_validation_perplexity": math.exp(best_val)}
    print(json.dumps(summary), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_save:
        atomic_save({"model": model.state_dict(), "config": cfg, "source_step": source_step,
                     "sft_step": global_step}, output / "final.pt")


if __name__ == "__main__":
    main()
