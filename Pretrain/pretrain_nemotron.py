#!/usr/bin/env python3
"""Single-GPU BF16 pretraining for the 1B Raptor model on Nemotron-CC."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from Pretrain.nemotron_dataset import PackedNemotronDataset
from Pretrain.raptor_model import RaptorConfig, RaptorForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint file or 'auto'")
    parser.add_argument("--max-steps", type=int, default=None, help="override for smoke tests")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile after eager smoke tests")
    parser.add_argument("--no-save", action="store_true", help="skip checkpoints for short benchmarks")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, *, peak: float, minimum: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(state: dict, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def checkpoint_state(model, optimizer, step: int, tokens_seen: int, config: dict):
    raw_model = getattr(model, "_orig_mod", model)
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int, amp_context):
    model.eval()
    losses = []
    iterator = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with amp_context():
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        losses.append(loss.float())
    model.train()
    if not losses:
        return float("nan")
    return torch.stack(losses).mean().item()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pretraining entrypoint")

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(int(cfg["training"]["seed"]))

    tokenizer_path = str(Path(cfg["data"]["tokenizer"]).resolve())
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model_cfg_values = dict(cfg["model"])
    model_cfg_values["vocab_size"] = tokenizer.vocab_size()
    model_config = RaptorConfig(**model_cfg_values)
    model = RaptorForCausalLM(model_config).to(device)
    counts = model.parameter_counts()
    print(json.dumps({k: f"{v / 1e6:.2f}M" for k, v in counts.items()}), flush=True)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    max_steps = args.max_steps or int(train_cfg["max_steps"])
    accumulation = int(train_cfg["gradient_accumulation_steps"])
    sequence_length = int(model_config.max_seq_len)

    train_dataset = PackedNemotronDataset(
        manifest=data_cfg["train_manifest"],
        tokenizer_path=tokenizer_path,
        sequence_length=sequence_length,
        seed=int(train_cfg["seed"]),
        repeat=True,
        document_shuffle_buffer=int(data_cfg.get("document_shuffle_buffer", 512)),
    )
    val_dataset = PackedNemotronDataset(
        manifest=data_cfg["validation_manifest"],
        tokenizer_path=tokenizer_path,
        sequence_length=sequence_length,
        seed=int(train_cfg["seed"]) + 99,
        repeat=False,
        document_shuffle_buffer=1,
    )
    loader_kwargs = {
        "batch_size": int(train_cfg["micro_batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    if int(data_cfg["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        num_workers=0,
        pin_memory=True,
    )

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer_kwargs = dict(
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["adam_eps"]),
    )
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(train_cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **optimizer_kwargs,
    )

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps({**cfg, "model": model.config_dict()}, indent=2), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    start_step = 0
    tokens_seen = 0

    resume = args.resume
    if resume == "auto":
        resume = str(output_dir / "checkpoints" / "latest.pt")
    if resume and Path(resume).exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        print(f"resumed from {resume} at step {start_step}", flush=True)

    if args.compile:
        model = torch.compile(model, dynamic=False)

    amp_context = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"received signal {signum}; saving after current optimizer step", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iterator = iter(train_loader)
    log_interval = int(train_cfg["log_interval"])
    eval_interval = int(train_cfg["eval_interval"])
    save_interval = int(train_cfg["save_interval"])
    running_loss = 0.0
    running_aux = 0.0
    interval_tokens = 0
    interval_start = time.perf_counter()

    for step in range(start_step, max_steps):
        lr = learning_rate(
            step,
            peak=float(train_cfg["learning_rate"]),
            minimum=float(train_cfg["min_learning_rate"]),
            warmup=int(train_cfg["warmup_steps"]),
            total=max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        step_aux = 0.0
        for _micro in range(accumulation):
            batch = next(data_iterator)
            inputs = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with amp_context():
                logits, aux_loss = model(inputs)
                ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = (ce_loss + aux_loss) / accumulation
            loss.backward()
            step_loss += ce_loss.detach().float().item() / accumulation
            step_aux += aux_loss.detach().float().item() / accumulation
            batch_tokens = labels.numel()
            tokens_seen += batch_tokens
            interval_tokens += batch_tokens

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        running_loss += step_loss
        running_aux += step_aux

        if completed_step % log_interval == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - interval_start
            record = {
                "step": completed_step,
                "tokens_seen": tokens_seen,
                "loss": running_loss / log_interval,
                "aux_loss": running_aux / log_interval,
                "perplexity": math.exp(min(20.0, running_loss / log_interval)),
                "learning_rate": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_second": interval_tokens / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(record), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            running_loss = running_aux = 0.0
            interval_tokens = 0
            interval_start = time.perf_counter()

        if completed_step % eval_interval == 0:
            val_loss = evaluate(
                model, val_loader, device, int(train_cfg["eval_batches"]), amp_context
            )
            val_record = {
                "step": completed_step,
                "tokens_seen": tokens_seen,
                "validation_loss": val_loss,
                "validation_perplexity": math.exp(min(20.0, val_loss)),
            }
            print(json.dumps(val_record), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(val_record) + "\n")
            interval_start = time.perf_counter()

        should_save = completed_step % save_interval == 0 or stop_requested or completed_step == max_steps
        if should_save and not args.no_save:
            state = checkpoint_state(model, optimizer, completed_step, tokens_seen, cfg)
            checkpoint_dir = output_dir / "checkpoints"
            atomic_torch_save(state, checkpoint_dir / "latest.pt")
            if completed_step % (save_interval * 5) == 0 or completed_step == max_steps:
                atomic_torch_save(state, checkpoint_dir / f"step_{completed_step:08d}.pt")
            print(f"saved checkpoint at step {completed_step}", flush=True)
        if stop_requested:
            break


if __name__ == "__main__":
    main()
