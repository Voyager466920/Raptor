# Raptor 1B Nemotron-CC Pretraining

This is the supported single-H200 pretraining path. The original 150M scripts
remain for checkpoint compatibility, but should not be used for new large runs.

## Model

- 1.027B total parameters, approximately 404M active parameters per token
- 18 layers, width 1024, latent attention dimension 256
- six SwiGLU experts per layer with top-2 routing
- causal Flash SDPA, RoPE, RMSNorm, tied token embeddings
- sequence length 2048 and BF16 activations with FP32 optimizer state

## Data

Download 128 training shards and four validation shards (roughly 40-50 GB):

```bash
cd /home/hoon/joonha_llm/KoRaptor
/home/hoon/joonha_llm/.venv/bin/python scripts/download_nemotron_subset.py \
  --output-dir /home/hoon/joonha_llm/data/nemotron_cc \
  --train-shards 128 \
  --validation-shards 4 \
  --synthetic-fraction 0.25 \
  --workers 8
```

The selection is deterministic. It mixes Nemotron high-quality actual web text
with high-quality synthetic diverse-QA shards. Documents are shuffled in a
bounded buffer and packed without padding into fixed 2048-token sequences.

## Training

```bash
tmux new-session -d -s raptor1b-train \
  '/home/hoon/joonha_llm/KoRaptor/scripts/run_pretrain_gpu7.sh'
```

Attach with `tmux attach -t raptor1b-train`. Metrics are appended to
`/home/hoon/joonha_llm/runs/raptor_1b_nemotron/metrics.jsonl`. Checkpoints are
atomic and include model, AdamW, step, token count, and RNG states. A terminated
run saves after the current optimizer step and the launcher resumes from
`checkpoints/latest.pt`.

The default configuration uses a 524,288-token global batch and trains for
approximately 20B tokens. On GPU 7, the verified micro-batch 16 benchmark used
about 75 GiB and reached roughly 67k-79k tokens/second after warmup.
