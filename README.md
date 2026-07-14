# Raptor

Raptor is a decoder-only Mixture-of-Experts language model trained from scratch on a single H200 GPU.

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Raptor-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Voyager466920/Raptor)

## Raptor 1B

- 1.027B total parameters and approximately 404M active parameters per token
- 18 layers with width 1,024 and latent attention dimension 256
- six SwiGLU experts per layer with top-2 routing
- RoPE, RMSNorm, causal Flash SDPA, and tied token embeddings
- 2,048-token context length
- pretrained for approximately 20B tokens on a deterministic Nemotron-CC subset
- BF16 activations with FP32 optimizer state

The supported pretraining path is documented in [PRETRAIN_1B.md](PRETRAIN_1B.md). The public pretrained weights are available at [`Voyager466920/Raptor`](https://huggingface.co/Voyager466920/Raptor).

## Setup

```bash
python -m pip install -r requirements-pretrain.txt
```

The supplied YAML files and launch scripts contain local example paths. Update the dataset, output, environment, and repository paths for your machine before launching.

## Pretraining

Prepare the deterministic Nemotron-CC subset and then launch the single-GPU run:

```bash
python scripts/download_nemotron_subset.py \
  --output-dir /path/to/nemotron_cc \
  --train-shards 128 \
  --validation-shards 4 \
  --synthetic-fraction 0.25 \
  --workers 8

python -m Pretrain.pretrain_nemotron \
  --config configs/raptor_1b_nemotron.yaml \
  --resume auto
```

## Supervised fine-tuning

The `Finetune` directory contains:

- `prepare_smoltalk_mix.py`: deterministic, source-stratified SmolTalk mixture preparation
- `sft_raptor_1b.py`: single-turn Alpaca SFT
- `sft_raptor_1b_chat.py`: multi-turn, assistant-only SmolTalk SFT with periodic validation and best-checkpoint retention

## Legacy KoRaptor 150M

The original 150M Korean LatentMoE implementation and checkpoints remain in the repository for compatibility. Its chatbot model is available at [`Voyager466920/KoRaptor_Chatbot`](https://huggingface.co/Voyager466920/KoRaptor_Chatbot).

## Limitations

- The released Raptor checkpoint is a pretrained base model, not an instruction-tuned assistant.
- It is English-focused and the current tokenizer has poor Korean coverage.
- Generated text may be inaccurate, repetitive, biased, or unsafe.




