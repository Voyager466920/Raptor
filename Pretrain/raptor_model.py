"""Scalable Raptor language model used by the Nemotron pretraining path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, latent_dim: int, rope_theta: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_down = nn.Linear(dim, 2 * latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, dim, bias=False)
        self.v_up = nn.Linear(latent_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_latent, v_latent = self.kv_down(x).chunk(2, dim=-1)
        k = self.k_up(k_latent).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(v_latent).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device, q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.out_proj(y.transpose(1, 2).contiguous().view(batch, seq_len, self.dim))


class SwiGLUExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = x.shape
        router_logits = self.router(x).float()
        router_probs = router_logits.softmax(dim=-1)
        top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        flat_x = x.reshape(-1, dim)
        flat_indices = top_indices.reshape(-1, self.top_k)
        flat_probs = top_probs.reshape(-1, self.top_k)
        output = torch.zeros_like(flat_x)

        # Six experts keep this simple dispatch practical while avoiding dense
        # execution of all experts. Empty tensors are valid Linear inputs, so no
        # GPU-synchronizing `if mask.any()` is needed.
        for expert_id, expert in enumerate(self.experts):
            assignment = flat_indices.eq(expert_id)
            rows, slots = assignment.nonzero(as_tuple=True)
            expert_out = expert(flat_x.index_select(0, rows))
            weights = flat_probs[rows, slots].to(expert_out.dtype).unsqueeze(-1)
            output.index_add_(0, rows, (expert_out * weights).to(output.dtype))

        importance = router_probs.mean(dim=(0, 1))
        load = F.one_hot(top_indices, self.num_experts).float().mean(dim=(0, 1, 2))
        load_balance_loss = self.num_experts * torch.sum(importance * load)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        return output.view(batch, seq_len, dim), load_balance_loss, router_z_loss


class RaptorBlock(nn.Module):
    def __init__(self, config: "RaptorConfig"):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = MultiHeadLatentAttention(
            config.dim, config.num_heads, config.latent_dim, config.rope_theta
        )
        self.moe_norm = RMSNorm(config.dim)
        self.moe = SparseMoE(
            config.dim, config.expert_hidden_dim, config.num_experts, config.top_k
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        moe_out, balance_loss, z_loss = self.moe(self.moe_norm(x))
        return x + moe_out, balance_loss, z_loss


@dataclass
class RaptorConfig:
    vocab_size: int = 35_000
    dim: int = 1_024
    latent_dim: int = 256
    expert_hidden_dim: int = 2_816
    num_layers: int = 18
    num_heads: int = 16
    num_experts: int = 6
    top_k: int = 2
    max_seq_len: int = 2_048
    rope_theta: float = 10_000.0
    load_balance_weight: float = 0.01
    router_z_weight: float = 0.001
    gradient_checkpointing: bool = True


class RaptorForCausalLM(nn.Module):
    def __init__(self, config: RaptorConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([RaptorBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / (2 * config.num_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            for expert in block.moe.experts:
                nn.init.normal_(expert.down_proj.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor):
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds {self.config.max_seq_len}")
        x = self.token_embedding(input_ids)
        # Embedding is not an autocast op. Explicitly move activations to BF16
        # so residual streams and sparse-expert outputs share one dtype while
        # master parameters and Adam states remain FP32.
        if x.is_cuda and torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())
        balance_total = x.new_zeros((), dtype=torch.float32)
        z_total = x.new_zeros((), dtype=torch.float32)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x, balance_loss, z_loss = checkpoint(block, x, use_reentrant=False)
            else:
                x, balance_loss, z_loss = block(x)
            balance_total = balance_total + balance_loss
            z_total = z_total + z_loss

        logits = self.lm_head(self.norm(x))
        aux_loss = (
            self.config.load_balance_weight * balance_total / len(self.blocks)
            + self.config.router_z_weight * z_total / len(self.blocks)
        )
        return logits, aux_loss

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        expert = sum(
            p.numel() for name, p in self.named_parameters() if ".experts." in name
        )
        active = total - expert + expert * self.config.top_k / self.config.num_experts
        return {"total": total, "expert": expert, "active_approx": int(active)}

    def config_dict(self):
        return asdict(self.config)
