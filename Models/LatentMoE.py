import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, rope_theta=10000.0, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.dkv_proj = nn.Linear(dim, 2 * latent_dim, bias=False)
        self.up_proj_k = nn.Linear(latent_dim, dim, bias=False)
        self.up_proj_v = nn.Linear(latent_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._rope_cache = None

    def _build_rope_cache(self, seq_len, device, dtype):
        if (
            self._rope_cache is None
            or self._rope_cache[0].size(0) < seq_len
            or self._rope_cache[0].dtype != dtype
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._rope_cache = (emb.sin().to(dtype), emb.cos().to(dtype))
        return self._rope_cache

    def forward(self, x, kv_cache=None, *, attn_mask=None, use_cache=False):
        b, s, _ = x.size()
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        d_k, d_v = self.dkv_proj(x).view(b, s, 2, self.latent_dim).permute(2, 0, 1, 3)
        k = self.up_proj_k(d_k).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.up_proj_v(d_v).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if kv_cache is None else kv_cache[0].size(2)
        total_k = past_len + k.size(-2)
        sin, cos = self._build_rope_cache(total_k, x.device, x.dtype)
        sin = sin[past_len:past_len + k.size(-2)]
        cos = cos[past_len:past_len + k.size(-2)]
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        # PyTorch SDPA boolean masks use True for positions that are allowed to
        # participate in attention.  The previous upper-triangular mask did the
        # opposite of causal language modelling and exposed future tokens.
        if kv_cache is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            allowed = torch.arange(total_k, device=x.device).unsqueeze(0) <= (
                past_len + torch.arange(s, device=x.device).unsqueeze(1)
            )
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=allowed.unsqueeze(0).unsqueeze(0),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(b, s, self.dim)
        out = self.out_proj(out)
        if use_cache:
            return out, (k.detach(), v.detach())
        else:
            return out, None

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_size, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, mlp_size)
        self.fc2 = nn.Linear(mlp_size, embedding_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def subsequent_mask(seq_len: int, device=None) -> torch.Tensor:
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1)

class Expert(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MoEBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_experts: int,
        experts_per_token: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.experts = nn.ModuleList(
            [Expert(embed_dim, mlp_dim, dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(embed_dim, num_experts)
        self.gate_dropout = nn.Dropout(p=0.1)
        self.experts_per_token = experts_per_token
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        if torch.is_autocast_enabled() and x_norm.dtype == torch.float32:
            x_norm = x_norm.to(torch.float16)
        gate_logits = self.gate(x_norm)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_probs = self.gate_dropout(gate_probs)
        gate_probs = gate_probs / (gate_probs.sum(dim=-1, keepdim=True) + 1e-8)
        topk_probs, topk_idx = torch.topk(
            gate_probs, self.experts_per_token, dim=-1
        )

        B, S, D = x_norm.shape
        K = self.experts_per_token
        x_flat = x_norm.view(-1, D)
        topk_idx_flat = topk_idx.view(-1, K)
        topk_probs_flat = topk_probs.view(-1, K)
        out_flat = torch.zeros_like(x_flat, dtype=x_flat.dtype)

        for eid, expert in enumerate(self.experts):
            mask = (topk_idx_flat == eid)
            if not mask.any():
                continue
            rows = mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            expert_in = x_flat.index_select(0, rows)
            expert_out = expert(expert_in)
            probs = (topk_probs_flat[rows] * mask[rows].float()).sum(dim=1)
            expert_out.mul_(probs.unsqueeze(-1))
            out_flat.index_add_(0, rows, expert_out)

        output = out_flat.view(B, S, D)
        importance = gate_probs.float().sum(dim=(0, 1))
        load = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        load.scatter_add_(0, topk_idx_flat.reshape(-1),
                          torch.ones_like(topk_idx_flat.reshape(-1), dtype=load.dtype))
        balance_loss = 0.5 * (
                torch.std(importance / (importance.sum() + 1e-8)) +
                torch.std(load / (load.sum() + 1e-8))
        )

        return output, balance_loss

class LatentGPTBlock(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, mlp_dim: int, dropout: float = 0.1, num_heads: int = 16, num_experts: int = 6, experts_per_token: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(
            dim=embed_dim,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.norm2 = RMSNorm(embed_dim)
        self.moe = MoEBlock(embed_dim, mlp_dim, num_experts, experts_per_token, dropout)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        y, _ = self.attn(self.norm1(x), attn_mask=mask)
        x = x + y
        moe_output, balance_loss = self.moe(self.norm2(x))
        x = x + moe_output
        return x, balance_loss

class LatentMoE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 4096,
        embed_dim: int = 1024,
        latent_dim: int = 256,
        mlp_dim: int = 4096,
        num_layers: int = 16,
        dropout: float = 0.1,
        num_heads: int = 16,
        num_experts: int = 6,
        experts_per_token: int = 2,
        balance_loss_weight: float = 0.001
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            LatentGPTBlock(
                embed_dim,
                latent_dim,
                mlp_dim,
                dropout,
                num_heads,
                num_experts,
                experts_per_token
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.balance_loss_weight = balance_loss_weight

    def forward(self, input_ids: torch.LongTensor) -> (torch.Tensor, torch.Tensor):
        b, s = input_ids.size()
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        mask = subsequent_mask(s, device=input_ids.device)
        total_balance_loss = 0.0
        for blk in self.blocks:
            x, balance_loss = blk(x, mask)
            total_balance_loss += balance_loss * self.balance_loss_weight
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_balance_loss / len(self.blocks)
