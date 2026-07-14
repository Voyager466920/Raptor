import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pretrain.raptor_model import RaptorConfig, RaptorForCausalLM


def test_forward_backward_and_causality():
    config = RaptorConfig(
        vocab_size=128,
        dim=64,
        latent_dim=16,
        expert_hidden_dim=96,
        num_layers=2,
        num_heads=4,
        num_experts=4,
        top_k=2,
        max_seq_len=32,
        gradient_checkpointing=False,
    )
    model = RaptorForCausalLM(config)
    model.eval()
    first = torch.randint(0, config.vocab_size, (2, 16))
    second = first.clone()
    second[:, 9:] = torch.randint(0, config.vocab_size, second[:, 9:].shape)
    with torch.no_grad():
        logits_first, _ = model(first)
        logits_second, _ = model(second)
    torch.testing.assert_close(logits_first[:, :9], logits_second[:, :9], atol=1e-5, rtol=1e-5)

    model.train()
    logits, aux = model(first)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size), first[:, 1:].reshape(-1)
    ) + aux
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


if __name__ == "__main__":
    test_forward_backward_and_causality()
    print("Nemotron pretraining smoke test passed")
