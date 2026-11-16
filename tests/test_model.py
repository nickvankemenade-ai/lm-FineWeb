import torch

from config import Config
from model import GPT, CausalSelfAttention, RMSNorm, RoPE, SwiGLU


def make_config() -> Config:
    return Config(
        vocab_size=32000,
        block_size=64,
        n_layers=2,
        n_heads=4,
        n_embd=128,
        dropout=0.1,
        batch_size=2,
        max_iters=100,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup=10,
        weight_decay=0.1,
        eval_interval=10,
        train_split=0.99,
        seed=42,
    )


def test_rmsnorm_and_rope():
    x = torch.randn(2, 4, 128)
    norm = RMSNorm(128)
    y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    rope = RoPE(head_dim=32, max_seq_len=128)
    q = torch.randn(2, 4, 16, 32)
    q_rotated = rope(q)

    assert q_rotated.shape == q.shape
    assert torch.isfinite(q_rotated).all()

    loss = y.square().mean() + q_rotated.square().mean()
    loss.backward()

    assert norm.weight.grad is not None


def test_attention():
    torch.manual_seed(42)
    attention = CausalSelfAttention(
        n_embd=128,
        n_heads=4,
        block_size=64,
        dropout=0.1,
        resid_dropout=0.1,
    )
    x = torch.randn(2, 32, 128, requires_grad=True)

    y = attention(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    assert x.grad is not None
    assert attention.qkv_proj.weight.grad is not None
    assert attention.out_proj.weight.grad is not None

    # Evaluation mode disables dropout.
    attention.eval()
    with torch.no_grad():
        assert torch.allclose(attention(x), attention(x))


def test_swiglu():
    torch.manual_seed(42)
    mlp = SwiGLU(n_embd=128, hidden_dim=352, dropout=0.1)
    x = torch.randn(2, 32, 128, requires_grad=True)

    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    assert x.grad is not None
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None

    mlp.eval()
    with torch.no_grad():
        assert torch.allclose(mlp(x), mlp(x))


def test_gpt_forward_backward():
    torch.manual_seed(42)
    model = GPT(make_config())

    input_ids = torch.randint(0, 32000, (2, 32))
    targets = torch.randint(0, 32000, (2, 32))

    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 32, 32000)
    assert loss is not None and loss.ndim == 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attn.qkv_proj.weight.grad is not None
    assert model.blocks[0].mlp.gate_proj.weight.grad is not None
    assert model.lm_head.weight.grad is not None

    model.eval()
    with torch.no_grad():
        logits_inference, loss_inference = model(input_ids)
    assert logits_inference.shape == (2, 32, 32000)
    assert loss_inference is None
