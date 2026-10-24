import torch
from model import RMSNorm, RoPE

from config import Config

config = Config(
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

# RMSNorm
x = torch.randn(2, 4, 128)
norm = RMSNorm(128)

y = norm(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

# RoPE
rope = RoPE(head_dim=32, max_seq_len=128)
q = torch.randn(2, 4, 16, 32)

q_rotated = rope(q)

assert q_rotated.shape == q.shape
assert torch.isfinite(q_rotated).all()

# Verify gradients propagate.
loss = y.square().mean() + q_rotated.square().mean()
loss.backward()

assert norm.weight.grad is not None

print("RMSNorm and RoPE tests passed.")

import torch
from model import CausalSelfAttention

torch.manual_seed(42)

attention = CausalSelfAttention(
    n_embd=128,
    n_heads=4,
    block_size=64,
    dropout=0.1,
    resid_dropout=0.1,
)

x = torch.randn(2, 32, 128, requires_grad=True)

# Forward pass
y = attention(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

# Backward pass
loss = y.square().mean()
loss.backward()

assert x.grad is not None
assert attention.qkv_proj.weight.grad is not None
assert attention.out_proj.weight.grad is not None

# Evaluation should disable dropout.
attention.eval()
with torch.no_grad():
    y1 = attention(x)
    y2 = attention(x)

assert torch.allclose(y1, y2)

print("Attention tests passed.")

if torch.cuda.is_available():
    device = torch.device("cuda")

    attention = attention.to(device)
    x_gpu = torch.randn(2, 32, 128, device=device)

    with torch.no_grad():
        y_gpu = attention(x_gpu)

    assert y_gpu.shape == x_gpu.shape
    assert torch.isfinite(y_gpu).all()

    print("GPU attention test passed.")

import torch
from model import SwiGLU

torch.manual_seed(42)

mlp = SwiGLU(
    n_embd=128,
    hidden_dim=352,
    dropout=0.1,
)

x = torch.randn(2, 32, 128, requires_grad=True)

# Forward pass
y = mlp(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

# Backward pass
loss = y.square().mean()
loss.backward()

assert x.grad is not None
assert mlp.gate_proj.weight.grad is not None
assert mlp.up_proj.weight.grad is not None
assert mlp.down_proj.weight.grad is not None

# Evaluation disables dropout.
mlp.eval()

with torch.no_grad():
    y1 = mlp(x)
    y2 = mlp(x)

assert torch.allclose(y1, y2)

print("SwiGLU tests passed.")

import torch
from model import GPT

torch.manual_seed(42)

model = GPT(config)

# Two sequences, each containing 32 tokens.
input_ids = torch.randint(0, 32000, (2, 32))
targets = torch.randint(0, 32000, (2, 32))

# Forward pass with targets.
logits, loss = model(input_ids, targets)

assert logits.shape == (2, 32, 32000)
assert loss is not None
assert loss.ndim == 0
assert torch.isfinite(logits).all()
assert torch.isfinite(loss)

# Backward pass.
loss.backward()

assert model.token_embedding.weight.grad is not None
assert model.blocks[0].attn.qkv_proj.weight.grad is not None
assert model.blocks[0].mlp.gate_proj.weight.grad is not None
assert model.lm_head.weight.grad is not None

# Forward pass without targets.
model.eval()

with torch.no_grad():
    logits_inference, loss_inference = model(input_ids)

assert logits_inference.shape == (2, 32, 32000)
assert loss_inference is None

print("Full model tests passed.")
print(f"Parameters: {model.count_parameters():,}")

import torch

for i, block in enumerate(model.blocks):
    attn_std = block.attn.out_proj.weight.std().item()
    mlp_std = block.mlp.down_proj.weight.std().item()

    print(
        f"Block {i}: "
        f"attention output std={attn_std:.5f}, "
        f"MLP output std={mlp_std:.5f}"
    )


from config import Config
from dataset import TextDataset

config = Config(
    vocab_size=32000,
    block_size=64,
    n_layers=2,
    n_heads=4,
    n_embd=128,
    dropout=0.1,
    batch_size=4,
    max_iters=1000,
    max_lr=3e-4,
    min_lr=3e-5,
    warmup=100,
    weight_decay=0.1,
    eval_interval=100,
    train_split=0.99,
    seed=42,
)

dataset = TextDataset(config)

x, y = dataset.get_batch("train")

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Input dtype:", x.dtype)
print("Target dtype:", y.dtype)
print("First input:", x[0])
print("First target:", y[0])