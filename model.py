import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + self.eps)

        output = x_fp32 * rms * self.weight.float()
        return output.to(dtype=x.dtype)


class RoPE(nn.Module):

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        angles = torch.cat((angles, angles), dim=-1)

        self.register_buffer(
            "cos_cached",
            angles.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            angles.sin(),
            persistent=False,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        seq_len = x.size(-2)
        end_position = position_offset + seq_len

        if position_offset < 0 or end_position > self.max_seq_len:
            raise ValueError(
                f"RoPE positions [{position_offset}, {end_position}) exceed "
                f"the supported range [0, {self.max_seq_len})."
            )

        cos = self.cos_cached[position_offset:end_position]
        sin = self.sin_cached[position_offset:end_position]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return (x * cos + self._rotate_half(x) * sin).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()

        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads.")

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.block_size = block_size
        self.dropout = dropout

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=True)

        self.rope = RoPE(
            head_dim=self.head_dim,
            max_seq_len=block_size,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block size "
                f"{self.block_size}."
            )

        qkv = self.qkv_proj(x)
        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_embd)

        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class SwiGLU(nn.Module):

    def __init__(
        self,
        n_embd: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=True)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        x = gate * value
        x = self.down_proj(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        mlp_hidden_dim: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()

        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            block_size=block_size,
            dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )

        self.ln_2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(
            n_embd=n_embd,
            hidden_dim=mlp_hidden_dim,
            dropout=resid_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_layers = config.n_layers
        self.n_embd = config.n_embd

        mlp_hidden_dim = (8 * config.n_embd + 2) // 3

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_heads=config.n_heads,
                block_size=config.block_size,
                mlp_hidden_dim=mlp_hidden_dim,
                attn_dropout=config.dropout,
                resid_dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.initialize_residual_projections()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch_size, seq_len].")

        batch_size, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block size "
                f"{self.block_size}."
            )

        if idx.dtype != torch.long:
            raise TypeError("idx must contain torch.long token IDs.")

        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx.")

        x = self.token_embedding(idx)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size).float(),
                targets.reshape(-1),
            )

        return logits, loss

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if not trainable_only or p.requires_grad
        )

    def initialize_residual_projections(self) -> None:
        scale = (2 * self.n_layers) ** -0.5

        with torch.no_grad():
            for block in self.blocks:
                nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
                nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=0.02 * scale)