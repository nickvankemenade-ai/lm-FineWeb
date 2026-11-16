from dataclasses import dataclass


@dataclass
class Config:
    # Model
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    dropout: float

    # Training
    batch_size: int
    max_iters: int | None
    max_lr: float
    min_lr: float
    warmup: int
    weight_decay: float
    eval_interval: int
    train_split: float
    seed: int
    warmup_fraction: float = 0.02
    grad_accum_steps: int = 1
    epochs: int = 1
    eval_batches: int = 100
    rollout_interval: int = 1000
    rollout_prompt: str = "The most important lesson I learned was"
    rollout_new_tokens: int = 120

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")

        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")

        if self.n_embd % self.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

        if self.max_iters is not None and self.max_iters <= 0:
            raise ValueError("max_iters must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.max_lr <= 0 or self.min_lr < 0:
            raise ValueError(
                "Learning rates must be non-negative, with max_lr > 0"
            )

        if self.min_lr > self.max_lr:
            raise ValueError("min_lr cannot exceed max_lr")

        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")

        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be between 0 and 1")

        if self.max_iters is not None and self.warmup >= self.max_iters:
            raise ValueError("warmup must be less than max_iters")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if self.eval_batches <= 0 or self.rollout_interval <= 0:
            raise ValueError("eval_batches and rollout_interval must be positive")

        if self.rollout_new_tokens <= 0:
            raise ValueError("rollout_new_tokens must be positive")

        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
