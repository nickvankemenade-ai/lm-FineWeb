from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    dropout: float
    batch_size: int
    max_iters: int
    max_lr: float
    min_lr: float
    warmup: int
    weight_decay: float
    eval_interval: int
    train_split: float
    seed: int

    def __post_init__(self):
        if self.n_embd % self.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        