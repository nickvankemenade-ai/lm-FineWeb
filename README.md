# lm

A decoder-only GPT language model trained from scratch on a filtered
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) corpus with a custom byte-level
BPE tokenizer.

The model uses RMSNorm, rotary position embeddings (RoPE), SwiGLU feed-forward layers and
PyTorch's fused `scaled_dot_product_attention`.

## Repository layout

```
├── configs/             Training profiles (default, rtx, a100)
├── tests/               Unit tests and a training/resume smoke test
├── tokenizer/           Trained BPE tokenizer (tokenizer.json)
├── config.py            Config dataclass and validation
├── model.py             GPT model
├── dataset.py           Token caching and batch sampling
├── prepare_fineweb.py   Download and filter FineWeb
├── train_tokenizer.py   Train the BPE tokenizer
├── train.py             Training loop
├── generate.py          Interactive text generation
└── dashboard.py         Live training dashboard
```

`dataset/` and `checkpoints/` are created at runtime and are not tracked.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

1. **Prepare data.** Writes `dataset/processed/fineweb_train.jsonl` (`--full`), or a small test
   file by default.

   ```bash
   python prepare_fineweb.py --full
   ```

2. **Train the tokenizer** (optional, `tokenizer/tokenizer.json` is already included).

   ```bash
   python train_tokenizer.py
   ```

3. **Train.** Without a flag, the tiny `configs/default.yaml` profile is used. The corpus is
   encoded to `dataset/encoded/fineweb_tokens.bin` on the first run.

   ```bash
   python train.py --rtx --compile
   ```

   | Flag                | Description                                                      |
   | ------------------- | ---------------------------------------------------------------- |
   | `--rtx` / `--a100`  | Load `configs/rtx.yaml` or `configs/a100.yaml`                   |
   | `--resume`          | Continue from `checkpoints/<profile>/latest.pt`                  |
   | `--compile`         | Use `torch.compile` (CUDA only)                                  |
   | `--compile-backend` | `auto` (default), `inductor` or `cudagraphs`                     |

   Checkpoints, metrics (`training_metrics.jsonl`) and rollout samples (`rollouts.jsonl`) are
   written to `checkpoints/<profile>/`.

4. **Monitor.** Serves a live dashboard at <http://127.0.0.1:8765>.

   ```bash
   python dashboard.py --profile rtx
   ```

5. **Generate.** Loads the newest `checkpoints/**/best.pt` unless `--checkpoint` is given.

   ```bash
   python generate.py --temperature 0.8 --top-k 50
   ```

## Tests

```bash
pytest
```

The smoke test trains a small model on the first 10M encoded tokens and checks that checkpoints
resume correctly. It needs `dataset/encoded/fineweb_tokens.bin`.

```bash
python -m tests.smoke_resume
```
