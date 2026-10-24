"""
Smoke test for training and checkpoint resumption using the first 10 million
tokens from dataset/encoded/fineweb_tokens.bin.

Run from the project root:
    python smoke_resume.py

This uses the real token binary but only samples from its first 10M tokens.
It does not modify the binary or your project checkpoints. Checkpoints are stored
in a temporary directory and removed automatically when the test finishes.
"""

import os
import time
import numpy as np
import torch

from config import Config
from dataset import TextDataset
from model import GPT
from train import (
    create_grad_scaler,
    create_optimizer,
    get_autocast_dtype,
    get_device,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


TOKEN_BIN = "dataset/encoded/fineweb_tokens.bin"
TOKEN_LIMIT = 10_000_000


def make_test_config() -> Config:
    """Small smoke-test model; does not change the project's config.py."""
    return Config(
        vocab_size=32000,
        block_size=512,
        n_layers=2,
        n_heads=4,
        n_embd=128,
        dropout=0.1,
        batch_size=4,
        max_iters=1000,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup=2,
        weight_decay=0.1,
        eval_interval=2,
        train_split=0.99,
        seed=42,
    )


class FirstTokensDataset:
    """Batch sampler over the first TOKEN_LIMIT token IDs in the existing .bin."""

    def __init__(self, config: Config, token_path: str, token_limit: int):
        if not os.path.isfile(token_path):
            raise FileNotFoundError(
                f"Token binary not found: {token_path}\n"
                "Wait for token encoding to finish, or check TOKEN_BIN."
            )

        file_size = os.path.getsize(token_path)
        token_count_in_file = file_size // np.dtype(np.uint16).itemsize
        if token_count_in_file < token_limit:
            raise ValueError(
                f"The binary currently contains {token_count_in_file:,} complete "
                f"tokens, fewer than the requested {token_limit:,}. "
                "Wait until encoding has written at least 10M tokens."
            )

        self.config = config
        self.tokens = np.memmap(
            token_path,
            dtype=np.uint16,
            mode="r",
            shape=(token_count_in_file,),
        )
        self.token_count = token_limit
        self.train_end = int(token_limit * config.train_split)

        if self.train_end <= config.block_size + 1:
            raise ValueError("Not enough tokens for the configured block size.")

        print(f"Token binary: {token_path}")
        print(f"Tokens visible in file: {token_count_in_file:,}")
        print(f"Tokens used by this test: {self.token_count:,}")
        print(f"Train split tokens: {self.train_end:,}")
        print(f"Validation split tokens: {self.token_count - self.train_end:,}")

    def get_batch(self, split: str, batch_size: int, device: torch.device):
        if split == "train":
            data_start = 0
            data_end = self.train_end
        elif split == "val":
            data_start = self.train_end
            data_end = self.token_count
        else:
            raise ValueError(f"Unknown split: {split}")

        # Each sample needs block_size input tokens and one shifted target token.
        max_start_exclusive = data_end - self.config.block_size
        min_start = data_start
        if max_start_exclusive <= min_start:
            raise ValueError(f"Not enough tokens in {split} split.")

        starts = np.random.randint(
            min_start,
            max_start_exclusive,
            size=batch_size,
        )

        x_np = np.stack([
            np.asarray(self.tokens[s:s + self.config.block_size], dtype=np.int64)
            for s in starts
        ])
        y_np = np.stack([
            np.asarray(
                self.tokens[s + 1:s + self.config.block_size + 1],
                dtype=np.int64,
            )
            for s in starts
        ])

        x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
        return x, y


@torch.no_grad()
def estimate_smoke_loss(model, dataset, config, device, amp_dtype, batches=2):
    was_training = model.training
    model.eval()
    results = {}

    try:
        for split in ("train", "val"):
            losses = []
            for _ in range(batches):
                x, y = dataset.get_batch(split, config.batch_size, device)
                if device.type == "cuda" and amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                losses.append(loss.float().item())
            results[split] = sum(losses) / len(losses)
    finally:
        model.train(was_training)

    return results


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    config = make_test_config()
    device = get_device()
    amp_dtype = get_autocast_dtype(device)

    print("TRAINING + RESUME SMOKE TEST")
    print(f"Device: {device}")
    print(f"Mixed precision: {amp_dtype or 'Disabled'}")

    dataset = FirstTokensDataset(config, TOKEN_BIN, TOKEN_LIMIT)

    model = GPT(config).to(device)
    optimizer = create_optimizer(model, config, device)
    scaler = create_grad_scaler(device, amp_dtype)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"GradScaler enabled: {scaler.is_enabled()}")

    # Evaluate once before training to verify real-token batches and loss work.
    initial_losses = estimate_smoke_loss(
        model, dataset, config, device, amp_dtype, batches=100
    )
    print(
        f"Initial loss | train: {initial_losses['train']:.4f} | "
        f"val: {initial_losses['val']:.4f}"
    )

    # Save only in a temporary directory so the real project checkpoints stay safe.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gpt_resume_smoke_") as temp_dir:
        latest_path = os.path.join(temp_dir, "latest.pt")
        best_path = os.path.join(temp_dir, "best.pt")
        best_val_loss = float("inf")
        start_iteration = 0
        start_time = time.time()

        # Short run using actual token batches.
        for iteration in range(start_iteration, config.max_iters):
            if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
                losses = estimate_smoke_loss(
                    model, dataset, config, device, amp_dtype, batches=2
                )
                print(
                    f"Step {iteration:>3} | train: {losses['train']:.4f} | "
                    f"val: {losses['val']:.4f}"
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                        best_val_loss=best_val_loss,
                        config=config,
                        checkpoint_path=best_path,
                        scaler=scaler,
                    )

                # Like train.py, this checkpoint stores the next iteration
                # to execute because saving happens before the current update.
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    best_val_loss=best_val_loss,
                    config=config,
                    checkpoint_path=latest_path,
                    scaler=scaler,
                )

            x, y = dataset.get_batch("train", config.batch_size, device)
            lr = get_lr(iteration, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss = train_step(
                model=model,
                optimizer=optimizer,
                x=x,
                y=y,
                device=device,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            print(f"  update {iteration:>3} | loss: {loss:.4f} | lr: {lr:.2e}")

        # Save completed position using the NEXT iteration number.
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=config.max_iters,
            best_val_loss=best_val_loss,
            config=config,
            checkpoint_path=latest_path,
            scaler=scaler,
        )

        # Restore into newly created model/optimizer/scaler, just as a fresh
        # process would do when --resume is used.
        resumed_model = GPT(config).to(device)
        resumed_optimizer = create_optimizer(resumed_model, config, device)
        resumed_scaler = create_grad_scaler(device, amp_dtype)

        resumed_iteration, resumed_best_loss = load_checkpoint(
            checkpoint_path=latest_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
            device=device,
            scaler=resumed_scaler,
        )

        assert resumed_iteration == config.max_iters, (
            f"Expected next iteration {config.max_iters}, got {resumed_iteration}"
        )
        assert abs(resumed_best_loss - best_val_loss) < 1e-9

        for name, expected in model.state_dict().items():
            actual = resumed_model.state_dict()[name]
            assert torch.equal(expected, actual), f"Model tensor mismatch: {name}"

        # Confirm the restored model can still execute a forward pass on real tokens.
        x, y = dataset.get_batch("val", config.batch_size, device)
        resumed_model.eval()
        with torch.no_grad():
            if device.type == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, resumed_loss = resumed_model(x, y)
            else:
                _, resumed_loss = resumed_model(x, y)

        assert torch.isfinite(resumed_loss), "Resumed model produced non-finite loss."

        print(f"\nElapsed time: {(time.time() - start_time) / 60:.2f} minutes")
        print("PASS: first 10M tokens loaded via memory mapping")
        print("PASS: real-token forward/evaluation and training updates")
        print("PASS: temporary checkpoints saved and loaded")
        print("PASS: iteration and best validation loss restored")
        print("PASS: model weights restored exactly")
        print("PASS: resumed model forward pass on real tokens")

    print("\nSMOKE TEST PASSED")
    print("Your original token binary and checkpoints were not modified.")


if __name__ == "__main__":
    main()
