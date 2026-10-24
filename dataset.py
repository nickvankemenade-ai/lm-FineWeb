import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


class TextDataset:

    def __init__(
        self,
        config,
        dataset_path="dataset/processed/fineweb_train.jsonl",
        tokenizer_path="tokenizer/tokenizer.json",
        cache_dir="dataset/encoded",
    ):
        self.config = config
        self.dataset_path = Path(dataset_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.cache_dir = Path(cache_dir)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path.resolve()}"
            )

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {self.tokenizer_path.resolve()}"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_path)
        )

        if self.tokenizer.get_vocab_size() != config.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary size does not match config.vocab_size: "
                f"{self.tokenizer.get_vocab_size()} != {config.vocab_size}"
            )

        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        if self.eos_id is None:
            raise ValueError(
                "Tokenizer does not contain the <|eos|> special token."
            )

        self.cache_path = self.cache_dir / "fineweb_tokens.bin"

        self.tokens = self._load_or_encode()

        if len(self.tokens) <= config.block_size + 1:
            raise ValueError(
                "The encoded corpus is too small for the configured "
                "block_size."
            )

        split_index = int(len(self.tokens) * config.train_split)

        self.train_tokens = self.tokens[:split_index]
        self.val_tokens = self.tokens[split_index:]

        if len(self.train_tokens) <= config.block_size:
            raise ValueError("Training split is too small for block_size.")

        if len(self.val_tokens) <= config.block_size:
            raise ValueError("Validation split is too small for block_size.")

        print("\nDataset ready.")
        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Training tokens: {len(self.train_tokens):,}")
        print(f"Validation tokens: {len(self.val_tokens):,}")

    def _load_or_encode(self):

        if self.cache_path.exists():
            print(f"Loading cached tokens: {self.cache_path}")

            tokens = np.memmap(
                self.cache_path,
                dtype=np.uint16,
                mode="r",
            )

            if len(tokens) == 0:
                raise ValueError(
                    "The token cache is empty. Delete it and re-encode."
                )

            return tokens

        print("Encoding corpus. This may take a while...")

        dtype = np.uint16
        buffer = []
        total_tokens = 0
        documents = 0

        with self.cache_path.open("wb") as output_file:
            with self.dataset_path.open(
                "r",
                encoding="utf-8",
            ) as input_file:

                for line_number, line in enumerate(
                    input_file,
                    start=1,
                ):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"Warning: skipping invalid JSON "
                            f"on line {line_number:,}"
                        )
                        continue

                    text = record.get("text", "")

                    if not isinstance(text, str) or not text.strip():
                        continue

                    encoding = self.tokenizer.encode(text)
                    buffer.extend(encoding.ids)
                    buffer.append(self.eos_id)

                    documents += 1

                    if len(buffer) >= 1_000_000:
                        array = np.asarray(buffer, dtype=dtype)
                        array.tofile(output_file)

                        total_tokens += len(array)
                        buffer.clear()

                    if documents % 10_000 == 0:
                        print(
                            f"Documents: {documents:,} | "
                            f"Tokens written: {total_tokens:,}"
                        )

            if buffer:
                array = np.asarray(buffer, dtype=dtype)
                array.tofile(output_file)
                total_tokens += len(array)

        print("Encoding complete.")
        print(f"Documents encoded: {documents:,}")
        print(f"Tokens written: {total_tokens:,}")

        if total_tokens == 0:
            self.cache_path.unlink(missing_ok=True)
            raise ValueError("No tokens were produced from the corpus.")

        return np.memmap(
            self.cache_path,
            dtype=np.uint16,
            mode="r",
        )

    def get_batch(self, split="train", batch_size=None, device="cpu"):

        if split == "train":
            data = self.train_tokens
        elif split in ("val", "validation"):
            data = self.val_tokens
        else:
            raise ValueError(
                "split must be 'train', 'val', or 'validation'"
            )

        if batch_size is None:
            batch_size = self.config.batch_size

        block_size = self.config.block_size

        if len(data) <= block_size:
            raise ValueError(
                f"The {split} split has too few tokens "
                f"for block_size={block_size}."
            )

        starts = torch.randint(
            low=0,
            high=len(data) - block_size,
            size=(batch_size,),
        ).tolist()

        x = np.stack(
            [data[start : start + block_size] for start in starts]
        ).astype(np.int64)

        y = np.stack(
            [
                data[start + 1 : start + block_size + 1]
                for start in starts
            ]
        ).astype(np.int64)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x.to(device), y.to(device)

    def training_batch_count(self, batch_size: int) -> int:
        """Return the number of sequential microbatches in one train epoch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        transitions = len(self.train_tokens) - 1
        full_chunks, remainder = divmod(transitions, self.config.block_size)
        full_batches = (full_chunks + batch_size - 1) // batch_size
        return full_batches + int(remainder > 0)

    def iter_train_batch_specs(self, batch_size: int):
        """Yield sequential token-window descriptions for one complete pass."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        block_size = self.config.block_size
        transitions = len(self.train_tokens) - 1
        full_chunks, remainder = divmod(transitions, block_size)

        for first_chunk in range(0, full_chunks, batch_size):
            last_chunk = min(first_chunk + batch_size, full_chunks)
            starts = np.arange(first_chunk, last_chunk, dtype=np.int64)
            starts *= block_size
            yield starts, block_size

        if remainder:
            yield np.asarray([full_chunks * block_size], dtype=np.int64), remainder

    def batch_from_spec(self, spec, device="cpu"):
        """Materialize one sequential microbatch from an epoch batch spec."""
        starts, seq_len = spec
        x = np.stack([
            self.train_tokens[start : start + seq_len]
            for start in starts
        ]).astype(np.int64)
        y = np.stack([
            self.train_tokens[start + 1 : start + seq_len + 1]
            for start in starts
        ]).astype(np.int64)

        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
