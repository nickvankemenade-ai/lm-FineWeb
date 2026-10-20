
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers import trainers, decoders, normalizers


# config

DATASET_PATH = Path("dataset/processed/fineweb_train.jsonl")
TOKENIZER_DIR = Path("tokenizer")
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"

VOCAB_SIZE = 32_000

SPECIAL_TOKENS = [
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
]


# iterator

def corpus_iterator(path):

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON on line "
                    f"{line_number:,}"
                )
                continue

            text = record.get("text", "")

            if isinstance(text, str) and text.strip():
                yield text


# training

def train_tokenizer():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print("Training BPE tokenizer")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Vocabulary size: {VOCAB_SIZE:,}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print()

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    tokenizer.normalizer = normalizers.NFC()

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Reading corpus and learning vocabulary...")
    tokenizer.train_from_iterator(
        corpus_iterator(DATASET_PATH),
        trainer=trainer,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print()
    print("Tokenizer training complete")
    print(f"Tokenizer saved to: {TOKENIZER_PATH.resolve()}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")

    return tokenizer


# testing

def test_tokenizer(tokenizer):
    examples = [
        "Hello, world!",
        "This is a test."
    ]

    print()
    print("=" * 60)
    print("Tokenizer tests")
    print("=" * 60)

    for text in examples:
        encoding = tokenizer.encode(text)

        print(f"\nText:   {text}")
        print(f"Tokens: {encoding.tokens}")
        print(f"IDs:    {encoding.ids}")
        print(f"Count:  {len(encoding.ids)}")


# main

if __name__ == "__main__":
    trained_tokenizer = train_tokenizer()
    test_tokenizer(trained_tokenizer)

