import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from model import GPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a validation-best GPT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to load. Defaults to the newest checkpoints/**/best.pt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each prompt (default: 200).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; use 0 for greedy decoding (default: 0.8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Sample only from the top K logits; use 0 to disable (default: 50).",
    )
    return parser.parse_args()


def find_newest_checkpoint(project_dir: Path) -> Path:
    checkpoint_dir = project_dir / "checkpoints"
    candidates = list(checkpoint_dir.glob("**/best.pt"))

    if not candidates:
        raise FileNotFoundError(
            f"No best.pt checkpoint found under '{checkpoint_dir}'. "
            "Train a model first, or pass --checkpoint PATH."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        # Compatibility with PyTorch versions that do not support mmap or
        # weights_only in torch.load.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def get_model_config(saved_config):
    if isinstance(saved_config, dict):
        values = saved_config
    else:
        values = vars(saved_config)

    required = (
        "vocab_size",
        "block_size",
        "n_layers",
        "n_heads",
        "n_embd",
        "dropout",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(
            "Checkpoint config is missing model settings: "
            + ", ".join(missing)
        )

    return SimpleNamespace(**{name: values[name] for name in required})


def choose_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    logits = logits.float()

    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.inference_mode()
def generate_tokens(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int | None,
    device: torch.device,
    autocast_dtype,
    token_callback=None,
) -> list[int]:
    generated = []
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = tokens[:, -model.block_size :]

        if device.type == "cuda" and autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, _ = model(context)
        else:
            logits, _ = model(context)

        next_token = choose_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
        )
        token_id = next_token.item()
        generated.append(token_id)
        tokens = torch.cat((tokens, next_token), dim=1)

        if token_callback is not None:
            token_callback(generated)

        if eos_id is not None and token_id == eos_id:
            break

    return generated


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")

    project_dir = Path(__file__).resolve().parent
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else find_newest_checkpoint(project_dir)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            f"'{checkpoint_path}' is not a training checkpoint with a model "
            "state and config."
        )

    config = get_model_config(checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    autocast_dtype = None
    if device.type == "cuda":
        autocast_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    tokenizer_path = project_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match the checkpoint model: "
            f"{tokenizer.get_vocab_size()} != {config.vocab_size}"
        )

    eos_id = tokenizer.token_to_id("<|eos|>")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | parameters: {model.count_parameters():,}")
    print("Enter a prompt to generate text, or /quit to exit.")

    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.strip().lower() == "/quit":
            break

        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            print("Please enter some text first.")
            continue

        printed_chars = 0

        def stream_decoded_text(generated_ids):
            nonlocal printed_chars
            decoded_so_far = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            # Keep a few trailing characters buffered in case a token ends in
            # the middle of a multi-byte UTF-8 character.
            stable_length = max(0, len(decoded_so_far) - 4)
            if stable_length > printed_chars:
                print(
                    decoded_so_far[printed_chars:stable_length],
                    end="",
                    flush=True,
                )
                printed_chars = stable_length

        print("\n", end="", flush=True)
        generated_ids = generate_tokens(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=eos_id,
            device=device,
            autocast_dtype=autocast_dtype,
            token_callback=stream_decoded_text,
        )
        continuation = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        print(continuation[printed_chars:])


if __name__ == "__main__":
    main()
