import argparse
import json
import math
import os
import random
import tempfile
import textwrap
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml

from config import Config
from dataset import TextDataset
from generate import generate_tokens
from model import GPT

CHECKPOINT_DIR = "checkpoints"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the GPT language model.")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--a100",
        action="store_true",
        help="Load model and training settings from configs/a100.yaml.",
    )
    profile_group.add_argument(
        "--rtx",
        action="store_true",
        help="Load model and training settings from configs/rtx.yaml.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint for the selected profile.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile for faster CUDA training.",
    )
    parser.add_argument(
        "--compile-backend",
        choices=("auto", "inductor", "cudagraphs"),
        default="auto",
        help="Compiler backend; auto selects CUDA graphs when Triton is unavailable.",
    )
    return parser.parse_args()


def create_config(profile: str | None = None) -> Config:
    name = profile or "default"
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.is_file():
        raise ValueError(f"Unknown configuration profile: {name}")

    with config_path.open("r", encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)

    if not isinstance(settings, dict):
        raise ValueError(f"Expected a mapping of settings in {config_path}.")

    try:
        return Config(**settings)
    except TypeError as error:
        raise ValueError(f"Invalid settings in {config_path}: {error}") from error


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_grad_scaler(device: torch.device, amp_dtype):
    use_scaler = (
        device.type == "cuda"
        and amp_dtype == torch.float16
    )

    return torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )


def get_autocast_dtype(device: torch.device):
    if device.type != "cuda":
        return None

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def autocast_context(device: torch.device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
        )

    return nullcontext()


def get_lr(
    iteration: int,
    config: Config,
    total_iters: int | None = None,
) -> float:
    total_iters = total_iters if total_iters is not None else config.max_iters
    if total_iters is None:
        raise ValueError("total_iters is required when max_iters is not set")

    if iteration < config.warmup:
        return config.max_lr * (iteration + 1) / config.warmup

    if iteration >= total_iters:
        return config.min_lr

    decay_ratio = (
        (iteration - config.warmup)
        / (total_iters - config.warmup)
    )

    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return config.min_lr + coeff * (
        config.max_lr - config.min_lr
    )


def create_optimizer(
    model: GPT,
    config: Config,
    device: torch.device,
):
    use_fused = device.type == "cuda"

    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Matrix weights receive decoupled weight decay. Biases and 1-D
        # parameters (including normalization scales) do not.
        (decay_params if parameter.ndim >= 2 and not name.endswith("bias") else no_decay_params).append(parameter)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=use_fused,
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )

    return optimizer


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    amp_dtype,
    scaler=None,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    batch_supplier=None,
    microbatch_weights=None,
    return_grad_norm: bool = False,
) -> float:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")

    if grad_accum_steps > 1 and batch_supplier is None:
        raise ValueError(
            "batch_supplier is required when grad_accum_steps is greater than 1"
        )

    if microbatch_weights is not None:
        if len(microbatch_weights) != grad_accum_steps:
            raise ValueError("microbatch_weights must match grad_accum_steps")
        if not math.isclose(sum(microbatch_weights), 1.0, rel_tol=1e-6):
            raise ValueError("microbatch_weights must sum to 1")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for microstep in range(grad_accum_steps):
        if microstep > 0:
            x, y = batch_supplier()

        with autocast_context(device, amp_dtype):
            _, loss = model(x, y)

        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(
                f"Non-finite training loss: {loss.detach().item()}"
            )

        loss_weight = (
            microbatch_weights[microstep]
            if microbatch_weights is not None
            else 1.0 / grad_accum_steps
        )
        total_loss += loss.detach().float().item() * loss_weight
        scaled_loss = loss * loss_weight

        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        optimizer.step()

    if return_grad_norm:
        return total_loss, float(grad_norm.detach().item())
    return total_loss


@torch.no_grad()


def estimate_loss(
    model: GPT,
    dataset: TextDataset,
    config: Config,
    device: torch.device,
    amp_dtype,
    eval_batches: int | None = None,
) -> dict[str, float]:

    eval_batches = eval_batches or config.eval_batches

    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")

    results = {}
    was_training = model.training

    model.eval()

    try:
        for split in ("train", "val"):
            losses = []

            for _ in range(eval_batches):
                x, y = dataset.get_batch(
                    split=split,
                    batch_size=config.batch_size,
                    device=device,
                )

                with autocast_context(device, amp_dtype):
                    _, loss = model(x, y)

                losses.append(loss.float().item())

            results[split] = sum(losses) / len(losses)

    finally:
        model.train(was_training)

    return results


def save_checkpoint(
    model,
    optimizer,
    iteration,
    best_val_loss,
    config,
    checkpoint_path,
    scaler=None,
):
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }

    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(checkpoint_path)}.",
        suffix=".tmp",
        dir=checkpoint_dir,
    )
    os.close(file_descriptor)

    try:
        torch.save(checkpoint, temporary_path)

        for attempt in range(8):
            try:
                os.replace(temporary_path, checkpoint_path)
                break
            except PermissionError as error:
                if attempt == 7:
                    raise PermissionError(
                        f"Could not replace checkpoint '{checkpoint_path}' "
                        "after several retries. Another process may be "
                        "holding the file open; the previous checkpoint "
                        "has been preserved."
                    ) from error
                time.sleep(0.5 * (attempt + 1))
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
    scaler=None,
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Cannot resume: checkpoint not found at '{checkpoint_path}'. "
            "Run python train.py first to create a checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    saved_optimizer_state = checkpoint["optimizer_state_dict"]
    current_optimizer_state = optimizer.state_dict()
    if len(saved_optimizer_state["param_groups"]) != len(current_optimizer_state["param_groups"]):
        # Migrate older checkpoints with one all-parameter group to the new
        # decay/no-decay groups by matching each optimizer slot to its name.
        names_by_object = {id(parameter): name for name, parameter in model.named_parameters()}
        old_ids = saved_optimizer_state["param_groups"][0]["params"]
        old_names = [name for name, _ in model.named_parameters()]
        if len(old_ids) != len(old_names):
            raise ValueError("Cannot migrate optimizer state: parameter layout changed.")
        state_by_name = {
            name: saved_optimizer_state["state"][parameter_id]
            for name, parameter_id in zip(old_names, old_ids)
            if parameter_id in saved_optimizer_state["state"]
        }
        current_ids_by_name = {}
        for group, parameter_group in zip(current_optimizer_state["param_groups"], optimizer.param_groups):
            for parameter_id, parameter in zip(group["params"], parameter_group["params"]):
                current_ids_by_name[names_by_object[id(parameter)]] = parameter_id
        current_optimizer_state["state"] = {
            current_ids_by_name[name]: state
            for name, state in state_by_name.items()
            if name in current_ids_by_name
        }
        optimizer.load_state_dict(current_optimizer_state)
        print("Migrated optimizer state from the previous all-parameter decay layout.")
    else:
        optimizer.load_state_dict(saved_optimizer_state)

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_iteration = checkpoint["iteration"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Resuming at iteration: {start_iteration}")
    print(f"Best validation loss so far: {best_val_loss:.4f}")

    return start_iteration, best_val_loss


def format_training_log(record: dict) -> str:
    grad_norm = record["last_update_grad_norm_pre_clip"]
    last_loss = record["last_update_loss"]
    throughput = record["tokens_per_second_since_log"]
    progress = 100.0 * record["progress"]
    last_loss_text = f"{last_loss:.4f}" if last_loss is not None else "n/a"
    grad_norm_text = f"{grad_norm:.3f}" if grad_norm is not None else "n/a"
    throughput_text = f"{throughput:,.0f} tok/s" if throughput is not None else "n/a"
    return (
        f"\n{'─' * 76}\n"
        f" STEP {record['step']:,}/{record['total_steps']:,}  ({progress:5.1f}%)"
        f"                         elapsed {record['elapsed_seconds'] / 3600:.2f} h\n"
        f" Loss     train {record['train_loss_sampled']:.4f}"
        f"   val {record['validation_loss']:.4f}"
        f"   last batch {last_loss_text}\n"
        f" Update   lr {record['learning_rate']:.2e}"
        f"   grad norm {grad_norm_text}"
        f"   throughput {throughput_text}\n"
        f" Data     {record['tokens_processed']:,} tokens seen"
        f"   {record['tokens_per_update']:,} tokens/update"
        f"   eval {record['eval_batches_per_split']} batches/split\n"
        f"{'─' * 76}"
    )


def write_metric_record(metrics_file, record: dict) -> None:
    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics_file.flush()


def main() -> None:
    args = parse_args()

    profile = "a100" if args.a100 else "rtx" if args.rtx else None
    config = create_config(profile=profile)
    set_seed(config.seed)

    checkpoint_dir = (
        os.path.join(CHECKPOINT_DIR, profile)
        if profile is not None
        else CHECKPOINT_DIR
    )

    device = get_device()
    amp_dtype = get_autocast_dtype(device)

    dataset = TextDataset(config)

    epoch_mode = config.max_iters is None
    updates_per_epoch = None
    if epoch_mode:
        microbatches_per_epoch = dataset.training_batch_count(config.batch_size)
        updates_per_epoch = (
            microbatches_per_epoch + config.grad_accum_steps - 1
        ) // config.grad_accum_steps
        total_iters = updates_per_epoch * config.epochs
    else:
        total_iters = config.max_iters

    config.warmup = max(1, round(total_iters * config.warmup_fraction))

    if config.warmup >= total_iters:
        raise ValueError(
            f"warmup ({config.warmup}) must be less than the total "
            f"optimizer updates ({total_iters})"
        )

    model = GPT(config).to(device)

    optimizer = create_optimizer(
        model=model,
        config=config,
        device=device,
    )

    scaler = create_grad_scaler(
        device=device,
        amp_dtype=amp_dtype,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = os.path.join(
        checkpoint_dir,
        "latest.pt",
    )

    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        "best.pt",
    )
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.jsonl")
    rollouts_path = os.path.join(checkpoint_dir, "rollouts.jsonl")

    best_val_loss = float("inf")
    start_iteration = 0

    if args.resume:
        start_iteration, best_val_loss = load_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        if start_iteration >= total_iters:
            print(
                f"Training already completed: "
                f"checkpoint iteration {start_iteration}, "
                f"total optimizer updates={total_iters}."
            )
            return

    execution_model = model
    compile_backend = None
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "--compile requires PyTorch 2.0 or newer with torch.compile."
            )
        if device.type != "cuda":
            raise RuntimeError("--compile is currently supported for CUDA runs only.")

        compile_backend = args.compile_backend
        if compile_backend == "auto":
            try:
                import triton  # noqa: F401
            except ImportError:
                compile_backend = "cudagraphs"
                print(
                    "Triton is unavailable; selecting PyTorch's cudagraphs "
                    "backend instead of Inductor."
                )
            else:
                compile_backend = "inductor"
        print(
            f"Compiling model with torch.compile backend={compile_backend} "
            "(initial steps will be slower)..."
        )
        execution_model = torch.compile(model, backend=compile_backend)

    print("TRAINING STARTED")
    print(f"Device:               {device}")
    print(f"Mixed precision:      {amp_dtype or 'Disabled'}")
    print(f"Vocabulary size:      {config.vocab_size:,}")
    print(f"Context length:       {config.block_size:,}")
    print(f"Transformer layers:   {config.n_layers}")
    print(f"Attention heads:      {config.n_heads}")
    print(f"Embedding dimension:  {config.n_embd}")
    effective_batch_size = config.batch_size * config.grad_accum_steps
    tokens_per_update = effective_batch_size * config.block_size
    print(f"Microbatch size:      {config.batch_size}")
    print(f"Gradient accumulation: {config.grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size:,} sequences")
    print(f"Tokens per update:    {tokens_per_update:,}")
    if epoch_mode:
        print(f"Training epochs:      {config.epochs}")
        print(f"Updates per epoch:    {updates_per_epoch:,}")
    print(f"Training iterations:  {total_iters:,} optimizer updates")
    print(f"Maximum learning rate: {config.max_lr}")
    print(f"Minimum learning rate: {config.min_lr}")
    print(f"Warmup iterations:    {config.warmup:,}")
    print(f"Warmup fraction:      {config.warmup_fraction:.2%}")
    print(f"Weight decay:         {config.weight_decay}")
    print(f"Model parameters:     {model.count_parameters():,}")
    compile_label = f"Enabled ({compile_backend})" if args.compile else "Disabled"
    print(f"Torch compilation:    {compile_label}")
    print(f"Gradient scaler:      {scaler.is_enabled()}")
    profile_label = f"{profile or 'default'} (configs/{profile or 'default'}.yaml)"
    print(f"Configuration profile: {profile_label}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    print("\nBeginning optimization...\n")

    training_start_time = time.time()
    metrics_file = open(metrics_path, "a", encoding="utf-8", buffering=1)
    last_log_time = training_start_time
    last_train_loss = None
    last_grad_norm = None

    def iter_update_batches():
        if not epoch_mode:
            for _ in range(total_iters):
                yield None
            return

        for _ in range(config.epochs):
            batch_specs = iter(
                dataset.iter_train_batch_specs(config.batch_size)
            )
            for _ in range(updates_per_epoch):
                group = list(islice(batch_specs, config.grad_accum_steps))
                if not group:
                    break
                yield group

    for iteration, batch_group in enumerate(iter_update_batches()):
        if iteration < start_iteration:
            continue

        should_evaluate = (
            iteration % config.eval_interval == 0
            or iteration == total_iters - 1
        )

        if should_evaluate:
            try:
                losses = estimate_loss(
                    model=execution_model,
                    dataset=dataset,
                    config=config,
                    device=device,
                    amp_dtype=amp_dtype,
                )
            except Exception as error:
                if (
                    compile_backend != "inductor"
                    or "Cannot find a working triton installation" not in str(error)
                ):
                    raise
                print(
                    "Inductor could not initialize Triton; retrying this "
                    "evaluation with the cudagraphs backend."
                )
                torch._dynamo.reset()
                compile_backend = "cudagraphs"
                execution_model = torch.compile(model, backend=compile_backend)
                losses = estimate_loss(
                    model=execution_model,
                    dataset=dataset,
                    config=config,
                    device=device,
                    amp_dtype=amp_dtype,
                )

            current_time = time.time()
            elapsed = current_time - last_log_time
            total_elapsed = current_time - training_start_time

            lr_for_step = get_lr(iteration, config, total_iters=total_iters)
            record = {
                "record_type": "evaluation",
                "step": iteration,
                "total_steps": total_iters,
                "progress": iteration / total_iters,
                "learning_rate": lr_for_step,
                "train_loss_sampled": losses["train"],
                "validation_loss": losses["val"],
                "last_update_loss": last_train_loss,
                "last_update_grad_norm_pre_clip": last_grad_norm,
                "tokens_processed": iteration * tokens_per_update,
                "tokens_per_update": tokens_per_update,
                "effective_batch_sequences": effective_batch_size,
                "elapsed_seconds": total_elapsed,
                "seconds_since_log": elapsed,
                "tokens_per_second_since_log": (
                    config.eval_interval * tokens_per_update / elapsed
                    if elapsed > 0 and iteration > start_iteration
                    else None
                ),
                "eval_batches_per_split": config.eval_batches,
            }
            write_metric_record(metrics_file, record)
            print(format_training_log(record), flush=True)

            if iteration > 0 and iteration % config.rollout_interval == 0:
                previous_mode = model.training
                model.eval()
                prompt_ids = dataset.tokenizer.encode(config.rollout_prompt).ids
                rollout_ids = generate_tokens(
                    model=model,
                    prompt_ids=prompt_ids,
                    max_new_tokens=config.rollout_new_tokens,
                    temperature=0.0,
                    top_k=0,
                    eos_id=dataset.eos_id,
                    device=device,
                    autocast_dtype=amp_dtype,
                )
                model.train(previous_mode)
                rollout = {
                    "step": iteration,
                    "prompt": config.rollout_prompt,
                    "continuation": dataset.tokenizer.decode(rollout_ids, skip_special_tokens=True),
                    "temperature": 0.0,
                    "max_new_tokens": config.rollout_new_tokens,
                }
                with open(rollouts_path, "a", encoding="utf-8") as rollout_file:
                    rollout_file.write(json.dumps(rollout, ensure_ascii=False) + "\n")
                print(
                    f"\n{'─' * 76}\n"
                    f" ROLLOUT SNAPSHOT  |  step {iteration:,}\n"
                    f" Prompt: {config.rollout_prompt}\n\n"
                    f"{textwrap.fill(rollout['continuation'], width=76, initial_indent=' ', subsequent_indent=' ')}\n"
                    f"{'─' * 76}",
                    flush=True,
                )

            last_log_time = current_time

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    best_val_loss=best_val_loss,
                    config=config,
                    checkpoint_path=best_checkpoint_path,
                    scaler=scaler,
                )

                print(
                    f"  New best validation loss: "
                    f"{best_val_loss:.4f}"
                )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                best_val_loss=best_val_loss,
                config=config,
                checkpoint_path=latest_checkpoint_path,
                scaler=scaler,
            )

        step_started_at = time.perf_counter()
        update_tokens = tokens_per_update
        microbatch_weights = None
        if batch_group is None:
            def get_train_batch():
                return dataset.get_batch(
                    split="train",
                    batch_size=config.batch_size,
                    device=device,
                )

            x, y = get_train_batch()
            train_batch_supplier = get_train_batch
            accumulation_steps = config.grad_accum_steps
        else:
            spec_iterator = iter(batch_group)
            first_spec = next(spec_iterator)
            x, y = dataset.batch_from_spec(first_spec, device=device)

            target_token_counts = [
                len(starts) * seq_len
                for starts, seq_len in batch_group
            ]
            total_target_tokens = sum(target_token_counts)
            update_tokens = total_target_tokens
            microbatch_weights = [
                token_count / total_target_tokens
                for token_count in target_token_counts
            ]

            def get_next_epoch_batch():
                return dataset.batch_from_spec(
                    next(spec_iterator),
                    device=device,
                )

            train_batch_supplier = get_next_epoch_batch
            accumulation_steps = len(batch_group)

        learning_rate = get_lr(iteration, config, total_iters=total_iters)

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        train_loss, grad_norm = train_step(
            model=execution_model,
            optimizer=optimizer,
            x=x,
            y=y,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            grad_accum_steps=accumulation_steps,
            batch_supplier=train_batch_supplier,
            microbatch_weights=microbatch_weights,
            return_grad_norm=True,
        )
        last_train_loss = train_loss
        last_grad_norm = grad_norm

        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {iteration}: "
                f"{train_loss}"
            )

        update_seconds = max(time.perf_counter() - step_started_at, 1e-9)
        write_metric_record(metrics_file, {
            "record_type": "update",
            "step": iteration + 1,
            "total_steps": total_iters,
            "progress": (iteration + 1) / total_iters,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "grad_norm_pre_clip": grad_norm,
            "tokens_processed": (iteration + 1) * tokens_per_update,
            "tokens_this_update": update_tokens,
            "tokens_per_update": tokens_per_update,
            "tokens_per_second": update_tokens / update_seconds,
            "update_seconds": update_seconds,
        })

    metrics_file.close()
    final_iteration = total_iters

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=final_iteration,
        best_val_loss=best_val_loss,
        config=config,
        checkpoint_path=latest_checkpoint_path,
        scaler=scaler,
    )

    total_time = time.time() - training_start_time

    print("\nTRAINING COMPLETE")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"Best checkpoint:   {best_checkpoint_path}")


if __name__ == "__main__":
    main()
