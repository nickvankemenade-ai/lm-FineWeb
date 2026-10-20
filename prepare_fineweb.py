import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


# config

DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "CC-MAIN-2024-10"

TARGET_DOCUMENTS = 4_700_000

OUTPUT_DIR = Path("dataset/processed")
TEST_OUTPUT = OUTPUT_DIR / "fineweb_test.jsonl"
FULL_OUTPUT = OUTPUT_DIR / "fineweb_train.jsonl"

SHUFFLE_BUFFER_SIZE = 10_000
RANDOM_SEED = 1337

PROGRESS_INTERVAL = 1_000


# cleaning

def clean_text(text):
    """Normalize line endings and remove excessive blank lines."""

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in text.split("\n")]

    result = []
    blank_count = 0

    for line in lines:
        if line.strip():
            blank_count = 0
            result.append(line)
        else:
            blank_count += 1

            if blank_count <= 2:
                result.append("")

    return "\n".join(result).strip()


# filter code

CODE_LINE_PATTERNS = [
    re.compile(r"^\s*def\s+\w+\s*\("),
    re.compile(r"^\s*class\s+\w+[\s(:]"),
    re.compile(r"^\s*function\s+\w+\s*\("),
    re.compile(r"^\s*(const|let|var)\s+\w+\s*="),
    re.compile(r"^\s*#include\s*[<\"]"),
    re.compile(r"^\s*<\?php"),
    re.compile(r"^\s*(SELECT|INSERT INTO|CREATE TABLE)\b", re.I),
    re.compile(r"^\s*import\s+\w+"),
    re.compile(r"^\s*from\s+\w+\s+import\s+"),
]

CODE_MARKERS = [
    "node_modules/",
    "<!DOCTYPE html>",
    "<?php",
    "#include <",
]


def is_code_heavy(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 5:
        return False

    code_lines = sum(
        any(pattern.search(line) for pattern in CODE_LINE_PATTERNS)
        for line in lines
    )

    code_ratio = code_lines / len(lines)

    if code_lines >= 5 and code_ratio >= 0.35:
        return True

    lower_text = text.lower()

    marker_count = sum(
        marker.lower() in lower_text
        for marker in CODE_MARKERS
    )

    if marker_count >= 2 and code_ratio >= 0.20:
        return True

    return False


# filters

def is_repetitive_or_corrupted(text):
    """Reject empty, highly repetitive, or obviously corrupted text."""

    if not text:
        return True

    if re.search(r"(.)\1{29,}", text):
        return True

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) >= 8:
        unique_ratio = len(set(lines)) / len(lines)

        if unique_ratio <= 0.30:
            return True

    if len(text) >= 500:
        characters = [
            character
            for character in text
            if not character.isspace()
        ]

        if not characters:
            return True

        alphabetic_ratio = (
            sum(character.isalpha() for character in characters)
            / len(characters)
        )

        symbol_ratio = (
            sum(not character.isalnum() for character in characters)
            / len(characters)
        )

        if alphabetic_ratio < 0.25 and symbol_ratio > 0.50:
            return True

    return False


def filter_document(text):
    """Return whether a document should be kept and the reason."""

    if not text:
        return False, "empty"

    if len(text) < 40 and len(text.split()) < 6:
        return False, "very_short"

    if is_code_heavy(text):
        return False, "code_heavy"

    if is_repetitive_or_corrupted(text):
        return False, "repetitive_or_corrupted"

    return True, "kept"


# counter

def count_existing_documents(path):

    if not path.exists():
        return 0

    count = 0

    print(f"Checking existing output: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if isinstance(record, dict) and isinstance(
                    record.get("text"), str
                ):
                    count += 1

            except json.JSONDecodeError:
                continue

    print(f"Existing valid documents: {count:,}")

    return count


# dataset preparation

def prepare_dataset(max_documents=None, resume=False):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    is_full_run = max_documents is None

    output_path = FULL_OUTPUT if is_full_run else TEST_OUTPUT

    existing_documents = 0

    if is_full_run and resume:
        existing_documents = count_existing_documents(output_path)

    if is_full_run and existing_documents >= TARGET_DOCUMENTS:
        print("The target document count has already been reached.")
        print(f"Output file: {output_path.resolve()}")
        return

    print()
    print("=" * 60)
    print("FineWeb dataset preparation")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Configuration: {DATASET_CONFIG}")
    print(f"Output: {output_path}")
    print(f"Resume mode: {resume}")

    if is_full_run:
        print(f"Target saved documents: {TARGET_DOCUMENTS:,}")
        print(f"Already saved: {existing_documents:,}")
        print(
            f"Remaining documents: "
            f"{TARGET_DOCUMENTS - existing_documents:,}"
        )
    else:
        print(f"Source document limit for this test: {max_documents:,}")

    print("=" * 60)
    print()

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=RANDOM_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    stats = Counter()

    documents_seen = 0
    documents_saved_this_run = 0
    documents_saved_total = existing_documents

    file_mode = "a" if is_full_run and resume else "w"

    try:
        with output_path.open(
            file_mode,
            encoding="utf-8",
            buffering=1024 * 1024,
        ) as output_file:

            for record in dataset:
                documents_seen += 1

                text = clean_text(record.get("text", ""))

                keep, reason = filter_document(text)
                stats[reason] += 1

                if keep:
                    output_record = {
                        "text": text,
                        "url": record.get("url", ""),
                        "dump": record.get("dump", ""),
                        "date": record.get("date", ""),
                        "language_score": record.get(
                            "language_score", 0.0
                        ),
                    }

                    output_file.write(
                        json.dumps(
                            output_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    documents_saved_this_run += 1
                    documents_saved_total += 1

                if documents_seen % PROGRESS_INTERVAL == 0:
                    if is_full_run:
                        print(
                            f"Scanned: {documents_seen:,} | "
                            f"Saved this run: "
                            f"{documents_saved_this_run:,} | "
                            f"Total saved: "
                            f"{documents_saved_total:,} / "
                            f"{TARGET_DOCUMENTS:,}"
                        )
                    else:
                        print(
                            f"Scanned: {documents_seen:,} | "
                            f"Saved: {documents_saved_this_run:,}"
                        )

                if not is_full_run:
                    if documents_seen >= max_documents:
                        break

                elif documents_saved_total >= TARGET_DOCUMENTS:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        print("The output file has been closed and flushed.")
        print("You can resume the full run with --full --resume.")
        raise

    print()
    print("=" * 60)
    print("Preparation finished")
    print("=" * 60)
    print(f"Documents examined this run: {documents_seen:,}")
    print(f"Documents saved this run: {documents_saved_this_run:,}")
    print(f"Total documents in target corpus: {documents_saved_total:,}")

    print("\nFiltering statistics:")

    for reason, count in sorted(stats.items()):
        print(f"  {reason}: {count:,}")

    print(f"\nOutput file: {output_path.resolve()}")

    if is_full_run and documents_saved_total < TARGET_DOCUMENTS:
        print(
            "\nThe dataset stream ended before reaching the target. "
            "You can try running again with --full --resume."
        )


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a filtered FineWeb corpus."
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Process documents until the target saved-document count.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20_000,
        help="Maximum source documents to examine in test mode.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the existing full output instead of overwriting it.",
    )

    args = parser.parse_args()

    if args.full:
        prepare_dataset(
            max_documents=None,
            resume=args.resume,
        )
    else:
        if args.resume:
            parser.error("--resume can only be used together with --full")

        prepare_dataset(
            max_documents=args.limit,
            resume=False,
        )
