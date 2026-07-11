import json
import random
import os
import argparse


def split_data(input_file: str, output_dir: str, split_ratio: float = 0.9, seed: int = 42):
    """Split a JSONL file into train/valid files at a fixed random seed.

    Single source of truth for the 90/10 split used by prepare_all_datasets.sh
    and the legacy CLI invocation. The split_ratio and seed are fixed (not
    configurable from the CLI) to keep training runs reproducible across the
    repo — change them here if you want a different split.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.jsonl")
    valid_file = os.path.join(output_dir, "valid.jsonl")

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = [json.loads(line) for line in lines]

    # Fixed seed for reproducibility so resuming training works correctly.
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    print(f"Total conversations: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")

    print(f"Writing to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    print(f"Writing to {valid_file}...")
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")

    print("Done.")


def _resolve_legacy_input(character_name: str) -> str:
    """Pick the highest-priority source file for legacy --character usage.
    New callers should pass --input explicitly so the priority is unambiguous."""
    base = f"raw_data/training_data_{character_name}"
    candidates = [
        f"{base}_augmented_curated.jsonl",  # curated first
        f"{base}_augmented.jsonl",
        f"{base}_full.jsonl",
        f"{base}.jsonl",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return f"{base}.jsonl"  # fall through; split_data will report "not found"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split training data into train/valid sets (90/10, seed 42)."
    )
    # Either pass --input + --output-dir explicitly, or pass --character for
    # the legacy single-character convenience mode that writes to
    # raw_data/<character>/.
    parser.add_argument("--input", help="Source JSONL file to split")
    parser.add_argument("--output-dir", help="Destination directory for train.jsonl + valid.jsonl")
    parser.add_argument("--character", help="Legacy: resolve input via priority + write to raw_data/<character>/")
    args = parser.parse_args()

    if args.input and args.output_dir:
        split_data(args.input, args.output_dir)
    elif args.character:
        input_file = _resolve_legacy_input(args.character)
        output_dir = os.path.join("raw_data", args.character)
        split_data(input_file, output_dir)
    else:
        parser.error("Provide either --input and --output-dir, or --character (legacy mode).")