import json
import random
import os
import argparse

def split_data(character_name):
    # Determine input file
    # Priority: augmented (largest) > full > augmented_curated > base
    base_path = f"raw_data/training_data_{character_name}"
    if os.path.exists(f"{base_path}_augmented.jsonl"):
        input_file = f"{base_path}_augmented.jsonl"
    elif os.path.exists(f"{base_path}_full.jsonl"):
        input_file = f"{base_path}_full.jsonl"
    elif os.path.exists(f"{base_path}_augmented_curated.jsonl"):
        input_file = f"{base_path}_augmented_curated.jsonl"
    else:
        input_file = f"{base_path}.jsonl"

    # Output to character-specific directory to avoid overwrites and allow resuming
    output_dir = os.path.join("raw_data", character_name)
    
    train_file = os.path.join(output_dir, "train.jsonl")
    valid_file = os.path.join(output_dir, "valid.jsonl")
    split_ratio = 0.9

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = [json.loads(line) for line in lines]
    
    # Use fixed seed for reproducibility so resuming training works correctly
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    
    print(f"Total conversations: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    
    print(f"Writing to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Writing to {valid_file}...")
    with open(valid_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split training data into train/valid sets")
    parser.add_argument("--character", type=str, required=True, help="Character name (e.g. baseline)")
    args = parser.parse_args()
    
    split_data(args.character)
