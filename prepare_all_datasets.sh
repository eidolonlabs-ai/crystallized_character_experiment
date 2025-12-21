#!/bin/bash
# ============================================================================
# Prepare All Datasets Script
# ============================================================================
# This script generates all dataset variants once and saves them to
# prepared_data/ directory. Data is versioned and consistent for all runs.
#
# Generates:
# - Split variants: base, augmented, augmented_curated (90/10 train/valid)
# - Pre-truncated at 512 tokens (intelligent truncation preserving context)
#
# Truncation strategy:
# - Keeps system prompt (character definition)
# - Works backwards to preserve recent conversation context
# - Truncates assistant response if needed (not conversation history)
#
# Output structure:
# raw_data/prepared_data/
# ├── baseline/
# │   ├── base_split/
# │   ├── base_split_truncated/
# │   ├── augmented_split/
# │   ├── augmented_split_truncated/
# │   └── full_split_truncated/
#
# Usage:
#   ./prepare_all_datasets.sh
#
# Then commit to git to lock in data consistency.
# ============================================================================

set -e

cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "============================================================================"
echo "Preparing All Datasets with Pre-Truncation (512 tokens)"
echo "============================================================================"
echo ""

# Create prepared_data directory
mkdir -p raw_data/prepared_data

# Configuration
CHARACTERS=("baseline")
MAX_SEQ_LENGTH=512  # Truncate to this length (Mistral standard)

# ============================================================================
# Process each character
# ============================================================================
for CHARACTER in "${CHARACTERS[@]}"; do
    echo "Processing $CHARACTER..."
    
    # Determine variants based on character
    if [ "$CHARACTER" = "baseline" ]; then
        VARIANTS="baseline baseline_augmented baseline_full"
    fi
    
    for VARIANT in $VARIANTS; do
        echo "  Variant: $VARIANT"
        
        # Determine source file - VARIANT already includes character name
        SOURCE_FILE="raw_data/training_data_${VARIANT}.jsonl"
        if [ ! -f "$SOURCE_FILE" ]; then
            echo "    ⚠️  Source file not found: $SOURCE_FILE, skipping"
            continue
        fi
        
        # Split the data
        VARIANT_NAME=$(echo "$VARIANT" | sed "s/${CHARACTER}_//" | sed "s/^${CHARACTER}$/base/")
        SPLIT_DIR="raw_data/prepared_data/${CHARACTER}/${VARIANT_NAME}_split"
        mkdir -p "$SPLIT_DIR"
        
        # Use split script (modified to accept output dir)
        python3 << PYEOF
import json
import random
import os

source = "$SOURCE_FILE"
output_dir = "$SPLIT_DIR"
split_ratio = 0.9

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"    Splitting {source}...")
with open(source, 'r') as f:
    lines = f.readlines()

data = [json.loads(line) for line in lines]
random.seed(42)
random.shuffle(data)

split_idx = int(len(data) * split_ratio)
train_data = data[:split_idx]
valid_data = data[split_idx:]

with open(os.path.join(output_dir, "train.jsonl"), 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "valid.jsonl"), 'w') as f:
    for item in valid_data:
        f.write(json.dumps(item) + "\n")

print(f"    ✓ Split: {len(train_data)} train + {len(valid_data)} valid")
PYEOF

        # Step 2: Pre-truncate to 512 and 768 tokens
        for SEQ_LEN in 512 768; do
            echo "    Pre-truncating to $SEQ_LEN tokens..."
            python3 scripts/truncate_training_data.py \
                --input-dir "$SPLIT_DIR" \
                --max-length "$SEQ_LEN" \
                --model "mistralai/Mistral-7B-Instruct-v0.3" 2>&1 | grep "Complete" || true
            # Move output from [SPLIT_DIR]_truncated to versioned folder
            TRUNC_SRC="$SPLIT_DIR"_truncated
            TRUNC_DIR="raw_data/prepared_data/${CHARACTER}/${VARIANT_NAME}_split_${SEQ_LEN}"
            mkdir -p "$TRUNC_DIR"
            mv "$TRUNC_SRC/train.jsonl" "$TRUNC_DIR/train.jsonl"
            mv "$TRUNC_SRC/valid.jsonl" "$TRUNC_DIR/valid.jsonl"
            rm -rf "$TRUNC_SRC"
        done
    done
    
    echo ""
done

echo "============================================================================"
echo "Dataset Preparation Complete!"
echo "============================================================================"
echo ""
echo "Generated datasets in: raw_data/prepared_data/{character}/{variant}_split_truncated/"
echo ""
echo "Each directory contains:"
echo "  • train.jsonl - 90% of examples, pre-truncated to 512 tokens"
echo "  • valid.jsonl - 10% of examples, pre-truncated to 512 tokens"
echo ""
echo "Pre-truncation benefits:"
echo "  ✓ No runtime tokenization overhead during training"
echo "  ✓ Consistent token limits across all runs"
echo "  ✓ System prompt + recent context preserved"
echo "  ✓ Assistant responses truncated intelligently (not conversation history)"
echo ""
echo "Next steps:"
echo "  1. git add raw_data/prepared_data/"
echo "  2. git commit -m 'Add pre-truncated datasets (512 tokens)'"
echo "  3. Training will automatically use truncated data"
echo ""
echo "============================================================================"
