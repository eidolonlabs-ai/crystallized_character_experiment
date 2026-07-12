#!/bin/bash
# ============================================================================
# Prepare All Datasets Script
# ============================================================================
# This script generates all dataset variants once and saves them to
# prepared_data/ directory. Data is versioned and consistent for all runs.
#
# Generates:
# - Split variants: base, augmented, augmented_curated, augmented_curated_thinking (90/10 train/valid)
# - Pre-truncated at 512, 768, 2048 tokens (intelligent truncation preserving context)
#
# Order of operations:
#   1. Run curate_training_data.py to derive augmented_curated.jsonl from augmented.jsonl
#   2. Run generate_thinking_data.py to derive augmented_curated_thinking.jsonl from augmented_curated.jsonl
#   3. For each source variant (base, augmented, augmented_curated, augmented_curated_thinking, full):
#        split 90/10 → variant_split/
#        truncate to 512, 768, and 2048 tokens → variant_split_512/, variant_split_768/, variant_split_2048/
#
# Truncation strategy:
# - Keeps system prompt (character definition)
# - Works backwards to preserve recent conversation context
# - Truncates assistant response if needed (not conversation history)
#
# Output structure:
# raw_data/prepared_data/
# ├── baseline/
# │   ├── base_split/                  (unsplit-cap input for tooling that wants full sequences)
# │   ├── base_split_512/, base_split_768/, base_split_2048/
# │   ├── augmented_split/
# │   ├── augmented_split_512/, augmented_split_768/, augmented_split_2048/
# │   ├── augmented_curated_split/     (only created if augmented_curated.jsonl exists)
# │   ├── augmented_curated_split_512/, augmented_curated_split_768/, augmented_curated_split_2048/
# │   ├── full_split/
# │   └── full_split_512/, full_split_768/, full_split_2048/
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
echo "Preparing All Datasets with Pre-Truncation (512, 768, 2048 tokens)"
echo "Includes thinking variant for reasoning models"
echo "============================================================================"
echo ""

# Create prepared_data directory
mkdir -p raw_data/prepared_data

# Configuration
CHARACTERS=("baseline")

# ============================================================================
# Process each character
# ============================================================================
for CHARACTER in "${CHARACTERS[@]}"; do
    echo "Processing $CHARACTER..."
    
    # Derive augmented_curated.jsonl from augmented.jsonl via the curate script.
    # This must run before the loop below, otherwise the augmented_curated variant
    # has no source file to read from.
    if [ -f "raw_data/training_data_${CHARACTER}_augmented.jsonl" ] && [ ! -f "raw_data/training_data_${CHARACTER}_augmented_curated.jsonl" ]; then
        echo "  Curating augmented data → augmented_curated (trim long responses, keep voice)..."
        python3 scripts/curate_training_data.py 2>&1 | grep -E "(kept|truncated|skipped|Curation)" || true
    fi

    # Derive augmented_curated_thinking.jsonl from augmented_curated.jsonl.
    # This must run after curation so it has the flattened single-turn data.
    if [ -f "raw_data/training_data_${CHARACTER}_augmented_curated.jsonl" ] && [ ! -f "raw_data/training_data_${CHARACTER}_augmented_curated_thinking.jsonl" ]; then
        echo "  Generating thinking blocks for reasoning models..."
        THINKING_MODEL="${THINKING_GEN_MODEL:-openai/gpt-4o}"
        python3 scripts/generate_thinking_data.py \
            --input "raw_data/training_data_${CHARACTER}_augmented_curated.jsonl" \
            --output "raw_data/training_data_${CHARACTER}_augmented_curated_thinking.jsonl" \
            --model "$THINKING_MODEL" 2>&1 || echo "  ⚠️  Thinking generation skipped (API key may be missing)"
    fi

    # Determine variants based on character
    if [ "$CHARACTER" = "baseline" ]; then
        VARIANTS="baseline baseline_augmented baseline_augmented_curated baseline_augmented_curated_thinking baseline_full"
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

        # Use the shared split script (single source of truth for the 90/10 split,
        # seed 42 — kept here so the split logic lives in exactly one place).
        python3 scripts/split_training_data.py \
            --input "$SOURCE_FILE" \
            --output-dir "$SPLIT_DIR" 2>&1 | sed 's/^/    /'

        # Step 2: Pre-truncate to 512, 768, and 2048 tokens
        for SEQ_LEN in 512 768 2048; do
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
echo "Generated datasets in: raw_data/prepared_data/{character}/{variant}_split_{seq_len}/"
echo ""
echo "Each directory contains:"
echo "  • train.jsonl - 90% of examples, pre-truncated to 512/768/2048 tokens"
echo "  • valid.jsonl - 10% of examples, pre-truncated to 512/768/2048 tokens"
echo ""
echo "Pre-truncation benefits:"
echo "  ✓ No runtime tokenization overhead during training"
echo "  ✓ Consistent token limits across all runs"
echo "  ✓ System prompt + recent context preserved"
echo "  ✓ Assistant responses truncated intelligently (not conversation history)"
echo ""
echo "Next steps:"
echo "  1. git add raw_data/prepared_data/"
echo "  2. git commit -m 'Add pre-truncated datasets (512, 768, 2048 tokens)'"
echo "  3. Training will automatically use truncated data"
echo ""
echo "============================================================================"
