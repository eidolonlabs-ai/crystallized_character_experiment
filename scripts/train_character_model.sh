#!/bin/bash
# ============================================================================
# Consolidated Character Training Script for MLX
# ============================================================================
# 
# This script consolidates all character training into a single parameterized
# script. The only differences between runs are the character name, model 
# name, and optional variant (standard vs deep tuning).
#
# Usage:
#   ./train_character_model.sh <character> <model_name> [variant] [quantize]
#
# Examples:
#   ./scripts/train_character_model.sh baseline mistral
#   ./scripts/train_character_model.sh baseline mistral deep 4bit
#
# ============================================================================

# Show usage if no arguments
if [ $# -lt 2 ]; then
    echo "============================================================================"
    echo "Consolidated MLX Character Training Script"
    echo "============================================================================"
    echo ""
    echo "Usage: $0 <character> <model_name> [variant] [quantize]"
    echo "       [--config PATH] [--fine-tune-type {lora,dora}] [--mask-prompt|--no-mask-prompt]"
    echo "       [--grad-checkpoint] [--optimizer NAME] [--seed N]"
    echo ""
    echo "CHARACTERS:"
    echo "  baseline      - Baseline Lyra character model"
    echo ""
    echo "MODEL NAMES:"
    echo "  Mistral family:"
    echo "    mistral       - Mistral 7B Instruct v0.3 (alias for mistral_v0_3)"
    echo "    mistral_v0_3  - Mistral 7B Instruct v0.3"
    echo "    mistral_v0_2  - Mistral 7B Instruct v0.2"
    echo "    mistral_v0_1  - Mistral 7B Instruct v0.1"
    echo "  Llama family:"
    echo "    llama         - Meta Llama 3.1 8B (alias for llama31_8b)"
    echo "    llama31_8b    - Meta Llama 3.1 8B Instruct (latest)"
    echo "    llama3_8b     - Meta Llama 3 8B Instruct"
    echo "    llama2_7b     - Meta Llama 2 7B Chat"
    echo ""
    echo "VARIANTS (optional):"
    echo "  standard      - Standard training (8-layer LoRA, 5e-5 LR, 512 seq len)"
    echo "  deep          - Deep training (16-layer LoRA, 2.5e-5 LR, 768 seq len)"
    echo "  (default)     - Uses 'standard' if not specified"
    echo ""
    echo "QUANTIZE (optional):"
    echo "  full          - Full precision FP16 (better quality, larger size)"
    echo "  4bit          - 4-bit quantized (smaller size, faster inference)"
    echo "  (default)     - Uses 'full' if not specified"
    echo ""
    echo "OPTIONAL FLAGS:"
    echo "  --config PATH         Path to a YAML config (default: auto-generated from"
    echo "                        scripts/model_config.py::DEFAULT_TRAINING)."
    echo "  --fine-tune-type F    lora | dora (default: lora; DoRA is supported natively)."
    echo "  --mask-prompt         Only compute loss on assistant tokens (modern SFT default; repo default)."
    echo "  --no-mask-prompt      Include the prompt in the loss. Note: with the post-Phase 0 fold,"
    echo "                        this causes AdamW to amplify the identical ~50-token system-prompt"
    echo "                        gradient across all examples and diverges Mistral v0.2/v0.3."
    echo "  --grad-checkpoint     Trade compute for memory on <32 GB unified-memory Macs."
    echo "  --optimizer NAME      adam | adamw | muon | sgd | adafactor (default: adamw)."
    echo "  --seed N              PRNG seed (default: 42)."
    echo ""
    echo "EXAMPLES:"
    echo "  $0 baseline mistral                # Train Baseline with Mistral (standard, full precision)"
    echo "  $0 baseline llama31_8b standard full # Train Baseline with Llama 3.1 8B (standard, full precision)"
    echo "  $0 baseline mistral --fine-tune-type dora  # Train with DoRA instead of LoRA"
    echo "  $0 baseline mistral --config configs/lyra_mistral7b_v3.yaml"
    echo ""
    echo "============================================================================"
    exit 1
fi

# Parse optional --flag arguments before the positional ones.
# Usage: train_character_model.sh <character> <model> [variant] [quantize] [--config PATH] [--fine-tune-type {lora,dora}] [--mask-prompt] [--grad-checkpoint] [--optimizer NAME] [--seed N]
CONFIG_PATH=""
FINE_TUNE_TYPE_OVERRIDE=""
MASK_PROMPT_OVERRIDE=""
GRAD_CHECKPOINT_OVERRIDE=""
OPTIMIZER_OVERRIDE=""
SEED_OVERRIDE=""
POSITIONAL=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --fine-tune-type) FINE_TUNE_TYPE_OVERRIDE="$2"; shift 2 ;;
        --mask-prompt) MASK_PROMPT_OVERRIDE="--mask-prompt"; shift ;;
        --no-mask-prompt) MASK_PROMPT_OVERRIDE="--no-mask-prompt"; shift ;;
        --grad-checkpoint) GRAD_CHECKPOINT_OVERRIDE="--grad-checkpoint"; shift ;;
        --optimizer) OPTIMIZER_OVERRIDE="$2"; shift 2 ;;
        --seed) SEED_OVERRIDE="$2"; shift 2 ;;
        --*) echo "Unknown flag: $1" >&2; exit 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

# Now POSITIONAL contains only the original positionals.
set -- "${POSITIONAL[@]}"
CHARACTER="${1:-}"
MODEL_NAME="${2:-}"
VARIANT="${3:-standard}"
QUANTIZE="${4:-full}"

# ============================================================================
# Setup & Validation
# ============================================================================

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Normalize aliases — "mistral" → "mistral_v0_3", "llama" → "llama31_8b"
if [ "$MODEL_NAME" = "mistral" ]; then
    MODEL_NAME="mistral_v0_3"
elif [ "$MODEL_NAME" = "llama" ]; then
    MODEL_NAME="llama31_8b"
fi

# Validate character
if [ "$CHARACTER" != "baseline" ]; then
    echo "Error: Unknown character '$CHARACTER'"
    echo "Available characters: baseline"
    exit 1
fi

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run: python -m venv .venv"
    exit 1
fi
source .venv/bin/activate

# ============================================================================
# Metal/MLX Configuration
# ============================================================================
export MLX_METAL_DEBUG=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Stop script on error
set -e

# ============================================================================
# Model Configuration — single source of truth in scripts/model_config.sh
# ============================================================================
SCRIPT_DIR_TCM="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR_TCM/model_config.sh"

# Alias: get_model_config is the historical name used by callers below
get_model_config() { get_hf_model "$1"; }

# Validate model
MODEL_CONFIG=$(get_model_config "$MODEL_NAME")
if [ -z "$MODEL_CONFIG" ]; then
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: mistral, mistral_v0_3, mistral_v0_2, mistral_v0_1, llama, llama31_8b, llama3_8b, llama2_7b"
    exit 1
fi

# Validate variant
if [ "$VARIANT" != "standard" ] && [ "$VARIANT" != "deep" ]; then
    echo "Error: Unknown variant '$VARIANT'"
    echo "Available variants: standard, deep"
    exit 1
fi

# Validate quantize option
if [ "$QUANTIZE" != "full" ] && [ "$QUANTIZE" != "4bit" ]; then
    echo "Error: Unknown quantize option '$QUANTIZE'"
    echo "Available options: full, 4bit"
    exit 1
fi

# ============================================================================
# Extract Model Configuration
# ============================================================================
BASE_MODEL=$(get_model_config "$MODEL_NAME")

# Llama-2 is numerically unstable in its native fp16 (activation outliers
# overflow fp16 -> inf -> NaN loss). Train against a local bf16 conversion,
# which shares fp32's exponent range and avoids the overflow.
if [ "$MODEL_NAME" = "llama2_7b" ]; then
    LLAMA2_BF16="models/llama-2-7b-chat-bf16"
    if [ ! -d "$LLAMA2_BF16" ]; then
        echo "  Converting $BASE_MODEL to bf16 (one-time) -> $LLAMA2_BF16"
        python -m mlx_lm convert \
            --hf-path "$BASE_MODEL" \
            --mlx-path "$LLAMA2_BF16" \
            --dtype bfloat16
    fi
    BASE_MODEL="$LLAMA2_BF16"
fi

# ============================================================================
# Training Parameters (Variant-Based)
# ============================================================================

if [ "$VARIANT" = "deep" ]; then
    # Deep variant: More aggressive fine-tuning for personality depth
    BATCH_SIZE=1
    GRAD_ACCUM_STEPS=4
    NUM_LAYERS=16
    LEARNING_RATE=2.5e-5
    MAX_SEQ_LENGTH=768
    EPOCHS=5
    VARIANT_SUFFIX="_deep"
else
    # Standard variant: Conservative, stable fine-tuning
    BATCH_SIZE=1
    GRAD_ACCUM_STEPS=2
    NUM_LAYERS=8
    LEARNING_RATE=5e-5
    MAX_SEQ_LENGTH=512
    EPOCHS=5
    VARIANT_SUFFIX=""
fi

# Modern SFT default: only compute loss on the assistant response. The
# `--no-mask-prompt` alternative trains on the full sequence including the
# (identical, repeated) system prompt, which AdamW's adaptive LR amplifies
# until training diverges on Mistral v0.2/v0.3 (see commit log for the
# empirical 10-iter test). Override via --no-mask-prompt on the CLI.
MASK_PROMPT="--mask-prompt"

# Set quantization suffix for output paths
if [ "$QUANTIZE" = "4bit" ]; then
    QUANTIZE_SUFFIX="_q4"
else
    QUANTIZE_SUFFIX="_fp16"
fi

# ============================================================================
# Construct Paths
# ============================================================================


# Select correct split directory based on max sequence length
SEQ_SUFFIX="_split_${MAX_SEQ_LENGTH}"
DATA_DIR=""
DATA_SOURCE=""

if [ -d "raw_data/prepared_data/baseline/augmented_curated${SEQ_SUFFIX}" ]; then
    DATA_DIR="raw_data/prepared_data/baseline/augmented_curated${SEQ_SUFFIX}"
    DATA_SOURCE="augmented_curated"
elif [ -d "raw_data/prepared_data/baseline/augmented${SEQ_SUFFIX}" ]; then
    DATA_DIR="raw_data/prepared_data/baseline/augmented${SEQ_SUFFIX}"
    DATA_SOURCE="augmented"
elif [ -d "raw_data/prepared_data/baseline/base${SEQ_SUFFIX}" ]; then
    DATA_DIR="raw_data/prepared_data/baseline/base${SEQ_SUFFIX}"
    DATA_SOURCE="base"
elif [ -d "raw_data/prepared_data/baseline/full${SEQ_SUFFIX}" ]; then
    DATA_DIR="raw_data/prepared_data/baseline/full${SEQ_SUFFIX}"
    DATA_SOURCE="full"
else
    echo "Error: No valid dataset found for baseline (expected split: $SEQ_SUFFIX)."
    exit 1
fi

OUTPUT_ADAPTER="adapters/${CHARACTER}_${MODEL_NAME}_qlora${VARIANT_SUFFIX}"
# train.log lives next to the adapter. Captured from mlx_lm lora stdout+stderr
# so the loss curve survives the shell session and parse_train_log.py can read it.
TRAIN_LOG="${OUTPUT_ADAPTER}/train.log"
OUTPUT_MLX="models/${CHARACTER}_${MODEL_NAME}_mlx${QUANTIZE_SUFFIX}${VARIANT_SUFFIX}"

# ============================================================================
# Display Configuration
# ============================================================================
echo ""
echo "============================================================================"
echo "MLX Character Training Pipeline"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Character: $CHARACTER"
echo "  Model: $MODEL_NAME (Variant: $VARIANT)"
echo "  Base Model: $BASE_MODEL"
echo "  Quantization: $QUANTIZE"
echo "  Dataset: $DATA_SOURCE"
echo ""
echo "Paths:"
echo "  Data Directory: $DATA_DIR"
echo "  Base Model: $BASE_MODEL"
echo "  LoRA Adapter Output: $OUTPUT_ADAPTER"
echo "  MLX Model Output: $OUTPUT_MLX"
echo ""
echo "Training Hyperparameters:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "  LoRA Layers: $NUM_LAYERS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Sequence Length: $MAX_SEQ_LENGTH"
echo "  Epochs: $EPOCHS"
echo ""
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Prepare Base Model (no quantization)
# ============================================================================
echo "[Step 1/4] Loading base model..."
echo "  • Using model from HuggingFace: $BASE_MODEL"
echo "  • Format: Full precision (FP16) during training"
echo ""
echo "  ✓ Base model will be loaded directly during training."
echo ""

# ============================================================================
# Step 2: Prepare Training Data
# ============================================================================
echo "[Step 2/4] Using pre-prepared and pre-truncated training data..."
echo "  ✓ Loading pre-split dataset: $DATA_SOURCE"
echo "  ✓ Data location: $DATA_DIR"
echo "  ✓ Pre-truncated to $MAX_SEQ_LENGTH tokens (intelligent truncation)"
echo "  ✓ System prompt + recent context preserved in each example"
echo ""

# ============================================================================
# Step 3: LoRA Fine-Tuning
# ============================================================================
echo "[Step 3/4] Starting LoRA fine-tuning..."
echo "  Model: $BASE_MODEL"
echo "  Data: $DATA_DIR"
echo ""

# Resolve the YAML config the trainer will consume:
#   1. If the user passed --config PATH, use that.
#   2. Otherwise build one on the fly from scripts/model_config.py defaults
#      so users always get the modern recipe (rank 16, alpha 32, adamw,
#      cosine schedule, etc.) without having to remember to pass --config.
EFFECTIVE_CONFIG="$CONFIG_PATH"
if [ -z "$EFFECTIVE_CONFIG" ]; then
    EFFECTIVE_CONFIG="$(mktemp -t lora_config.XXXXXX.yaml)"
    python -c "
import sys
sys.path.insert(0, 'scripts')
from model_config import render_yaml_config
print(render_yaml_config('$MODEL_NAME', variant='$VARIANT'), end='')
" > "$EFFECTIVE_CONFIG"
    echo "  Auto-generated config: $EFFECTIVE_CONFIG"
fi

# Resolve mask-prompt / fine-tune-type / grad-checkpoint / optimizer / seed:
# CLI flag override > YAML value > repo default. The YAML is the modern
# recipe; the CLI flag wins when present.
MASK_PROMPT_FLAG="$MASK_PROMPT"
if [ -n "$MASK_PROMPT_OVERRIDE" ]; then MASK_PROMPT_FLAG="$MASK_PROMPT_OVERRIDE"; fi
FINE_TUNE_TYPE_FLAG=""
if [ -n "$FINE_TUNE_TYPE_OVERRIDE" ]; then FINE_TUNE_TYPE_FLAG="$FINE_TUNE_TYPE_OVERRIDE"; fi
GRAD_CHECKPOINT_FLAG=""
if [ -n "$GRAD_CHECKPOINT_OVERRIDE" ]; then GRAD_CHECKPOINT_FLAG="$GRAD_CHECKPOINT_OVERRIDE"; fi
OPTIMIZER_FLAG=""
if [ -n "$OPTIMIZER_OVERRIDE" ]; then OPTIMIZER_FLAG="$OPTIMIZER_OVERRIDE"; fi
SEED_FLAG=""
if [ -n "$SEED_OVERRIDE" ]; then SEED_FLAG="$SEED_OVERRIDE"; fi

TRAIN_CMD=(python scripts/train_mlx.py
    --model "$BASE_MODEL"
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_ADAPTER"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --num-layers "$NUM_LAYERS"
    --learning-rate "$LEARNING_RATE"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --config "$EFFECTIVE_CONFIG"
    $MASK_PROMPT_FLAG
)
[ -n "$FINE_TUNE_TYPE_FLAG" ] && TRAIN_CMD+=(--fine-tune-type "$FINE_TUNE_TYPE_FLAG")
[ -n "$GRAD_CHECKPOINT_FLAG" ] && TRAIN_CMD+=($GRAD_CHECKPOINT_FLAG)
[ -n "$OPTIMIZER_FLAG" ] && TRAIN_CMD+=(--optimizer "$OPTIMIZER_FLAG")
[ -n "$SEED_FLAG" ] && TRAIN_CMD+=(--seed "$SEED_FLAG")

# Tee stdout+stderr to train.log so the loss curve survives the shell session.
# parse_train_log.py reads from there; plot_loss.py visualizes the result.
mkdir -p "$OUTPUT_ADAPTER"
echo "  Train log: $TRAIN_LOG"
"${TRAIN_CMD[@]}" 2>&1 | tee "$TRAIN_LOG"

echo ""
echo "  ✓ LoRA training complete."
echo ""

# ============================================================================
# Step 4: Merge Adapter and Convert to MLX
# ============================================================================
echo "[Step 4/4] Merging adapter with base model..."
rm -rf "$OUTPUT_MLX"
mkdir -p "$OUTPUT_MLX"

# Merge adapter with base model
if [ "$QUANTIZE" = "4bit" ]; then
    python -m mlx_lm fuse \
      --model "$BASE_MODEL" \
      --adapter-path "$OUTPUT_ADAPTER" \
      --save-path "$OUTPUT_MLX" \
      --quantize \
      --q-bits 4
else
    python -m mlx_lm fuse \
      --model "$BASE_MODEL" \
      --adapter-path "$OUTPUT_ADAPTER" \
      --save-path "$OUTPUT_MLX"
fi

echo "  ✓ Model saved to: $OUTPUT_MLX"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "Training Complete!"
echo "============================================================================"
echo ""
echo "Artifacts Created:"
echo "  • LoRA Adapter (Lightweight, ~5-10MB): $OUTPUT_ADAPTER"
if [ "$QUANTIZE" = "4bit" ]; then
    echo "  • Merged MLX Model (4-bit quantized): $OUTPUT_MLX"
else
    echo "  • Merged MLX Model (Full precision, FP16): $OUTPUT_MLX"
fi
echo ""
echo "Next Steps:"
echo ""
echo "  To test with LoRA adapter:"
echo "    python -m mlx_lm chat --model $BASE_MODEL --adapter-path $OUTPUT_ADAPTER"
echo ""
echo "  To test merged model:"
echo "    python -m mlx_lm chat --model $OUTPUT_MLX"
echo ""
echo "  To export to GGUF (optional):"
echo "    Use: mlx_lm convert --model-path $OUTPUT_MLX --mlx-path <output>"
echo ""
echo "============================================================================"
