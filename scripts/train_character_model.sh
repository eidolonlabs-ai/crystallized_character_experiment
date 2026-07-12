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
#   ./scripts/train_character_model.sh baseline mistral_v0_3
#   ./scripts/train_character_model.sh baseline mistral_v0_3 deep 4bit
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
    echo "    mistral_v0_3  - Mistral 7B Instruct v0.3"
    echo "    llama31_8b    - Meta Llama 3.1 8B Instruct"
    echo "    qwen25_7b     - Qwen2.5-7B-Instruct"
    echo "    qwen3_8b      - Qwen3-8B"
    echo ""
    echo "VARIANTS (optional):"
    echo "  standard      - Standard training (8-layer LoRA, 5e-5 LR, 2048 seq len)"
    echo "  deep          - Deep training (16-layer LoRA, 2.5e-5 LR, 2048 seq len)"
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
    echo "  --mask-prompt         Only compute loss on assistant tokens."
    echo "  --no-mask-prompt      Include the prompt in the loss (repo default; cross-model comparable)."
    echo "  --learning-rate LR    Override the per-model learning rate (default 5e-5; 2.5e-5 for"
    echo "                        mistral_v0_3 — see PER_MODEL_LEARNING_RATE)."
    echo "  --grad-checkpoint     Trade compute for memory on <32 GB unified-memory Macs."
    echo "  --optimizer NAME      adam | adamw | muon | sgd | adafactor (default: adamw)."
    echo "  --seed N              PRNG seed (default: 42)."
    echo ""
    echo "EXAMPLES:"
    echo "  $0 baseline mistral_v0_3                # Train Baseline with Mistral (standard, full precision)"
    echo "  $0 baseline llama31_8b standard full # Train Baseline with Llama 3.1 8B (standard, full precision)"
    echo "  $0 baseline mistral_v0_3 --fine-tune-type dora  # Train with DoRA instead of LoRA"
    echo "  $0 baseline mistral_v0_3 --config configs/lyra_mistral7b_v3.yaml"
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
LEARNING_RATE_OVERRIDE=""
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
        --learning-rate) LEARNING_RATE_OVERRIDE="$2"; shift 2 ;;
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
    echo "Available models: mistral_v0_3, llama31_8b, qwen25_7b, qwen3_8b"
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

# ============================================================================
# Training Parameters (Variant-Based)
# ============================================================================

if [ "$VARIANT" = "deep" ]; then
    # Deep variant: More aggressive fine-tuning for personality depth
    BATCH_SIZE=1
    GRAD_ACCUM_STEPS=4
    NUM_LAYERS=16
    LEARNING_RATE=2.5e-5
    MAX_SEQ_LENGTH=2048
    EPOCHS=5
    VARIANT_SUFFIX="_deep"
else
    # Standard variant: Conservative, stable fine-tuning
    BATCH_SIZE=1
    GRAD_ACCUM_STEPS=2
    NUM_LAYERS=8
    LEARNING_RATE=5e-5
    MAX_SEQ_LENGTH=2048
    EPOCHS=5
    VARIANT_SUFFIX=""
fi

# Apply per-model learning-rate override (Mistral v0.3 only — see
# model_config.py::PER_MODEL_LEARNING_RATE for why). The standard variant's
# 5e-5 default diverges on it because AdamW amplifies the gradient
# on the identical post-Phase-0 system-prompt block until loss explodes.
# 2.5e-5 converges cleanly with the same loss recipe.
if [ -n "$LEARNING_RATE_OVERRIDE" ]; then
    LEARNING_RATE="$LEARNING_RATE_OVERRIDE"
    echo "  CLI --learning-rate override: $LEARNING_RATE"
else
    PER_MODEL_LR=$(python -c "
import sys
sys.path.insert(0, 'scripts')
from model_config import get_learning_rate
print(get_learning_rate('$MODEL_NAME'))
")
    if [ "$PER_MODEL_LR" != "5e-05" ]; then
        LEARNING_RATE="$PER_MODEL_LR"
        echo "  Per-model learning rate for $MODEL_NAME: $LEARNING_RATE"
    fi
fi

# Repo default is --no-mask-prompt (parity with the existing 8 trained
# adapters, all 6 supported base models converge under this). --mask-prompt
# is exposed for users who want modern SFT defaults; it changes the loss
# recipe and breaks cross-model comparison, so don't switch the default.
MASK_PROMPT="--no-mask-prompt"

# Set quantization suffix for output paths
if [ "$QUANTIZE" = "4bit" ]; then
    QUANTIZE_SUFFIX="_q4"
else
    QUANTIZE_SUFFIX="_fp16"
fi

# ============================================================================
# Construct Paths
# ============================================================================


# Select correct split directory based on max sequence length.
# Reasoning models (Qwen3) prefer augmented_curated_thinking — the dataset
# variant that includes <think> blocks so the character learns to reason
# in-voice. Non-reasoning models use the standard augmented_curated.
SEQ_SUFFIX="_split_${MAX_SEQ_LENGTH}"
DATA_DIR=""
DATA_SOURCE=""

if is_reasoning_model "$MODEL_NAME" && [ -d "raw_data/prepared_data/baseline/augmented_curated_thinking${SEQ_SUFFIX}" ]; then
    DATA_DIR="raw_data/prepared_data/baseline/augmented_curated_thinking${SEQ_SUFFIX}"
    DATA_SOURCE="augmented_curated_thinking"
elif [ -d "raw_data/prepared_data/baseline/augmented_curated${SEQ_SUFFIX}" ]; then
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
    "$MASK_PROMPT_FLAG"
)
[ -n "$FINE_TUNE_TYPE_FLAG" ] && TRAIN_CMD+=(--fine-tune-type "$FINE_TUNE_TYPE_FLAG")
[ -n "$GRAD_CHECKPOINT_FLAG" ] && TRAIN_CMD+=("$GRAD_CHECKPOINT_FLAG")
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
