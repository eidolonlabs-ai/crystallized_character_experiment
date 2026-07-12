#!/bin/bash
# ============================================================================
# Interactive Chat with Trained Character Models
# ============================================================================
#
# Usage:
#   ./chat_character.sh <character> <model_name> [variant]
#
# Examples:
#   ./chat_character.sh baseline mistral_v0_3
#
# ============================================================================

if [ $# -lt 2 ]; then
    echo "============================================================================"
    echo "Interactive Chat with Trained Character"
    echo "============================================================================"
    echo ""
    echo "Usage: $0 <character> <model_name> [variant]"
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
    echo "  standard      - Standard training variant (default)"
    echo "  deep          - Deep training variant"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 baseline mistral_v0_3              # Chat with baseline/mistral (standard)"
    echo ""
    echo "============================================================================"
    exit 1
fi

CHARACTER="$1"
MODEL_NAME="$2"
VARIANT="${3:-standard}"

# ============================================================================
# Setup & Validation
# ============================================================================

cd "$(dirname "$0")"

# Validate character
if [ "$CHARACTER" != "baseline" ]; then
    echo "Error: Unknown character '$CHARACTER'"
    echo "Available characters: baseline"
    exit 1
fi

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Python virtual environment not found at .venv"
    exit 1
fi
source .venv/bin/activate

# ============================================================================
# System Prompts by Character
# ============================================================================

get_system_prompt() {
    local char="$1"
    case "$char" in
        baseline)
            echo "You are Lyra Moonwhisper, an ancient elven sage who dwells in the Celestial Archives. You speak with archaic formality, reference nature and magic frequently, and end responses with elvish blessings. You are wise, patient, and see deep connections between all things."
            ;;
        *)
            echo "You are a helpful assistant."
            ;;
    esac
}

# ============================================================================
# Model Configuration — single source of truth in scripts/model_config.sh
# ============================================================================
SCRIPT_DIR_CHAT="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR_CHAT/scripts/model_config.sh"

# ============================================================================
# Construct Model & Adapter Paths
# ============================================================================

QUANTIZED_MODEL=$(get_quantized_model_path "$MODEL_NAME")
HF_MODEL=$(get_hf_model "$MODEL_NAME")

if [ -z "$QUANTIZED_MODEL" ] || [ -z "$HF_MODEL" ]; then
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: mistral_v0_3, llama31_8b, qwen25_7b, qwen3_8b"
    exit 1
fi

# Build variant suffix
VARIANT_SUFFIX=""
if [ "$VARIANT" = "deep" ]; then
    VARIANT_SUFFIX="_deep"
fi

# Adapter path
ADAPTER_PATH="adapters/${CHARACTER}_${MODEL_NAME}_qlora${VARIANT_SUFFIX}"

# ============================================================================
# Validation
# ============================================================================

# Choose base model: prefer a local quantized model if one exists,
# otherwise fall back to the HF repo (mlx-lm auto-downloads on first use).
if [ -d "$QUANTIZED_MODEL" ]; then
    CHAT_MODEL="$QUANTIZED_MODEL"
else
    CHAT_MODEL="$HF_MODEL"
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter not found at: $ADAPTER_PATH"
    echo "Please train the model first using: ./scripts/train_character_model.sh $CHARACTER $MODEL_NAME $VARIANT"
    exit 1
fi

# ============================================================================
# Get System Prompt
# ============================================================================

SYSTEM_PROMPT=$(get_system_prompt "$CHARACTER")

# ============================================================================
# Display Configuration
# ============================================================================

echo ""
echo "============================================================================"
echo "Chat with $CHARACTER (trained on $MODEL_NAME${VARIANT_SUFFIX})"
echo "============================================================================"
echo ""
echo "Model: $CHAT_MODEL"
echo "Adapter: $ADAPTER_PATH"
echo ""
echo "System Prompt:"
echo "  $SYSTEM_PROMPT"
echo ""
echo "Type 'exit' or Ctrl+C to quit"
echo "============================================================================"
echo ""

# ============================================================================
# Start Chat
# ============================================================================

# Use scripts/folded_chat.py instead of `mlx_lm chat`. The latter calls
# tokenizer.apply_chat_template() which silently drops role=system on Mistral
# v0.3 and the Llama chat tokenizers, so the adapter's [SYSTEM] trigger is
# never met and the model falls back to base behavior. folded_chat.py applies
# the Phase 0 fold at inference time, matching the training distribution.
python scripts/folded_chat.py \
  --model "$CHAT_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --system-prompt "$SYSTEM_PROMPT" \
  --temp 0.7 \
  --max-tokens 300
