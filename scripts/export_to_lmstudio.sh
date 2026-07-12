#!/bin/bash
# ============================================================================
# Export MLX Model to LM Studio
# ============================================================================
#
# Usage: ./export_to_lmstudio.sh <character> <model_name> [variant]
#
# Examples:
#   ./export_to_lmstudio.sh baseline mistral_v0_3
#   ./export_to_lmstudio.sh baseline llama31_8b
#
# This script copies the merged and quantized MLX model to LM Studio's
# models directory for use in the LM Studio GUI.

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <character> <model_name> [variant]"
    echo ""
    echo "Characters: baseline"
    echo "Models: mistral_v0_3, llama31_8b, qwen25_7b, qwen3_8b"
    echo "Variant: standard (default) or deep"
    exit 1
fi

CHARACTER="$1"
MODEL_NAME="$2"
VARIANT="${3:-standard}"
REPO_ROOT="$(dirname "$0")/.."

# MLX output path pattern
MLX_PATH="$REPO_ROOT/models/${CHARACTER}_${MODEL_NAME}_mlx_q4"
[ "$VARIANT" = "deep" ] && MLX_PATH="$REPO_ROOT/models/${CHARACTER}_${MODEL_NAME}_mlx_q4_deep"

# LM Studio models directory
LMSTUDIO_MODELS="$HOME/.lmstudio/models"

if [ ! -d "$MLX_PATH" ]; then
    echo "❌ MLX model not found: $MLX_PATH"
    echo ""
    echo "Make sure you've trained and merged the model first:"
    echo "  ./scripts/train_character_model.sh $CHARACTER $MODEL_NAME $VARIANT"
    exit 1
fi

if [ ! -d "$LMSTUDIO_MODELS" ]; then
    echo "❌ LM Studio models directory not found: $LMSTUDIO_MODELS"
    echo ""
    echo "Please install and configure LM Studio first:"
    echo "  https://lmstudio.ai"
    exit 1
fi

# Destination path in LM Studio
DEST_PATH="$LMSTUDIO_MODELS/cynthia/${CHARACTER}-${MODEL_NAME}-${VARIANT}"

echo "========================================"
echo "Exporting to LM Studio"
echo "========================================"
echo ""
echo "Source:      $MLX_PATH"
echo "Destination: $DEST_PATH"
echo ""

# Create destination directory
mkdir -p "$DEST_PATH"

# Copy model
echo "Copying MLX model..."
cp -r "$MLX_PATH"/* "$DEST_PATH/"
echo "✅ Model copied to LM Studio"
echo ""
echo "Next steps:"
echo "  1. Open LM Studio"
echo "  2. Go to 'Local Models' section"
echo "  3. Refresh the model list"
echo "  4. Select '${CHARACTER}-${MODEL_NAME}-${VARIANT}' model"
echo "  5. Click 'Load Model'"
echo ""
echo "The model will be available for chat once loaded!"
