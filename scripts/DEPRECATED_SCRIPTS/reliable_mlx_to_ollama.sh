#!/bin/bash
set -e

# Usage Check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <character_name> <adapter_directory>"
    echo "Example: $0 baseline adapters/baseline_llama31_8b_qlora"
    exit 1
fi

CHARACTER_NAME="$1"
ADAPTER_PATH="$2"

# Configuration
BASE_MODEL_GGUF="gguf_models/llama3.1_base.gguf"
OUTPUT_ADAPTER_GGUF="gguf_models/${CHARACTER_NAME}_adapter.gguf"
OUTPUT_MERGED_GGUF="gguf_models/${CHARACTER_NAME}_merged.gguf"
MODELFILE_NAME="Modelfile.${CHARACTER_NAME}"
LLAMA_CPP_BUILD="third_party/llama.cpp/build/bin"

echo "=== Starting Reliable Workflow for Character: $CHARACTER_NAME ==="
echo "Adapter Source: $ADAPTER_PATH"

# 1. Ensure HF Adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter directory $ADAPTER_PATH not found."
    exit 1
fi

# 2. Convert HF Adapter to GGUF
echo "Step 1: Converting HF Adapter to GGUF..."
# Always re-convert to ensure we have the latest, or check if newer? 
# For reliability, let's overwrite if it exists or just run it.
# We need the config.json in the adapter dir.
if [ ! -f "$ADAPTER_PATH/config.json" ]; then
    echo "Error: config.json missing in $ADAPTER_PATH"
    echo "Please ensure the base model config.json is present in the adapter directory."
    exit 1
fi

.venv/bin/python third_party/llama.cpp/convert_lora_to_gguf.py \
    "$ADAPTER_PATH" \
    --base "$ADAPTER_PATH" \
    --outfile "$OUTPUT_ADAPTER_GGUF"

# 3. Merge GGUF Adapter into Base GGUF
echo "Step 2: Merging Adapter into Base Model..."
if [ ! -f "$BASE_MODEL_GGUF" ]; then
    echo "Error: Base model $BASE_MODEL_GGUF not found."
    exit 1
fi

# Using llama-export-lora to bake the weights in
"$LLAMA_CPP_BUILD/llama-export-lora" \
    -m "$BASE_MODEL_GGUF" \
    --lora "$OUTPUT_ADAPTER_GGUF" \
    -o "$OUTPUT_MERGED_GGUF"

echo "Merged model created at $OUTPUT_MERGED_GGUF"

# 4. Create Character-Specific Modelfile
echo "Step 3: Creating $MODELFILE_NAME..."
cat > "$MODELFILE_NAME" <<EOF
FROM ./$OUTPUT_MERGED_GGUF

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 5. Create Ollama Model
echo "Step 4: Creating Ollama Model '$CHARACTER_NAME'..."
ollama create "$CHARACTER_NAME" -f "$MODELFILE_NAME"

echo "=== Workflow Complete ==="
echo "Try running: ollama run $CHARACTER_NAME"
