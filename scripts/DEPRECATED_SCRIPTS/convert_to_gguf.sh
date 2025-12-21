#!/bin/bash
set -e

# Directory setup
WORKSPACE_ROOT=$(pwd)
LLAMA_CPP_DIR="$WORKSPACE_ROOT/third_party/llama.cpp"
MERGED_MODELS_DIR="$WORKSPACE_ROOT/merged_models"
GGUF_MODELS_DIR="$WORKSPACE_ROOT/gguf_models"

# Create directories
mkdir -p "$GGUF_MODELS_DIR"
mkdir -p "$(dirname "$LLAMA_CPP_DIR")"

# 1. Setup llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

# Build if binary missing
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/build/llama-quantize"
fi

if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "Building llama.cpp..."
    cd "$LLAMA_CPP_DIR"
    mkdir -p build
    cd build
    cmake ..
    cmake --build . --config Release --target llama-quantize
    cd "$WORKSPACE_ROOT"
    
    # Install python requirements
    echo "Installing llama.cpp python requirements..."
    pip install -r "$LLAMA_CPP_DIR/requirements.txt"
else
    echo "llama.cpp binary found at $QUANTIZE_BIN"
fi

# 2. Convert and Quantize Models
if [ ! -d "$MERGED_MODELS_DIR" ]; then
    echo "Error: $MERGED_MODELS_DIR not found. Please run merge_adapters.py first."
    exit 1
fi

for model_dir in "$MERGED_MODELS_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "Processing $model_name..."
        
        # Define output paths
        f16_output="$GGUF_MODELS_DIR/${model_name}-f16.gguf"
        q4_output="$GGUF_MODELS_DIR/${model_name}-Q4_K_M.gguf"
        
        # Convert to GGUF (F16)
        if [ ! -f "$f16_output" ]; then
            echo "Converting $model_name to GGUF (F16)..."
            python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$model_dir" --outfile "$f16_output"
        else
            echo "$f16_output already exists. Skipping conversion."
        fi
        
        # Quantize to Q4_K_M
        if [ ! -f "$q4_output" ]; then
            echo "Quantizing $model_name to Q4_K_M..."
            QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
            if [ ! -f "$QUANTIZE_BIN" ]; then
                QUANTIZE_BIN="$LLAMA_CPP_DIR/build/llama-quantize"
            fi
            
            if [ ! -f "$QUANTIZE_BIN" ]; then
                echo "Error: llama-quantize binary not found at $QUANTIZE_BIN"
                exit 1
            fi

            "$QUANTIZE_BIN" "$f16_output" "$q4_output" Q4_K_M
        else
            echo "$q4_output already exists. Skipping quantization."
        fi
    fi
done

echo "Done! GGUF models are in $GGUF_MODELS_DIR"
