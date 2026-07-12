#!/bin/bash
# Shared model configuration — single source of truth for model mappings.
# Source this file from any script that needs model configs.
#
# Usage: source "$(dirname "$0")/model_config.sh"

# Map model name to HuggingFace repo ID
get_hf_model() {
    local model="$1"
    case "$model" in
        mistral_v0_3)    echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
        llama31_8b)       echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        qwen25_7b)        echo "Qwen/Qwen2.5-7B-Instruct" ;;
        qwen3_8b)         echo "Qwen/Qwen3-8B" ;;
        *)                echo "" ;;
    esac
}

get_quantized_model_path() {
    local model="$1"
    case "$model" in
        mistral_v0_3)    echo "models/mistral-7b-instruct-v0.3-4bit" ;;
        llama31_8b)       echo "models/llama-3.1-8b-instruct-4bit" ;;
        qwen25_7b)        echo "models/qwen2.5-7b-instruct-4bit" ;;
        qwen3_8b)         echo "models/qwen3-8b-4bit" ;;
        *)                echo "" ;;
    esac
}

# Whether the model natively emits <think> blocks (reasoning/CoT models).
# Training data for these models should include thinking blocks so the
# character learns to reason in-voice.
is_reasoning_model() {
    local model="$1"
    case "$model" in
        qwen3_8b)         return 0 ;;
        *)                return 1 ;;
    esac
}

VALID_MODELS="mistral_v0_3 llama31_8b qwen25_7b qwen3_8b"
VALID_CHARACTERS="baseline"
VALID_VARIANTS="standard deep"
