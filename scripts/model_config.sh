#!/bin/bash
# Shared model configuration — single source of truth for model mappings.
# Source this file from any script that needs model configs.
#
# Usage: source "$(dirname "$0")/model_config.sh"

# Map model name to HuggingFace repo ID
get_hf_model() {
    local model="$1"
    case "$model" in
        # Mistral family
        mistral|mistral_v0_3) echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
        mistral_v0_2)         echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
        mistral_v0_1)         echo "mistralai/Mistral-7B-Instruct-v0.1" ;;
        # Llama family
        llama|llama31_8b)     echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        llama3_8b)            echo "meta-llama/Meta-Llama-3-8B-Instruct" ;;
        llama2_7b)            echo "meta-llama/Llama-2-7b-chat-hf" ;;
        *)                    echo "" ;;
    esac
}

get_quantized_model_path() {
    local model="$1"
    case "$model" in
        # Mistral family
        mistral|mistral_v0_3) echo "models/mistral-7b-instruct-v0.3-4bit" ;;
        mistral_v0_2)         echo "models/mistral-7b-instruct-v0.2-4bit" ;;
        mistral_v0_1)         echo "models/mistral-7b-instruct-v0.1-4bit" ;;
        # Llama family
        llama|llama31_8b)     echo "models/llama-3.1-8b-instruct-4bit" ;;
        llama3_8b)            echo "models/llama-3-8b-instruct-4bit" ;;
        llama2_7b)            echo "models/llama-2-7b-chat-4bit" ;;
        *)                    echo "" ;;
    esac
}

VALID_MODELS="mistral_v0_3 mistral_v0_2 mistral_v0_1 llama31_8b llama3_8b llama2_7b"
# Backward-compat aliases
VALID_MODELS="$VALID_MODELS mistral llama"
VALID_CHARACTERS="baseline"
VALID_VARIANTS="standard deep"
