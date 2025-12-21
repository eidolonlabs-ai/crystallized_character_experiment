#!/bin/bash
# ============================================================================
# Test All Trained Models - Inference Quality Check
# ============================================================================
#
# This script tests all successfully trained models by running them through
# standard test prompts and evaluating response quality.
#
# Usage:
#   ./test_trained_models.sh [character] [num_prompts]
#
# Examples:
#   ./test_trained_models.sh baseline        # Test all baseline models
#   ./test_trained_models.sh baseline 3      # Test with 3 prompts each
#
# ============================================================================

CHARACTER="${1:-baseline}"
NUM_PROMPTS="${2:-5}"

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Validate character
if [ "$CHARACTER" != "baseline" ]; then
    echo "Error: Unknown character '$CHARACTER'"
    echo "Available: baseline"
    exit 1
fi

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found"
    exit 1
fi
source .venv/bin/activate

# Source shared model config (provides get_quantized_model_path, get_hf_model, etc.)
SCRIPT_DIR_TTM="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR_TTM/scripts/model_config.sh"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test prompts by character
get_test_prompts() {
    local char="$1"
    local num="$2"
    
    case "$char" in
        baseline)
            echo "Who are you?"
            echo "What is your name and role?"
            echo "Tell me about your wisdom"
            echo "How do you greet people?"
            echo "What matters most to you?"
            ;;
    esac | head -n "$num"
}

# Colors for output
print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_model() {
    echo -e "${YELLOW}▶ Testing: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Find all trained adapters for this character
print_header "Testing $CHARACTER Models"

echo "Character: $CHARACTER"
echo "Test prompts: $NUM_PROMPTS per model"
echo ""

ADAPTERS=(adapters/${CHARACTER}_*_qlora)
MODELS_TESTED=0
MODELS_PASSED=0
MODELS_FAILED=()

# Check if any adapters exist
if [ ! -d "${ADAPTERS[0]%/*}" ]; then
    echo "Error: No adapters directory found"
    exit 1
fi

# Test each adapter
for adapter_dir in "${ADAPTERS[@]}"; do
    if [ ! -d "$adapter_dir" ]; then
        continue
    fi
    
    MODELS_TESTED=$((MODELS_TESTED + 1))
    adapter_name=$(basename "$adapter_dir")
    
    # Extract model name from adapter path (baseline_mistral_qlora -> mistral)
    model_name=$(echo "$adapter_name" | sed "s/${CHARACTER}_\(.*\)_qlora/\1/")
    
    # Normalize underscores back to dots for display (llama31_8b -> llama31.8b)
    model_display="${model_name//_/.}"
    
    print_model "$model_display"
    
    # Find corresponding quantized model
    # Map model names to quantized model paths via scripts/model_config.sh
    quantized="$(get_quantized_model_path "$model_name")"
    hf_model="$(get_hf_model "$model_name")"
    
    # Use quantized model if available, otherwise fall back to HF repo
    if [ -d "$quantized" ]; then
        chat_model="$quantized"
    elif [ -n "$hf_model" ]; then
        chat_model="$hf_model"
    else
        echo "  ✗ Model not found: $quantized"
        MODELS_FAILED+=("$model_name")
        continue
    fi
    
    # Check adapter files exist
    if [ ! -f "$adapter_dir/adapters.safetensors" ]; then
        echo "  ✗ Adapter weights not found"
        MODELS_FAILED+=("$model_name")
        continue
    fi
    
    # Run test prompts
    prompts_passed=0
    prompts_tested=0
    
    while IFS= read -r prompt; do
        prompts_tested=$((prompts_tested + 1))
        
        # Try inference with timeout
        output=$(timeout 30 python -m mlx_lm generate \
            --model "$chat_model" \
            --adapter-path "$adapter_dir" \
            --prompt "$prompt" \
            --max-tokens 50 \
            2>&1 | tail -1)
        
        if [ $? -eq 0 ] && [ -n "$output" ] && [ ${#output} -gt 10 ]; then
            prompts_passed=$((prompts_passed + 1))
        fi
    done < <(get_test_prompts "$CHARACTER" "$NUM_PROMPTS")
    
    # Report results
    if [ "$prompts_passed" -ge $((prompts_tested / 2)) ]; then
        print_success "Inference working ($prompts_passed/$prompts_tested prompts)"
        MODELS_PASSED=$((MODELS_PASSED + 1))
    else
        echo "  ✗ Inference issues ($prompts_passed/$prompts_tested prompts)"
        MODELS_FAILED+=("$model_name")
    fi
done

# Summary
print_header "Summary"

echo "Models tested: $MODELS_TESTED"
echo "Models passed: $MODELS_PASSED"
echo "Models failed: $((MODELS_TESTED - MODELS_PASSED))"
echo ""

if [ $MODELS_PASSED -eq $MODELS_TESTED ]; then
    print_success "All models passed inference testing!"
else
    if [ ${#MODELS_FAILED[@]} -gt 0 ]; then
        echo "Failed models:"
        for model in "${MODELS_FAILED[@]}"; do
            echo "  ✗ $model"
        done
    fi
fi

echo ""
echo "To chat with a model interactively:"
echo "  ./chat_character.sh $CHARACTER <model_name>"
echo ""
echo "Example:"
echo "  ./chat_character.sh $CHARACTER mistral"
echo ""
