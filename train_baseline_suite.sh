#!/bin/bash
# Baseline Character Model - Full Training Suite
# Trains all appropriate models with baseline augmented dataset
# Usage: ./train_baseline_suite.sh [test|run]

set -e

REPO_ROOT="$(dirname "$0")"
cd "$REPO_ROOT"

# Validate we're in the right directory
if [ ! -f "scripts/train_character_model.sh" ]; then
    echo "ERROR: scripts/train_character_model.sh not found"
    echo "This script must be run from the project root directory"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "ERROR: Python virtual environment not found at .venv"
    echo "Please create it with: python -m venv .venv"
    exit 1
fi

MODELS=(
    "mistral_v0_3"  # Mistral 7B v0.3
    "llama31_8b"    # Llama 3.1 8B Instruct
)

CHARACTERS=("baseline")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse arguments
MODE="${1:-test}"

if [ "$MODE" = "test" ]; then
    print_header "BASELINE TRAINING SUITE - TEST MODE (DRY RUN)"
    print_info "No models will be trained. Showing what would be executed:"
    echo ""
    
    for character in "${CHARACTERS[@]}"; do
        echo -e "${YELLOW}Character: $character${NC}"
        for model in "${MODELS[@]}"; do
            echo "  → ./scripts/train_character_model.sh $character $model"
        done
        echo ""
    done
    
    exit 0
    
elif [ "$MODE" = "run" ]; then
    print_header "BASELINE TRAINING SUITE - RUNNING"
    
    total_models=0
    for character in "${CHARACTERS[@]}"; do
        total_models=$((total_models + ${#MODELS[@]}))
    done
    
    print_info "Total training runs: $total_models (approx 2-3 hours total)"
    print_info "Each model: 12-18 minutes"
    print_info "Models: ${MODELS[*]}"
    print_info "Characters: ${CHARACTERS[*]}"
    print_warning "This will consume significant compute. Press Ctrl+C to cancel."
    echo ""
    read -p "Press Enter to start training, or Ctrl+C to cancel..."
    echo ""
    
    start_time=$(date +%s)
    failed=()
    succeeded=()
    
    for character in "${CHARACTERS[@]}"; do
        print_header "Training $character dataset"
        
        for model in "${MODELS[@]}"; do
            run_start=$(date +%s)
            
            print_info "Starting: $character / $model"
            if ./scripts/train_character_model.sh "$character" "$model" 2>&1; then
                run_end=$(date +%s)
                duration=$((run_end - run_start))
                print_success "$character / $model (${duration}s)"
                succeeded+=("$character/$model")
            else
                run_end=$(date +%s)
                duration=$((run_end - run_start))
                print_error "$character / $model (${duration}s)"
                failed+=("$character/$model")
            fi
            echo ""
        done
    done
    
    end_time=$(date +%s)
    total_duration=$((end_time - start_time))
    hours=$((total_duration / 3600))
    minutes=$(((total_duration % 3600) / 60))
    
    print_header "TRAINING COMPLETE"
    
    print_success "Succeeded: ${#succeeded[@]}"
    for item in "${succeeded[@]}"; do
        echo "  ✅ $item"
    done
    echo ""
    
    if [ ${#failed[@]} -gt 0 ]; then
        print_error "Failed: ${#failed[@]}"
        for item in "${failed[@]}"; do
            echo "  ❌ $item"
        done
        echo ""
    fi
    
    print_info "Total time: ${hours}h ${minutes}m"
    
else
    echo "Usage: $0 [test|run]"
    echo ""
    echo "  test   - Show what would be trained (dry run, no actual training)"
    echo "  run    - Actually train all models"
    echo ""
    echo "Example:"
    echo "  $0 test  # Preview the training plan"
    echo "  $0 run   # Execute full training suite"
    exit 1
fi
