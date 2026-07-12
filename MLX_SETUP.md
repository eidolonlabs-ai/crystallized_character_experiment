# MLX Training Setup

## Quick Start

The easiest way to train models is using the consolidated training script:

```bash
# Train a single model
./scripts/train_character_model.sh baseline mistral_v0_3

# Train all baseline models at once
./train_baseline_suite.sh test    # Preview
./train_baseline_suite.sh run     # Execute

# Chat with trained models
./chat_character.sh baseline mistral_v0_3
./chat_character.sh baseline mistral_v0_3 deep
```

## Installation

Install MLX dependencies via requirements.txt:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install transformers mlx-lm langchain-openai langchain-core python-dotenv
```

## Training with Consolidated Script (Recommended)

The consolidated training script handles everything automatically:

```bash
# Train with standard variant (8-layer LoRA, 512 seq length)
./scripts/train_character_model.sh baseline mistral_v0_3

# Train with deep variant (16-layer LoRA, 768 seq length)
./scripts/train_character_model.sh baseline mistral_v0_3 deep

# Train on different models
./scripts/train_character_model.sh baseline llama31_8b
./scripts/train_character_model.sh baseline llama31_8b
```

What the script does automatically:
1. Loads base model
2. Selects best available training data (augmented_curated > augmented > base > full)
3. Uses pre-prepared and pre-truncated training data
4. Runs LoRA fine-tuning with optimal hyperparameters
5. Merges adapter with base model to MLX format

## Direct MLX Training (Advanced)

For direct control over MLX training, run the Python script:
```bash
python scripts/train_mlx.py \
    --model models/mistral-7b-instruct-v0.3-4bit \
    --data_dir raw_data/prepared_data/baseline/augmented_split_512 \
    --output_dir adapters/baseline_mistral_qlora \
    --epochs 5 \
    --batch-size 1 \
    --num-layers 8 \
    --learning-rate 5e-5
```

## Command-Line Options

### train_character_model.sh

```bash
./scripts/train_character_model.sh <character> <model_name> [variant]
```

**Characters**: baseline
**Models**: mistral_v0_3, llama31_8b
**Variants**: standard (default), deep

**Example**:
```bash
./scripts/train_character_model.sh baseline mistral_v0_3 deep
```

### train_mlx.py (Advanced)

Required Arguments:
- `--model`: Base model path (e.g., `models/mistral-7b-instruct-v0.3-4bit`)
- `--data_dir`: Directory containing `train.jsonl` and `valid.jsonl`
- `--output_dir`: Output directory for adapters

Training Configuration:
- `--epochs`: Number of training epochs (default: 10)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--num-layers`: Number of layers to fine-tune (default: 16)
- `--batch-size`: Batch size (default: 1)
- `--max-seq-length`: Max sequence length (default: 4096)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 2)

### Advanced Options
- `--qlora`: Use QLoRA (quantized LoRA) - experimental, may cause NaN loss

## Memory Guidelines

**Llama 3.1 8B on Apple Silicon:**
- **64GB RAM**: Standard settings work well (batch=1, grad_accum=2, seq_len=1024)
- **32GB RAM**: Use lower memory profile (batch=1, grad_accum=1, seq_len=768)
- **Peak memory**: ~18GB (model) + ~40GB (system) during training

**Training stops at high memory?** Reduce in this order:
1. `--gradient-accumulation-steps 1`
2. `--max-seq-length 768`
3. `--num-layers 8`

## Inference / Chat with Trained Model

### Using Final Trained Adapter

**Interactive chat (recommended):**
```bash
./chat_character.sh baseline mistral
```

**With custom parameters:**
```bash
python -m mlx_lm chat \
    --model models/mistral-7b-instruct-v0.3-4bit \
    --adapter-path adapters/baseline_mistral_v0_3_qlora \
    --temp 0.7 \
    --max-tokens 500
```

**Single prompt generation:**
```bash
python -m mlx_lm generate \
    --model models/mistral-7b-instruct-v0.3-4bit \
    --adapter-path adapters/baseline_mistral_v0_3_qlora \
    --prompt "Who are you?" \
    --max-tokens 200
```

### Inference Parameters

**Control generation quality:**
```bash
python -m mlx_lm chat \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --adapter-path adapters/baseline_llama31_8b_qlora \
    --max-tokens 500 \
    --temp 0.7 \
    --top-p 0.9
```

**Useful parameters:**
- `--temp` - Temperature (0.1-1.0, higher = more creative)
- `--top-p` - Nucleus sampling threshold
- `--max-tokens` / `-m` - Max response length
- `--system-prompt` - Override system prompt
- `--max-kv-size` - KV cache size for long conversations
- `--seed` - For reproducible generation

## Notes

- MLX uses its own data format and LoRA implementation
- Optimized specifically for Apple Silicon unified memory
- Adapters saved in MLX format
- QLoRA support via mlx-lm-lora but unstable with Llama 3.1 8B
- Training checkpoints saved every 50 iterations
- Use `--mask-prompt` flag NOT recommended (causes NaN loss with Llama 3.1)
