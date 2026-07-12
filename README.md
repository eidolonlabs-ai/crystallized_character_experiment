# Crystallized Character Experiment (E002)

This project explores the "crystallization" of AI character personalities by fine-tuning efficient LLMs on their specific conversation history using **MLX** on Apple Silicon. The goal is to "bake" the character's emergent voice, quirks, and identity directly into the model weights.

## Documentation

| Document | What it covers |
|----------|---------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the training pipeline works — read this first |
| [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | Data preparation, splitting, and truncation |
| [`MLX_SETUP.md`](MLX_SETUP.md) | MLX installation, training commands, memory tuning |
| [`A_B_TEST_COMMANDS.md`](A_B_TEST_COMMANDS.md) | Comparing standard vs deep variants, test prompts |
| [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) | Knob-by-knob rationale: rank, alpha, scheduler, optimizer, etc. |
| [`docs/CHAT_TEMPLATES.md`](docs/CHAT_TEMPLATES.md) | The Phase 0 system-prompt gotcha and how to verify your model |
| [`docs/MULTI_TURN.md`](docs/MULTI_TURN.md) | Why training data is single-turn and how to extend it |
| [`docs/PREFERENCE_TUNING.md`](docs/PREFERENCE_TUNING.md) | DPO/IPO/KTO — what's missing from mlx-lm 0.30 and how to add it |
| [`configs/README.md`](configs/README.md) | Per-model YAML overrides for the training recipe |
| [`E002_crystallized_character.md`](E002_crystallized_character.md) | Original experiment proposal and results |
| [`CLAUDE.md`](CLAUDE.md) | Reference for AI coding assistants |

## Project Structure

```
.
├── ARCHITECTURE.md                 # How the training pipeline works (read this!)
├── raw_data/                       # Training data (JSONL)
├── scripts/                        # Training and utility scripts
│   ├── train_character_model.sh   # Consolidated MLX training script
│   ├── train_mlx.py               # MLX LoRA training engine
│   ├── split_training_data.py     # Prepare character data
│   └── truncate_training_data.py  # Limit sequence length
├── adapters/                      # LoRA adapter outputs (~5-10MB each)
├── models/                        # MLX quantized models & merged results
└── requirements.txt               # Python dependencies
```

## Quick Start (Clone & Train)

If you're cloning this repo on a new machine and want to run the full training pipeline:

```bash
# 1. Clone and navigate
git clone <repository-url>
cd crystallized_character_experiment

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full baseline training suite (trains all models)
./train_baseline_suite.sh run

# Or train a single model
./scripts/train_character_model.sh baseline mistral_v0_3

# 5. Chat with your trained model
./chat_character.sh baseline mistral_v0_3
```

## MLX Training (Apple Silicon)

The project uses **MLX** exclusively for optimal performance on Apple Silicon Macs.

### Quick Start

**Train all baseline models at once:**
```bash
./train_baseline_suite.sh test    # Preview what will train
./train_baseline_suite.sh run     # Start full training suite
```

**Train individual models:**
```bash
# Train Baseline with Mistral 7B (standard)
./scripts/train_character_model.sh baseline mistral_v0_3

# Deep variant
./scripts/train_character_model.sh baseline mistral_v0_3 deep

# Chat with trained models
./chat_character.sh baseline mistral_v0_3
./chat_character.sh baseline mistral_v0_3 deep
```

### Available Models

| Model ID | Base Model | Size |
|----------|-----------|------|
| `mistral_v0_3` | Mistral 7B v0.3 | 7B |
| `llama31_8b` | Llama 3.1 8B | 8B |

### Training Variants

- **`standard`** (default): Balanced fine-tuning
  - 8-layer LoRA, 5e-5 learning rate, 512 max sequence length, 5 epochs
  
- **`deep`**: Deeper personality embedding
  - 16-layer LoRA, 2.5e-5 learning rate, 768 max sequence length, 5 epochs

### Detailed Workflow

#### Step 1: Prepare Data
```bash
# One-shot data preparation
./prepare_all_datasets.sh
```

#### Step 2: Run Training
```bash
# All-in-one training (auto-trains, merges)
./scripts/train_character_model.sh baseline mistral_v0_3```

The script will:
1. Load base model
2. Use pre-prepared and pre-truncated training data
3. Run LoRA fine-tuning
4. Merge adapter with base model to MLX format

#### Step 3: Chat/Test
```bash
# With LoRA adapter (lightweight, ~10MB):
python -m mlx_lm chat --model models/mistral-7b-instruct-v0.3-4bit --adapter-path adapters/baseline_mistral_v0_3_qlora

# With merged model (self-contained):
python -m mlx_lm chat --model models/baseline_mistral_v0_3_mlx_q4
```

## Setup (Local - macOS with Apple Silicon)

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Environment Variables
For synthetic data generation:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

## Data Pipeline

### 1. Prepare Character Data
Extract and organize character-specific training data:
```bash
python scripts/split_training_data.py --character baseline
```

### 2. Generate Synthetic Data (Optional)
Augment real data with LLM-generated examples:
```bash
python scripts/generate_synthetic_data.py --character baseline
```

### 3. Curate Data (Optional)
Filter and clean the dataset:
```bash
python scripts/curate_training_data.py --input raw_data/training_data_baseline_augmented.jsonl
```

### 4. Train the Model
Start consolidated training:
```bash
./scripts/train_character_model.sh baseline mistral_v0_3 deep
```

## Model Outputs

Each training run produces:

**LoRA Adapter** (`adapters/`)
- Lightweight (~5-10MB)
- Contains only personality differences
- Can be applied to base model at inference

**Merged MLX Model** (`models/`)
- Standalone model
- Ready for deployment
- Use for: production, offline use, LM Studio deployment

## Deployment to LM Studio

### Export to LM Studio
```bash
./scripts/export_to_lmstudio.sh baseline mistral_v0_3
./scripts/export_to_lmstudio.sh baseline llama31_8b deep
```

This copies the merged MLX model to LM Studio's models directory.

## Notes

- **Hardware**: Optimized for Apple Silicon (M1/M2/M3/M4 Macs) using MLX
- **No PyTorch**: MLX-exclusive for better performance
- **No QLoRA**: mlx-lm uses native fp16/bf16 on Apple Silicon's unified memory; NF4 quantization (Dettmers et al. 2023) isn't needed and isn't implemented here. If you want to experiment with QLoRA-style training on a non-Apple GPU, this repo won't help — that's a different stack (bitsandbytes + HuggingFace PEFT).
