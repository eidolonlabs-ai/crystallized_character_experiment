# Architecture: LoRA Character Fine-tuning on MLX

This document explains how the training pipeline works so you can understand, modify, and extend it. It assumes you've read the README and have basic familiarity with fine-tuning concepts.

## High-level overview

The pipeline takes conversation history for a character, formats it as training data, and uses LoRA (Low-Rank Adaptation) to bake the character's voice into a model's weights. All of this runs on Apple Silicon via MLX — no GPU cluster, no Docker, no PyTorch.

```
                  ┌─────────────────────┐
                  │  conversation log    │  (JSONL, checked into git)
                  │  training_data_*.jsonl│
                  └──────────┬──────────┘
                             │
              ┌──────────────▼──────────────┐
              │    prepare_all_datasets.sh   │  (run once, commit output)
              │    split 90/10 + truncate    │
              └──────────────┬──────────────┘
                             │
         raw_data/prepared_data/baseline/{variant}_split_512/
                  ┌──────────┴──────────┐
                  │ train.jsonl         │ valid.jsonl
                  └──────────┬──────────┘
                             │
              ┌──────────────▼──────────────┐
              │ train_character_model.sh    │  (one command per model)
              │   → train_mlx.py            │
              │   → python -m mlx_lm lora   │
              │   → python -m mlx_lm fuse   │
              └──────────────┬──────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                                     ▼
adapters/baseline_mistral_v0_3_qlora/    models/baseline_mistral_v0_3_mlx_fp16/
(LoRA weights, ~10 MB)              (full merged model, ~13 GB)
        │
        ▼
chat_character.sh baseline mistral_v0_3
(loads base model + applies LoRA adapter at inference time)
```

## Core concepts

### Why LoRA?

Fine-tuning a full 7B-parameter model would require ~56 GB of GPU memory just for the optimizer states. LoRA instead freezes the original weights and trains only small "adapter" matrices injected into specific layers. This means:

- Training fits on a MacBook with unified memory
- The adapter file is ~5-10 MB (vs. ~13 GB for the full model)
- You can have many adapters for different characters sharing one base model
- Swapping characters is instant — just change the adapter path

Our adapters target 8 or 16 layers with rank 8 (configurable in the training script).

### The MLX difference

MLX (Apple's machine learning framework) is designed specifically for unified memory — the GPU and CPU share the same physical RAM on Apple Silicon. This eliminates the GPU→CPU data copies that dominate PyTorch training, and lets us train models that would otherwise OOM on the same hardware.

## The training script walkthrough

`scripts/train_character_model.sh` is the entry point. It takes a character name, model name, and optional variant/quantize flags:

```bash
./scripts/train_character_model.sh baseline mistral_v0_3
./scripts/train_character_model.sh baseline llama31_8b deep 4bit
```

Here's what happens step by step:

### Step 1: Configuration resolution

The script sources `scripts/model_config.sh` which maps model names to HuggingFace repos:

| You type    | Resolves to                               |
|-------------|-------------------------------------------|
| `mistral_v0_3` | `mistralai/Mistral-7B-Instruct-v0.3`   |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct`    |

All model names use their canonical form directly — no aliases. Adapter paths are always consistent.

Training hyperparameters come from the variant selection:

| Parameter          | standard (default) | deep              |
|--------------------|--------------------|--------------------|
| LoRA layers        | 8                  | 16                 |
| Learning rate      | 5e-5               | 2.5e-5             |
| Max seq length     | 512                | 768                |
| Epochs             | 5                  | 5                   |
| Effective batch    | 2 (1×2 accum)      | 4 (1×4 accum)      |

### Step 2: Dataset selection

Training reads from `raw_data/prepared_data/baseline/{variant}_split_{seq}/`. The script picks the first available variant matching the model's sequence length:

1. `augmented_curated_split_512` — curated high-quality examples (if available)
2. `augmented_split_512` — augmented with synthetic data
3. `base_split_512` — original data only
4. `full_split_512` — largest raw dataset

The deep variant uses `_768` suffix instead. The data was pre-split and pre-truncated by `prepare_all_datasets.sh`, so training never does this work at runtime.

### Step 3: LoRA training

The script calls `scripts/train_mlx.py`, which wraps `python -m mlx_lm lora` with the correct flags:

```
python -m mlx_lm lora \
    --model <huggingface-repo> \
    --train --data <prepared-data-dir> \
    --adapter-path adapters/baseline_mistral_v0_3_qlora \
    --epochs 5 --batch-size 1 --gradient-accumulation-steps 2 \
    --num-layers 8 --learning-rate 5e-5 --max-seq-length 512 \
    --no-mask-prompt --save-every 50
```

Key details:
- `--no-mask-prompt` means loss is computed on the full sequence including the system prompt. This is intentional — we want the model to learn the character definition.
- `--save-every 50` creates checkpoints every 50 iterations for resume capability and A/B testing of intermediate states.
- Training auto-resumes from the latest checkpoint if it detects existing `*_adapters.safetensors` files.

### Step 4: Adapter merging (fuse)

After training, the LoRA weights are "fused" (merged) into the base model to create a standalone model file:

```bash
rm -rf models/baseline_mistral_v0_3_mlx_fp16         # clean stale output
mkdir -p models/baseline_mistral_v0_3_mlx_fp16
python -m mlx_lm fuse \
    --model <huggingface-repo> \
    --adapter-path adapters/baseline_mistral_v0_3_qlora \
    --save-path models/baseline_mistral_v0_3_mlx_fp16
```

The output directory is cleaned first to avoid write conflicts with leftover files from previous runs.

With `4bit` quantization mode, the fused model is also quantized:
```
python -m mlx_lm fuse ... --quantize --q-bits 4
# Output: models/baseline_mistral_v0_3_mlx_q4/
```

## Path conventions

Every artifact follows a predictable naming scheme so scripts can find each other without configuration:

```
adapters/{character}_{model}_qlora{_deep}/
  ├── adapters.safetensors      ← LoRA weights
  ├── adapter_config.json       ← LoRA hyperparams
  └── 0000050_adapters.safetensors  ← checkpoint (every 50 iters)

models/{character}_{model}_mlx_{fp16|q4}{_deep}/
  ├── model.safetensors         ← merged weights
  └── model.safetensors.index.json
```

Examples:
- `adapters/baseline_mistral_v0_3_qlora/` — standard Mistral adapter
- `adapters/baseline_llama31_8b_qlora_deep/` — deep Llama adapter
- `models/baseline_mistral_v0_3_mlx_fp16/` — full-precision fused model
- `models/baseline_llama31_8b_mlx_q4_deep/` — 4-bit quantized deep model

## Chat and inference

`chat_character.sh` combines a base model with a LoRA adapter for interactive chat:

```
python -m mlx_lm chat \
    --model <base-model-or-hf-repo> \
    --adapter-path adapters/baseline_mistral_v0_3_qlora \
    --system-prompt "You are Lyra Moonwhisper..."
```

If the quantized 4-bit model exists locally, it's used for faster loading. Otherwise the script falls back to the HuggingFace repo ID, which mlx-lm auto-downloads (and on first use, quantizes to 4-bit into the local `models/` cache). The local-quantized path is a performance optimization — the common case today is the HF fallback, since `train_character_model.sh` produces fused fp16 outputs (`models/<char>_<model>_mlx_fp16[_deep]`) and not 4-bit conversions. Run `python -m mlx_lm convert --hf-path <hf_repo> --mlx-path models/<name>-4bit --quantize --q-bits 4` once per model to pre-create the local quantized cache.

The system prompt is hand-written in `chat_character.sh` — it's not extracted from training data. If you update the character definition in the training data, update it in both scripts too.

## Data pipeline in detail

Source data lives as JSONL files at the repo root: `raw_data/training_data_baseline*.jsonl`. Each line is a conversation in ChatML format:

```json
{
    "messages": [
        {"role": "system", "content": "You are Lyra Moonwhisper..."},
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am Lyra Moonwhisper, keeper of the Celestial Archives..."}
    ]
}
```

`prepare_all_datasets.sh` processes these once (you commit the output):

1. **Split**: 90% train / 10% validation with `random.seed(42)` for reproducibility
2. **Truncate**: Each conversation is trimmed to a max token count using the model's tokenizer, keeping the system prompt and most recent turns
3. **Version**: Output goes to `raw_data/prepared_data/baseline/{variant}_split_{seq}/`

Three variants exist:
- `base` — original data only (~22 examples)
- `augmented` — original + synthetic augmentation (~319 examples)
- `full` — largest untruncated dataset (~270 examples)

An `augmented_curated` variant can be created by running `scripts/curate_training_data.py`, which filters low-quality examples and trims long responses.

## Adding a new model

To add a new model (e.g., a new Llama version):

1. **`scripts/model_config.sh`**: Add entries in both `get_hf_model()` and `get_quantized_model_path()`:
   ```bash
   llama4_8b) echo "meta-llama/Llama-4-8B-Instruct" ;;
   ```

2. **`scripts/model_config.py`**: Add a matching entry in `MODEL_CONFIGS`:
   ```python
   "llama4_8b": {
       "hf": "meta-llama/Llama-4-8B-Instruct",
       "quantized": "models/llama-4-8b-instruct-4bit",
   }
   ```

3. **`scripts/test_inference_quality.py`**: Reads `MODEL_CONFIGS` from `scripts/model_config.py`, so no change needed here as long as the config entry exists. (The old hardcoded dict was removed.)

4. **`train_baseline_suite.sh`**: Add the model to the `MODELS` array if you want the suite to train it.

That's it — the training script, chat script, and test scripts all derive their paths from these configs dynamically.

## Adding a new character

1. **Create training data**: `raw_data/training_data_newchar.jsonl` in ChatML format
2. **Add system prompt** in `scripts/character_config.sh` and `chat_character.sh`
3. **Prepare data**: Run `./prepare_all_datasets.sh` (edit it to include the new character)
4.  **Train**: `./scripts/train_character_model.sh newchar mistral_v0_3`

## Metal environment variables

Three environment variables are critical for stable MLX training on Apple Silicon:

| Variable | Value | Why |
|----------|-------|-----|
| `MLX_METAL_DEBUG` | `1` | Enables Metal shader validation (catches GPU errors early) |
| `OMP_NUM_THREADS` | `1` | Prevents OpenMP from spawning extra threads that compete with Metal |
| `MKL_NUM_THREADS` | `1` | Same for MKL-based operations |

These are set automatically by `train_character_model.sh`.

## Files of interest

| File | Purpose |
|------|---------|
| `scripts/model_config.sh` | Single source of truth for model→HF repo and →quantized path mappings |
| `scripts/model_config.py` | Python equivalent of the above, used by analysis scripts |
| `scripts/character_config.sh` | System prompts per character |
| `scripts/train_character_model.sh` | Entry point: orchestrates the full pipeline |
| `scripts/train_mlx.py` | Thin Python wrapper around `mlx_lm lora` |
| `prepare_all_datasets.sh` | One-shot data prep (split + truncate) |
| `train_baseline_suite.sh` | Runs training across both supported models |
| `chat_character.sh` | Interactive chat with a trained adapter |
| `test_trained_models.sh` | Smoke-tests all adapters for a character |
| `scripts/split_training_data.py` | Splits JSONL into train/valid (standalone utility) |
| `scripts/truncate_training_data.py` | Token-level truncation of conversations |
| `scripts/generate_synthetic_data.py` | LLM-powered data augmentation |
| `scripts/curate_training_data.py` | Filter and clean training examples |
| `scripts/validate_all_jsonl.py` | Check all JSONL files for parse errors |
| `scripts/validate_dataset_quality.py` | Deep validation of ChatML format and character consistency |
| `scripts/export_to_lmstudio.sh` | Copy fused model to LM Studio for GUI chat |

## Training suite flow

`train_baseline_suite.sh` trains both supported models sequentially, tracking successes and failures:

```
mistral_v0_3 → llama31_8b
```

This provides a full comparison matrix — 2 models × 2 variants (standard/deep) = 4 configurations. The suite is designed for A/B testing: train everything, then compare voice quality across architectures.
