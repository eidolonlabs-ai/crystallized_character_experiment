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
         raw_data/prepared_data/baseline/{variant}_split_2048/
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

### A note on "QLoRA" naming

The adapter paths and filenames use `_qlora` (e.g., `adapters/baseline_mistral_v0_3_qlora/`), but this is **not QLoRA** in the Dettmers et al. (2023) sense. True QLoRA uses NF4 quantization of the base model *during training* to fit larger models on consumer GPUs, implemented via bitsandbytes + HuggingFace PEFT on NVIDIA hardware.

This repo uses **standard LoRA** — the base model stays in FP16 during training. MLX leverages Apple Silicon's unified memory architecture, where the GPU and CPU share RAM, so the ~14 GB needed for a 7B model in FP16 fits comfortably without quantization tricks. The `_qlora` suffix is a historical artifact from an early naming decision and was kept to avoid breaking every script and adapter directory that depends on it.

If you see `_qlora` in a path, read it as "LoRA adapter" — no quantization is applied to the base model during training.

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
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen3_8b` | `Qwen/Qwen3-8B` |

All model names use their canonical form directly — no aliases. Adapter paths are always consistent.

Training hyperparameters come from the variant selection:

| Parameter          | standard (default) | deep              |
|--------------------|--------------------|--------------------|
| LoRA layers        | 8                  | 16                 |
| Learning rate      | 5e-5               | 2.5e-5             |
| Max seq length     | 2048               | 2048               |
| Epochs             | 5                  | 5                   |
| Effective batch    | 2 (1×2 accum)      | 4 (1×4 accum)      |

### Step 2: Dataset selection

Training reads from `raw_data/prepared_data/baseline/{variant}_split_{seq}/`. The script picks the first available variant matching the model's sequence length:

1. `augmented_curated_split_2048` — curated high-quality examples (if available)
2. `augmented_split_2048` — augmented with synthetic data
3. `base_split_2048` — original data only
4. `full_split_2048` — largest raw dataset

Both variants use `_2048` suffix. All 355 examples fit within 2048 tokens without truncation. The data was pre-split and pre-truncated by `prepare_all_datasets.sh`, so training never does this work at runtime.

### Step 3: LoRA training

The script calls `scripts/train_mlx.py`, which wraps `python -m mlx_lm lora` with the correct flags:

```
python -m mlx_lm lora \
    --model <huggingface-repo> \
    --train --data <prepared-data-dir> \
    --adapter-path adapters/baseline_mistral_v0_3_qlora \
    --epochs 5 --batch-size 1 --gradient-accumulation-steps 2 \
    --num-layers 8 --learning-rate 5e-5 --max-seq-length 2048 \
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
- `augmented` — original + synthetic augmentation (~355 examples)
- `full` — largest untruncated dataset (~270 examples)

An `augmented_curated` variant is created by running `scripts/curate_training_data.py`, which flattens multi-turn to single-turn and trims long responses. An `augmented_curated_thinking` variant extends this further with `<think>` blocks for reasoning models.

## Synthetic data generation: best practices

The seed of 22 hand-written conversations is the anchor. Everything else is LLM-generated scaffolding built on that voice sample. Each layer is additive — it derives from the previous layer, never replaces the seed. The full chain:

```
22 hand-written (base)
    ↓  generate_synthetic_data.py  (few-shot generation)
355 augmented
    ↓  curate_training_data.py     (flatten to single-turn, trim length)
355 augmented_curated
    ↓  generate_thinking_data.py   (prepend <think> inner monologue)
355 augmented_curated_thinking
```

All source JSONL files are committed to git. `prepare_all_datasets.sh` processes them once and commits the output. The entire chain is reproducible.

### Layer 1: Base (hand-written)

**Script**: N/A — created manually in `raw_data/training_data_baseline.jsonl`.

**Purpose**: The ground-truth voice sample. These 22 examples define the character's register (archaic formality, elvish blessings, nature metaphors), response length, and conversational range.

**Best practices:**
- Write 15-30 examples covering different scenarios (greetings, emotional support, lore questions, casual chat)
- Each example should be 3-6 turns — enough to show conversational rhythm, but not so long they dominate the few-shot prompt
- Vary user personas (traveler, scholar, child, warrior) so the character's voice generalizes across audiences
- Include edge cases: the user is rude, confused, or asks something the character can't answer
- **Don't** include system messages — the character definition is baked into the character prompt, not the seed data

### Layer 2: Augmented (synthetic generation)

**Script**: `scripts/generate_synthetic_data.py`.

**Purpose**: Scale from 22 to 355 examples by having GPT-4o write new conversations in the same voice.

**Technique**: Few-shot generation with random sampling.
- For each new conversation, randomly sample 3 seed examples as in-context demonstrations
- Pass the character description (from `character_config.py`) alongside the examples
- Request 3-6 turn JSON output with explicit format instructions

**Best practices:**
- **Random few-shot prevents memorization** — sampling different examples each time ensures the LLM generalizes the voice pattern rather than copying specific phrasing
- **Temperature 0.7** — enough variation that conversations don't feel templated, not so high that the voice drifts
- **Prompt includes both character definition and examples** — the definition sets constraints (register, tone, topics), the examples show rhythm, length, and specific mannerisms
- **Validate JSON output** — wrap generation in try/except for `json.JSONDecodeError`; strip markdown code fences (` ```json `)
- **Shuffle after merging** — interleave real + synthetic so the train/valid split isn't biased by source
- **Async with sequential loop** — one at a time (not concurrent) to avoid rate-limiting and keep prompt context size manageable
- **Commit the output** — `training_data_baseline_augmented.jsonl` is versioned so retraining is deterministic

### Layer 3: Curated (flatten + trim)

**Script**: `scripts/curate_training_data.py`.

**Purpose**: Standardize every example to single-turn `[system, user, assistant]` triplets and trim overly long responses.

**Technique**:
- Extract the last user-assistant pair from each conversation (drops earlier turns)
- If the assistant response exceeds `max_response_tokens` (default 600), trim the middle — keeps first `keep_start_tokens` (personality setup) and last `keep_end_tokens` (conclusion/elvish blessing)
- Outputs `training_data_baseline_augmented_curated.jsonl`

**Best practices:**
- **Trim the middle, not the end** — the character's closing blessing is a key voice marker; truncating it would strip identity
- **Keep start tokens high enough** — 300 tokens ensures the character's name, setting references, and opening framing survive
- **Run before thinking generation** — thinking data assumes single-turn curated input; generating thinking on multi-turn data would produce inconsistent `<think>` placement
- **Verify after curation** — spot-check a few examples to make sure the truncation ellipsis (`[...]`) doesn't land mid-sentence in a way that creates training noise
- **Commit the output** — `training_data_baseline_augmented_curated.jsonl` is versioned

### Layer 4: Thinking (reasoning prepend)

**Script**: `scripts/generate_thinking_data.py`.

**Purpose**: Prepend `<think>...</think>` inner monologue to each assistant response so reasoning models (Qwen3) learn to think in-character before speaking.

**Technique**: For each curated example, ask GPT-4o "what was the character thinking before they said this?" and prepend the result.
- Uses the character definition as system context
- Processes examples concurrently with `asyncio.Semaphore` (max 5 concurrent) for speed
- Cleans the output: strips stray `<think>` tags, leading labels, quotes
- Wraps with `<think>\n...\n</think>\n\n` before the original response

**Best practices:**
- **Concurrent with semaphore** — 5 concurrent requests balances speed against rate limits and prompt context memory
- **Keep thinking brief** — 2-4 sentences. Long thinking blocks inflate the token count without improving training quality
- **Match the character's voice** — the thinking prompt explicitly says "use her voice — archaic, mystical, nature-infused" so the inner monologue uses the same register as the outer speech
- **Clean aggressively** — LLMs love to add labeling ("Think: ...", `<think>...</think>`); strip all wrappers before prepending your own to avoid double-wrapping
- **Validate before committing** — spot-check that thinking blocks are character-appropriate and not hallucinated facts
- **Don't generate thinking for non-reasoning models** — Mistral/Llama/Qwen2.5 would tokenize `<think>` as literal text and produce it at inference, which looks broken

### Layer 5: Tool calling (future)

**Script**: TBD — `scripts/generate_tooluse_data.py`.

**Purpose**: Generate multi-turn conversations where the character decides to call a tool, processes the result, and responds.

**Best practices (anticipated):**
- **Define a tool catalog first** — `search_archives`, `calculate_alignment`, `fetch_lore`. The LLM needs to know what tools exist before it can generate examples of using them
- **Generate from scratch, not transform** — unlike thinking (which prepends to existing responses), tool calling introduces new messages (tool call + tool result) that don't exist in the source data
- **Multi-turn format** — `[system, user, assistant(tool_call), tool(result), assistant(response)]`
- **Include non-tool examples too** — if every training example has a tool call, the model will call tools even when not needed. Mix tool and non-tool examples to teach discretion
- **Pass tool definitions at inference** — training teaches the model *how* to format tool calls; inference supplies *which* tools are available via the system prompt

### General principles across all layers

- **Each layer is additive** — a new script reads the previous layer's output and writes a new file. Never overwrite the source.
- **Seed determinism** — `random.seed(42)` on splits, fixed prompts on generation. Reproducibility matters.
- **Commit source JSONL** — the input files (`training_data_baseline_*.jsonl`) are versioned so any commit is a checkpoint you can re-split from
- **`prepare_all_datasets.sh` is the single entry point** — it knows the layer order and only runs generation scripts when the output file is missing
- **Validate at each layer** — `validate_all_jsonl.py` catches parse errors; `validate_dataset_quality.py` checks role patterns and character consistency
- **Spot-check aggressively** — after any generation run, manually review 5-10 random examples. Voice drift is invisible to automated validation but obvious to a human reader

### How much data?

There's no universal formula, but these heuristics work in practice for LoRA character fine-tuning:

| Layer | Minimum | Sweet spot | Diminishing returns | Rationale |
|-------|---------|------------|---------------------|-----------|
| **Base** | 10 | 20-30 | 50+ | You need enough variety for few-shot to generalize. Below 10, the LLM latches onto specific phrases. Above 50, the marginal new voice signal is near zero. |
| **Augmented** | 50 | 300-500 | 1000+ | LoRA with rank 16 has limited capacity — it can learn a voice from 300 examples just as well as 3000. More data increases training time linearly with negligible quality gain. |
| **Curated** | N/A | N/A | N/A | Curate the same number as augmented (flatten, don't drop). Every input produces one output. |
| **Thinking** | N/A | N/A | N/A | Same as curated — one `<think>` block per existing response. |
| **Tool calling** (future) | 50 | 100-200 | 500+ | Tool calling is a narrower skill than voice — you need enough examples to learn the format but far fewer than learning a personality. |

**Key insight**: LoRA's low rank (16) is the bottleneck, not data volume. With 7.6B frozen parameters and only ~11M trainable, the adapter physically can't absorb more voice information past a few hundred examples. More data after that point just makes training slower — val loss plateaus but never improves.

**When to add more data:**
- If the character has distinct sub-personalities (formal vs. casual, different emotional registers), you need more examples to cover the range — aim for the high end of the sweet spot
- If eval shows the model defaulting to base-model behavior on certain question types, add targeted examples for those gaps rather than scaling uniformly
- If adding a new layer (thinking, tool calling), generate fresh — don't mix layers in one script, keep each layer as a separate dataset variant

**When to stop:**
- When val loss stops dropping (current models all converge well before 5 epochs)
- When spot-checking shows voice consistency across all prompt types
- When you can't tell the difference between epoch 3 and epoch 5 in chat

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

### Reasoning models (thinking blocks)

Reasoning models like Qwen3 natively emit `<think>...</think>` blocks. To make the character learn to reason in-voice:

1. **Generate thinking data**: `python scripts/generate_thinking_data.py` — uses an LLM (OpenRouter, default `openai/gpt-4o`) to prepend `<think>` inner monologues to each assistant response.
2. **Mark the model**: Set `"is_reasoning": True` in `MODEL_CONFIGS` (Python) and add an `is_reasoning_model()` case in `model_config.sh` (bash).
3. **Prepare the dataset**: `prepare_all_datasets.sh` runs the generation script, then splits and truncates the output into `augmented_curated_thinking_split_2048/`.
4. **Training auto-routes**: `train_character_model.sh` detects reasoning models and prefers the thinking variant.

The `<think>` blocks are part of the training target, so the model learns to produce in-character reasoning before responding. Non-reasoning models (Mistral, Llama, Qwen2.5) use the standard dataset — no `<think>` tags in their training data.

### Future: tool calling

Tool calling would follow the same pattern but isn't implemented yet:

1. **Generate tool-use examples** — script that produces multi-turn conversations with `<tool_call>` / `<tool_result>` messages in the assistant's voice.
2. **New dataset variant** — `augmented_curated_tooluse_split_2048/`.
3. **Mark tool-capable models** — add `supports_tool_calling: True` in `model_config.py`.
4. **Inference** — `folded_chat.py` would need to pass tool definitions and handle tool-call loops.
5. **Training format** — system/user/assistant/tool/assistant multi-turn where the character decides when to call a tool, processes the result, and responds in-voice.

### Future: multi-turn training

All training data is currently single-turn (`[system, user, assistant]` triplets). The curate script flattens every conversation to its last user-assistant pair, even if the source had multiple turns. Multi-turn works at inference because the base model's pre-training understands turn continuations, but the adapter has never seen a reply to a reply.

**Gap**: The model doesn't learn to self-reference ("as I said before..."), maintain topic across turns, or remember details from earlier in the conversation. Multi-turn turns at inference rely entirely on the base model's pre-training, not the adapter.

**Fix** would follow the same dataset-variant pattern:

1. **Preserve multi-turn in curate** — add a `--keep-turns N` flag to `curate_training_data.py` (currently it always takes the last pair) so it emits `[system, user1, assistant1, user2, assistant2, ...]` rows.
2. **Or generate synthetic multi-turn** — script using OpenRouter to extend existing single-turn examples into 3-5 turn conversations with consistent character memory.
3. **New dataset variant** — `augmented_curated_multiturn_split_2048/`.
4. **Training auto-routes** — or expose a CLI flag (`--multiturn`) on the training wrapper. Multi-turn likely improves all models, not just specific ones.

### Future: vision support

None of the current models support vision — they're all text-only. The vision-capable variants (Qwen2.5-VL, Llama 3.2 Vision) are separate models entirely: different HuggingFace repos, different tokenizers, different input formats. Adding vision is a fundamentally different shape of training data, not a data augmentation.

**Training format** — images embedded alongside text in user messages:
```json
{"messages": [
  {"role": "system", "content": "You are Lyra..."},
  {"role": "user", "content": [
    {"type": "image", "image": "<base64 or file path>"},
    {"type": "text", "text": "What do you see in this ancient scroll?"}
  ]},
  {"role": "assistant", "content": "Ah, the runes speak of a forgotten kingdom..."}
]}
```

**Requirements:**
1. **Multimodal base model** — new entry in `model_config.py`/`.sh` (e.g., `qwen25_vl_7b` → `Qwen/Qwen2.5-VL-7B-Instruct`), different tokenizer, different `MAX_SEQ_LENGTH` considerations (images consume tokens).
2. **Image assets** — character-relevant images (fantasy art, scrolls, maps, potions) to include in training examples.
3. **Synthetic data generation** — LLM generates image descriptions + captions; images sourced from Stable Diffusion/DALL-E or pre-existing fantasy art paired with machine captions.
4. **New dataset variant** — `augmented_curated_vision/` with base64-encoded images or image file references.
5. **mlx-lm vision compatibility** — not all vision models are supported by `mlx-lm lora`; the fuse/quantize pipeline would need verification.
6. **Inference** — `folded_chat.py` would need to accept image input alongside text.
7. **`is_vision=True` marker in config** — since training data format is a different shape from text-only models, routing must be explicit.

**Heavier lift than thinking or tool calling** — vision is a separate model class, not a data augmentation. The image pipeline (generation, embedding, pre-processing for tokenizer) is the bulk of the work.

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
| `train_baseline_suite.sh` | Runs training across all supported models |
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

`train_baseline_suite.sh` trains all supported models sequentially, tracking successes and failures:

```
mistral_v0_3 → llama31_8b → qwen25_7b → qwen3_8b
```

This provides a full comparison matrix — 4 models × 2 variants (standard/deep) = 8 configurations. The suite is designed for A/B testing: train everything, then compare voice quality across architectures.
