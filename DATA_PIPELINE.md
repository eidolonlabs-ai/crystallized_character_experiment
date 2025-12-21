# Data Pipeline Architecture

## Overview

The data pipeline has been restructured to ensure **consistency and reproducibility** across all training runs. Instead of regenerating and potentially corrupting data on each training run, all datasets are now prepared once and committed to git.

## Pipeline Flow

```
training_data_*.jsonl (source files)
    ↓
prepare_all_datasets.sh (one-time setup)
    ↓
raw_data/prepared_data/{character}/{variant}_split/
    ├── train.jsonl (90% of data)
    └── valid.jsonl (10% of data)
    ↓
[Committed to git - locked for reproducibility]
    ↓
train_character_model.sh (reuses prepared data)
    ├── Reads from raw_data/prepared_data/{character}/{variant}_split/
    ├── Applies truncation at MAX_SEQ_LENGTH tokens (model-specific)
    └── Trains model
```

## Key Changes

### Before (Problematic)
- ❌ `train_character_model.sh` would call `split_training_data.py` each run
- ❌ `split_training_data.py` would regenerate data, potentially corrupting it
- ❌ `truncate_training_data.py` would regenerate truncated data
- ❌ No version control of prepared data
- ❌ Inconsistent data across training runs (reproducibility issue)

### After (Fixed)
- ✅ Run `prepare_all_datasets.sh` once to split all data
- ✅ Datasets are committed to git (`raw_data/prepared_data/`)
- ✅ All training uses consistent, pre-prepared data
- ✅ No regeneration on each training run
- ✅ Data is versioned and reproducible

## Dataset Organization

### Prepared Data Structure

```
raw_data/prepared_data/
└── baseline/
    ├── base_split/               (19 examples)
    │   ├── train.jsonl
    │   └── valid.jsonl
    ├── base_split_512/            (19 truncated)
    ├── base_split_768/
    ├── augmented_split/          (319 examples)
    │   ├── train.jsonl
    │   └── valid.jsonl
    ├── augmented_split_512/       (319 truncated)
    ├── augmented_split_768/
    ├── full_split/               (270 examples)
    │   ├── train.jsonl
    │   └── valid.jsonl
    ├── full_split_512/            (270 truncated)
    └── full_split_768/
```

## Data Variant Selection

When training, the script automatically selects the best available dataset variant using this priority:

### Baseline
1. **augmented_curated** (best quality) - if available
2. **augmented** - if available
3. **base** - if available
4. **full** - fallback

## Training Process

### Step 1: Prepare Datasets (One-Time)

```bash
./prepare_all_datasets.sh
```

This script:
- Reads all `training_data_*.jsonl` source files
- Splits each 90% train / 10% validation (consistent seed)
- Outputs to `raw_data/prepared_data/{character}/{variant}_split/`
- Should be run when source data changes

### Step 2: Commit to Git

```bash
git add raw_data/prepared_data/
git commit -m "Update prepared datasets"
```

This locks in the data for reproducibility.

### Step 3: Train Model

```bash
./scripts/train_character_model.sh baseline mistral
```

The training script:
- Reads pre-prepared data from `raw_data/prepared_data/{character}/{variant}_split/`
- Selects best available variant (augmented_curated > augmented > base > full)
- Applies token truncation during training (model-specific)
- Never regenerates data

## Data Loss Analysis

### Split Data Integrity

All splits preserve 100% of the original data (90/10 random split with seed 42).

### Token Truncation

Truncation occurs **during training**, not during preparation. Token counts depend on the model's tokenizer:

**Example (Llama 3.1 8B tokenizer):**

| Character | Variant | Total | Split Train | Loss @ 512 tokens |
|-----------|---------|-------|-------------|-------------------|
| Baseline | augmented | 319 | 287 | ~10% |

Longer sequences are truncated at the token limit specified by the training variant:
- **Standard variant**: MAX_SEQ_LENGTH = 512 tokens
- **Deep variant**: MAX_SEQ_LENGTH = 768 tokens

## Important Notes

### What Gets Versioned
- ✅ `raw_data/prepared_data/` - Split datasets (in git)
- ✅ `prepare_all_datasets.sh` - The preparation script (in git)
- ✅ `training_data_*.jsonl` - Source data (in git)

### What Does NOT Get Regenerated
- ❌ Split/train valid datasets - reused from git
- ❌ No calls to `split_training_data.py` during training
- ❌ Prepared data is never regenerated unless explicitly running `prepare_all_datasets.sh`

### Data Reproducibility

To ensure reproducible training:

1. **Prepared data is in git**: Same data for all team members
2. **Consistent random seed**: 90/10 split uses seed 42
3. **No dynamic regeneration**: Training never modifies prepared data
4. **Model-specific tokenization**: Truncation uses the actual model's tokenizer

## Updating Data

When source data changes:

```bash
# Update training_data_*.jsonl files as needed
# Then regenerate prepared datasets:
./prepare_all_datasets.sh

# Review changes:
git status

# Commit new data:
git add raw_data/prepared_data/
git commit -m "Update prepared datasets with new source data"
```

## Migration Path

If retraining with old data is needed:

```bash
# Old prepared data is still in git history
git log --oneline raw_data/prepared_data/

# Checkout old data if needed
git checkout <commit-hash> -- raw_data/prepared_data/
```

## File Size Reference

Prepared datasets are compressed JSONL format:

- `baseline/base_split/train.jsonl` - ~27 KB
- `baseline/augmented_split/train.jsonl` - ~350 KB

Total size: ~1 MB for all prepared datasets
