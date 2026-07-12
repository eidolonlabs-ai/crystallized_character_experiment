# Training configs

Per-character / per-model YAML overrides for the modern fine-tune recipe.

## When to use one

By default, `scripts/train_character_model.sh` builds a YAML on the fly from
[`scripts/model_config.py`](../scripts/model_config.py) `DEFAULT_TRAINING` —
one recipe applied to every model. You only need a custom YAML when you
want to override a single model (e.g. trying DoRA on Mistral only, or
bumping rank on Llama 3.1).

```bash
# Default (no config needed) — uses DEFAULT_TRAINING
./scripts/train_character_model.sh baseline mistral_v0_3

# Override with a YAML
./scripts/train_character_model.sh baseline mistral_v0_3 \
    --config configs/lyra_mistral7b_v3.yaml
```

## Schema

Every key in the YAML is a key from
`mlx_lm.lora.CONFIG_DEFAULTS`. The full schema is in the installed library:

```python
python -c "from mlx_lm.lora import CONFIG_DEFAULTS; import json; print(json.dumps(CONFIG_DEFAULTS, indent=2, default=str))"
```

The subset we actually use:

| Key | Meaning | Default |
|-----|---------|---------|
| `model` | HF repo or local path | required |
| `train` | Boolean | `true` |
| `fine_tune_type` | `lora` or `dora` | `lora` |
| `optimizer` | `adam`, `adamw`, `muon`, `sgd`, `adafactor` | `adamw` |
| `seed` | PRNG seed | `42` |
| `mask_prompt` | Mask prompt in loss (only train on completion) | `false` |
| `grad_checkpoint` | Gradient checkpointing for memory | `false` |
| `lora_parameters.rank` | LoRA rank (typical 8-64) | `16` |
| `lora_parameters.dropout` | LoRA dropout | `0.05` |
| `lora_parameters.scale` | α (alpha), doubles rank is a common default | `32` |

Run-specific knobs (num_layers, learning_rate, max_seq_length, batch_size,
grad_accumulation_steps, epochs, save_every, steps_per_report,
steps_per_eval) are passed as CLI flags by `train_character_model.sh` —
they're not part of the YAML. See `docs/HYPERPARAMETERS.md` for the rationale
behind every value.

## Adding a new model-specific config

1. Copy `lyra_mistral7b_v3.yaml` to `<char>_<model>.yaml`.
2. Edit the `model:` line.
3. Tweak the knobs you want to override.
4. Pass it with `--config configs/<char>_<model>.yaml`.

The CLI overrides YAML when both are present — `--rank`, `--lr-schedule`,
etc. on the bash wrapper take precedence over the file.