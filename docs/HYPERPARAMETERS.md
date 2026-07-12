# Hyperparameters

Every knob in the training pipeline, what it does, what value we use, and why.
Source of truth for defaults: [`scripts/model_config.py`](../scripts/model_config.py)
`DEFAULT_TRAINING`. To change defaults across the whole suite, edit that
dict; to change a single model, drop a YAML in [`configs/`](../configs/).

The values below are written for the `baseline` (Lyra) character — 355
training examples, 7-8B base model, Apple Silicon M-series. They transfer
to similar small-dataset style-transfer tasks but should be re-tuned for
substantially different setups.

---

## Structural knobs (in `lora_parameters`, set via YAML)

### `rank` — LoRA rank (default `16`)

The rank of the low-rank update matrices `A` and `B` injected into each
targeted linear layer. Higher rank = more trainable parameters = more
capacity to learn style/format, but slower training and larger adapter
files. mlx-lm's default is `8`; we use `16` because Lyra's voice has
distinct lexical markers (Sindarin closers, archaic English) that benefit
from extra capacity.

The relationship between rank and trainable parameter count:
- rank 8 → ~2.6M trainable params (the original "standard" variant)
- rank 16 → ~5.2M trainable params (modern default)
- rank 32 → ~10.4M trainable params
- rank 64 → ~20.8M trainable params

For datasets in the 100-1000 example range, ranks 16-32 are the sweet spot.
Below rank 8, style learning underfits; above rank 64, you risk the LoRA
re-learning what the base model already knows.

### `scale` (α / alpha) — LoRA scaling (default `32`)

mlx-lm calls this `scale`; in other LoRA implementations it's `alpha`.
The effective learning rate multiplier is `scale / rank`, so the default
`scale=32, rank=16` gives `α/rank = 2.0`. This is the standard
recommendation from the LoRA paper (Hu et al. 2021). Setting α/rank much
higher (>4) destabilizes training; much lower (<0.5) makes the adapter
too conservative.

### `dropout` — LoRA dropout (default `0.05`)

Dropout inside the LoRA adapter branches (applied to the input of `A`).
Helps generalization on small datasets. The original LoRA paper used 0.0;
modern recipes (Tulu, OpenHermes) commonly use 0.05-0.1. Keep this small
— large dropout (e.g. 0.5) actively hurts style learning.

---

## Loss & optimization knobs

### `mask_prompt` — only train on the assistant response (default `true`)

When `true`, only assistant tokens contribute to the loss. This is the
**modern SFT default** (Tulu, OpenHermes, SmolTalk all mask the prompt)
and the repo default (`MASK_PROMPT="--mask-prompt"` in
`scripts/train_character_model.sh`).

**Why we switched the default.** The Phase 0 fold puts the character
system prompt into every training row's first user turn as
`[SYSTEM] You are Lyra Moonwhisper, ... \n\n[USER] <q>`. That
~50-token system-prompt block is byte-identical across all 319 training
examples. With `--no-mask-prompt`, AdamW computes 319 near-identical
gradient signals for "predict the next token of the system prompt" and
amplifies those weights until training diverges on Mistral v0.2/v0.3:

| iter | `--no-mask-prompt` (diverging) | `--mask-prompt` (converging) |
|------|--------------------------------|------------------------------|
| 1 (val) | **5.434** | 2.354 |
| 5 (train) | 4.315 | 2.096 |
| 10 (val) | 7.561 (catastrophic) | **1.638** ✓ |
| 10 (train) | 8.778 | **1.368** ✓ |

Empirically verified in commit `a6d6988`. Llama family (lower base
perplexity on the system-prompt tokens) doesn't diverge without mask,
but is also probably memorizing the prompt rather than learning from
it. `--mask-prompt` is unambiguously better for this dataset shape.

**Backward-compat:** the 8 pre-Phase-0 adapters in `adapters/` were
trained with `--no-mask-prompt` and represent a known-bad baseline.
Re-train them under the new default if you need them; do not re-train
the diverged post-Phase-0 Mistral v0.2/v0.3 adapters under the old
default — they will diverge again.

Historical note: an older version of mlx-lm produced NaN loss with
`--mask-prompt` on Llama 3.1. mlx-lm 0.30.0 doesn't have that bug.

### `optimizer` — optimizer choice (default `adamw`)

mlx-lm 0.30.0 ships five: `adam`, `adamw`, `muon`, `sgd`, `adafactor`.
- `adamw` is the modern default for SFT — decoupled weight decay, robust
  to the LR range we use here (1e-5 to 5e-5).
- `muon` (Jordan et al. 2024) is a Newton-Schulz-orthogonalized momentum
  optimizer that has been shown to converge faster than AdamW on dense
  pre-training, with promising small-scale SFT results. Worth trying if
  your LR feels hard to tune.
- `sgd` and `adafactor` are there for completeness; not recommended for
  this scale.

### `learning_rate` — peak LR (CLI flag, default `5e-5` for standard, `2.5e-5` for deep)

Set in `train_character_model.sh` per-variant. Standard variant uses
`5e-5`; deep variant halves it because it has more trainable parameters
(16 LoRA layers vs 8) and tends to overfit faster. Both are passed
straight through to `mlx_lm.lora --learning-rate`.

### `lr_schedule` — schedule + warmup (YAML block, default cosine + warmup 20)

Currently we don't pass a schedule to mlx-lm; the default is constant LR.
For better convergence on longer runs, switch to `cosine` decay with a
short warmup. (To enable, edit the YAML `lr_schedule` block in
`scripts/model_config.py::render_yaml_config()` and uncomment the
schedule lines in `mlx_lm.lora` — mlx-lm 0.30 supports `linear`,
`cosine_decay`, and `join_schedules` via YAML.)

### `seed` — PRNG seed (default `42`)

mlx-lm 0.30.0 doesn't expose `--seed` cleanly across all subcommands; we
set it via YAML. Same seed → same training run (modulo non-determinism
in Metal ops, which is small but nonzero).

---

## Capacity knobs (CLI flags, per-variant)

### `num_layers` — how many transformer layers get LoRA (default `8` standard, `16` deep)

The standard variant targets 8 of the (typically 32-40) transformer
layers. The deep variant doubles to 16. Targeting more layers increases
trainable parameter count proportionally and tends to learn more style
nuance; targeting all layers (`-1`) is full fine-tuning and requires much
more memory.

### `max_seq_length` — context window (default `512` standard, `768` deep)

Long enough to fit system prompt + 1-2 user/assistant turns at 5e-5 LR.
The deep variant goes to 768 because the curated training examples can
be longer. Going beyond 1024 starts hurting throughput on 32 GB unified
memory.

### `batch_size` + `gradient_accumulation_steps` — effective batch (1 × 2 or 1 × 4)

Effective batch size 2 for standard, 4 for deep. mlx-lm is constrained
to batch size 1 on Apple Silicon for 7-8B models at fp16; we accumulate
gradients to recover a reasonable effective batch.

### `epochs` — passes through the training set (default `5`)

With 319 training examples and effective batch 2, that's ~800 iters/epoch
× 5 = ~4000 total iters. `--save-every 50` means ~80 checkpoints; the
auto-resume logic in `train_mlx.py` picks the latest.

---

## Memory & checkpointing

### `grad_checkpoint` — gradient checkpointing (default `false`)

Trades compute for memory by recomputing activations during backward.
Enable on <32 GB unified-memory Macs, especially at `max_seq_length=1024`.
Has a 20-30% throughput cost; safe to enable on M2 Pro and above.

---

## Swapping to DoRA

[DoRA](https://arxiv.org/abs/2402.09353) (Liu et al. 2024) decomposes
the pretrained weight into magnitude and direction components, applying
LoRA only to the direction. Empirically, DoRA at the same rank
consistently matches or beats LoRA on style/format learning tasks
(per the original paper + follow-ups).

To run DoRA on a single model:

```bash
./scripts/train_character_model.sh baseline mistral_v0_3 \
    --fine-tune-type dora
```

Or in a custom YAML:

```yaml
fine_tune_type: dora
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 32
```

No DoRA adapter is trained in this PR — see [A_B_TEST_COMMANDS.md](../A_B_TEST_COMMANDS.md)
for the A/B protocol when you do train one.

---

## Reading list

- LoRA: [Hu et al. 2021](https://arxiv.org/abs/2106.09685) — foundational paper.
- QLoRA: [Dettmers et al. 2023](https://arxiv.org/abs/2305.14314) — not applicable on Apple Silicon (mlx-lm uses native fp16/bf16; no NF4 needed).
- DoRA: [Liu et al. 2024](https://arxiv.org/abs/2402.09353) — decomposed LoRA.
- Muon optimizer: [Jordan et al. 2024](https://kellerjordan.github.io/posts/muon/) — orthogonalized momentum, worth experimenting with on small SFT.
- NEFTune: [Zhang et al. 2023](https://arxiv.org/abs/2310.05914) — adds noise to embeddings during training; not exposed by mlx-lm 0.30.0 (re-evaluate when upstream adds it).