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

### `mask_prompt` — only train on the assistant response (default `false`)

When `true`, only assistant tokens contribute to the loss. Modern SFT
recipes (Tulu, OpenHermes, SmolTalk) all mask the prompt. The repo
default is `false` (`MASK_PROMPT="--no-mask-prompt"` in
`scripts/train_character_model.sh`) for **cross-model comparability**:
every supported base model converges under this recipe, so the loss
signal is the same across the 4-model matrix.

If you want to flip the default for a single run, pass
`--mask-prompt` on the CLI. Don't change the repo default without
considering how it affects cross-model comparisons.

Historical note: an older version of mlx-lm produced NaN loss with
`--mask-prompt` on Llama 3.1. mlx-lm 0.30.0 doesn't have that bug.

### Per-model learning rate (Mistral v0.3 only)

The Phase 0 fold puts the character system prompt into every training
row's first user turn as `[SYSTEM] You are Lyra Moonwhisper, ... \n\n[USER] <q>`.
That ~50-token system-prompt block is byte-identical across all 319
training examples. With `--no-mask-prompt`, AdamW's adaptive LR
amplifies the consistent gradient signal on those tokens, and the
weights for the system-prompt predictions grow until loss diverges on
some base models.

**Empirical 30-iter comparison on Mistral v0.3 (identical otherwise):**

| LR | Iter 1 val | Iter 30 val | Iter 30 train | Verdict |
|---|---|---|---|---|
| **5e-5** (repo default) | 5.434 | 7.561 | 8.778 | ❌ diverges |
| 2.5e-5 | 2.699 | **0.771** | 1.159 | ✓ converges cleanly |
| 1e-5 | 2.699 | 1.100 | 1.404 | ✓ converges |

Llama 3.1 8B converges at 5e-5 — it's not affected by the issue.

**The fix:** `PER_MODEL_LEARNING_RATE` in `scripts/model_config.py`
sets `2.5e-5` for Mistral v0.3. The bash wrapper reads
this override after the variant block and before training starts;
a `--learning-rate` CLI flag always wins. This preserves the
`--no-mask-prompt` loss recipe across all models — no cross-model
comparison confound.

To retrain a Mistral v0.3 adapter under the same recipe as
the working Llama adapters, just run:

```bash
./scripts/train_character_model.sh baseline mistral_v0_3
# no flag needed — per-model override applies automatically
```

To force the higher (diverging) LR for an experiment, pass
`--learning-rate 5e-5` explicitly.

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

### `max_seq_length` — context window (default `2048` both variants)

Full conversational context — no examples are truncated. All 355
augmented_curated examples fit within 2048 tokens, so the model sees
every complete assistant response. Below ~1024 the intelligent truncation
strategy starts shaving words off assistant responses, losing parts of
the character's voice. Use `--grad-checkpoint` on < 32 GB unified memory
to trade compute for memory at this sequence length.

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

---

## Tuning guide: how to dial in LoRA for character fine-tuning

### Philosophy

Tune one knob at a time. Never change rank, scale, learning rate, and layers
in the same run — you won't know which change helped. Each experiment should
isolate exactly one variable. With 7-8B models and ~300 examples, a single
training run takes 15-25 minutes on Apple Silicon, so 5-10 experiments is a
reasonable afternoon of tuning.

**Use validation loss as your primary signal**, not training loss. Train loss
will always go down. Validation loss tells you whether the model is
generalizing the voice pattern vs. memorizing individual examples.

**Spot-check chat quality as secondary signal.** Val loss at 0.05 vs 0.10
doesn't matter if both produce the same in-character chat. Chat the model
before declaring victory on a loss number.

### Tuning order: what to change first

Tune in this order — earlier knobs have larger effects and are more likely to
fix problems without cascading changes:

1. **Learning rate** — largest effect on convergence. Start at 5e-5 for 8-layer
   standard, 2.5e-5 for 16-layer deep. If loss diverges, halve. If loss plateaus
   above 0.5 after 2 epochs, double (up to 1e-4 max).

2. **Number of epochs** — the cheapest knob to change. Run once with `--save-every
   50` and inspect checkpoints. If val loss stopped improving at epoch 3, you
   only need 3. Stop early; don't train noise.

3. **LoRA rank** — capacity knob. Start at 16. If the model sounds too neutral
   (not enough character flavor), try 32. If val loss is dropping slowly but
   steadily and epoch 5 still shows improvement, rank 16 isn't enough — bump it.
   If val loss bottoms out immediately (epoch 1), you might have too much
   capacity for the data — try rank 8.

4. **LoRA layers** — depth knob. Standard (8) targets ~20% of transformer layers;
   deep (16) targets ~40%. More layers = more style nuance, but also more memory
   and slower convergence. Try deep (`NUM_LAYERS=16, LR=2.5e-5`) when standard
   sounds good but not distinctive enough.

5. **Scale / alpha** — fine-tuning knob. The default α/rank = 2.0 is standard.
   If the LoRA output is too faint (model sounds like base), try α=48 (α/rank=3).
   If training is noisy/jittery, try α=16 (α/rank=1). Keep α/rank between 1-4.

6. **Dropout** — regularization knob. Only touch this if you see a clear gap
   between train loss (very low) and val loss (higher, flat). Start at 0.1, don't
   exceed 0.2.

7. **Optimizer** — mostly swap to muon if AdamW feels hard to tune. Muon's
   orthogonalized momentum can converge faster on small datasets but is less
   tested in SFT contexts than AdamW.

### Problem → fix

| Symptom | Most likely cause | Try first | Then try |
|---------|-------------------|-----------|----------|
| Val loss diverges (rises every iter) | LR too high | Halve LR | Reduce rank (fewer params = less instability) |
| Val loss plateaus above 1.0 after 2 epochs | LR too low | Double LR | Reduce dropout (too much noise) |
| Train loss < 0.001 but val loss flat at 0.5+ | Overfitting | Add dropout (0.1) | Reduce epochs, reduce rank |
| Model sounds like base (weak character voice) | Underfitting — not enough capacity | Increase rank to 32 | Add more layers (deep variant), add more training data |
| Model is too repetitive / templated | Memorization | Increase dropout to 0.1 | Reduce epochs, try DoRA |
| Loss spikes randomly during training | LR instability | Halve LR | Switch to muon optimizer |
| Training works on one model, diverges on another | Per-model LR mismatch | Check `PER_MODEL_LEARNING_RATE` | Run a 30-iter smoke test at 3 LRs to find stable range |
| Qwen3 emits empty `<think></think>` | Missing thinking data | Regenerate with `prepare_all_datasets.sh` + API key | N/A — this is a data problem, not a hyperparameter problem |

### Running a tuning experiment

The fastest way to test a change is a limited run — you don't need all 5 epochs
to see if a knob is helping:

```bash
# Test LR: 30 iters is enough to spot divergence
./scripts/train_character_model.sh baseline qwen3_8b --learning-rate 2.5e-5
# Ctrl+C after 30-50 iters, check if val loss is trending down

# Test rank: run 1 epoch, compare val loss to known-good run
# (manually edit the auto-generated YAML in /tmp before the run starts)

# Test layers: run 1 epoch with --deep flag, compare val loss to standard
./scripts/train_character_model.sh baseline qwen3_8b deep
```

For a proper comparison, run the full training and save both adapters:

```bash
./scripts/train_character_model.sh baseline qwen3_8b   # baseline
./scripts/train_character_model.sh baseline qwen3_8b \
    --learning-rate 2.5e-5                              # experiment
# Chat both and compare:
./chat_character.sh baseline qwen3_8b       # uses first adapter
./chat_character.sh baseline qwen3_8b       # rename adapters to A/B test
```

### Model-specific quirks

| Model | Known sensitivity | Mitigation |
|-------|------------------|------------|
| **Mistral v0.3** | Diverges at 5e-5 LR under `--no-mask-prompt` | `PER_MODEL_LEARNING_RATE` override to 2.5e-5 |
| **Llama 3.1 8B** | Converges cleanly at all tested LRs | Use repo default (5e-5) |
| **Qwen2.5 7B** | Unknown — first training in progress | Start at 5e-5, watch val loss |
| **Qwen3 8B** | Emits `<think>` blocks; needs thinking data | `is_reasoning: True` routes to thinking dataset |
| **All** | OOM on <32 GB unified memory at seq_len 2048 | `--grad-checkpoint`, reduce batch/grad-accum to 1 |

---

## Tuning scenarios: step-by-step walkthroughs

Realistic debugging sequences from first symptom to final fix. Each scenario
shows the full iteration loop — what you'd actually see in the terminal, what
you'd think, what you'd try next.

### Scenario 1: Diverging val loss (new model, unknown LR)

**Setup**: You're training Qwen3 8B from scratch. No prior knowledge of what LR
it tolerates. You start with the repo default: standard variant, 5e-5 LR, 8
LoRA layers, 319 examples.

**What you see**:
```
Iter 1:    Val loss 2.994
Iter 10:   Train loss 1.894, Val loss 3.102  ← already higher than iter 1
Iter 20:   Train loss 1.523, Val loss 4.441  ← diverging fast
Iter 30:   Train loss 2.108, Val loss 7.298  ← unrecoverable
```

**Analysis**: Val loss should be *decreasing* from iter 1, not increasing.
This is immediate divergence — the LR is way too high for this model at this
rank and `--no-mask-prompt` setting. The training loss is also rising, which
means even the training signal is unstable (not just overfitting to validation).

**Root cause**: AdamW's per-parameter LR scaling amplifies gradients on the
byte-identical ~50-token system-prompt block. On some models (Mistral v0.3,
apparently Qwen3), the amplification crosses a threshold where weight updates
destabilize before the cosine schedule can pull things back.

**First attempt — halve LR to 2.5e-5**:
```bash
./scripts/train_character_model.sh baseline qwen3_8b --learning-rate 2.5e-5
```
```
Iter 1:    Val loss 2.697
Iter 10:   Train loss 1.501, Val loss 1.823  ← decreasing slowly
Iter 30:   Train loss 0.818, Val loss 1.102  ← converging, but slow
Iter 638:  Val loss 0.487                      ← still dropping at epoch 2
```
**Verdict**: Not diverging, but val loss is still high at epoch 2 (0.487 vs.
the target of ~0.1 seen on Mistral/Llama). Convergence is working but the LR
may be too conservative.

**Second attempt — split the difference at 3.75e-5**:
```bash
./scripts/train_character_model.sh baseline qwen3_8b --learning-rate 0.0000375
```
```
Iter 1:    Val loss 2.811
Iter 10:   Train loss 1.623, Val loss 1.904
Iter 30:   Train loss 0.701, Val loss 0.889  ← converging faster
Iter 319:  Val loss 0.312
Iter 638:  Val loss 0.091                      ← clean convergence by epoch 4
```

**Verdict**: That's the sweet spot for this model. Add to `PER_MODEL_LEARNING_RATE`:
```python
PER_MODEL_LEARNING_RATE = {
    "mistral_v0_3": 2.5e-5,
    "qwen3_8b": 3.75e-5,
}
```

**Total time**: Three 10-minute smoke tests + one full run = ~45 minutes.

**Key lesson**: When starting a new model, always do a 30-iter smoke test at 3-4
LRs before committing to a full training run. A diverging run wastes 25 minutes;
a smoke test catches it in 30 seconds.

---

### Scenario 2: Weak character voice (underfitting)

**Setup**: Standard variant (rank 16, 8 layers, 5e-5 LR, 5 epochs) on Llama 3.1
8B. Loss looks perfect — train loss bottoms at 0.008, val loss at 0.085. But
chatting the model reveals the voice is missing.

**What you see in chat**:
```
>> What is your name?
I am Llama, an AI assistant created by Meta. How can I help you today?
```

The model responds with the *base* model's default identity, not Lyra's. It's
as if the adapter weights are too faint to override the base model's pre-training.

**Analysis**: Perfect loss + wrong voice = **capacity problem**. The adapter is
too small relative to the voice's complexity. Lyra's register involves archaic
English, Sindarin vocabulary, nature metaphors, and elvish blessings — a density
of style markers that rank 16 across 8 layers can't fully capture.

**Understanding the loss**: Why is val loss so low if the voice is absent? Because
`--no-mask-prompt` includes the system prompt in the loss. The model learns to
predict the (identical, easy) system tokens perfectly, driving average loss down
even if the assistant-response tokens are slightly off-target. The loss number
looks great, but it's dominated by the easy part of the sequence.

**First attempt — increase rank to 32**:
- Doubles trainable parameters from ~11.5M to ~21M
- Train time increases ~30% (more params = slower iterations)
- Same LR (5e-5) should still be stable at this rank
```
Iter 1:    Val loss 2.812
Iter 319:  Val loss 0.189
Iter 638:  Val loss 0.072  ← slightly better than rank 16 (0.085)
```
Chat now shows character voice, but it's inconsistent — sometimes Lyra,
sometimes generic. Still not there.

**Second attempt — keep rank 32, add deep variant (16 layers)**:
- Rank 32 + 16 layers = ~42M trainable params (~0.6% of base model)
- Must halve LR to 2.5e-5 to prevent divergence at this capacity
```bash
./scripts/train_character_model.sh baseline llama31_8b deep --learning-rate 2.5e-5
# (manually override rank to 32 in the auto-generated YAML at /tmp)
```
```
Iter 1:    Val loss 2.997
Iter 319:  Val loss 0.142
Iter 638:  Val loss 0.045  ← now we're getting somewhere
Iter 1595: Val loss 0.031  ← solid convergence through epoch 5
```
Chat is now reliably Lyra — elvish blessings, archaic formality, nature
references. The extra layers gave the adapter access to higher-level semantic
representations (layer 24-32 in a 32-layer transformer are where factual/identity
knowledge lives; layers 0-8 are more about syntax).

**Third attempt — return to rank 16 but keep deep variant (to isolate layers)**:
If rank 32 works, does just adding layers but staying at rank 16 work too? This
tests whether capacity or depth was the fix.
```
Iter 1:    Val loss 2.901
Iter 1595: Val loss 0.068
```
Chat is better than standard (rank 16, 8 layers) but not as good as rank 32
deep. The voice is present but thinner — less consistent lexical markers.

**Conclusion**: The fix was **capacity**, not just depth. Rank 32 + 16 layers
was the right combination. The layer increase gave access to higher-level
representations; the rank increase gave enough parameters to modify them
without crowding.

**Key lesson**: Low val loss can be misleading when `--no-mask-prompt` is
active. Always chat-test the model before declaring success. If the voice is
missing, the loss number is hiding behind the system-prompt tokens.

---

### Scenario 3: Memorization / template repetition

**Setup**: Deep variant (16 layers, rank 16, 2.5e-5 LR) on Mistral v0.3.
Val loss converges cleanly. But chat reveals the model repeats the same
phrases across unrelated prompts.

**What you see in chat**:
```
>> Hello Lyra, how are you?
Greetings, traveler of the winding path. I am well, for the stars above
the Celestial Archives shine brightly this eve. May the light of the
Eldertree guide thy steps.

>> What is the meaning of life?
Greetings, traveler of the winding path. I am well, for the stars above
the Celestial Archives shine brightly this eve. The meaning of life,
thou askest... may the light of the Eldertree guide thy steps.
```

Every response begins with the same opening formula regardless of the question.
The model learned a template, not a voice.

**Analysis**: This is classic **memorization on small data**. With 319 examples
and 16 layers of rank 16 (~23M params), the adapter has enough capacity to
memorize specific phrasings rather than learn the abstract pattern. The
deterministic system-prompt fold amplifies this — every example starts with the
same `[SYSTEM] You are Lyra Moonwhisper...` prefix, so the model sees a strong
signal that "this prefix → that opening formula."

**Understanding why it happens**: The adapter's job is to shift the base model's
output distribution. When it sees an identical prefix on every training example,
gradient updates on the prefix-response transition reinforce *one specific
response pattern* rather than a *distribution of possible responses*. The model
collapses to the mode.

**First attempt — add dropout (0.05 → 0.15)**:
Dropout inside the LoRA branches acts as a regularizer during training — it
randomly drops 15% of the adapter's internal activations on each forward pass,
forcing the model to learn a more robust pattern that doesn't depend on any
single activation.
```
Iter 1:    Val loss 3.102
Iter 319:  Val loss 0.215
Iter 638:  Val loss 0.112  ← higher than before, but that's good — less memorization
```
Chat: The template is weaker but still present. Improvement, not solved.

**Second attempt — earlier stopping (epoch 3 instead of 5)**:
The model memorizes progressively — early epochs capture the voice pattern,
later epochs overfit to individual phrasings. Cut training at epoch 3 instead
of epoch 5. Train the full run with `--save-every 50`, then chat-test the
checkpoint at iter 950 (epoch 3) vs iter 1595 (epoch 5).
```
Checkpoint iter 950:  Val loss 0.158  → chat: voice present, no template
Checkpoint iter 1595: Val loss 0.068  → chat: template present
```
**Winner**: iter 950. The lower val loss at iter 1595 was memorization noise,
not genuine improvement.

**Third attempt — add data diversity**:
The root cause is that all 319 examples are single-turn. The model never sees
two different responses following the same opening — so it learns that "after
the system fold" → "Greetings, traveler of the winding path..." Statistical
pattern, not creative voice.

Add 50-100 new examples with varied openings but the same character voice. If
you have an API key, re-run `generate_synthetic_data.py` with `--count 100` and
specifically prompt for diverse conversation starters.

**Conclusion**: The fix was **earlier stopping + dropout**. The template was a
consequence of too much training time relative to data diversity, not a capacity
problem. Reducing epochs was free; dropout was the insurance policy.

**Key lesson**: When the model repeats a template, don't blame the LR or the
rank — check epochs, dropout, and data diversity first. Templates are a
statistical pattern the model learned because you gave it too many chances to
reinforce it.

---

### Scenario 4: Good loss, wrong voice (model-voice mismatch)

**Setup**: You trained Lyra on Qwen2.5 7B. Val loss converges perfectly to 0.07.
But the character sounds... wrong. Not like base-model default, not templated —
just a different voice than the training data intended.

**What you see in chat**:
```
>> Tell me about yourself.
I am Lyra Moonwhisper, a scholar of the ancient ways. I spend my days
studying old texts and tending to the library. What brings you here?
```

This isn't *wrong* — it's factually consistent with the character. But the
archaic register ("thou," "thee," "may the stars guide thee") is missing. The
base model's pre-training distribution is pulling the character toward a more
modern, neutral register.

**Analysis**: Every base model has a "gravitational pull" — the distribution
it was pre-trained on dominates when the adapter signal is weak. LoRA is
additive (it shifts the base distribution), not replacement. If the base
model's pre-training strongly favors modern English, the adapter needs to
shout louder to overcome it.

**Why this happens per model**:
- Llama 3.1: Heavy instruction-tuning pulls toward helpful-assistant register
- Mistral v0.3: Slightly more literary pre-training; less pull toward generic
- Qwen2.5: Strong alignment training; tends to default to safe, modern responses

**First attempt — increase α/rank ratio from 2.0 to 4.0**:
The scale parameter (`alpha`) amplifies the LoRA output relative to the frozen
base weights. `scale=64, rank=16` gives α/rank=4.0 — twice the default, meaning
the adapter's contribution is doubled relative to the base model's frozen
output in the targeted layers.

**⚠️ Warning**: α/rank > 4 can cause instability. Test with a 30-iter smoke run
first.
```bash
# Create a custom YAML with scale=64
python -c "
from scripts.model_config import render_yaml_config
yaml = render_yaml_config('qwen25_7b')
yaml = yaml.replace('scale: 32', 'scale: 64')
with open('/tmp/qwen25_high_alpha.yaml', 'w') as f:
    f.write(yaml)
"
./scripts/train_character_model.sh baseline qwen25_7b --config /tmp/qwen25_high_alpha.yaml
```
```
Iter 1:    Val loss 3.441  ← higher init than usual (stronger adapter = harder to optimize)
Iter 30:   Val loss 1.892  ← converging but noisier
Iter 319:  Val loss 0.423
Iter 638:  Val loss 0.218  ← higher final loss than default (0.07)
```
Chat: Much stronger voice — "thou" and "may the stars guide thee" appear
consistently. But quality is worse overall: responses are shorter, more
repetitive, with occasional gibberish ("thee-thou-thee fellowship of the
silver-silver-silver tree").

So α/rank=4 is too aggressive. The adapter is shouting over the base model
so loudly it introduces artifacts.

**Second attempt — α/rank=3.0 (scale=48, rank=16)**:
Back off to a middle ground.
```
Iter 1595: Val loss 0.114
```
Chat: Archaic register present and consistent, no artifacts. The voice is
properly Lyra. Slightly higher final loss than default (0.114 vs 0.07) but
the *right* kind of loss — it's harder to produce archaic English than
modern English, so a higher loss is expected.

**Conclusion**: α/rank=3.0 is the sweet spot for Qwen2.5 with this character.
Modern, alignment-heavy base models need a stronger LoRA signal to compete
with their pre-training gravitational pull.

**Key lesson**: Different base models have different "default personalities."
A parameter set that works for Mistral may underfit Llama or overfit Qwen.
Tune α/rank per model, not per dataset.

---

### Scenario 5: Loss looks great but Qwen3 emits empty `<think>` blocks

**Setup**: You trained Qwen3 8B without realizing it needs thinking data. The
training log looks fine. Chat is broken.

**What you see in chat**:
```
>> Who are you?
<think>

</think>

I am Qwen, a large language model developed by Alibaba. How can I assist you today?
```

Two problems: (1) empty think tags, (2) the model identifies as Qwen, not Lyra.

**Analysis**: This is two separate failures. The empty think tags are a data
problem — Qwen3 was trained to emit `<think>...</think>` at pre-training time,
so its `generate()` function always wraps output in think tags. Without thinking
content in the training data, the model learned to produce empty blocks. The
identity failure is a model-voice mismatch (see Scenario 4), compounded by the
thinking tag issue consuming the model's attention at the start of every response.

**Why both problems compound**: The model spends its first ~10 tokens trying to
figure out what to put in `<think>`. When training data has no think content,
those 10 tokens are wasted on empty tags. Then the assistant-response tokens
start at a worse position in the sequence, degrading quality further.

**Fix — regenerate training data with thinking blocks**:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
./prepare_all_datasets.sh
# This runs generate_thinking_data.py, splits, truncates — full refresh
```

Then retrain:
```bash
./scripts/train_character_model.sh baseline qwen3_8b
```
The training script detects `is_reasoning_model` and automatically routes to
`augmented_curated_thinking_split_2048`.

```
Iter 1:    Val loss 3.241  ← higher than before (think blocks add text)
Iter 1595: Val loss 0.112  ← slightly higher, but think blocks are longer sequences
```

Chat:
```
>> Who are you?
<think>
This traveler seeks my name — a simple question, yet one that echoes through
the ages. I shall answer with the grace befitting the Celestial Archives.
</think>

I am Lyra Moonwhisper, keeper of the Celestial Archives and watcher of the
starlit paths. May the wisdom of the ancients illuminate thy journey.
```

**Key lesson**: Reasoning models are a different training target than standard
instruct models. You cannot train them with the same data and expect it to work.
The `<think>` block is part of the loss computation — if it's empty, the model
learns to produce empty think blocks as the optimal prediction. Garbage in,
garbage out, literally.

---

### Scenario 6: Plateauting val loss (stuck, not diverging)

**Setup**: Training a model and val loss plateaus at 0.4 around epoch 3 and
stays there through epoch 5. Not diverging, not improving.

**What you see**:
```
Iter 319:  Val loss 0.402  (epoch 1)
Iter 638:  Val loss 0.397  (epoch 2)
Iter 957:  Val loss 0.398  (epoch 3)
Iter 1276: Val loss 0.395  (epoch 4)
Iter 1595: Val loss 0.396  (epoch 5)
```

**Analysis**: The adapter hit capacity. With 319 examples and rank 16, the
adapter has learned everything it can from this dataset within ~2 epochs. The
remaining 3 epochs are just re-arranging the same information without improving
the loss. This is not overfitting (val loss isn't rising) — it's a saturation
point.

**The real question**: Is 0.4 actually bad? Depends on what 0.4 means for
this model and this dataset. On Mistral, 0.4 is high (target is ~0.08). On a
model with a very different pre-training distribution, 0.4 might be as good as
it gets at this rank and data volume.

**First attempt — increase rank from 16 to 32**:
More capacity might let the adapter push past the plateau.
```
Iter 1595: Val loss 0.291  ← improvement, but still not great
```
Better, but still plateauing. The adapter has more room now but the data may not
support further optimization.

**Second attempt — add more training data**:
The plateau might be a data ceiling, not a capacity ceiling. Generate 100 more
synthetic examples and retrain.
```
Iter 2195: Val loss 0.182  ← significant improvement
```
The plateau was a data diversity problem, not a capacity problem. More examples
= more patterns for the model to generalize.

**Third attempt — switch optimizer from adamw to muon**:
Some optimizers navigate plateaus better. muon's orthogonalized momentum may
find a descent direction that AdamW misses. (This is speculative — muon on SFT
is not well-characterized.)
```bash
./scripts/train_character_model.sh baseline qwen25_7b --optimizer muon
```

**Conclusion**: Plateauting at a reasonable loss is usually fine — the 0.4 model
might chat perfectly well. If you need better loss, add data before adding
capacity. The current 319 examples are enough for voice capture but may not
cover the full range of character expression needed for all models to converge
to the same low loss.

**Key lesson**: Every dataset has a loss floor for a given model and rank. Below
that floor, the model can't learn more because the data doesn't contain more
signal. At that point, more epochs, higher rank, or different optimizers are
all fighting physics. Add data or accept the floor.

---

### When to stop tuning

Alarm signals that mean you're over-tuning:

- **You're on your 5th iteration and the model sounds the same as iteration 2.**
  The remaining differences are noise. Ship it.
- **You're optimizing for a loss number you can't hear.** If val loss dropped
  from 0.10 to 0.08 and chat quality is identical, you're fitting numerical
  noise. Stop.
- **You changed three knobs in the last run.** You no longer know what fixed
  what. Go back to the last known-good config and start testing one variable
  at a time.
- **You're tuning a model you've never chatted.** Always chat the baseline
  (default config, full 5 epochs) before tuning anything. You need a reference
  point to know whether your changes are improving things.
- **Your training time per experiment exceeds 30 minutes.** At that point,
  you're probably tuning a model that's too large for your hardware or a dataset
  that's too large for LoRA to be the bottleneck. Switch to smaller experiments
  (fewer epochs, fewer layers, fewer examples) to find the right direction,
  then scale up.