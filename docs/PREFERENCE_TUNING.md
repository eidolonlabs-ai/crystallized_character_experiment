# Preference Tuning

Beyond SFT (Supervised Fine-Tuning)
-----------------------------------

Once a character model has been trained with LoRA on conversation data,
the next step in modern pipelines is **preference tuning** — training the
model to prefer some outputs over others given the same prompt, using a
dataset of (prompt, chosen, rejected) triples.

The dominant algorithms are:
- **DPO** (Direct Preference Optimization, Rafailov et al. 2023) — the
  simplest and most common. Removes the RLHF reward model entirely;
  trains a classifier-style loss on preference pairs directly.
- **IPO** (Identity Preference Optimization, Azar et al. 2023) — fixes a
  known DPO overfitting failure mode.
- **KTO** (Kahneman-Tversky Optimization, Ethayarajh et al. 2024) —
  works on per-example good/bad labels instead of paired preferences,
  useful when you don't have ranked pairs.
- **SimPO** (Simple Preference Optimization, Meng et al. 2024) —
  reference-free variant, simpler than DPO.

Why we don't ship preference tuning today
----------------------------------------

`mlx-lm 0.30.0` does not include a `dpo`, `ipo`, `kto`, or `simpo`
subcommand. Implementing these from scratch on top of mlx is
straightforward (~200 lines per algorithm) but is a separate body of
work from the SFT pipeline this repo focuses on.

Recommended path if you want to add preference tuning
-----------------------------------------------------

1. Generate a small preference dataset (~200-1000 triples) for Lyra:
   - Take existing training prompts.
   - Generate 2-4 candidate responses per prompt with the SFT model.
   - Use GPT-4o or a strong open model to label which response best
     matches the character definition.

2. Implement DPO on top of `mlx_lm.tuner.lora`. The algorithm is:

   ```python
   # Pseudo-code, NOT mlx-lm's actual API
   loss = -log_sigmoid(
       beta * (logp_chosen - logp_rejected) -
       beta * (logp_chosen_ref - logp_rejected_ref)
   )
   ```

   With the LoRA path being applied to both the policy and reference
   model (the reference is just the policy with the LoRA adapter
   detached).

3. Train on Apple Silicon using the same `--grad-checkpoint` /
   `--batch-size 1` recipe as SFT.

4. Evaluate with the same `evaluate_character.py` and compare.

Tracking upstream
-----------------

Watch the [mlx-examples](https://github.com/ml-explore/mlx-examples) repo
for first-party DPO/IPO implementations. As of mlx-lm 0.30.0 (2026-07)
no stable support exists. Until then, the unofficial DPO notebook in
mlx-examples is the experimental path.