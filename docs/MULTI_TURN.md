# Multi-Turn Training

Current state
-------------

All 355 training examples in `raw_data/training_data_baseline_augmented.jsonl`
are **single-turn** (verified: `min=3, max=3, avg=3.0 messages`). The
prepared splits at `raw_data/prepared_data/baseline/augmented_split_*/`
all share that property. The trained adapters therefore never see a
follow-up user message during training, even though `chat_character.sh`
runs multi-turn inference.

This is a known limitation of the current data corpus, not a bug in the
training pipeline. The truncation logic in
[`scripts/truncate_training_data.py`](../scripts/truncate_training_data.py)
is built for multi-turn (backwards-grow from the last assistant message,
keeping system prompt + as many recent turns as fit) — it's just not
exercised by the data we have.

What multi-turn training would look like
----------------------------------------

A multi-turn row in the JSONL:

```json
{"messages": [
  {"role": "system", "content": "You are Lyra Moonwhisper."},
  {"role": "user", "content": "Who are you?"},
  {"role": "assistant", "content": "I am Lyra Moonwhisper, Keeper of the Celestial Archives..."},
  {"role": "user", "content": "Tell me about the archives."},
  {"role": "assistant", "content": "The Celestial Archives hold the wisdom of ages..."}
]}
```

After the Phase 0 fold, this becomes:

```json
{"messages": [
  {"role": "user", "content": "[SYSTEM] You are Lyra Moonwhisper.\n\n[USER] Who are you?"},
  {"role": "assistant", "content": "I am Lyra Moonwhisper, Keeper of the Celestial Archives..."},
  {"role": "user", "content": "Tell me about the archives."},
  {"role": "assistant", "content": "The Celestial Archives hold the wisdom of ages..."}
]}
```

When tokenized by `mlx_lm`'s data loader, the turns get concatenated into
one long sequence with the model's chat template separators. The
`--mask-prompt` flag (now supported cleanly in mlx-lm 0.30.0) is what
distinguishes user-side loss masking across turns: without it, every
token contributes to loss; with it, only assistant tokens do.

Why we don't have multi-turn data yet
-------------------------------------

The 22 real `baseline` conversations are all single-turn by design — each
is one user question + one Lyra answer, no follow-ups. The synthetic
generation pipeline (`scripts/generate_synthetic_data.py`) was configured
to produce 3-6 turn conversations but the curated split ended up taking
only the most recent user-assistant pair.

Adding multi-turn data
----------------------

Two practical paths:

1. **Hand-craft a few dozen multi-turn conversations** in Lyra's voice
   that show character continuity across turns (callback to an earlier
   topic, building on a metaphor, refusing and redirecting). Add them
   to `raw_data/training_data_baseline.jsonl` and re-run
   `prepare_all_datasets.sh`.

2. **Modify `scripts/generate_synthetic_data.py`** to request 4-6 turn
   conversations explicitly (the current prompt says "3-6 turns" but
   the post-processing in `curate_training_data.py` flattens to the
   last user/assistant pair — that flattening step would need to be
   removed or relaxed for multi-turn data to survive).

Either path is straightforward but is out of scope for the current
modernization PR.