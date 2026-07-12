# Chat Templates

Why this doc exists
-------------------

Modern SFT pipelines depend on the tokenizer's chat template producing
predictable output — the same input string should tokenize to the same
sequence at training time and at inference time. But several popular
chat tokenizers **silently drop messages with `role="system"`** when
called via `tokenizer.apply_chat_template()`.

That bug bites anyone who trains a character model on data that includes
a system prompt: the model never sees the character definition during
training, so the "crystallization" hypothesis (system prompt + LoRA =
consistent voice) silently fails. We hit this in `baseline` and fixed it
in Phase 0 of the modernization (the `fold_system_prompt` transform).

Verified status per model
-------------------------

| Model | System message rendered? | Workaround applied |
|-------|--------------------------|--------------------|
| `mistral_v0_3` (Mistral 7B Instruct v0.3) | **No — silently dropped** | Folded into first user turn (`[SYSTEM] ... [USER] ...`) |
| `llama31_8b` | **Untested — gated repos, 401 in our env** | Fold applied defensively in `truncate_training_data.py` |

How the fold works
------------------

`scripts/fold_system_prompt.py::fold()` rewrites a ChatML row like:

```json
{"messages": [
  {"role": "system", "content": "You are Lyra Moonwhisper."},
  {"role": "user", "content": "Who are you?"},
  {"role": "assistant", "content": "I am Lyra."}
]}
```

into:

```json
{"messages": [
  {"role": "user", "content": "[SYSTEM] You are Lyra Moonwhisper.\n\n[USER] Who are you?"},
  {"role": "assistant", "content": "I am Lyra."}
]}
```

Now when Mistral's template renders `[INST] {content} [/INST]`, the
character definition is in the user content and survives. The transform
is idempotent — running it twice produces the same output.

The fold is applied automatically:
- At data prep time (`scripts/truncate_training_data.py` — every row folded)
- At eval time (`scripts/evaluate_character.py::render_prompt_for_eval` — every prompt wrapped)
- The raw `messages` field is preserved in source JSONLs in case a future
  mlx-lm release fixes the system-message bug upstream; rerunning
  `prepare_all_datasets.sh` would just produce the same folded output.

Verifying it for your model
---------------------------

Use the `scripts/render_template_check.py` smoke test (added in Phase 4
of the modernization) against a multi-turn example to confirm the
system text appears in the rendered template for each base model you
care about. CI-style usage:

```bash
python scripts/render_template_check.py data/examples/lyra_multiturn_sample.jsonl
# exits non-zero if any model's chat template dropped the system text
```

Inference-time gotcha: chat_character.sh
----------------------------------------

`chat_character.sh` uses `mlx_lm chat --system-prompt "..."`. With Mistral
v0.3, that system prompt is also dropped by the chat template. The
model therefore sees only the user turns — which still works because the
LoRA adapter internalized character voice from the training data, but it
doesn't see the explicit system prompt at inference.

If you want system-prompt-aware inference today:
1. Pre-fold your prompt manually: `[SYSTEM] <system>\n\n[USER] <query>`
2. Or write a small wrapper script that reads user input, folds it, and
   forwards to `mlx_lm generate` (no chat template in the way).

A `folded_chat.py` wrapper that does this is a clean follow-up PR — not
included here.

Upgrade path: custom chat template
----------------------------------

If you want template cleanliness instead of fold-and-go, the path is:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tok.chat_template = """{% for m in messages %}[INST] {{m['content']}} [/INST]{% endfor %}"""
```

…and supply your own `system` rendering. This works but needs testing
across every base model in the matrix (2 currently) and breaks the moment
mlx-lm updates its template rendering. Fold-and-go avoids that surface
area.