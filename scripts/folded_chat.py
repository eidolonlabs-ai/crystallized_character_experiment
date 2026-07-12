#!/usr/bin/env python3
"""Folded interactive chat for mlx-lm — matches the Phase 0 training distribution.

Why this exists
---------------
`mlx_lm chat` builds its prompt via `tokenizer.apply_chat_template()`, and
Mistral v0.3 (and the Llama chat tokenizers) silently drop messages with
`role="system"` there. So `--system-prompt` never reaches the model — the
adapter's [SYSTEM] trigger isn't met, and the response degrades to base-
model behavior with weak character flavor.

`truncate_training_data.py` solved this at *training* time by folding the
system message into the first user turn as `[SYSTEM] ... \\n\\n[USER] ...`.
This script applies the same fold at *inference* time.

The fold is applied to the FIRST user turn only — subsequent turns continue
with the model's own chat template (which Llama 3 and Mistral handle
correctly for non-system messages). The model was trained single-turn so
it never saw a turn continuation in training data, but the chat template
turn-separator ([INST] [/INST], <|eot_id|>, etc.) is in-distribution for
the base model, so multi-turn works at inference.

Usage
-----
    python scripts/folded_chat.py \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --adapter-path adapters/baseline_mistral_qlora \\
        --system-prompt "You are Lyra Moonwhisper, ..."

Reads user input from stdin, prints model responses to stdout. Type `exit`,
`quit`, or `q` (or Ctrl+C) to leave.
"""
import argparse
import sys
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

FOLD_MARKER_SYSTEM = "[SYSTEM]"
FOLD_MARKER_USER = "[USER]"


def fold_first_turn(system: str, user: str) -> str:
    """Build the first user-turn content with the system prompt folded in.

    Mirrors scripts/fold_system_prompt.py::fold() so training and inference
    see the exact same prompt shape.
    """
    return f"{FOLD_MARKER_SYSTEM} {system.strip()}\n\n{FOLD_MARKER_USER} {user}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="HF repo or local path for the base model")
    p.add_argument("--adapter-path", default=None,
                   help="Path to LoRA adapter directory (optional)")
    p.add_argument("--system-prompt", required=True,
                   help="Character system prompt (will be folded into first turn)")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading model: {args.model}", flush=True)
    if args.adapter_path:
        print(f"Loading adapter: {args.adapter_path}", flush=True)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    # Conversation state: list of {role, content} dicts.
    # The first user message is folded with [SYSTEM]...; subsequent ones
    # are plain. We rebuild the prompt from this list each turn using
    # apply_chat_template(tokenize=False). Mistral/Llama tokenizers drop
    # role=system messages (hence the fold), but Qwen's template renders
    # them — so on Qwen we inject an explicit system message for language
    # constraint while keeping the fold intact for training consistency.
    messages: list[dict] = []
    first_turn = True

    # Qwen models auto-insert "You are Qwen, created by Alibaba Cloud..."
    # into the system slot when no system message is present (the fold
    # eliminates the system role). This creates a dual-identity conflict
    # with the folded character prompt and causes the model to fall back
    # to Chinese for self-correction. Provide a system message that
    # matches the training distribution while constraining language.
    is_qwen = "qwen" in args.model.lower()
    QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. All your responses must be in English only. Do not use Chinese, Japanese, or any other language."

    print()
    print("Folded chat — Phase 0 fold applied on every first turn.")
    print("Commands: 'q' exit, 'r' reset, 'h' help.")
    print(f"System prompt: {args.system_prompt[:80]}...")
    print()

    while True:
        try:
            user = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not user:
            continue
        if user in ("q", "quit", "exit"):
            print("Bye.")
            return 0
        if user == "r":
            messages = []
            first_turn = True
            print("(conversation reset)")
            continue
        if user == "h":
            print("Commands: 'q' exit, 'r' reset, 'h' help.")
            continue

        if is_qwen and first_turn:
            messages.append({"role": "system", "content": QWEN_SYSTEM})

        if first_turn:
            content = fold_first_turn(args.system_prompt, user)
            first_turn = False
        else:
            content = user
        messages.append({"role": "user", "content": content})

        # Render the conversation. The fold puts the character prompt inside
        # the first user turn's content. For Qwen models we also inject a
        # system message to prevent the template from auto-inserting a
        # conflicting default and to constrain language.
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Stream the response.
        print()
        chunks: list[str] = []
        for resp in stream_generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            sys.stdout.write(resp.text)
            sys.stdout.flush()
            chunks.append(resp.text)
        print("\n")
        assistant_text = "".join(chunks).strip()
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    sys.exit(main() or 0)