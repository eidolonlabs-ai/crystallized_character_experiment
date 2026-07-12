#!/usr/bin/env python3
"""Fold the system message into the first user turn of a ChatML row.

Why this exists
---------------
`tokenizer.apply_chat_template()` on Mistral v0.3 (and most Llama-family
chat tokenizers) silently drops messages with `role == "system"`. So if your
training JSONL has a system prompt and you tokenize via apply_chat_template,
the character definition never reaches the model.

We pre-process the JSONL to fold the system message into the first user turn
as `[SYSTEM] ...\\n\\n[USER] ...`. This survives every chat template that
renders user content as-is, including Mistral's `[INST] {content} [/INST]`.

The transform is idempotent: a row that already starts with `[SYSTEM]` is left
alone, so running the fold twice is safe.

Usage
-----
    from fold_system_prompt import fold
    folded = fold(row)            # in-process

    python scripts/fold_system_prompt.py input.jsonl output.jsonl   # CLI

The CLI is for back-filling existing source files; the in-process `fold()` is
what `truncate_training_data.py`, `generate_synthetic_data.py`, and
`evaluate_character.py` import so the transform is applied automatically.
"""
import argparse
import json
import sys
from typing import Any

FOLD_MARKER_SYSTEM = "[SYSTEM]"
FOLD_MARKER_USER = "[USER]"


def fold(row: dict[str, Any]) -> dict[str, Any]:
    """If `row["messages"]` starts with a system message, fold its content into
    the first user message using the `[SYSTEM] ... \\n\\n [USER] ...` pattern.
    Returns a new dict; does not mutate the input.

    Idempotent: if the first user content already starts with `[SYSTEM]`,
    the row is returned as a system-less message list (the fold already
    happened, drop the redundant system entry).
    """
    if "messages" not in row:
        return row
    msgs = row["messages"]
    if not msgs or msgs[0].get("role") != "system":
        return row

    system_content = (msgs[0].get("content") or "").strip()
    if not system_content:
        # Empty system message — drop it, no fold needed.
        return {"messages": list(msgs[1:])}

    # Find the first user message after the system message.
    for i, m in enumerate(msgs[1:], start=1):
        if m.get("role") == "user":
            user_content = m.get("content", "") or ""
            if user_content.lstrip().startswith(FOLD_MARKER_SYSTEM):
                # Already folded. Drop the system entry and re-emit.
                return {"messages": list(msgs[1:])}
            folded_user = {
                "role": "user",
                "content": f"{FOLD_MARKER_SYSTEM} {system_content}\n\n{FOLD_MARKER_USER} {user_content}",
            }
            return {"messages": [folded_user] + list(msgs[i + 1:])}

    # No user message after the system message — drop the orphan system entry.
    return {"messages": list(msgs[1:])}


def _fold_file(input_path: str, output_path: str) -> tuple[int, int, int]:
    """Stream JSONL through fold(); return (total, folded, skipped) counts."""
    total = folded = skipped = 0
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            original = row
            out = fold(row)
            if out is not original:
                folded += 1
            fout.write(json.dumps(out) + "\n")
    return total, folded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold system messages into the first user turn (JSONL → JSONL).")
    parser.add_argument("input", help="Input JSONL file (one ChatML row per line)")
    parser.add_argument("output", help="Output JSONL file")
    args = parser.parse_args()

    total, folded, skipped = _fold_file(args.input, args.output)
    print(f"Read {total} rows: {folded} folded, {skipped} skipped (parse errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())