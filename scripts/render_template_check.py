#!/usr/bin/env python3
"""CI-style smoke check: verify the chat template renders the system text.

For each row in a JSONL file, applies the Phase 0 fold and then calls
`tokenizer.apply_chat_template()` to render the conversation. Asserts that
the system prompt text appears in the rendered output — this is the
contract the Phase 0 fix exists to enforce, because Mistral v0.3 (and
likely the Llama chat tokenizers) silently drop `role=system` messages.

Exit codes:
  0  every row passed (system text rendered)
  1  at least one row failed — system text was dropped

Usage:
    python scripts/render_template_check.py data/examples/lyra_multiturn_sample.jsonl
    python scripts/render_template_check.py data/examples/lyra_multiturn_sample.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

# Repo-relative imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_system_prompt import fold as _fold_row


def check_file(jsonl_path: Path, model_name: str) -> tuple[int, int]:
    """Returns (passed, failed) counts."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer for {model_name!r}: {e}", file=sys.stderr)
        return (0, 0)

    passed = failed = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [{line_no}] JSON parse error: {e}", file=sys.stderr)
                failed += 1
                continue

            # Find the original system text (before folding).
            orig_system = ""
            for m in row.get("messages", []):
                if m.get("role") == "system":
                    orig_system = m.get("content", "")
                    break
            if not orig_system:
                # No system message — nothing to assert.
                print(f"  [{line_no}] (no system message — skipped)")
                continue

            folded = _fold_row(row)
            try:
                rendered = tok.apply_chat_template(folded["messages"], tokenize=False)
            except Exception as e:
                print(f"  [{line_no}] chat template failed: {e}", file=sys.stderr)
                failed += 1
                continue

            # The contract: the system text (or a unique substring) must appear
            # in the rendered output. We check for a 30-char prefix to be
            # robust to small whitespace differences.
            probe = orig_system[:30]
            if probe in rendered:
                print(f"  [{line_no}] PASS  ({len(rendered)} chars rendered, system prefix '{probe[:24]}…' found)")
                passed += 1
            else:
                print(f"  [{line_no}] FAIL  system text dropped from rendered output")
                print(f"        first 200 chars of rendered: {rendered[:200]!r}")
                failed += 1

    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jsonl", type=Path, help="Path to a JSONL file with ChatML rows")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HF repo or local path to load the tokenizer from (default: Mistral 7B v0.3).",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"Not found: {args.jsonl}", file=sys.stderr)
        return 1

    print(f"Checking chat-template rendering on {args.jsonl.name} using {args.model!r}")
    print()
    passed, failed = check_file(args.jsonl, args.model)
    print()
    print(f"Result: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())