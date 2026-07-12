#!/usr/bin/env python3
"""Parse an mlx_lm training log into structured JSONL.

mlx_lm 0.30.0 prints two log line shapes during training (from
`mlx_lm/tuner/trainer.py::train`):

  Iter {it}: Train loss {loss}, Learning Rate {lr}, It/sec {rate},
             Tokens/sec {tokens}, Trained Tokens {n}, Peak mem {mem} GB
  Iter {it}: Val loss {loss}, Val took {time}s

This script extracts every such line from a log file and writes one JSON
record per line to stdout (or to --output PATH). The output is suitable
input for `scripts/plot_loss.py` or any external charting tool.

Usage
-----
    python scripts/parse_train_log.py adapters/baseline_mistral_qlora/train.log
    python scripts/parse_train_log.py train.log --output loss.jsonl

The output schema:
    {"iteration": int, "train_loss": float|null, "val_loss": float|null,
     "learning_rate": float|null, "tokens_per_sec": float|null,
     "peak_mem_gb": float|null, "val_seconds": float|null}
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Regex for the two line shapes mlx-lm 0.30.0 emits. Compiled once.
_TRAIN_RE = re.compile(
    r"^Iter\s+(\d+):\s+"
    r"Train loss\s+([\d.]+),\s+"
    r"Learning Rate\s+([\d.eE+-]+),\s+"
    r"It/sec\s+([\d.]+),\s+"
    r"Tokens/sec\s+([\d.]+),\s+"
    r"Trained Tokens\s+(\d+),\s+"
    r"Peak mem\s+([\d.]+)\s+GB",
    re.IGNORECASE,
)
_VAL_RE = re.compile(
    r"^Iter\s+(\d+):\s+"
    r"Val loss\s+([\d.]+),\s+"
    r"Val took\s+([\d.]+)s",
    re.IGNORECASE,
)


def parse_log(path: Path) -> list[dict]:
    """Stream a log file; return a list of parsed records (one per matched line)."""
    records: dict[int, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TRAIN_RE.match(line)
            if m:
                it = int(m.group(1))
                rec = records.setdefault(it, {"iteration": it})
                rec["train_loss"] = float(m.group(2))
                rec["learning_rate"] = float(m.group(3))
                rec["it_per_sec"] = float(m.group(4))
                rec["tokens_per_sec"] = float(m.group(5))
                rec["trained_tokens"] = int(m.group(6))
                rec["peak_mem_gb"] = float(m.group(7))
                continue
            m = _VAL_RE.match(line)
            if m:
                it = int(m.group(1))
                rec = records.setdefault(it, {"iteration": it})
                rec["val_loss"] = float(m.group(2))
                rec["val_seconds"] = float(m.group(3))
    return [records[k] for k in sorted(records)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="Path to mlx_lm training log")
    parser.add_argument("--output", type=Path, help="Write JSONL here (default: stdout)")
    parser.add_argument("--quiet", action="store_true", help="Don't print summary to stderr")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        return 1

    records = parse_log(args.log)
    train_count = sum(1 for r in records if r.get("train_loss") is not None)
    val_count = sum(1 for r in records if r.get("val_loss") is not None)

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        for r in records:
            out.write(json.dumps(r) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()

    if not args.quiet:
        print(f"Parsed {len(records)} records ({train_count} train, {val_count} val) from {args.log}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())