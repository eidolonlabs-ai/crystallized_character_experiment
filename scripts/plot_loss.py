#!/usr/bin/env python3
"""Plot training loss from a parsed mlx_lm log.

Default output: an ASCII sparkline + tabular summary printed to stdout —
useful when matplotlib isn't installed or when you want a quick eyeball
check in a terminal.

With `--png PATH`, also writes a matplotlib PNG (only if matplotlib is
importable; if not, prints a hint to stderr and skips the PNG).

Input formats accepted:
  - Path to a raw mlx_lm train.log (auto-parsed via parse_train_log.parse_log)
  - Path to a JSONL produced by parse_train_log.py

Usage
-----
    python scripts/plot_loss.py adapters/baseline_mistral_qlora/train.log
    python scripts/plot_loss.py loss.jsonl --png loss.png
"""
import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/plot_loss.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_train_log import parse_log

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 60) -> str:
    """Render a list of floats as a fixed-width Unicode bar chart."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    # Downsample to `width` buckets.
    if len(values) > width:
        bucket_size = len(values) / width
        sampled = [
            sum(values[int(i * bucket_size):int((i + 1) * bucket_size)]) /
            max(int((i + 1) * bucket_size) - int(i * bucket_size), 1)
            for i in range(width)
        ]
    else:
        sampled = values
    return "".join(SPARK_CHARS[min(int((v - lo) / span * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1)]
                   for v in sampled)


def _load_records(path: Path) -> list[dict]:
    """Load records from either a raw log or a JSONL file."""
    if path.suffix == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    return parse_log(path)


def print_summary(records: list[dict]) -> None:
    train = [(r["iteration"], r["train_loss"]) for r in records if r.get("train_loss") is not None]
    val = [(r["iteration"], r["val_loss"]) for r in records if r.get("val_loss") is not None]

    if not train:
        print("No train-loss records found.")
        return

    print(f"Records: {len(records)} ({len(train)} train, {len(val)} val)")
    if train:
        first, last = train[0][1], train[-1][1]
        print(f"Train loss: first={first:.4f}  last={last:.4f}  delta={last - first:+.4f}")
    if val:
        first, last = val[0][1], val[-1][1]
        print(f"Val   loss: first={first:.4f}  last={last:.4f}  delta={last - first:+.4f}")
    if train:
        print()
        print(f"Train: {_sparkline([v for _, v in train])}")
    if val:
        print(f"Val:   {_sparkline([v for _, v in val])}")
    print()

    # Last 10 records tabular.
    print(f"{'iter':>6}  {'train':>8}  {'val':>8}  {'lr':>10}  {'peak_mem':>8}")
    for r in records[-10:]:
        tr = f"{r['train_loss']:.4f}" if r.get("train_loss") is not None else "—"
        vl = f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else "—"
        lr = f"{r['learning_rate']:.2e}" if r.get("learning_rate") is not None else "—"
        pm = f"{r['peak_mem_gb']:.2f}" if r.get("peak_mem_gb") is not None else "—"
        print(f"{r['iteration']:>6}  {tr:>8}  {vl:>8}  {lr:>10}  {pm:>8} GB")


def write_png(records: list[dict], out_path: Path) -> bool:
    """Try to write a matplotlib PNG. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping PNG. Records: {out_path}",
              file=sys.stderr)
        return False

    train_it = [r["iteration"] for r in records if r.get("train_loss") is not None]
    train_loss = [r["train_loss"] for r in records if r.get("train_loss") is not None]
    val_it = [r["iteration"] for r in records if r.get("val_loss") is not None]
    val_loss = [r["val_loss"] for r in records if r.get("val_loss") is not None]

    plt.figure(figsize=(8, 4))
    if train_it:
        plt.plot(train_it, train_loss, label="train", alpha=0.6)
    if val_it:
        plt.plot(val_it, val_loss, label="val", linewidth=2)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("mlx_lm training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Wrote {out_path}", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Path to train.log or loss.jsonl")
    parser.add_argument("--png", type=Path, help="Also write a PNG here (requires matplotlib)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Not found: {args.path}", file=sys.stderr)
        return 1

    records = _load_records(args.path)
    print_summary(records)

    if args.png:
        write_png(records, args.png)

    return 0


if __name__ == "__main__":
    sys.exit(main())