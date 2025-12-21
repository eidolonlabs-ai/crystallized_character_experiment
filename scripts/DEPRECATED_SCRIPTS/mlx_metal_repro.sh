#!/bin/bash

# Minimal MLX/Metal crash repro runner
# Creates a tiny dataset and runs mlx_lm lora with modest settings to surface Metal Internal Error 0x0e.

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

# Metal debug for clearer errors
export MLX_METAL_DEBUG=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# M2/earlier GPU timeout workarounds (uncomment to try alternatives)
export MLX_MAX_OPS_PER_BUFFER=1
# export MLX_MAX_MB_PER_BUFFER=10  # Lower MB threshold for command buffer commits
# export MLX_METAL_FAST_SYNCH=1    # Enable faster Metal synchronization 

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR="raw_data/min_repro"
OUTPUT_DIR="adapters/min_repro_lora"
MAX_SEQ_LENGTH=4096
ITERS=1200
SAVE_EVERY=200
EVAL_EVERY=400
BATCH_SIZE=1
NUM_LAYERS=16
LR=1e-4

mkdir -p "$DATA_DIR"
python - <<PY
import json
from pathlib import Path

data_dir = Path("$DATA_DIR")
data_dir.mkdir(parents=True, exist_ok=True)
long_text = " ".join(["longtext"] * 1200)

train = [
  {"messages": [{"role": "user", "content": long_text}, {"role": "assistant", "content": "ack " + long_text}]},
  {"messages": [{"role": "user", "content": "Tell me a very long story " + long_text}, {"role": "assistant", "content": "story " + long_text}]},
  {"messages": [{"role": "user", "content": long_text}, {"role": "assistant", "content": long_text}]},
  {"messages": [{"role": "user", "content": "repeat " + long_text}, {"role": "assistant", "content": "repeat back " + long_text}]},
]

valid = [
  {"messages": [{"role": "user", "content": long_text}, {"role": "assistant", "content": "val " + long_text}]}
]

with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
  for ex in train:
    f.write(json.dumps(ex) + "\n")

with open(data_dir / "valid.jsonl", "w", encoding="utf-8") as f:
  for ex in valid:
    f.write(json.dumps(ex) + "\n")
PY

python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$OUTPUT_DIR" \
  --iters $ITERS \
  --steps-per-report 10 \
  --steps-per-eval $EVAL_EVERY \
  --save-every $SAVE_EVERY \
  --batch-size $BATCH_SIZE \
  --num-layers $NUM_LAYERS \
  --learning-rate $LR \
  --max-seq-length $MAX_SEQ_LENGTH \
  --mask-prompt \
  --fine-tune-type lora
