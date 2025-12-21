#!/usr/bin/env python3
"""
MLX-based training script for Mac using official mlx-lm package
Optimized for Apple Silicon using MLX framework
"""
import argparse
import time
import subprocess
import sys
import os
from pathlib import Path
import shutil

def prepare_data(data_dir, output_dir):
    """Copy data to output directory for mlx-lm"""
    data_path = Path(data_dir)
    train_file = data_path / "train.jsonl"
    valid_file = data_path / "valid.jsonl"
    
    # Validate data directory exists
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        sys.exit(1)
    
    # Validate required files exist
    if not train_file.exists():
        print(f"ERROR: Training file not found: {train_file}")
        sys.exit(1)
    if not valid_file.exists():
        print(f"ERROR: Validation file not found: {valid_file}")
        sys.exit(1)
    
    # Count samples
    try:
        with open(train_file, encoding='utf-8') as f:
            train_count = sum(1 for _ in f)
        with open(valid_file, encoding='utf-8') as f:
            valid_count = sum(1 for _ in f)
    except Exception as e:
        print(f"ERROR: Failed to read data files: {e}")
        sys.exit(1)
    
    if train_count == 0:
        print(f"ERROR: Training file is empty: {train_file}")
        sys.exit(1)
    if valid_count == 0:
        print(f"ERROR: Validation file is empty: {valid_file}")
        sys.exit(1)
    
    # Copy files to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy(train_file, output_path / "train.jsonl")
        shutil.copy(valid_file, output_path / "valid.jsonl")
    except Exception as e:
        print(f"ERROR: Failed to copy data files: {e}")
        sys.exit(1)
    
    return train_count, valid_count

def train(config):
    # Enable Metal debug logging for better crash diagnostics
    os.environ['MLX_METAL_DEBUG'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("Training with official MLX-LM (Apple Silicon optimized)")
    print(f"Model: {config.model}")
    print(f"Using prompt masking: {config.mask_prompt}")
    
    # Prepare data
    output_dir = Path(config.output_dir)
    train_count, valid_count = prepare_data(config.data_dir, output_dir)
    
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {valid_count}")
    
    # Calculate iterations based on epochs
    iters_per_epoch = train_count // config.batch_size
    total_iters = iters_per_epoch * config.epochs
    
    # Check for existing checkpoints to resume from
    resume_file = None
    completed_iters = 0
    checkpoint_files = list(output_dir.glob("[0-9]*_adapters.safetensors"))
    if checkpoint_files:
        # Find the latest checkpoint by iteration number
        latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[0]))
        completed_iters = int(latest_checkpoint.stem.split('_')[0])
        resume_file = latest_checkpoint
        print(f"Found checkpoint at iter {completed_iters}: {resume_file}")
        print(f"Resuming training from iter {completed_iters}")
    
    remaining_iters = total_iters - completed_iters
    print(f"Total iterations: {total_iters} ({iters_per_epoch} per epoch)")
    if resume_file:
        print(f"Remaining iterations: {remaining_iters}")
    
    # Build command using supported entrypoint (python -m mlx_lm lora)
    cmd = [
        sys.executable,
        "-m", "mlx_lm", "lora",
        "--model", config.model,
        "--train",
        "--data", str(output_dir),
        "--adapter-path", str(output_dir),
        "--iters", str(remaining_iters if resume_file else total_iters),
        "--steps-per-report", "10",
        "--steps-per-eval", str(iters_per_epoch),  # Eval once per epoch
        "--save-every", "50",  # Save more frequently to minimize loss on crash
        "--batch-size", str(config.batch_size),
        "--num-layers", str(config.num_layers),
        "--learning-rate", str(config.learning_rate),
        "--max-seq-length", str(config.max_seq_length),
    ]
    
    # Add resume file if found
    if resume_file:
        cmd.extend(["--resume-adapter-file", str(resume_file)])
    
    # Add prompt masking if requested
    if config.mask_prompt:
        cmd.append("--mask-prompt")
    
    # Add gradient accumulation if > 1
    if config.gradient_accumulation_steps > 1:
        cmd.extend(["--grad-accumulation-steps", str(config.gradient_accumulation_steps)])
    
    # Add fine-tune type
    if config.dora:
        cmd.extend(["--fine-tune-type", "dora"])
    else:
        cmd.extend(["--fine-tune-type", "lora"])
    
    masking_status = "with prompt masking" if config.mask_prompt else "without prompt masking"
    print(f"Running {'DoRA' if config.dora else 'LoRA'} training {masking_status}...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        print("Check the output above for error details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time / 60
    
    print(f"\nTraining completed in {minutes:.2f} minutes ({elapsed_time:.2f} seconds)")
    print(f"Adapters saved to {config.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with official MLX-LM LoRA")
    parser.add_argument("--model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with train.jsonl and valid.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for adapters")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers to fine-tune")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length (default 4096 for long conversations)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--mask-prompt", action="store_true", default=False, help="Mask prompt during training (only compute loss on completion)")
    parser.add_argument("--no-mask-prompt", action="store_false", dest="mask_prompt", help="Disable prompt masking")
    parser.add_argument("--qlora", action="store_true", help="Legacy flag (ignored, use --dora if you want DoRA)")
    parser.add_argument("--dora", action="store_true", help="Use DoRA instead of standard LoRA")
    
    args = parser.parse_args()
    train(args)
