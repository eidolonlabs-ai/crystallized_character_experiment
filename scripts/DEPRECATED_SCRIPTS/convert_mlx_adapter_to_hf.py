#!/usr/bin/env python3
"""
Convert MLX LoRA adapter to HuggingFace PEFT format.
This allows using the adapter with llama.cpp's convert_lora_to_gguf.py.

MLX LoRA shapes:
  lora_a: [in_features, rank]
  lora_b: [rank, out_features]

HF PEFT shapes:
  lora_A.weight: [rank, in_features]
  lora_B.weight: [out_features, rank]
"""

import argparse
import json
import os
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

def convert_mlx_to_hf(adapter_path, output_path):
    print(f"Converting MLX adapter at {adapter_path} to HF format at {output_path}")
    
    # Load config
    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load weights
    weights_path = Path(adapter_path) / "adapters.safetensors"
    mlx_weights = load_file(weights_path)
    
    hf_weights = {}
    
    for key, tensor in mlx_weights.items():
        # MLX keys: model.layers.0.self_attn.q_proj.lora_a
        # HF keys: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # But convert_lora_to_gguf.py might expect simpler keys or standard PEFT keys.
        # Standard PEFT keys usually look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # Or just: model.layers.0.self_attn.q_proj.lora_A.weight depending on how it's loaded.
        
        # Let's try to match what convert_lora_to_gguf.py expects.
        # It usually expects keys that match the base model's keys but with lora suffix.
        
        new_key = key
        
        # Transpose weights
        # MLX lora_a [in, rank] -> HF lora_A [rank, in]
        # MLX lora_b [rank, out] -> HF lora_B [out, rank]
        
        if key.endswith(".lora_a"):
            new_key = key.replace(".lora_a", ".lora_A.weight")
            hf_weights[new_key] = tensor.T.contiguous()
        elif key.endswith(".lora_b"):
            new_key = key.replace(".lora_b", ".lora_B.weight")
            hf_weights[new_key] = tensor.T.contiguous()
        else:
            hf_weights[key] = tensor.contiguous()
            
    # Save weights
    os.makedirs(output_path, exist_ok=True)
    save_file(hf_weights, Path(output_path) / "adapter_model.safetensors")
    
    # Save config
    # HF PEFT config might need some adjustments
    hf_config = config.copy()
    # Ensure required fields are present
    if "lora_alpha" not in hf_config and "lora_parameters" in config:
        hf_config["lora_alpha"] = config["lora_parameters"]["scale"]
        hf_config["r"] = config["lora_parameters"]["rank"]
        hf_config["lora_dropout"] = config["lora_parameters"].get("dropout", 0.0)
        
    if "base_model_name_or_path" not in hf_config and "model" in config:
        hf_config["base_model_name_or_path"] = config["model"]
    
    with open(Path(output_path) / "adapter_config.json", 'w') as f:
        json.dump(hf_config, f, indent=2)
        
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Path to MLX adapter")
    parser.add_argument("--output-path", required=True, help="Output path for HF adapter")
    args = parser.parse_args()
    
    convert_mlx_to_hf(args.adapter_path, args.output_path)
