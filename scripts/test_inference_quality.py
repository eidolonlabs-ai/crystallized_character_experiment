#!/usr/bin/env python3
"""
Comprehensive inference testing for trained character models.

Usage:
    python scripts/test_inference_quality.py [character] [num_samples]

Examples:
    python scripts/test_inference_quality.py baseline
    python scripts/test_inference_quality.py baseline 3
"""

import sys
import subprocess
import json
from pathlib import Path
from collections import defaultdict

def find_adapters(character):
    """Find all trained adapters for a character"""
    adapters_dir = Path("adapters")
    pattern = f"{character}_*_qlora"
    return sorted(adapters_dir.glob(pattern))

def get_model_config(model_name):
    """Map model name to paths"""
    configs = {
        "mistral": ("models/mistral-7b-instruct-v0.3-4bit", "mistralai/Mistral-7B-Instruct-v0.3"),
        "mistral_v0_2": ("models/mistral-7b-instruct-v0.2-4bit", "mistralai/Mistral-7B-Instruct-v0.2"),
        "mistral_v0_1": ("models/mistral-7b-instruct-v0.1-4bit", "mistralai/Mistral-7B-Instruct-v0.1"),
        "llama31_8b": ("models/llama-3.1-8b-instruct-4bit", "meta-llama/Llama-3.1-8B-Instruct"),
        "llama3_8b": ("models/llama-3-8b-instruct-4bit", "meta-llama/Meta-Llama-3-8B-Instruct"),
        "llama2_7b": ("models/llama-2-7b-chat-4bit", "meta-llama/Llama-2-7b-chat-hf"),
    }
    return configs.get(model_name, (None, None))

def get_test_prompts(character, num_prompts):
    """Get test prompts for a character"""
    prompts = {
        "baseline": [
            "Who are you?",
            "What is your name and role?",
            "Tell me about your wisdom",
            "How do you greet people?",
            "What matters most to you?",
        ],
    }
    return prompts.get(character, [])[:num_prompts]

def test_inference(quantized_model, adapter_path, prompt, timeout=30):
    """Test inference and return output"""
    try:
        cmd = [
            "python", "-m", "mlx_lm", "generate",
            "--model", str(quantized_model),
            "--adapter-path", str(adapter_path),
            "--prompt", prompt,
            "--max-tokens", "100",
            "--temp", "0.7"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            # Extract the actual response (last line usually has the output)
            lines = result.stdout.strip().split('\n')
            return lines[-1] if lines else None
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    character = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    num_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Validate character
    valid_chars = ["baseline"]
    if character not in valid_chars:
        print(f"Error: Unknown character '{character}'")
        print(f"Available: {', '.join(valid_chars)}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(f"Inference Quality Testing - {character.upper()}")
    print("="*70)
    print(f"Test prompts: {num_prompts}")
    print()
    
    # Find adapters
    adapters = find_adapters(character)
    if not adapters:
        print(f"No trained adapters found for {character}")
        sys.exit(1)
    
    test_prompts = get_test_prompts(character, num_prompts)
    results = defaultdict(dict)
    
    # Test each adapter
    for adapter_dir in adapters:
        adapter_name = adapter_dir.name
        # Extract model name: baseline_mistral_qlora -> mistral
        model_name = adapter_name.replace(f"{character}_", "").replace("_qlora", "")
        
        # Pretty name for display
        model_display = model_name.replace("_", ".")
        
        print(f"\n▶ Testing: {model_display}")
        print("-" * 70)
        
        # Get model paths — prefer quantized local model, fall back to HF repo
        quantized_model, hf_model = get_model_config(model_name)
        if not quantized_model and not hf_model:
            print(f"  ✗ Model not found for: {model_name}")
            results[model_name]["status"] = "FAILED"
            continue
        
        chat_model = quantized_model if Path(quantized_model).exists() and quantized_model else hf_model
        
        if not (adapter_dir / "adapters.safetensors").exists():
            print(f"  ✗ Adapter weights not found")
            results[model_name]["status"] = "FAILED"
            continue
        
        # Test prompts
        successful_prompts = 0
        responses = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  [{i}/{len(test_prompts)}] Testing: \"{prompt[:40]}...\"")
            
            response = test_inference(chat_model, adapter_dir, prompt)
            if response and len(response) > 10:
                successful_prompts += 1
                responses.append(response[:100])
                print(f"       ✓ Response received ({len(response)} chars)")
            else:
                print(f"       ✗ No response or too short")
        
        # Store results
        success_rate = successful_prompts / len(test_prompts) if test_prompts else 0
        results[model_name]["status"] = "PASSED" if success_rate >= 0.5 else "FAILED"
        results[model_name]["success_rate"] = success_rate
        results[model_name]["responses"] = responses
        
        # Report
        status_icon = "✓" if results[model_name]["status"] == "PASSED" else "✗"
        print(f"\n  {status_icon} Result: {successful_prompts}/{len(test_prompts)} prompts successful")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    total = len(results)
    
    print(f"\nModels tested: {total}")
    print(f"Models passed: {passed}")
    print(f"Models failed: {total - passed}\n")
    
    for model_name, result in sorted(results.items()):
        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        rate = result.get("success_rate", 0) * 100
        print(f"  {status_icon} {model_name:15} - {rate:.0f}% success rate")
    
    print("\n" + "="*70)
    print("To interact with models:")
    print(f"  ./chat_character.sh {character} <model_name>")
    print("\nExample:")
    print(f"  ./chat_character.sh {character} mistral")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
