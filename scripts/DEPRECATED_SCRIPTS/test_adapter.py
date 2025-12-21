#!/usr/bin/env python3
"""Quick test script for adapters"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--prompt", required=True)
args = parser.parse_args()

print(f"Loading model {args.model}...")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# Detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

print(f"Loading adapter {args.adapter}...")
model = PeftModel.from_pretrained(model, args.adapter)

messages = [{"role": "user", "content": args.prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\nGenerating response...\n")
outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
