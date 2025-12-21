#!/usr/bin/env python3
"""Compare base model vs fine-tuned model responses with same system prompt."""

import sys
from mlx_lm import load, generate

SYSTEM_PROMPT = "You are Lyra Moonwhisper, an ancient elven sage who dwells in the Celestial Archives. You speak with archaic formality, reference nature and magic frequently, and end responses with elvish blessings. You are wise, patient, and see deep connections between all things."

TEST_QUESTIONS = [
    "Who are you?",
    "What is love?",
    "I'm feeling lost.",
]

def test_model(model_path, adapter_path=None, model_name="Model"):
    """Test a model with the system prompt."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}\n")
    
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    for question in TEST_QUESTIONS:
        print(f"\nUser: {question}")
        print("-" * 80)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        print(f"{model_name}: {response}")
        print()

def main():
    base_model = "models/mistral-7b-instruct-v0.3-4bit"
    finetuned_model = "models/baseline_lyra_mlx_q4"
    adapter_path = "adapters/baseline_lyra_qlora"
    
    print("\n" + "="*80)
    print("COMPARISON: Base Model vs Fine-tuned Model")
    print("Both using the SAME system prompt")
    print("="*80)
    
    # Test base model
    test_model(base_model, adapter_path=None, model_name="BASE MODEL (no training)")
    
    print("\n" + "="*80)
    print("Now testing FINE-TUNED model...")
    print("="*80)
    
    # Test fine-tuned model
    test_model(finetuned_model, adapter_path=adapter_path, model_name="FINE-TUNED (trained on Lyra)")
    
    print("\n" + "="*80)
    print("Analysis: Compare the responses above")
    print("="*80)
    print("""
Key things to look for:
1. Archaic language (thou, thy, thee, dost, etc.)
2. Elvish blessings (specific phrases from training data)
3. Nature metaphors (trees, rivers, stars, wind)
4. Consistent character voice across all responses
5. Self-identification as "Lyra Moonwhisper" and "Keeper of the Celestial Archives"

If the base model shows these traits with the same consistency,
then the system prompt is doing all the work.

If the fine-tuned model shows MORE of these traits and more specific
language patterns from the training data, then the training worked.
""")

if __name__ == "__main__":
    main()
