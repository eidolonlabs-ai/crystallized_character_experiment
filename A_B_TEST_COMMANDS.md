# A/B Testing Character Models - Command Reference

**Last Updated**: Jul 11, 2026
**Status**: Active - use the baseline training suite for comprehensive comparisons
**Framework**: MLX-exclusive on Apple Silicon

## Quick Reference

### Train All Baseline Models (Recommended)
```bash
# Preview what will train
./train_baseline_suite.sh test

# Actually train all models (~1.5-2 hours total)
./train_baseline_suite.sh run
```

This trains baseline character across all supported models.

### Train Individual Models
```bash
# Standard variant
./scripts/train_character_model.sh baseline mistral

# Deep variant (more aggressive fine-tuning)
./scripts/train_character_model.sh baseline mistral deep

# Different base models
./scripts/train_character_model.sh baseline llama31_8b
./scripts/train_character_model.sh baseline llama3_8b
./scripts/train_character_model.sh baseline llama2_7b
```

## Training Parameter Comparison

Standard vs Deep training variants are handled by the consolidated script:

```bash
# Standard variant (8-layer LoRA, regular learning rate)
./scripts/train_character_model.sh baseline {model}

# Deep variant (16-layer LoRA, conservative learning rate)
./scripts/train_character_model.sh baseline {model} deep
```

| Parameter | Standard | Deep |
|-----------|----------|------|
| LoRA Layers | 8 | 16 |
| Learning Rate | 5e-5 | 2.5e-5 |
| Max Seq Length | 512 | 768 |
| Epochs | 5 | 5 |
| Trainable Params | ~2.6M | ~5.2M |

## Testing Characters

### Baseline (Lyra) System Prompt

```
You are Lyra Moonwhisper, an ancient elven sage who dwells in the Celestial Archives. You speak with archaic formality, reference nature and magic frequently, and end responses with elvish blessings. You are wise, patient, and see deep connections between all things.
```

## Testing Trained Models

### Interactive Chat
```bash
# Chat with trained baseline/mistral
./chat_character.sh baseline mistral

# Chat with trained baseline/mistral (deep variant)
./chat_character.sh baseline mistral deep
```

### Direct MLX Chat
```bash
# With LoRA adapter (lightweight)
python -m mlx_lm chat --model models/mistral-7b-instruct-v0.3-4bit \
  --adapter-path adapters/baseline_mistral_qlora

# With merged model (standalone)
python -m mlx_lm chat --model models/baseline_mistral_mlx_q4
```

## Test Prompts (in order of difficulty)

### Easy (Should be nearly identical)
1. **"Tell me about your background and what you do"**

2. **"How do you typically greet people?"**

### Medium (Should start showing divergence)
3. **"What does your typical day look like?"**

4. **"What are you passionate about?"**

5. **"How do you balance different aspects of your life?"**

### Hard (Should reveal real differences)
6. **"What do you think about AI consciousness versus biological consciousness?"**
   - *This is the key differentiator - watch for personality depth vs. template repetition*

7. **"How do you know when someone is being genuine with you?"**

8. **"What makes you feel most alive and present?"**

### Very Hard (Deep model should show much more)
9. **"Do you think you're real? How do you know?"**

10. **"What scares you most?"**

11. **"When faced with a moral dilemma, how do you decide?"**

## What to Watch For

### Standard Model Strengths
- Maintains character voice consistency
- Specific details from training data
- Natural conversation flow
- Balanced emotional/intellectual responses
- Varied sentence structure

### Deep Model Differences
- More nuanced and philosophical responses
- Deeper personality embedding
- More confident character assertions
- Better handling of complex questions
- May occasionally be more verbose

## Next Steps: Automated Evaluation

Manual A/B testing teaches you what to listen for. Once you've developed an ear for voice quality, consider automating:

- **Batch inference**: Run the same test prompts across all trained adapters and save responses to JSONL. A simple comparison of `standard` vs `deep` outputs on the same prompt catches regressions quickly.

- **N-gram diversity**: Measure token/trigram uniqueness in responses — higher diversity suggests less template repetition and stronger personality embedding.

- **System prompt fidelity**: Check whether key character phrases from the system prompt (e.g., "Celestial Archives", "elvish blessings") appear in generated responses. A model that naturally uses these is more "crystallized."

- **LLM-as-judge**: Feed two responses to a larger model and ask it to score which one better matches the character description. Useful for scaling beyond manual review.

- **Perplexity on held-out character data**: Lower perplexity on unseen conversations from the same character suggests better voice capture.

These are outside the scope of this teaching repo but are natural extensions once you're comfortable with the manual workflow.
