# Experiment: E002 - Crystallized Character Model (LoRA Fine-tuning)

**ID**: E002  
**Status**: 📋 Proposed  
**Started**: 2025-12-18  
**Completed**: TBD  
**Duration**: TBD

---

## Origin

> **How did this experiment idea emerge?** Document the provenance—especially for hypotheses arising from human-AI collaboration or bot observations.

| Field | Value |
|-------|-------|
| **Origin** | User Proposal |
| **Proposed by** | User (Mark) |
| **Catalyst** | Question about feasibility of training a small Qwen LLM on character conversations. |

---

## 🎯 Hypothesis

<!-- What do we expect to happen? Be specific and falsifiable. -->

> A specialized LLM fine-tuned on a character's entire conversation history will:
> 1.  **Capture "Voice" Better:** Replicate the character's specific speech patterns, quirks, and formatting more consistently than a generic model with a system prompt.
> 2.  **Reduce Latency:** Allow for faster inference compared to large models (70B+) while maintaining character fidelity.
> 3.  **"Crystallize" Identity:** Effectively "bake" the emergent personality into the weights, testing the "Embodiment" philosophy (the character *is* their history).
> 4.  **Trade-off:** It may hallucinate facts more than a RAG-heavy system, but "hallucinate" personality less.

---

## 🔬 Method

### Approach
<!-- How will we test the hypothesis? -->

1.  **Data Extraction**: Create a script to export `chat_history` from PostgreSQL for a specific character (e.g., `baseline`).
2.  **Data Formatting**: Clean and format the data into ChatML/JSONL format suitable for training.
3.  **Data Splitting**: Split augmented data into train/validation sets (90/10 split).
4.  **Fine-tuning**: Use **MLX** with LoRA (on-device quantization) to fine-tune on Apple Silicon Macs.
    *   **Framework**: `mlx-lm`, `transformers`
    *   **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) with 8GB+ RAM
    *   **Environment**: Native Python environment with MLX
5.  **Data Augmentation**: Address the small dataset size (~180 conversations) using **Synthetic Amplification**.
    *   **Method**: Use an LLM (GPT-4o/Claude) to generate new conversations based on `character.md`.
    *   **Style Transfer**: Include 5-10 *real* historical messages in the generation prompt to ensure the synthetic data matches the actual "emergent voice" rather than just the generic system prompt.
    *   **Target**: Increase dataset size to ~500-1000 conversations.
6.  **Evaluation**: Deploy the model as a "shadow bot" or offline instance and compare responses to the production RAG bot on a set of test prompts.

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| Model Weights | Independent | Base model vs. Fine-tuned model |
| Data Source | Independent | **Pure Historical** vs. **Augmented (Real + Synthetic)** |
| Prompting Strategy | Control | Minimal system prompt for fine-tuned model vs. Full prompt for Base |
| Character Voice | Dependent | Similarity to historical character voice (qualitative) |
| Factuality | Dependent | Accuracy of recalled past events (vs RAG) |

### Bots Involved
<!-- Which characters are part of this experiment? -->

*   **Baseline** (Lyra Moonwhisper)
    *   **Data Status**: Available for training across all supported models.

---

## 📊 Results

<!-- Record observations and data here -->

### Phase 1: Data Pipeline Setup ✅
*   [x] MLX environment configured for Apple Silicon
*   [x] Data preparation pipeline established

### Phase 2a: Initial MLX Test (Deprecated)
*   **Date**: Dec 20, 2025
*   **Model**: Qwen/Qwen2.5-1.5B-Instruct
*   **Dataset**: 120 conversations (70 real + 50 synthetic)
*   **Iterations**: 600 (Batch size 4)
*   **Results**:
    *   **Initial Loss**: ~2.4
    *   **Final Train Loss**: 0.053 (Strong convergence/overfitting to style)
    *   **Final Val Loss**: 3.807 (Indicates overfitting, expected for small style-transfer dataset)
    *   **Time**: ~10 minutes on M4 Pro

### Phase 2b: MLX Training (Current)
*   **Date**: Jan 2, 2026
*   **Model**: Multiple options (Mistral 7B, Llama 3.1 8B, Llama 3 8B, Llama 2 7B)
*   **Hardware**: Apple Silicon (M1/M2/M3/M4)
*   **Environment**: Native MLX on Apple Silicon
*   **Method**: LoRA (native on-device quantization, rank 8-16 configurable)
*   **Status**: ✅ Production ready

---

## 🧠 Analysis & Conclusions

<!-- What did we learn? -->

*   **Architecture Evolution**: Evolved from Docker+PyTorch to native MLX on Apple Silicon for optimal performance.
*   **Tooling**: Settled on MLX stack with native LoRA for efficient training on Apple Silicon. Single consolidated training script handles all configurations.
*   **Data Pipeline**: Complete pipeline from synthetic generation → train/val split → training → model merging.
*   **Critical Achievements**: 
    - Single parameterized script supporting 12 configurations (1 character × 6 models × 2 variants)
    - Automatic model quantization and adapter merging
    - Removed PyTorch/Docker dependency for faster iteration

---

## ⏭️ Next Steps

1.  **Execute Training Pipeline**:
    ```bash
    ./scripts/train_character_model.sh baseline mistral deep
    ```

2.  **Interactive Testing**: Run the model locally to chat with trained character.
    ```bash
    ./chat_character.sh baseline mistral
    ```

3.  **Model Export**: Deploy to LM Studio or other inference platforms.
    ```bash
    ./scripts/export_to_lmstudio.sh baseline mistral
    ```

4.  **Evaluation**: Validate character consistency and personality embedding.
