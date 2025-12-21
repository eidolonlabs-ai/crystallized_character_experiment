#!/usr/bin/env python3
"""Shared Python model and character configuration — single source of truth."""

MODEL_CONFIGS = {
    # Mistral family
    "mistral_v0_3": {
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
        "quantized": "models/mistral-7b-instruct-v0.3-4bit",
    },
    "mistral": {
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
        "quantized": "models/mistral-7b-instruct-v0.3-4bit",
    },
    "mistral_v0_2": {
        "hf": "mistralai/Mistral-7B-Instruct-v0.2",
        "quantized": "models/mistral-7b-instruct-v0.2-4bit",
    },
    "mistral_v0_1": {
        "hf": "mistralai/Mistral-7B-Instruct-v0.1",
        "quantized": "models/mistral-7b-instruct-v0.1-4bit",
    },
    # Llama family
    "llama": {
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "quantized": "models/llama-3.1-8b-instruct-4bit",
    },
    "llama31_8b": {
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "quantized": "models/llama-3.1-8b-instruct-4bit",
    },
    "llama3_8b": {
        "hf": "meta-llama/Meta-Llama-3-8B-Instruct",
        "quantized": "models/llama-3-8b-instruct-4bit",
    },
    "llama2_7b": {
        "hf": "meta-llama/Llama-2-7b-chat-hf",
        "quantized": "models/llama-2-7b-chat-4bit",
    },
}

CHARACTER_CONFIGS = {
    "baseline": {
        "system_prompt": (
            "You are Lyra Moonwhisper, an ancient elven sage who dwells in the Celestial Archives. "
            "You speak with archaic formality, reference nature and magic frequently, "
            "and end responses with elvish blessings."
        ),
        "description": (
            "Lyra Moonwhisper is an ancient elven sage who dwells in the Celestial Archives. "
            "She speaks with archaic formality, uses nature and magic metaphors, "
            "and ends her responses with elvish blessings."
        ),
    },
}

VALID_MODELS = list(MODEL_CONFIGS.keys())
VALID_CHARACTERS = list(CHARACTER_CONFIGS.keys())


def get_model_config(model_name):
    return MODEL_CONFIGS.get(model_name)


def get_hf_model(model_name):
    cfg = MODEL_CONFIGS.get(model_name, {})
    return cfg.get("hf")


def get_quantized_model(model_name):
    cfg = MODEL_CONFIGS.get(model_name, {})
    return cfg.get("quantized")


def get_system_prompt(character):
    cfg = CHARACTER_CONFIGS.get(character, {})
    return cfg.get("system_prompt", "You are a helpful assistant.")


def get_character_description(character):
    cfg = CHARACTER_CONFIGS.get(character, {})
    return cfg.get("description", "A helpful assistant.")
