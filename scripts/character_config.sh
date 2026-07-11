#!/bin/bash
# Character configuration — system prompts and per-character metadata.
# Source this from chat_character.sh and any other script that needs the
# character system prompts.
#
# Placeholder restored after an in-session reset. The original file held the
# canonical system prompts as bash heredocs; this version mirrors them and
# stays in sync with scripts/model_config.py::CHARACTER_CONFIGS.

get_system_prompt() {
    local char="$1"
    case "$char" in
        baseline)
            echo "You are Lyra Moonwhisper, an ancient elven sage who dwells in the Celestial Archives. You speak with archaic formality, reference nature and magic frequently, and end responses with elvish blessings. You are wise, patient, and see deep connections between all things."
            ;;
        *)
            echo "You are a helpful assistant."
            ;;
    esac
}

get_character_description() {
    local char="$1"
    case "$char" in
        baseline) echo "Lyra Moonwhisper — ancient elven sage of the Celestial Archives." ;;
        *)        echo "A helpful assistant." ;;
    esac
}

VALID_CHARACTERS="baseline"
