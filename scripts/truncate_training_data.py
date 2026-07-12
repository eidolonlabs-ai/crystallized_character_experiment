#!/usr/bin/env python3
"""
Intelligently truncate/split training data to fit sequence length limits
while preserving character fidelity and complete responses.

Multi-turn support
------------------
`process_file()` accepts rows of any turn count. The truncate strategy
walks the conversation backwards from the last assistant message, keeping
as many recent turns as fit. The full system prompt is preserved as long as
it fits; otherwise the row is dropped (you can't truncate the character
definition without losing the character's voice).

Phase 0 chat-template fix
-------------------------
Every row is passed through `fold_system_prompt.fold()` before tokenization,
so the system message ends up inside the first user turn as
`[SYSTEM] ... \\n\\n [USER] ...`. This is required because Mistral v0.3 (and
most Llama-family chat tokenizers) silently drop `role=system` messages
when `apply_chat_template()` is called. See `docs/CHAT_TEMPLATES.md` for the
full explanation.
"""
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import shutil

# Same directory as this file → import fold without packaging the repo.
import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from fold_system_prompt import fold as _fold_system

def count_tokens(text, tokenizer):
    """Count tokens in text including chat template formatting"""
    return len(tokenizer.encode(text))

def fix_message_roles(messages):
    """
    Fix messages to ensure roles alternate properly.
    Merges consecutive messages from the same role.
    Ensures the sequence starts with User (after optional System).
    """
    if not messages:
        return messages
    
    fixed = []
    
    # Handle system message
    start_idx = 0
    if messages[0]["role"] == "system":
        fixed.append(messages[0])
        start_idx = 1
    
    # Collect remaining messages
    remaining = messages[start_idx:]
    if not remaining:
        return fixed

    # 1. Merge consecutive messages first
    merged = []
    for msg in remaining:
        if not merged:
            merged.append(msg.copy())
        else:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg.copy())
    
    # 2. Ensure first message is User
    # If the first merged message is Assistant, we must drop it for valid training data
    if merged and merged[0]["role"] == "assistant":
        merged.pop(0)
        
    # 3. Ensure we still have messages
    if not merged:
        return fixed # Just system message or empty
        
    fixed.extend(merged)
    return fixed

def format_messages_as_text(messages, tokenizer):
    """
    Format messages using chat template to count tokens accurately.
    Does NOT fix messages - assumes they are already fixed.
    Raises exception if template fails.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        # Re-raise the exception instead of falling back
        raise ValueError(f"Failed to apply chat template: {e}") from e

def truncate_conversation(messages, max_length, tokenizer, safety_margin=50):
    """
    Intelligently truncate a conversation to fit within max_length tokens.
    Strategy:
    - Always keep system message (character definition)
    - Keep as many recent turns as possible
    - Ensure the final assistant response is complete
    - Apply safety margin to account for tokenization edge cases
    """
    if not messages:
        return messages
    
    # Apply safety margin to be conservative
    effective_max = max_length - safety_margin
    
    # Separate system message and conversation turns
    system_msg = None
    conversation = messages
    
    if messages[0]["role"] == "system":
        system_msg = messages[0]
        conversation = messages[1:]
    
    # If conversation fits, return as-is
    full_text = format_messages_as_text(messages, tokenizer)
    if count_tokens(full_text, tokenizer) <= effective_max:
        return messages
    
    # Build result starting with system message
    result = [system_msg] if system_msg else []
    
    # Work backwards from the end to keep recent context
    # Always include the last assistant response
    turns_to_include = []
    
    for i in range(len(conversation) - 1, -1, -1):
        turns_to_include.insert(0, conversation[i])
        
        # Check if this fits - fix roles before testing
        test_messages = result + turns_to_include
        test_messages_fixed = fix_message_roles(test_messages)
        
        try:
            test_text = format_messages_as_text(test_messages_fixed, tokenizer)
            test_tokens = count_tokens(test_text, tokenizer)
        except ValueError:
            # If template fails even after fixing, this combination is invalid
            # Remove what we just added and stop
            turns_to_include.pop(0)
            break
        
        if test_tokens > effective_max:
            # Too long, remove the oldest turn we just added
            turns_to_include.pop(0)
            break
    
    result.extend(turns_to_include)
    
    # Ensure result starts with User (after System)
    # If we kept an Assistant message but not the preceding User message, drop it
    start_idx = 1 if system_msg else 0
    if len(result) > start_idx and result[start_idx]["role"] == "assistant":
        result.pop(start_idx)
    
    # Ensure we have at least user + assistant pair
    if len(result) < 2 or result[-1]["role"] != "assistant":
        # Keep last user-assistant pair at minimum
        if system_msg:
            result = [system_msg, conversation[-2], conversation[-1]]
        else:
            result = [conversation[-2], conversation[-1]]
    
    # Fix any non-alternating roles before formatting
    result = fix_message_roles(result)
    
    # Final check: if still too long, truncate the assistant's response text
    final_text = format_messages_as_text(result, tokenizer)
    final_tokens = count_tokens(final_text, tokenizer)
    
    if final_tokens > effective_max and len(result) >= 2:
        # Find the assistant message and truncate its content
        for i in range(len(result) - 1, -1, -1):
            if result[i]["role"] == "assistant":
                # Truncate this message's content
                original_content = result[i]["content"]
                words = original_content.split()
                
                # Binary search to find the right length
                left, right = 0, len(words)
                best_length = 0
                
                while left <= right:
                    mid = (left + right) // 2
                    test_result = result.copy()
                    test_result[i] = {"role": "assistant", "content": " ".join(words[:mid])}
                    test_text = format_messages_as_text(test_result, tokenizer)
                    test_tokens = count_tokens(test_text, tokenizer)
                    
                    if test_tokens <= effective_max:
                        best_length = mid
                        left = mid + 1
                    else:
                        right = mid - 1
                
                if best_length > 0:
                    result[i] = {"role": "assistant", "content": " ".join(words[:best_length])}
                break
    
    return result

def validate_conversation(messages, tokenizer=None):
    """
    Validate conversation structure.
    Returns (is_valid, reason)
    Assumes messages have ALREADY been through fix_message_roles.
    """
    if not messages:
        return False, "Empty messages list"
    
    # Check for empty content
    for msg in messages:
        if not msg.get("content", "").strip():
            return False, "Empty content in message"
            
    # Check for at least one user message
    has_user = any(msg["role"] == "user" for msg in messages)
    if not has_user:
        return False, "No user message found"
    
    # Check strict alternation (messages should already be fixed)
    last_role = None
    for msg in messages:
        if msg["role"] == last_role:
             return False, f"Consecutive {msg['role']} messages found"
        last_role = msg["role"]
        
    # Check first non-system is user
    start_idx = 0
    if messages[0]["role"] == "system":
        start_idx = 1
    
    if start_idx < len(messages) and messages[start_idx]["role"] != "user":
        return False, f"First non-system message is {messages[start_idx]['role']}"

    has_assistant = any(msg["role"] == "assistant" for msg in messages)
    if not has_assistant:
        return False, "No assistant message found"
    
    # CRITICAL: Actually test if the chat template will accept this
    if tokenizer:
        try:
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            return False, f"Chat template rejected: {str(e)}"
        
    return True, ""

def process_file(input_file, output_file, max_length, tokenizer):
    """Process a single training file and truncate long sequences"""
    print(f"\nProcessing {input_file.name}...")
    
    with open(input_file) as f:
        data = [json.loads(line) for line in f]
    
    truncated_count = 0
    skipped_count = 0
    invalid_count = 0
    processed_data = []
    
    for item in data:
        if "messages" not in item:
            processed_data.append(item)
            continue

        # Phase 0: fold the system message into the first user turn so the
        # character definition survives tokenizers that drop `role=system`
        # (Mistral v0.3, most Llama families). Idempotent.
        item = _fold_system(item)
        messages = item["messages"]
        
        # Fix roles FIRST to ensure we are working with clean data
        messages = fix_message_roles(messages)
        if not messages:
            invalid_count += 1
            continue
        
        # Validate after fixing (with tokenizer for template check)
        is_valid, reason = validate_conversation(messages, tokenizer)
        if not is_valid:
            # print(f"  ⚠️  Skipped invalid: {reason}")
            invalid_count += 1
            continue

        # Check if truncation needed
        try:
            full_text = format_messages_as_text(messages, tokenizer)
            original_tokens = count_tokens(full_text, tokenizer)
        except ValueError as e:
            # Template failed even after fixing - skip this conversation
            print(f"  ⚠️  Skipped: {e}")
            invalid_count += 1
            continue
        
        if original_tokens > max_length:
            truncated_messages = truncate_conversation(messages, max_length, tokenizer)
            
            # Fix roles again after truncation (truncation might introduce issues)
            truncated_messages = fix_message_roles(truncated_messages)
            
            # Validate after truncation and fixing (with tokenizer for template check)
            is_valid, reason = validate_conversation(truncated_messages, tokenizer)
            if not is_valid:
                print(f"  ⚠️  Skipped after truncation: {reason}")
                skipped_count += 1
                continue

            try:
                new_text = format_messages_as_text(truncated_messages, tokenizer)
                new_tokens = count_tokens(new_text, tokenizer)
            except ValueError as e:
                # Template failed after truncation - skip
                print(f"  ⚠️  Skipped after truncation: {e}")
                skipped_count += 1
                continue
                
            # Skip if still too long (e.g., single very long user message)
            if new_tokens > max_length:
                print(f"  ⚠️  Skipped: {original_tokens} tokens (cannot truncate further without losing prompt)")
                skipped_count += 1
                continue
            
            
            print(f"  Truncated: {original_tokens} → {new_tokens} tokens ({len(item['messages'])} → {len(truncated_messages)} messages)")
            truncated_count += 1
            
            processed_data.append({"messages": truncated_messages})
        else:
            # Use the fixed messages, not the original item
            processed_data.append({"messages": messages})
    
    with open(output_file, 'w') as f:
        for item in processed_data:
            # Data is already fixed and validated, write directly
            f.write(json.dumps(item) + '\n')
    
    print(f"  {len(data)} examples: {truncated_count} truncated, {skipped_count} skipped, {invalid_count} invalid, {len(processed_data)} kept")
    return len(processed_data), truncated_count

def process_directory(input_dir, output_dir, max_length, model_name, in_place=False):
    """Process a directory containing train.jsonl and valid.jsonl"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Setup output directory
    if in_place:
        output_path = input_path
        print(f"Processing in-place: {input_dir}")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Processing {input_dir} → {output_dir}")
    
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from HuggingFace: {e}")
        print("Trying to use locally cached tokenizer...")
        # Try to load from MLX cache
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)
    
    total_examples = 0
    total_truncated = 0
    
    # Process train.jsonl
    train_file = input_path / "train.jsonl"
    if train_file.exists():
        output_train = output_path / "train.jsonl"
        examples, truncated = process_file(train_file, output_train, max_length, tokenizer)
        total_examples += examples
        total_truncated += truncated
    else:
        print(f"Warning: {train_file} not found")
    
    # Process valid.jsonl
    valid_file = input_path / "valid.jsonl"
    if valid_file.exists():
        output_valid = output_path / "valid.jsonl"
        examples, truncated = process_file(valid_file, output_valid, max_length, tokenizer)
        total_examples += examples
        total_truncated += truncated
    else:
        print(f"Warning: {valid_file} not found")
    
    print(f"\n✓ Complete:")
    print(f"  Total examples: {total_examples}")
    print(f"  Truncated: {total_truncated}")
    print(f"  Unchanged: {total_examples - total_truncated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate training data directory to fit sequence length")
    parser.add_argument("--input-dir", required=True, help="Input directory with train.jsonl and valid.jsonl")
    parser.add_argument("--output-dir", help="Output directory (default: input_dir with '_truncated' suffix)")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length in tokens")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name for tokenizer")
    parser.add_argument("--in-place", action="store_true", help="Modify files in-place (overwrites originals)")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.in_place:
        output_dir = args.input_dir
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        # Auto-generate output directory name
        output_dir = args.input_dir.rstrip('/') + '_truncated'
    
    process_directory(args.input_dir, output_dir, args.max_length, args.model, args.in_place)
