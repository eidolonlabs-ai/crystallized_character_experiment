#!/usr/bin/env python3
"""
Curate training data by extracting high-signal turn pairs and limiting response length.
Creates shorter, focused training examples that fit in memory without truncation.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

def curate_conversations(input_file: str, output_file: str, 
                        max_response_tokens: int = 600,
                        keep_start_tokens: int = 300,
                        keep_end_tokens: int = 300,
                        tokenizer_name: str = "models/mistral-7b-instruct-v0.3-4bit"):
    """
    Extract individual conversation turns and limit response length by truncating the middle.
    
    Keeps:
    - System prompt
    - Recent user message
    - Assistant response with smart truncation:
      - Keeps first keep_start_tokens (personality setup)
      - Keeps last keep_end_tokens (conclusion)
      - Removes middle content if needed
    
    This preserves character voice (beginning) and narrative payoff (ending).
    """
    try:
        # Try local model directory first (faster, no download)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except:
        # Fall back to HuggingFace remote
        print("Loading tokenizer from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    
    examples_read = 0
    examples_kept = 0
    examples_truncated = 0
    examples_skipped = 0
    total_tokens_before = 0
    total_tokens_after = 0
    
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                examples_read += 1
                
                messages = data.get("messages", [])
                if len(messages) < 2:
                    examples_skipped += 1
                    continue
                
                # Extract system prompt
                system_msg = None
                user_msgs = []
                assistant_msgs = []
                
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        system_msg = msg
                    elif role == "user":
                        user_msgs.append(msg)
                    elif role == "assistant":
                        assistant_msgs.append(msg)
                
                # Need at least one user-assistant pair
                if not user_msgs or not assistant_msgs:
                    examples_skipped += 1
                    continue
                
                # Use most recent user message and corresponding assistant response
                # (or the last assistant response if counts don't match)
                user_msg = user_msgs[-1]
                assistant_msg = assistant_msgs[-1]
                
                # Build new conversation with truncated assistant response
                new_messages = []
                
                if system_msg:
                    new_messages.append(system_msg)
                
                new_messages.append(user_msg)
                
                # Truncate assistant response to max_response_tokens by removing middle
                assistant_content = assistant_msg.get("content", "")
                assistant_tokens = tokenizer.encode(assistant_content)
                
                if len(assistant_tokens) > max_response_tokens:
                    # Keep start and end, remove middle
                    start_tokens = assistant_tokens[:keep_start_tokens]
                    end_tokens = assistant_tokens[-keep_end_tokens:]
                    
                    # Build truncated content: start + [middle removed] + end
                    truncated_tokens = start_tokens + end_tokens
                    truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    
                    # Add ellipsis to indicate removed content
                    truncated_content = truncated_content.rstrip() + "\n\n[...]\n\n" + tokenizer.decode(end_tokens, skip_special_tokens=True)
                    
                    new_assistant_msg = {
                        "role": "assistant",
                        "content": truncated_content
                    }
                    examples_truncated += 1
                else:
                    new_assistant_msg = assistant_msg
                
                new_messages.append(new_assistant_msg)
                
                # Calculate tokens
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                new_text = tokenizer.apply_chat_template(new_messages, tokenize=False)
                
                total_tokens_before += len(tokenizer.encode(full_text))
                total_tokens_after += len(tokenizer.encode(new_text))
                
                # Write curated example
                output_data = {"messages": new_messages}
                f_out.write(json.dumps(output_data) + "\n")
                examples_kept += 1
                
            except Exception as e:
                examples_skipped += 1
                continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Curation Summary: {output_file}")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Examples read:     {examples_read}")
    print(f"Examples kept:     {examples_kept} ({examples_kept*100/max(examples_read, 1):.1f}%)")
    print(f"  - Truncated:     {examples_truncated}")
    print(f"  - Unchanged:     {examples_kept - examples_truncated}")
    print(f"Examples skipped:  {examples_skipped}")
    print(f"\nToken reduction:")
    print(f"  Before curation: {total_tokens_before:,} tokens")
    print(f"  After curation:  {total_tokens_after:,} tokens")
    if total_tokens_before > 0:
        reduction = (1 - total_tokens_after / total_tokens_before) * 100
        print(f"  Reduction:       {reduction:.1f}%")
    print(f"  Avg tokens/sample: {total_tokens_after / max(examples_kept, 1):.0f}")
    print(f"\nOutput: {output_file}")

def main():
    characters = ["baseline"]
    max_response_tokens = 600  # Total: 300 start + 300 end
    keep_start_tokens = 300    # Preserve personality setup
    keep_end_tokens = 300      # Preserve conclusion/payoff
    
    for character in characters:
        input_file = f"raw_data/training_data_{character}_augmented.jsonl"
        output_file = f"raw_data/training_data_{character}_augmented_curated.jsonl"
        
        if not Path(input_file).exists():
            print(f"⚠️  Input file not found: {input_file}")
            continue
        
        print(f"\nCurating {character.upper()} training data...")
        curate_conversations(input_file, output_file, max_response_tokens=max_response_tokens,
                           keep_start_tokens=keep_start_tokens, keep_end_tokens=keep_end_tokens)

if __name__ == "__main__":
    main()
