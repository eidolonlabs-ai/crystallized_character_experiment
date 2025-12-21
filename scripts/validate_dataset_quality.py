#!/usr/bin/env python3
"""
Comprehensive dataset validation script.
Checks:
1. System prompts present
2. Character consistency
3. Proper user/assistant alternation
4. Complete conversation turns
5. No empty messages
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def validate_dataset(dataset_path, character_name):
    """Validate a single dataset for quality."""
    results = {
        'total_examples': 0,
        'has_system_prompt': 0,
        'missing_system_prompt': 0,
        'proper_alternation': 0,
        'improper_alternation': 0,
        'complete_turns': 0,
        'incomplete_turns': 0,
        'has_empty_messages': 0,
        'character_consistency': 0,
        'character_inconsistency': 0,
        'issues': []
    }
    
    # Character-specific system prompts
    system_prompts = {
        'baseline': ['lyra', 'moonwhisper', 'elven', 'archives', 'celestial'],
    }
    
    try:
        with open(dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line)
                    results['total_examples'] += 1
                    
                    if 'messages' not in example:
                        results['issues'].append(f"Line {line_num}: No 'messages' field")
                        continue
                    
                    messages = example['messages']
                    
                    # Check 1: System prompt present
                    if messages and messages[0].get('role') == 'system':
                        results['has_system_prompt'] += 1
                        
                        # Check character consistency
                        system_text = messages[0].get('content', '').lower()
                        char_keywords = system_prompts.get(character_name, [])
                        if any(kw.lower() in system_text for kw in char_keywords):
                            results['character_consistency'] += 1
                        else:
                            results['character_inconsistency'] += 1
                            results['issues'].append(
                                f"Line {line_num}: System prompt doesn't mention {character_name}"
                            )
                    else:
                        results['missing_system_prompt'] += 1
                        results['issues'].append(f"Line {line_num}: Missing system prompt")
                        continue
                    
                    # Check 2: Empty messages
                    has_empty = False
                    for msg in messages:
                        content = msg.get('content', '').strip()
                        if not content:
                            has_empty = True
                            results['issues'].append(
                                f"Line {line_num}: Empty message (role: {msg.get('role')})"
                            )
                    if has_empty:
                        results['has_empty_messages'] += 1
                    
                    # Check 3: Message alternation (system, user, assistant, user, ...)
                    is_valid_alternation = True
                    expected_role = 'user'  # After system
                    
                    for idx, msg in enumerate(messages[1:], 1):  # Skip system
                        role = msg.get('role', '').lower()
                        
                        if role not in ['user', 'assistant']:
                            is_valid_alternation = False
                            results['issues'].append(
                                f"Line {line_num}: Invalid role '{role}' at position {idx}"
                            )
                            break
                        
                        if role != expected_role:
                            is_valid_alternation = False
                            results['issues'].append(
                                f"Line {line_num}: Expected '{expected_role}' at position {idx}, got '{role}'"
                            )
                            break
                        
                        # Toggle expected role
                        expected_role = 'assistant' if expected_role == 'user' else 'user'
                    
                    if is_valid_alternation:
                        results['proper_alternation'] += 1
                    else:
                        results['improper_alternation'] += 1
                    
                    # Check 4: Complete turns (should end with assistant message)
                    if len(messages) >= 3:  # At least system + user + assistant
                        if messages[-1].get('role') == 'assistant':
                            results['complete_turns'] += 1
                        else:
                            results['incomplete_turns'] += 1
                            results['issues'].append(
                                f"Line {line_num}: Incomplete turn (ends with {messages[-1].get('role')})"
                            )
                    else:
                        results['incomplete_turns'] += 1
                        results['issues'].append(f"Line {line_num}: Too few messages (only {len(messages)})")
                
                except json.JSONDecodeError as e:
                    results['issues'].append(f"Line {line_num}: JSON decode error - {e}")
    
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {dataset_path}")
        return None
    
    return results

def format_results(character, dataset_name, results):
    """Format validation results for display."""
    if results is None:
        return f"\n❌ {character}/{dataset_name}: FILE NOT FOUND\n"
    
    total = results['total_examples']
    if total == 0:
        return f"\n⚠️  {character}/{dataset_name}: EMPTY DATASET\n"
    
    # Calculate pass/fail metrics
    system_prompt_pass = results['has_system_prompt'] == total
    alternation_pass = results['proper_alternation'] == total
    complete_turns_pass = results['complete_turns'] == total
    char_consistency_pass = results['character_consistency'] == total
    no_empty_pass = results['has_empty_messages'] == 0
    
    overall_pass = (
        system_prompt_pass and 
        alternation_pass and 
        complete_turns_pass and 
        char_consistency_pass and 
        no_empty_pass
    )
    
    status = "✅" if overall_pass else "⚠️ "
    
    output = f"\n{status} {character}/{dataset_name}\n"
    output += f"   Total examples: {total}\n"
    output += f"   • System prompts: {results['has_system_prompt']}/{total} {'✓' if system_prompt_pass else '❌'}\n"
    output += f"   • Character consistency: {results['character_consistency']}/{total} {'✓' if char_consistency_pass else '❌'}\n"
    output += f"   • Proper alternation: {results['proper_alternation']}/{total} {'✓' if alternation_pass else '❌'}\n"
    output += f"   • Complete turns: {results['complete_turns']}/{total} {'✓' if complete_turns_pass else '❌'}\n"
    output += f"   • No empty messages: {total - results['has_empty_messages']}/{total} {'✓' if no_empty_pass else '❌'}\n"
    
    if results['issues']:
        output += f"   Issues ({len(results['issues'])} found):\n"
        for issue in results['issues'][:5]:  # Show first 5 issues
            output += f"     - {issue}\n"
        if len(results['issues']) > 5:
            output += f"     ... and {len(results['issues']) - 5} more\n"
    
    return output

def main():
    """Validate all character datasets."""
    base_path = Path("raw_data/prepared_data")
    
    characters = ['baseline']
    
    # Preferred dataset selection: augmented_curated > augmented > base
    dataset_variants = {
        'baseline': ['augmented_curated_split_512', 'augmented_split_512', 'base_split_512'],
    }
    
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              DATASET QUALITY VALIDATION REPORT                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    all_results = {}
    
    for character in characters:
        print(f"\n{'='*80}")
        print(f"Character: {character.upper()}")
        print('='*80)
        
        variants = dataset_variants[character]
        for variant in variants:
            dataset_path = base_path / character / variant / "train.jsonl"
            
            # Only validate if the variant exists
            if not dataset_path.exists():
                continue
            
            results = validate_dataset(str(dataset_path), character)
            all_results[f"{character}/{variant}"] = results
            
            output = format_results(character, variant, results)
            print(output)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    total_datasets = len(all_results)
    passing = 0
    
    for dataset_name, results in all_results.items():
        if results is None:
            continue
        
        total = results['total_examples']
        if total == 0:
            continue
        
        # Check if all validations pass
        if (results['has_system_prompt'] == total and
            results['proper_alternation'] == total and
            results['complete_turns'] == total and
            results['character_consistency'] == total and
            results['has_empty_messages'] == 0):
            passing += 1
    
    print(f"\nDatasets passing all checks: {passing}/{total_datasets}")
    
    if passing == total_datasets:
        print("\n✅ ALL DATASETS PASSED VALIDATION")
    else:
        print(f"\n⚠️  {total_datasets - passing} dataset(s) need review")

if __name__ == "__main__":
    main()
