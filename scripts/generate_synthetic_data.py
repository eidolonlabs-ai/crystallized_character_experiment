import json
import os
import random
import asyncio
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv() 

# Configuration
NUM_CONVERSATIONS_TO_GENERATE = 50  # Start small for testing, user can increase
MODEL_NAME = "openai/gpt-4o"  # OpenRouter model name

async def generate_synthetic_data(character_name: str):
    real_data_file = f"raw_data/training_data_{character_name}.jsonl"
    synthetic_data_file = f"raw_data/training_data_{character_name}_synthetic.jsonl"
    augmented_data_file = f"raw_data/training_data_{character_name}_augmented.jsonl"
    character_file = f"characters/{character_name}/character.md"
    
    # Load specific env if exists
    load_dotenv(f".env.{character_name}")

    print(f"Loading real data from {real_data_file}...")
    try:
        with open(real_data_file, 'r', encoding='utf-8') as f:
            real_conversations = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: {real_data_file} not found. Ensure training data exists at raw_data/training_data_{character_name}.jsonl")
        return

    print(f"Loading character prompt from {character_file}...")
    try:
        with open(character_file, 'r', encoding='utf-8') as f:
            character_prompt_raw = f.read()
            # Pre-fill variables to avoid LangChain parsing errors
            character_prompt = character_prompt_raw.replace("{user_name}", "User").replace("{current_datetime}", "2025-12-20 12:00:00")
            # Escape any remaining braces just in case
            character_prompt = character_prompt.replace("{", "{{").replace("}", "}}")
    except FileNotFoundError:
        print(f"Error: {character_file} not found.")
        return

    # Initialize LLM with OpenRouter configuration
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: LLM_API_KEY or OPENROUTER_API_KEY not found in environment variables.")
        return

    llm = ChatOpenAI(
        model=MODEL_NAME, 
        temperature=0.7,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-Title": "WhisperEngine Synthetic Data Gen",
            "HTTP-Referer": "https://github.com/whisperengine-ai/whisperengine-v2"
        }
    )

    print(f"Generating {NUM_CONVERSATIONS_TO_GENERATE} synthetic conversations...")
    
    synthetic_conversations = []
    
    for i in range(NUM_CONVERSATIONS_TO_GENERATE):
        # Pick 3 random real conversations as few-shot examples (or fewer if not enough data)
        num_examples = min(3, len(real_conversations))
        examples = random.sample(real_conversations, num_examples)
        examples_text = ""
        for idx, ex in enumerate(examples):
            examples_text += f"--- Example {idx+1} ---\n"
            for msg in ex['messages']:
                examples_text += f"{msg['role']}: {msg['content']}\n"
            examples_text += "\n"

        # Create prompt for generation
        # We use f-string for the prompt content, but we need to be careful with braces for JSON
        # LangChain expects {{ and }} for literal braces in templates
        
        # Escape examples text for LangChain
        examples_text_safe = examples_text.replace("{", "{{").replace("}", "}}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert creative writer and AI trainer. Your task is to generate a realistic, multi-turn conversation between a user and the character described below. The conversation should mimic the style, tone, and length of the provided examples."),
            ("user", f"""
Character Description:
{character_prompt}

Real Conversation Examples (mimic this style):
{examples_text_safe}

Task:
Generate a new, unique conversation between a 'user' and 'assistant' ({character_name}).
- The conversation should have 3-6 turns.
- The user should ask relevant questions, discuss personal topics, or just chat casually.
- {character_name} should respond exactly as described in the profile and matching the examples.
- Output ONLY the conversation in JSON format:
{{{{
  "messages": [
    {{{{ "role": "user", "content": "..." }}}},
    {{{{ "role": "assistant", "content": "..." }}}},
    ...
  ]
}}}}
""")
        ])

        chain = prompt | llm | StrOutputParser()
        
        try:
            print(f"Generating conversation {i+1}/{NUM_CONVERSATIONS_TO_GENERATE}...")
            result = await chain.ainvoke({})
            
            # Clean up result to ensure it's valid JSON
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            conversation = json.loads(result)
            synthetic_conversations.append(conversation)
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error generating conversation {i+1}: {e}")

    # Save synthetic data
    print(f"Writing {len(synthetic_conversations)} synthetic conversations to {synthetic_data_file}...")
    with open(synthetic_data_file, 'w', encoding='utf-8') as f:
        for conv in synthetic_conversations:
            f.write(json.dumps(conv) + "\n")

    # Merge and save augmented data
    augmented_data = real_conversations + synthetic_conversations
    random.shuffle(augmented_data)
    
    print(f"Writing {len(augmented_data)} total conversations to {augmented_data_file}...")
    with open(augmented_data_file, 'w', encoding='utf-8') as f:
        for conv in augmented_data:
            f.write(json.dumps(conv) + "\n")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--character", type=str, default="baseline", help="Character name")
    args = parser.parse_args()
    
    asyncio.run(generate_synthetic_data(args.character))
