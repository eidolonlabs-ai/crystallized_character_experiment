#!/usr/bin/env python3
"""
Generate thinking/reasoning blocks for assistant responses in training data.

Takes existing character conversation data and prepends <think>...</think>
blocks to each assistant response using an LLM (OpenRouter by default).
This produces training data suitable for reasoning models (e.g., Qwen3) that
natively emit thinking blocks.

Usage:
    python scripts/generate_thinking_data.py

Input:  raw_data/training_data_baseline_augmented_curated.jsonl
Output: raw_data/training_data_baseline_augmented_thinking.jsonl

Requires OPENROUTER_API_KEY or LLM_API_KEY in .env.
"""

import json
import os
import asyncio
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MODEL_NAME = "openai/gpt-4o"
MAX_CONCURRENT = 5
CHARACTER_NAME = "Lyra Moonwhisper"

CHARACTER_CONTEXT = (
    "Lyra Moonwhisper is an ancient elven sage who dwells in the Celestial Archives. "
    "She speaks with archaic formality, uses nature and magic metaphors, "
    "and ends her responses with elvish blessings."
)

THINKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"You are helping generate training data for an AI character named {CHARACTER_NAME}. {CHARACTER_CONTEXT} For each user message and assistant response, write a brief inner monologue (2-4 sentences) showing {CHARACTER_NAME}'s thought process before responding. Use her voice — archaic, mystical, nature-infused. Output ONLY the thinking text, no labels or formatting."),
    ("user", "User message: {user_message}\n\nAssistant response: {assistant_response}\n\nWrite Lyra's inner thoughts before this response:"),
])


def _clean_thinking(text: str) -> str:
    text = text.strip()
    for prefix in ("<think>", "Think:", "Thinking:", "Inner thoughts:", '"'):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    for suffix in ("</think>", '"'):
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
    return text


async def _generate_one(llm, user: str, assistant: str, sem: asyncio.Semaphore, idx: int) -> str | None:
    async with sem:
        try:
            chain = THINKING_PROMPT | llm | StrOutputParser()
            result = await chain.ainvoke({"user_message": user, "assistant_response": assistant})
            return _clean_thinking(result)
        except Exception as e:
            print(f"  [{idx}] error: {e}")
            return None


async def generate_thinking(input_file: str, output_file: str, model: str = MODEL_NAME, limit: int | None = None):
    with open(input_file) as f:
        rows = [json.loads(line) for line in f]

    if limit:
        rows = rows[:limit]

    examples = []
    for i, row in enumerate(rows):
        msgs = row["messages"]
        user_msg = None
        assistant_msg = None
        for m in msgs:
            if m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                assistant_msg = m["content"]
        if user_msg and assistant_msg:
            examples.append((i, user_msg, assistant_msg))

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY or LLM_API_KEY not set.")
        return

    llm = ChatOpenAI(
        model=model,
        temperature=0.7,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-Title": "Crystallized Character Thinking Gen",
            "HTTP-Referer": "https://github.com/eidolonlabs-ai/crystallized_character_experiment",
        },
    )

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [_generate_one(llm, u, a, sem, idx) for idx, u, a in examples]
    results = await asyncio.gather(*tasks)

    thinking_map = {}
    for (idx, _, _), thinking in zip(examples, results):
        if thinking:
            thinking_map[idx] = thinking

    written = 0
    with open(output_file, "w") as f:
        for i, row in enumerate(rows):
            if i in thinking_map:
                msgs = row["messages"]
                for j in range(len(msgs) - 1, -1, -1):
                    if msgs[j]["role"] == "assistant":
                        thinking = thinking_map[i]
                        msgs[j] = {
                            "role": "assistant",
                            "content": f"<think>\n{thinking}\n</think>\n\n{msgs[j]['content']}",
                        }
                        break
            f.write(json.dumps(row) + "\n")
            written += 1

    print(f"\nGenerated thinking for {len(thinking_map)}/{written} examples")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate thinking blocks for character training data")
    parser.add_argument("--input", default="raw_data/training_data_baseline_augmented_curated.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", default="raw_data/training_data_baseline_augmented_thinking.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to process (for testing)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"OpenRouter model name (default: {MODEL_NAME})")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    print(f"Generating thinking blocks for {args.input}...")
    print(f"Model: {args.model}, character: {CHARACTER_NAME}")
    asyncio.run(generate_thinking(args.input, args.output, model=args.model, limit=args.limit))


if __name__ == "__main__":
    main()
