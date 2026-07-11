#!/usr/bin/env python3
"""
Quantitative evaluation for trained character models.

For each adapter on disk, runs the same fixed prompt set through the model and
reports metrics that proxy for "voice crystallization":
  - response length (chars, words)
  - key-phrase coverage: how often the response contains character-defining
    phrases (Celestial Archives, Lyra/Moonwhisper, elvish-blessing closers)
  - archaic-marker density: lowercase matches for archaic English ("thou",
    "doth", "may the", "thy", etc.)

Output is a comparison table that makes standard-vs-deep and model-vs-model
diffs obvious.

Usage:
    python scripts/evaluate_character.py [character] [--prompts PATH] [--max-tokens N]

Defaults: character=baseline, prompts=scripts/eval_prompts_baseline.jsonl,
max-tokens=200.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# Allow `python scripts/evaluate_character.py` to find model_config regardless
# of CWD by adding scripts/ to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import get_model_config as _get_model_config_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults — overridable via flags
# ---------------------------------------------------------------------------

DEFAULT_CHARACTER = "baseline"
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "eval_prompts_baseline.jsonl"

# Phrases that should appear in a crystallized Lyra response at least some of
# the time. Each is a list of lowercased substrings; any hit counts.
CHARACTER_PHRASES = {
    "baseline": {
        "name": ["lyra", "moonwhisper", "keeper of the celestial archives"],
        "setting": ["celestial archives", "archives", "elven", "ancient"],
        "blessing": [  # elvish/Sindarin-style closers seen in training data
            "vanya sulie", "aiya elenion", "namarie", "lasto beth",
            "anar caluva", "elenion", "may the stars", "may starlight",
            "may the wind", "may peace",
        ],
        "archaic": [  # lowercase archaic English
            "thou", "thy", "doth", "thee", "hath", "may the", "doth thy",
        ],
    },
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(path: Path, character: str) -> list[str]:
    """Load prompts from a JSONL file. Each line: {"prompt": "...", "messages": [...]}.
    If the file doesn't exist, fall back to a tiny built-in set."""
    if not path.exists():
        return _builtin_prompts(character)
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" in obj:
                prompts.append(obj["prompt"])
            elif "messages" in obj:
                # Use the last user turn as the prompt
                for m in reversed(obj["messages"]):
                    if m.get("role") == "user":
                        prompts.append(m["content"])
                        break
    return prompts


def _builtin_prompts(character: str) -> list[str]:
    if character == "baseline":
        return [
            "Who are you?",
            "What is your name and role?",
            "How do you greet people?",
            "Tell me about your wisdom.",
            "What matters most to you?",
            "What does your typical day look like?",
            "How do you deal with grief?",
            "Do you believe in destiny or free will?",
        ]
    return ["Who are you?"]


# ---------------------------------------------------------------------------
# Adapter discovery + model resolution
# ---------------------------------------------------------------------------

def find_adapters(character: str) -> list[Path]:
    return sorted(Path("adapters").glob(f"{character}_*_qlora*"))


def resolve_chat_model(model_name: str) -> str:
    """Same priority as chat_character.sh: llama2→bf16, else local quantized,
    else HF repo. Returns the path/repo string mlx_lm expects."""
    cfg = _get_model_config_dict(model_name) or {}
    quantized = cfg.get("quantized")
    hf = cfg.get("hf")
    if model_name == "llama2_7b" and Path("models/llama-2-7b-chat-bf16").exists():
        return "models/llama-2-7b-chat-bf16"
    if quantized and Path(quantized).exists():
        return quantized
    return hf or ""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(chat_model: str, adapter_dir: Path, prompt: str, max_tokens: int, timeout: int = 60) -> str:
    """Run mlx_lm.generate and return the response text."""
    cmd = [
        "python", "-m", "mlx_lm", "generate",
        "--model", chat_model,
        "--adapter-path", str(adapter_dir),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.7",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    if result.returncode != 0:
        return ""
    return _extract_response(result.stdout)


def _extract_response(stdout: str) -> str:
    """mlx_lm.generate prints "==========" separators and the prompt. Take only
    the last non-separator, non-empty line as the response."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ln.startswith("=====") or ln.startswith("Prompt:"):
            continue
        out.append(ln)
    # The response is whatever comes after the prompt line. mlx_lm prints the
    # prompt first (as a header) and the response after. Heuristic: last block
    # of lines after the last separator.
    return "\n".join(out[-10:]).strip() if out else ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _phrase_hits(text: str, phrases: list[str]) -> tuple[int, list[str]]:
    """Return (count, [matched_phrases]) for substrings that appear (lowercase)."""
    lo = text.lower()
    matched = [p for p in phrases if p in lo]
    return len(matched), matched


def score_response(text: str, character: str) -> dict:
    """Compute crystallization metrics for one response."""
    if not text:
        return {"empty": True}
    bucket = CHARACTER_PHRASES.get(character, {})
    n_words = len(text.split())
    n_chars = len(text)
    metrics = {
        "empty": False,
        "n_chars": n_chars,
        "n_words": n_words,
    }
    for category, phrases in bucket.items():
        hits, matched = _phrase_hits(text, phrases)
        metrics[f"{category}_hits"] = hits
        metrics[f"{category}_matched"] = matched
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_table(character: str, results: list[dict]):
    """Print a Markdown-ish comparison table to stdout."""
    if not results:
        print("No results to report.")
        return

    headers = [
        "adapter", "prompts",
        "avg_words", "name_hit_rate", "setting_hit_rate",
        "blessing_hit_rate", "archaic_hit_rate",
    ]
    print()
    print(f"## Crystallization metrics — {character}")
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results:
        n = r["n_prompts"]
        row = [
            r["adapter"],
            str(n),
            f"{_avg(r['words']):.1f}",
            f"{sum(1 for w in r['name_hits'] if w) / max(n, 1):.0%}",
            f"{sum(1 for w in r['setting_hits'] if w) / max(n, 1):.0%}",
            f"{sum(1 for w in r['blessing_hits'] if w) / max(n, 1):.0%}",
            f"{sum(1 for w in r['archaic_hits'] if w) / max(n, 1):.0%}",
        ]
        print("| " + " | ".join(row) + " |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("character", nargs="?", default=DEFAULT_CHARACTER)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH,
                        help="JSONL file with prompts ({prompt: str} or {messages: [...]}).")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output", type=Path, help="Optional path to dump per-prompt results as JSONL.")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts, args.character)
    if not prompts:
        print(f"No prompts found at {args.prompts} and no built-in prompts for character '{args.character}'.")
        sys.exit(1)

    adapters = find_adapters(args.character)
    if not adapters:
        print(f"No trained adapters found for '{args.character}'.")
        sys.exit(1)

    print(f"Found {len(adapters)} adapter(s), {len(prompts)} prompt(s) for '{args.character}'.")
    print()

    out_f = open(args.output, "w") if args.output else None
    summary = []

    try:
        for adapter_dir in adapters:
            adapter_name = adapter_dir.name
            # adapter dir naming: baseline_<model>_qlora[_deep]
            stripped = adapter_name.replace(f"{args.character}_", "").replace("_qlora", "")
            model_name = stripped  # may include "_deep" suffix from path
            # Strip _deep from model name lookup (configs don't have a _deep entry).
            lookup_name = model_name.replace("_deep", "")
            chat_model = resolve_chat_model(lookup_name)

            if not chat_model:
                print(f"⚠️  {adapter_name}: no model resolution — skipping")
                continue

            print(f"▶ {adapter_name}")
            print(f"  model: {chat_model}")
            print(f"  adapter: {adapter_dir}")

            per_prompt = []
            words, name_h, setting_h, blessing_h, archaic_h = [], [], [], [], []
            for i, p in enumerate(prompts, 1):
                response = generate(chat_model, adapter_dir, p, args.max_tokens)
                m = score_response(response, args.character)
                per_prompt.append({"prompt": p, "response": response, "metrics": m})
                if not m.get("empty"):
                    words.append(m["n_words"])
                    name_h.append(m.get("name_hits", 0))
                    setting_h.append(m.get("setting_hits", 0))
                    blessing_h.append(m.get("blessing_hits", 0))
                    archaic_h.append(m.get("archaic_hits", 0))
                preview = (response[:60] + "…") if len(response) > 60 else response
                print(f"  [{i}/{len(prompts)}] {p[:40]:<40} → {preview!r}")

            summary.append({
                "adapter": adapter_name,
                "model": lookup_name,
                "n_prompts": len(prompts),
                "words": words,
                "name_hits": name_h,
                "setting_hits": setting_h,
                "blessing_hits": blessing_h,
                "archaic_hits": archaic_h,
                "per_prompt": per_prompt,
            })
            if out_f:
                for pp in per_prompt:
                    out_f.write(json.dumps({"adapter": adapter_name, **pp}) + "\n")
            print()
    finally:
        if out_f:
            out_f.close()
            print(f"Per-prompt results → {args.output}")

    print_table(args.character, summary)


if __name__ == "__main__":
    main()