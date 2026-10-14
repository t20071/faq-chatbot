"""Shared helpers for the evaluation scripts. Run every script from chatbot/ as a module:

    python -m evaluation.generate_conversations
"""

import json
import os
import sys
from pathlib import Path

from faqbot.llm import chat_completion

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"
CONVERSATIONS_PATH = DATASETS_DIR / "conversations.jsonl"
HANDWRITTEN_PATH = DATASETS_DIR / "handwritten.jsonl"

# A bigger model than the bot uses, so the judge is stronger than the judged.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

# LLM output often contains characters the Windows console codepage can't print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_conversations(include_handwritten: bool = True) -> list[dict]:
    convs = read_jsonl(CONVERSATIONS_PATH) if CONVERSATIONS_PATH.exists() else []
    if include_handwritten and HANDWRITTEN_PATH.exists():
        convs += read_jsonl(HANDWRITTEN_PATH)
    if not convs:
        sys.exit("No conversations found. Run: python -m evaluation.generate_conversations")
    return convs


def llm_json(client, model: str, system: str, user: str) -> dict:
    """Call the LLM in JSON mode and parse the result."""
    text = chat_completion(
        client,
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(text)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
