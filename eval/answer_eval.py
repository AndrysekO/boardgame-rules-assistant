"""Answer eval: run every question through ask.answer() and save a CSV to grade by hand.

Run with: uv run python eval/answer_eval.py baseline
           uv run python eval/answer_eval.py agent-v1 --agent   (uses agent.agent_answer())
"""
import argparse
import csv
import json
import pathlib
import sys
import time

sys.path.insert(0, "src")
import agent
import ask
from agent import agent_answer
from ask import answer

parser = argparse.ArgumentParser(description="Run the answer eval.")
parser.add_argument("label", nargs="?", default="run")
parser.add_argument("--agent", action="store_true",
                    help="answer with agent_answer() instead of answer()")
args = parser.parse_args()
label = args.label
questions = json.loads(pathlib.Path("eval/questions.json").read_text())

rows = []
for q in questions:
    start = time.time()
    if args.agent:
        text, calls = agent_answer(q["question"], q["game"])
        # Every chunk any tool returned counts as retrieved.
        chunks = [chunk for call in calls for chunk in call["chunks"]]
        usage = agent.LAST_USAGE  # tokens summed over all of the agent's model calls
    else:
        text, chunks = answer(q["question"], q["game"])
        usage = ask.LAST_USAGE  # model + token counts from the call we just made
    pages = ", ".join(f"p.{meta.get('page')}" for _, meta in chunks)
    needles = q.get("must_contain", [])
    if isinstance(needles, str):
        needles = [needles]
    retrieved_text = " ".join(text.lower() for text, _ in chunks)
    found = all(n.lower() in retrieved_text for n in needles) if needles else ""
    row = {
        "game": q["game"],
        "question": q["question"],
        "type": q.get("type", ""),
        "expected": q.get("answer", ""),
        "model": usage["model"],
        "model_answer": text,
        "retrieved_pages": pages,
        "seconds": round(time.time() - start, 1),
        "input_tokens": usage["input_tokens"] or "",   # Claude only
        "output_tokens": usage["output_tokens"] or "",
        "grade": "",   # fill in: correct / partial / wrong
        "cause": "",   # fill in for wrong ones: retrieval / model / both
        "chunk_retrieved": found,
    }
    if args.agent:
        row["tool_calls"] = len(calls)
        row["tool_queries"] = " | ".join(f"{call['tool']}({call['query']})" for call in calls)
    rows.append(row)
    print(f"done: {q['question'][:60]}")

out = pathlib.Path(f"eval/answers_{label}.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"saved {out}")
