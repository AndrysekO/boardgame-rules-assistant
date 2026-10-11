"""Answer eval: run every question through ask.answer() and save a CSV to grade by hand.

Run with: uv run python eval/answer_eval.py baseline
"""
import csv
import json
import pathlib
import sys
import time

sys.path.insert(0, "src")
from ask import answer

label = sys.argv[1] if len(sys.argv) > 1 else "run"
questions = json.loads(pathlib.Path("eval/questions.json").read_text())

rows = []
for q in questions:
    start = time.time()
    text, chunks = answer(q["question"], q["game"])
    pages = ", ".join(f"p.{meta.get('page')}" for _, meta in chunks)
    needles = q.get("must_contain", [])
    if isinstance(needles, str):
        needles = [needles]
    retrieved_text = " ".join(text.lower() for text, _ in chunks)
    found = all(n.lower() in retrieved_text for n in needles) if needles else ""
    rows.append({
        "game": q["game"],
        "question": q["question"],
        "type": q.get("type", ""),
        "expected": q.get("answer", ""),
        "model_answer": text,
        "retrieved_pages": pages,
        "seconds": round(time.time() - start, 1),
        "grade": "",   # fill in: correct / partial / wrong
        "cause": "",   # fill in for wrong ones: retrieval / model / both
        "chunk_retrieved": found,
    })
    print(f"done: {q['question'][:60]}")

out = pathlib.Path(f"eval/answers_{label}.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"saved {out}")