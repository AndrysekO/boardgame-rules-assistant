"""Retrieval eval: where does the chunk with the expected text rank for each question?

Run with: uv run python eval/retrieval_eval.py baseline
"""
import csv
import json
import pathlib
import sys

sys.path.insert(0, "src")  # ask.py lives in src/, which is not an installed package

from ask import retrieve

TOP_K = 20
label = sys.argv[1] if len(sys.argv) > 1 else "run"

questions = json.loads(pathlib.Path("eval/questions.json").read_text())

rows = []
for q in questions:
    if "must_contain" not in q:
        continue  # "not in the rules" questions are for answer evals later

    hits = retrieve(q["question"], q["game"], k=TOP_K)
    docs = [text.lower() for text, _ in hits]

    needles = q["must_contain"] if isinstance(q["must_contain"], list) else [q["must_contain"]]
    ranks = [next((i for i, d in enumerate(docs, 1) if n.lower() in d), None) for n in needles]
    found = [r for r in ranks if r]
    rank = max(found) if len(found) == len(needles) else None  # all pieces found: rank of the last one
    coverage = sum(1 for r in ranks if r and r <= 5) / len(needles)

    rows.append({"game": q["game"], "question": q["question"], "type": q.get("type", ""),
                 "rank": rank, "coverage@5": round(coverage, 2)})
    print(f"{str(rank or '-'):>3}  {coverage:4.0%}  {q['game'][:18]:18}  {q['question']}")

found_in_5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5)
mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
avg_coverage = sum(r["coverage@5"] for r in rows) / len(rows)
print(f"\nrecall@5: {found_in_5}/{len(rows)}   MRR: {mrr:.2f}   coverage@5: {avg_coverage:.0%}")

out = pathlib.Path(f"eval/results_{label}.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"saved {out}")