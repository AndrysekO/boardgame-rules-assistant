"""Answer eval: run every question through ask.answer() and save a CSV and an
xlsx to grade by hand.

Run with: uv run python eval/answer_eval.py baseline
           uv run python eval/answer_eval.py agent-v1 --agent   (uses agent.agent_answer())
           uv run python eval/answer_eval.py v3 --previous v2   (a label or a file path)
Saves eval/results/answers/<label>.csv and eval/results/answers/<label>.xlsx.
"""
import argparse
import csv
import json
import pathlib
import sys
import time

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, "src")
import agent
import ask
from agent import agent_answer
from ask import answer

RESULTS_DIR = pathlib.Path("eval/results/answers")

# xlsx column order: the columns to read and grade first, then the details.
XLSX_FIRST = ["game", "question", "type", "expected", "model_answer", "grade", "cause",
              "previous_grade", "previous_cause", "answer_changed"]
XLSX_REST = ["chunk_retrieved", "retrieved_pages", "model", "seconds",
             "input_tokens", "output_tokens", "tool_calls", "tool_queries"]
# Filled from the --previous file, only for reference; shaded grey.
REFERENCE_COLUMNS = ["previous_grade", "previous_cause", "answer_changed"]
WIDTHS = {"question": 50, "expected": 55, "model_answer": 60,
          "retrieved_pages": 30, "tool_queries": 40}
NARROW = 14
DROPDOWNS = {"grade": ["correct", "partial", "wrong", "bad question"],
             "cause": ["retrieval", "model"]}


def find_previous(name):
    """Turn --previous into a file: a path as given, or a label's results file
    in RESULTS_DIR, preferring the .xlsx (that's where grades get filled in)."""
    path = pathlib.Path(name)
    if path.exists():
        return path
    for suffix in [".xlsx", ".csv"]:
        path = RESULTS_DIR / f"{name}{suffix}"
        if path.exists():
            return path
    raise SystemExit(f"--previous {name!r}: no such file, and no {name}.xlsx "
                     f"or {name}.csv in {RESULTS_DIR}")


def read_previous(path):
    """Return the rows of an earlier results file (.xlsx or .csv), keyed by question."""
    path = pathlib.Path(path)
    if path.suffix == ".xlsx":
        values = load_workbook(path, read_only=True).active.iter_rows(values_only=True)
        header = next(values)
        rows = [dict(zip(header, row)) for row in values]
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    return {row["question"]: row for row in rows}


def write_xlsx(rows, previous, path):
    """Save the rows as a formatted sheet, with grade/cause dropdowns and the
    grades from the `previous` rows next to them."""
    columns = XLSX_FIRST + [name for name in XLSX_REST if name in rows[0]]
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(columns)
    for row in rows:
        values = dict(row)
        earlier = previous.get(row["question"])
        if earlier:  # questions missing from the earlier file keep these empty
            values["previous_grade"] = earlier.get("grade") or ""
            values["previous_cause"] = earlier.get("cause") or ""
            same = earlier.get("model_answer") == row["model_answer"]
            values["answer_changed"] = "no" if same else "yes"
        # Write "" as a truly empty cell, so Excel's filter shows it as blank.
        sheet.append([None if values.get(name, "") == "" else values[name]
                      for name in columns])

    top_wrapped = Alignment(wrap_text=True, vertical="top")
    light_grey = PatternFill("solid", fgColor="EEEEEE")
    for number, name in enumerate(columns, start=1):
        letter = get_column_letter(number)
        sheet.column_dimensions[letter].width = WIDTHS.get(name, NARROW)
        for cell in sheet[letter]:
            cell.alignment = top_wrapped
            if name in REFERENCE_COLUMNS:
                cell.fill = light_grey

    for name, options in DROPDOWNS.items():
        letter = get_column_letter(columns.index(name) + 1)
        dropdown = DataValidation(type="list", formula1=f'"{",".join(options)}"')
        dropdown.add(f"{letter}2:{letter}{len(rows) + 1}")
        sheet.add_data_validation(dropdown)

    sheet.freeze_panes = "A2"  # keep the header row visible when scrolling
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


parser = argparse.ArgumentParser(description="Run the answer eval.")
parser.add_argument("label", nargs="?", default="run")
parser.add_argument("--agent", action="store_true",
                    help="answer with agent_answer() instead of answer()")
parser.add_argument("--previous", help="an earlier graded answers file (.xlsx or .csv), "
                                       "or its label, to compare against")
args = parser.parse_args()
label = args.label
# Read the earlier file first, so a wrong path fails before the slow eval runs.
previous = read_previous(find_previous(args.previous)) if args.previous else {}
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = RESULTS_DIR / f"{label}.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"saved {out}")

xlsx = RESULTS_DIR / f"{label}.xlsx"
write_xlsx(rows, previous, xlsx)
print(f"saved {xlsx}")
