# Board Game Rules Assistant

RAG + agent that answers board game rules questions from PDF rulebooks with page citations.

## Commands
- Run a script: `uv run python src/<script>.py`
- Add a dependency: `uv add <package>` (never `pip install`)
- Rebuild the index: `uv run python src/ingest.py` (deletes and rebuilds data/chroma)
- Retrieval eval: `uv run python eval/retrieval_eval.py <label>`
- Answer eval: `uv run python eval/answer_eval.py <label>`
- Use a Claude model: `RULES_MODEL=claude-haiku-5-5 uv run python ...` (default is the local Ollama model)

## Conventions
- Every chunk keeps metadata: game, source (rules/faq), page, section.
- Answers must cite as [Game p.12]; if the answer isn't in the excerpts, say "not in the rules".
- Use `import pymupdf`, not the deprecated `fitz`.
- Keep code simple and readable; I need to explain every line in interviews.
- Never overwrite eval results; each run gets a new label.

## Secrets
- Never write API keys, tokens or passwords into any file: code, config, notes, tests or commit messages.
- Read the Anthropic key only from the `ANTHROPIC_API_KEY` environment variable.
- Never print, log or save the key, not even partly, and never include it in eval CSVs or error messages.
- If a key is needed and the variable is missing, stop with a clear error; don't ask for it to be pasted into a file.

## Never commit
- `data/` (copyrighted rulebooks, Chroma index)
- `.venv/`, `__pycache__/`, `*.pyc`
- `.env` or any file containing secrets
