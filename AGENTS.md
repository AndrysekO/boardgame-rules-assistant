
# Board Game Rules Assistant

RAG + agent that answers board game rules questions from PDF rulebooks with page citations.

## Commands
- Run: `uv run python src/<script>.py`
- Add a dependency: `uv add <package>`

## Conventions
- Every chunk keeps metadata: game, source (rules/faq), page, section.
- Answers must cite as [Game p.12]; if the answer isn't in the excerpts, say "not in the rules".
- Never commit anything in data/ (rulebooks are copyrighted).
- Keep code simple and readable; I need to explain every line in interviews.
