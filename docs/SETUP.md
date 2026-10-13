# Setup and usage

Full guide to running the Board Game Rules Assistant. Back to the [README](../README.md).

Requires [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) (running), and for Claude models an `ANTHROPIC_API_KEY` in the environment.

## 1. Install

```bash
git clone https://github.com/AndrysekO/boardgame-rules-assistant
cd boardgame-rules-assistant
uv sync
ollama pull nomic-embed-text      # embeddings (always needed)
ollama pull llama3.1:8b           # local answering model (default)
```

## 2. Add rulebooks and build the index

Rulebooks aren't included (copyright). Create the folder and add your own PDFs, named `<game>_<source>.pdf`:

```bash
mkdir -p data/pdfs
# e.g. data/pdfs/bang_rules.pdf, data/pdfs/everdell_rules.pdf,
#      data/pdfs/terraforming_mars_rules.pdf, data/pdfs/terraforming_mars_faq.pdf
uv run python src/ingest.py       # builds data/chroma; rebuilds from scratch each run
```

The game name comes from the file name (underscores become spaces).

## 3. Ask questions

```bash
# Single search: one retrieval, one model call
uv run python src/ask.py --game bang "What is the distance with Panic?"

# Agent: chooses between rulebook, FAQ and keyword search (up to 6 tool calls)
uv run python src/agent.py --game everdell "How many cards can I have in my city?"
```

| Option | Meaning |
|---|---|
| `--game <name>` | Search only one game: `bang`, `everdell`, `"terraforming mars"` (quotes needed for spaces) |
| `RULES_MODEL=<model>` | Answering model: `llama3.1:8b` (default, local) or `claude-haiku-5-5` (needs `ANTHROPIC_API_KEY`) |

Example with Claude:

```bash
RULES_MODEL=claude-haiku-5-5 uv run python src/agent.py --game bang "Which characters have 3 life points?"
```

In zsh, avoid `!` inside double quotes, or use single quotes.

## 4. API

```bash
RULES_MODEL=claude-haiku-5-5 uv run uvicorn api:app --app-dir src --reload
```

Open http://localhost:8000/docs. `POST /ask` takes `question`, `game`, `mode` (`single` or `agent`, default `agent`) and optionally `model`. If `RULES_API_KEY` is set, requests need a matching `X-API-Key` header.

## 5. Docker

Build the index first (step 2); the image copies `data/chroma`. Ollama keeps running on the host for embeddings.

```bash
docker build -t rules-assistant .
docker run --rm -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e RULES_MODEL=claude-haiku-5-5 -e ANTHROPIC_API_KEY \
  rules-assistant
```

Then open http://localhost:8000/docs. `-e ANTHROPIC_API_KEY` passes the key from your shell; it's never stored in the image. Rebuild after code changes.

## 6. MCP server (Claude Code)

Connect it once (replace `/path` with where you cloned the repo), then start Claude Code anywhere:

```bash
claude mcp add rules-assistant -- uv run --directory /path/boardgame-rules-assistant python src/mcp_server.py
claude mcp list                   # rules-assistant should show as connected
claude
```

Then ask, for example:

> Using the rules-assistant tools, what does the Barrel card do in Bang?

Claude Code calls `list_games`, `search_rulebook`, `search_faq` or `search_exact` and answers with page citations. Inside Claude Code, `/mcp` shows the server and its tools.

## 7. Tests and evals

```bash
uv run pytest                                          # unit tests, no models needed
uv run python eval/retrieval_eval.py <label>           # → eval/results/retrieval/<label>.csv
uv run python eval/answer_eval.py <label> [--agent]    # → eval/results/answers/<label>.csv + .xlsx
```

`answer_eval.py --previous <file>` shows earlier grades and whether each answer changed.
