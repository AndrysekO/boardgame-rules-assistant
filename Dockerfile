# Rules assistant API in a container.
#
# Build (from the repo root; needs data/chroma, so run src/ingest.py first):
#   docker build -t rules-assistant .
#
# Run against Ollama on this machine (Docker Desktop reaches the host as
# host.docker.internal):
#   docker run --rm -p 8000:8000 \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     rules-assistant
#
# Run with Claude and an API key for the endpoints. -e NAME without a value
# passes the variable through from your shell, so no key is typed here:
#   docker run --rm -p 8000:8000 \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -e RULES_MODEL=claude-haiku-5-5 -e ANTHROPIC_API_KEY -e RULES_API_KEY \
#     rules-assistant
#
# Ollama is still needed with Claude: questions are embedded with nomic-embed-text.
# Then open http://127.0.0.1:8000/docs

FROM python:3.14-slim

# uv as a single binary, pinned to the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

WORKDIR /app

# Dependencies first: this layer is only rebuilt when pyproject.toml or
# uv.lock change, not on every code change. --no-install-project skips
# installing this project itself as a package; the API runs from src/ directly.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY data/chroma/ data/chroma/

# Use the virtual environment uv created, without needing "uv run".
ENV PATH="/app/.venv/bin:$PATH"

# No API keys here: ANTHROPIC_API_KEY, RULES_API_KEY, RULES_MODEL and
# OLLAMA_HOST all come from the environment at run time (docker run -e ...).

EXPOSE 8000
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
