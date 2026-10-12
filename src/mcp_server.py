"""MCP server that lets an MCP client (e.g. Claude Code) search the rulebooks.

It only searches; the client's own model reads the excerpts and answers.
Ollama must be running, because queries are embedded with nomic-embed-text.

Connect it to Claude Code (run once, with the path to this repo):
    claude mcp add rules-assistant -- \
        uv run --directory /path/to/rules-assistant python src/mcp_server.py
Add --scope project before the name to share it through a .mcp.json file in
the repo, or --scope user to have it in every project. Check it with
`claude mcp list`, or /mcp inside Claude Code. Then ask e.g.
"Using the rules-assistant tools, what does the Barrel card do in Bang?"

Run it by hand (it waits for MCP messages on stdin; Ctrl+C to stop):
    uv run python src/mcp_server.py

Built with MCPServer from the official `mcp` SDK - FastMCP's name since mcp 2.x.
stdout carries the MCP protocol, so nothing else may print there; logs go to stderr.
"""

import logging
import os
import pathlib
import sys

from mcp.server.mcpserver import MCPServer

from agent import search
from api import GAMES

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("rules-assistant-mcp")

# Same reminder the agent gets in its system prompt.
CITE = " Answers must quote the excerpt and cite its page, like [Terraforming Mars p.12]."

server = MCPServer("rules-assistant")


# The descriptions below are the agent's tool docstrings from agent.py's
# agent_answer(), word for word, plus CITE. They live inside that function,
# so they can't be imported - keep the two in sync when changing either.
#
# Each search gets a fresh list for agent.search() to record into: the agent's
# 6-searches-per-question limit doesn't apply to the client's own model.

@server.tool(description="Return the names of the games that can be searched.")
def list_games() -> list[str]:
    return GAMES


@server.tool(description="""Search a game's rulebook and return the 5 closest excerpts.

game must be one of: "Terraforming Mars", "Everdell", "Bang".
query is what to look for, e.g. "greenery tile placement".""" + CITE)
def search_rulebook(game: str, query: str) -> str:
    log.info("search_rulebook %r %r", game, query)
    return search([], "search_rulebook", game, query, "rules")


@server.tool(description="""Search a game's FAQ and return the 5 closest excerpts.

Only Terraforming Mars has a FAQ, so don't use this for other games.
game must be one of: "Terraforming Mars", "Everdell", "Bang".
query is what to look for, e.g. "greenery tile placement".""" + CITE)
def search_faq(game: str, query: str) -> str:
    log.info("search_faq %r %r", game, query)
    return search([], "search_faq", game, query, "faq")


@server.tool(description="""Return every excerpt of a game that contains the exact text (any case), up to 15.

Use for specific numbers, values or names, e.g. "3 life points" or a card
name. Use search_rulebook for questions about how rules work.
game must be one of: "Terraforming Mars", "Everdell", "Bang".""" + CITE)
def search_exact(game: str, text: str) -> str:
    log.info("search_exact %r %r", game, text)
    return search([], "search_exact", game, text, None)


if __name__ == "__main__":
    # ask.py opens the index at the relative path data/chroma, so work from
    # the repo root whichever directory the client starts us in.
    os.chdir(pathlib.Path(__file__).resolve().parents[1])
    server.run("stdio")
