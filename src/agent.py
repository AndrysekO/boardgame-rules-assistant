"""Answer a rules question with an agent that decides what to search for.

Unlike ask.py, which retrieves once and answers, the agent can search the
rulebook and the FAQ several times with different words before it answers.

Run with: uv run python src/agent.py "can a greenery tile go next to a city?"
           uv run python src/agent.py --game everdell "how many workers do I get?"
"""

import argparse
import os
import re
import textwrap
import threading

import chromadb
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError

from ask import CHAT_MODEL, COLLECTION, DB_DIR, MAX_TOKENS, TOP_K, retrieve

MAX_TOOL_CALLS = 6
MAX_EXACT_MATCHES = 15

# Every search made while answering the current question, in order. The tools
# append to it and agent_answer() resets it, so the tools can enforce the limit.
TOOL_CALLS = []

# Model and token counts summed over every model call of the most recent
# agent_answer(), like ask.LAST_USAGE. None if the model reported no usage.
LAST_USAGE = {"model": CHAT_MODEL, "input_tokens": None, "output_tokens": None}

# When the model asks for several searches at once, LangGraph runs them in
# parallel threads. Chroma's client isn't safe to open from several threads at
# the same time, and the limit check needs a consistent count, so searches take turns.
SEARCH_LOCK = threading.Lock()

SYSTEM = """You answer board game rules questions.

You have three tools: search_rulebook, search_faq and search_exact. Use them to find excerpts
from the rules, and answer only from those excerpts. Never answer from anything
you know about these games from elsewhere, and do not guess.

How to search:
- If a search doesn't find the answer, search again with different words,
  e.g. "components" or "contents" instead of "resources".
- If the question is about a list of things, search for each item separately.
- You can make at most 6 searches per question.

Reply in exactly this format:
Quote: the exact sentence(s) from the excerpts that answer the question, with
their citation, like [Terraforming Mars p.12].
Answer: a short answer based only on the quote.

If no excerpt contains a sentence that answers the question, reply exactly:
not in the rules"""


def find_exact(game, text):
    """Return up to 15 chunks of `game` that contain `text`, ignoring case.

    Chroma's $contains is case-sensitive, so this uses $regex instead: "(?i)"
    turns off case-sensitivity and re.escape makes characters like "." or "!"
    in the text match literally.
    """
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    result = collection.get(
        where={"game": game.title()},
        where_document={"$regex": "(?i)" + re.escape(text)},
        limit=MAX_EXACT_MATCHES,
    )
    return list(zip(result["documents"], result["metadatas"]))


def search(tool_name, game, query, source):
    """Run one search for a tool and return its chunks as numbered excerpts."""
    with SEARCH_LOCK:
        if len(TOOL_CALLS) >= MAX_TOOL_CALLS:
            return ("Search limit reached. Answer from the excerpts you already have, "
                    "or reply: not in the rules")
        if tool_name == "search_exact":
            chunks = find_exact(game, query)
        else:
            chunks = retrieve(query, game, k=TOP_K, source=source)
        # The chunks are kept so the eval can check what the agent saw.
        TOOL_CALLS.append({"tool": tool_name, "game": game, "query": query, "chunks": chunks})

    if not chunks:
        return f"No excerpts found for {query!r} in game {game!r}."
    # Same excerpt format as ask.py's answer().
    return "\n\n".join(
        f"[{number}] {metadata['game']} p.{metadata['page']}\n{text}"
        for number, (text, metadata) in enumerate(chunks, start=1)
    )


@tool
def search_rulebook(game: str, query: str) -> str:
    """Search a game's rulebook and return the 5 closest excerpts.

    game must be one of: "Terraforming Mars", "Everdell", "Bang".
    query is what to look for, e.g. "greenery tile placement".
    """
    return search("search_rulebook", game, query, "rules")


@tool
def search_faq(game: str, query: str) -> str:
    """Search a game's FAQ and return the 5 closest excerpts.

    Only Terraforming Mars has a FAQ, so don't use this for other games.
    game must be one of: "Terraforming Mars", "Everdell", "Bang".
    query is what to look for, e.g. "greenery tile placement".
    """
    return search("search_faq", game, query, "faq")


@tool
def search_exact(game: str, text: str) -> str:
    """Return every excerpt of a game that contains the exact text (any case), up to 15.

    Use for specific numbers, values or names, e.g. "3 life points" or a card
    name. Use search_rulebook for questions about how rules work.
    game must be one of: "Terraforming Mars", "Everdell", "Bang".
    """
    return search("search_exact", game, text, None)


def make_model():
    """Pick the chat model from RULES_MODEL, the same way ask.py does."""
    if CHAT_MODEL.startswith("claude-"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                f"RULES_MODEL={CHAT_MODEL} needs ANTHROPIC_API_KEY in the environment."
            )
        # No temperature: Claude Haiku 5.5 rejects it (see ask.py's ask_claude).
        # The client reads the key from the environment itself.
        return ChatAnthropic(model=CHAT_MODEL, max_tokens=MAX_TOKENS)
    return ChatOllama(model=CHAT_MODEL, temperature=0)


def agent_answer(question, game=None):
    """Answer `question` by letting the agent search the rules.

    Returns (answer text, tool calls), where each tool call is a dict with
    the tool name, game, query and the chunks it returned, in the order the
    agent made them. Summed token counts land in LAST_USAGE.
    """
    TOOL_CALLS.clear()
    agent = create_agent(
        make_model(), [search_rulebook, search_faq, search_exact], system_prompt=SYSTEM
    )

    user_message = question
    if game:
        user_message = f"Game: {game.title()}\nQuestion: {question}"

    # Each search round is two graph steps (model, then tools). Allow the 6
    # searches, one more round that only gets "limit reached", and the answer.
    recursion_limit = 2 * (MAX_TOOL_CALLS + 1) + 1
    # stream() hands back the conversation after every step. Keeping the latest
    # one means we still have every message, and its tokens, if the run is cut off.
    state = {"messages": []}
    try:
        for state in agent.stream(
            {"messages": [("user", user_message)]},
            {"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            pass
        text = state["messages"][-1].text.strip()
    except GraphRecursionError:
        # The agent kept searching past the limit instead of answering.
        text = "not in the rules"

    # Only the model's messages carry usage; tool results and the question don't.
    usages = [m.usage_metadata for m in state["messages"]
              if getattr(m, "usage_metadata", None)]
    LAST_USAGE.update(
        model=CHAT_MODEL,
        input_tokens=sum(u["input_tokens"] for u in usages) if usages else None,
        output_tokens=sum(u["output_tokens"] for u in usages) if usages else None,
    )
    return text, list(TOOL_CALLS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a board game rules question with an agent.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--game", help="the game the question is about, e.g. everdell")
    args = parser.parse_args()
    question = " ".join(args.question)

    text, calls = agent_answer(question, args.game)

    print(f"question: {question}")
    print(f"\n--- {len(calls)} tool calls ---")
    for number, call in enumerate(calls, start=1):
        print(f"[{number}] {call['tool']}(game={call['game']!r}, query={call['query']!r})")

    print("\n--- answer ---")
    print(textwrap.fill(text, 92))
