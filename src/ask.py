"""Answer a rules question from the chunks stored by ingest.py.

Run with: uv run python src/ask.py "can a greenery tile go next to a city?"
           uv run python src/ask.py --game everdell "how many workers do I get?"
"""

import argparse
import os
import textwrap

import anthropic
import chromadb
import ollama

DB_DIR = "data/chroma"
COLLECTION = "rules"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = os.environ.get("RULES_MODEL", "llama3.1:8b")
TOP_K = 5
MAX_TOKENS = 2048

# Model and token counts from the most recent answer() call. answer() still
# returns just (text, chunks), so the eval reads the usage numbers from here.
# Ollama reports no tokens, so they stay None unless Claude answered.
LAST_USAGE = {"model": CHAT_MODEL, "input_tokens": None, "output_tokens": None}

SYSTEM = """You answer board game rules questions.

Use only the numbered excerpts you are given. Do not use anything you know about
these games from elsewhere, and do not guess.

Reply in exactly this format:
Quote: the exact sentence(s) from the excerpts that answer the question, with
their citation, like [Terraforming Mars p.12].
Answer: a short answer based only on the quote.

If no excerpt contains a sentence that answers the question, reply exactly:
not in the rules"""


def retrieve(question, game=None, k=TOP_K):
    """Return the k chunks closest to `question`, as (text, metadata) pairs.

    The metadata is what ingest.py stored - game, source, page, section - plus
    a `score`: the collection uses cosine distance, so similarity is 1 minus it.
    """
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    vector = ollama.embed(model=EMBED_MODEL, input=question).embeddings[0]
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where={"game": game.title()} if game else None,
    )
    # Chroma nests one list per query; we only ever send one.
    return [
        (text, dict(metadata, score=1 - distance))
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def ask_claude(user_message):
    """Send one question to the Anthropic API - the project uses claude-haiku-5-5.

    No temperature is sent: Claude Haiku 5.5 rejects temperature, top_p and
    top_k with a 400. It thinks adaptively instead, so there is no knob to pin.

    Returns (answer text, input tokens, output tokens).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            f"RULES_MODEL={CHAT_MODEL} needs ANTHROPIC_API_KEY in the environment."
        )
    # The key is read from the environment by the client; never pass it around.
    reply = anthropic.Anthropic().messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in reply.content if block.type == "text")
    return text.strip(), reply.usage.input_tokens, reply.usage.output_tokens


def ask_ollama(user_message):
    """Send one question to the local Ollama server.

    Returns (answer text, None, None) - token counts are only tracked for Claude.
    """
    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0},
    )
    return reply.message.content.strip(), None, None


def answer(question, game=None):
    """Answer `question` from the retrieved chunks.

    Returns (answer text, chunks) so the caller can show the evidence it used.
    Which model answers comes from RULES_MODEL; token counts land in LAST_USAGE.
    """
    chunks = retrieve(question, game)
    if not chunks:
        # Nothing retrieved (an unknown game, an empty database): there are no
        # excerpts to read an answer out of, which is the same as not finding one.
        LAST_USAGE.update(model=CHAT_MODEL, input_tokens=None, output_tokens=None)
        return "not in the rules", chunks

    excerpts = "\n\n".join(
        f"[{number}] {metadata['game']} p.{metadata['page']}\n{text}"
        for number, (text, metadata) in enumerate(chunks, start=1)
    )
    user_message = f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"

    send = ask_claude if CHAT_MODEL.startswith("claude-") else ask_ollama
    text, input_tokens, output_tokens = send(user_message)
    LAST_USAGE.update(
        model=CHAT_MODEL, input_tokens=input_tokens, output_tokens=output_tokens
    )
    return text, chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a board game rules question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--game", help="only search one game, e.g. everdell")
    args = parser.parse_args()
    question = " ".join(args.question)

    text, chunks = answer(question, args.game)
    if not chunks:
        raise SystemExit(f"no chunks for game {args.game!r} - did you run src/ingest.py?")

    print(f"question: {question}")
    print(f"\n--- top {len(chunks)} chunks ---")
    for number, (chunk, metadata) in enumerate(chunks, start=1):
        print(f"\n[{number}] {metadata['game']} {metadata['source']} p.{metadata['page']} "
              f"score={metadata['score']:.3f} section={metadata['section'] or '-'}")
        print(textwrap.indent(textwrap.fill(chunk, 92), "    "))

    print("\n--- answer ---")
    print(textwrap.fill(text, 92))
