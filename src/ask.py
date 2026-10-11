"""Answer a rules question from the chunks stored by ingest.py.

Run with: uv run python src/ask.py "can a greenery tile go next to a city?"
           uv run python src/ask.py --game everdell "how many workers do I get?"
"""

import argparse
import textwrap

import chromadb
import ollama

DB_DIR = "data/chroma"
COLLECTION = "rules"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"
TOP_K = 5

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


def answer(question, game=None):
    """Answer `question` from the retrieved chunks.

    Returns (answer text, chunks) so the caller can show the evidence it used.
    """
    chunks = retrieve(question, game)
    if not chunks:
        # Nothing retrieved (an unknown game, an empty database): there are no
        # excerpts to read an answer out of, which is the same as not finding one.
        return "not in the rules", chunks

    excerpts = "\n\n".join(
        f"[{number}] {metadata['game']} p.{metadata['page']}\n{text}"
        for number, (text, metadata) in enumerate(chunks, start=1)
    )
    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return reply.message.content.strip(), chunks


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
