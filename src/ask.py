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

Cite every claim with the game and page of the excerpt it came from, like
[Mars p.12]. If the excerpts do not answer the question, reply exactly:
not in the rules"""


def retrieve(question, game=None):
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    vector = ollama.embed(model=EMBED_MODEL, input=question).embeddings[0]
    result = collection.query(
        query_embeddings=[vector],
        n_results=TOP_K,
        where={"game": game.title()} if game else None,
    )
    # Chroma nests one list per query; we only ever send one.
    return list(zip(result["documents"][0], result["metadatas"][0], result["distances"][0]))


def show(hits):
    for number, (text, meta, distance) in enumerate(hits, start=1):
        # The collection uses cosine distance, so 1 - distance is the similarity.
        print(f"\n[{number}] {meta['game']} {meta['source']} p.{meta['page']} "
              f"score={1 - distance:.3f} section={meta['section'] or '-'}")
        print(textwrap.indent(textwrap.fill(text, 92), "    "))


def answer(question, hits):
    excerpts = "\n\n".join(
        f"[{number}] {meta['game']} p.{meta['page']}\n{text}"
        for number, (text, meta, _) in enumerate(hits, start=1)
    )
    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return reply.message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Answer a board game rules question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--game", help="only search one game, e.g. mars")
    args = parser.parse_args()
    question = " ".join(args.question)

    hits = retrieve(question, args.game)
    if not hits:
        raise SystemExit(f"no chunks for game {args.game!r} - did you run src/ingest.py?")
    print(f"question: {question}")
    print(f"\n--- top {len(hits)} chunks ---")
    show(hits)
    print("\n--- answer ---")
    print(textwrap.fill(answer(question, hits), 92))


if __name__ == "__main__":
    main()
