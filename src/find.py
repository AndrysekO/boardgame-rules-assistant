import sys
import chromadb
import ollama

client = chromadb.PersistentClient(path="data/chroma")  # adjust if your ingest.py uses another path
col = client.get_collection(client.list_collections()[0].name)

question, needle = sys.argv[1], sys.argv[2]

# 1. Which chunks contain the text?
hits = col.get(where_document={"$contains": needle})
print("Chunks containing the text:", hits["ids"])

# 2. Where do they rank for the question?
emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
res = col.query(query_embeddings=[emb], n_results=30)
ranks = {}
for rank, (cid, meta) in enumerate(zip(res["ids"][0], res["metadatas"][0]), 1):
    mark = "  <-- contains it" if cid in hits["ids"] else ""
    if mark:
        ranks[cid] = rank
    print(rank, cid, meta.get("page"), meta.get("section"), mark)

# 3. Show the full text of each matching chunk
for cid, doc in zip(hits["ids"], hits["documents"]):
    print(f"\n===== {cid} | rank {ranks.get(cid, 'not in top 30')} | {len(doc)} characters =====")
    print(doc)