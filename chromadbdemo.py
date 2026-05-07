import chromadb
import ollama

client = chromadb.Client()

collection = client.create_collection(name="docs")

doc = "Ollama is used to run local LLMs."

embedding = ollama.embed(
    model="nomic-embed-text",
    input=doc
)["embeddings"][0]

collection.add(
    documents=[doc],
    embeddings=[embedding],
    ids=["1"]
)

query_embedding = ollama.embed(
    model="nomic-embed-text",
    input="What runs local AI models?"
)["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(results)