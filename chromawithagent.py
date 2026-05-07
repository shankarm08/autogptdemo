import chromadb
import ollama

# -----------------------------
# Create Vector Database
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="knowledge")

# -----------------------------
# Documents
# -----------------------------
docs = [
    "Ollama is used to run local LLMs.",
    "ChromaDB is a vector database.",
    "Embeddings convert text into vectors.",
    "RAG means Retrieval Augmented Generation."
]

# -----------------------------
# Store Embeddings
# -----------------------------
for i, doc in enumerate(docs):

    embedding = ollama.embed(
        model="nomic-embed-text",
        input=doc
    )["embeddings"][0]

    collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding]
    )

# -----------------------------
# Agent Function
# -----------------------------
def rag_agent(question):

    # Convert question to embedding
    query_embedding = ollama.embed(
        model="nomic-embed-text",
        input=question
    )["embeddings"][0]

    # Search relevant docs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    context = "\n".join(results["documents"][0])

    # Prompt for LLM
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}
    """

    # Generate response
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

# -----------------------------
# Ask Agent
# -----------------------------
question = "What is RAG?"

answer = rag_agent(question)

print("\nAnswer:\n")
print(answer)