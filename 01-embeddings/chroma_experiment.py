import chromadb
from sentence_transformers import SentenceTransformer
from sentences import sentences

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings for all sentences...")
embeddings = model.encode(sentences).tolist()

print("Setting up Chroma...")
client = chromadb.PersistentClient(path="./chroma_db")

# Delete collection if exists — clean run every time
try:
    client.delete_collection("docs")
except:
    pass

collection = client.create_collection(
    "docs",
    metadata={"hnsw:space": "cosine"}
)

# Add all 20 sentences
collection.add(
    embeddings=embeddings,
    documents=sentences,
    ids=[f"id{i}" for i in range(len(sentences))]
)

print(f"Stored {collection.count()} sentences in Chroma\n")

# 3 query questions
queries = [
    "What is artificial intelligence and machine learning?",
    "Tell me about food and cooking traditions.",
    "How does nature and the environment work?",
]

print("=" * 60)
print("QUERY RESULTS — TOP 5 SIMILAR SENTENCES")
print("=" * 60)

for query in queries:
    q_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=5
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        score = round(1 - dist, 4)
        print(f"  {i+1}. [score: {score}] {doc}")

print("\n" + "=" * 60)
print("ANALYSIS QUESTIONS FOR STAND-UP:")
print("  1. Are all top 5 results from the correct topic?")
print("  2. Which query got the most accurate results?")
print("  3. Did any off-topic sentence appear in top 5?")
print("  4. Why might an off-topic result appear?")
print("=" * 60)