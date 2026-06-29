from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

sentences = [
    # AI and tech
    "Machine learning helps computers learn from data",
    "Neural networks are inspired by the human brain",
    "Deep learning requires large amounts of training data",
    "Python is the most popular language for AI development",
    "Artificial intelligence is transforming many industries",

    # Food
    "Pizza is one of the most loved foods in the world",
    "Indian cuisine uses a variety of spices like turmeric",
    "Biryani is a fragrant rice dish cooked with meat",
    "Chocolate cake is a popular dessert at celebrations",
    "Sushi is a traditional Japanese dish with fresh fish",

    # Nature
    "The Amazon rainforest produces 20 percent of Earths oxygen",
    "The ocean covers more than 70 percent of the Earth",
    "Climate change is causing sea levels to rise",
    "Photosynthesis allows plants to convert sunlight into energy",
    "Bees are essential for pollinating flowers and crops",

    # Sports
    "Cricket is the most popular sport in India",
    "Football is watched by billions of fans worldwide",
    "The Olympic Games bring athletes from every country",
    "Tennis requires both physical fitness and mental focus",
    "Swimming is one of the best full body exercises",
]

print("Generating embeddings...")
embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")
print(f"First 5 numbers of sentence 1: {embeddings[0][:5].tolist()}\n")

# ============================================================
# KEY FIX — use cosine similarity in Chroma
# By default Chroma uses L2 distance — we want cosine
# ============================================================
client = chromadb.PersistentClient(path="./chroma_db_fixed")

try:
    client.delete_collection("sentences")
except:
    pass

collection = client.create_collection(
    name="sentences",
    metadata={
        # THIS IS THE FIX — tells Chroma to use cosine similarity
        "hnsw:space": "cosine"
    }
)

collection.add(
    embeddings=embeddings.tolist(),
    documents=sentences,
    ids=[f"id_{i}" for i in range(len(sentences))]
)

print(f"Stored {collection.count()} sentences in Chroma\n")

# ============================================================
# MANUAL COSINE SIMILARITY
# Let's understand what cosine similarity means
# before using Chroma's built-in search
# ============================================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("="*60)
print("MANUAL SIMILARITY COMPARISON")
print("="*60)

pairs = [
    (0, 1),   # Both AI sentences — should be HIGH
    (0, 5),   # AI vs Food — should be LOW
    (5, 6),   # Both food sentences — should be HIGH
    (10, 11), # Both nature sentences — should be HIGH
    (0, 15),  # AI vs Sports — should be LOW
]

for i, j in pairs:
    sim = cosine_similarity(embeddings[i], embeddings[j])
    bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
    print(f"\n[{bar}] {sim:.3f}")
    print(f"  A: {sentences[i]}")
    print(f"  B: {sentences[j]}")

# ============================================================
# CHROMA SEMANTIC SEARCH
# Now with correct cosine similarity
# ============================================================
queries = [
    "What is artificial intelligence and machine learning?",
    "Tell me about food and cooking traditions",
    "How does nature and the environment work?",
]

print("\n" + "="*60)
print("SEMANTIC SEARCH RESULTS")
print("="*60)

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    for rank, (doc, dist) in enumerate(zip(documents, distances), 1):
        # With cosine space — distance IS the cosine distance
        # similarity = 1 - cosine_distance
        similarity = 1 - dist
        bar = "█" * int(similarity * 20) + "░" * (20 - int(similarity * 20))
        print(f"  {rank}. [{bar}] {similarity:.3f}")
        print(f"     {doc}")

# ============================================================
# DIMENSION ANALYSIS
# Let's see what a single embedding looks like
# ============================================================
print("\n" + "="*60)
print("EMBEDDING ANALYSIS")
print("="*60)
emb = embeddings[0]
print(f"\nSentence: '{sentences[0]}'")
print(f"Dimensions: {len(emb)}")
print(f"Min value:  {emb.min():.4f}")
print(f"Max value:  {emb.max():.4f}")
print(f"Mean:       {emb.mean():.4f}")
print(f"First 10 numbers: {emb[:10].tolist()}")

print("\n✅ Run complete!")
print("\nAnalysis for standup:")
print("1. Did AI queries return AI sentences? (they should)")
print("2. Did food queries return food sentences? (they should)")
print("3. Are scores between 0 and 1 now? (yes — cosine space)")
print("4. What is the highest similarity score you see?")