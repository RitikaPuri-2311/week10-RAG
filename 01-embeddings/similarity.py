from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pairs = [
    (
        "Machine learning helps computers learn.",
        "Deep learning uses neural networks.",
        "SHOULD BE HIGH — both AI topics"
    ),
    (
        "Pizza is a popular food.",
        "Cricket is played in India.",
        "SHOULD BE LOW — different topics"
    ),
    (
        "Python is used for AI development.",
        "Learning to code opens opportunities.",
        "SHOULD BE MEDIUM — loosely related"
    ),
    (
        "Biryani is a fragrant rice dish.",
        "Indian cuisine uses spices like turmeric.",
        "SHOULD BE HIGH — both Indian food"
    ),
    (
        "Climate change is causing sea levels to rise.",
        "A good resume highlights your achievements.",
        "SHOULD BE LOW — completely unrelated"
    ),
]

print("=" * 60)
print("COSINE SIMILARITY EXPERIMENT")
print("=" * 60)

for s1, s2, expected in pairs:
    a = model.encode(s1)
    b = model.encode(s2)
    score = cosine_similarity(a, b)

    if score >= 0.6:
        label = "HIGH"
    elif score >= 0.3:
        label = "MEDIUM"
    else:
        label = "LOW"

    print(f"\nExpected : {expected}")
    print(f"Score    : {score:.4f}  [{label}]")
    print(f"Sentence1: {s1}")
    print(f"Sentence2: {s2}")

print("\n" + "=" * 60)
print("Analysis: scores above 0.6 = similar meaning")
print("          scores below 0.3 = different topics")
print("=" * 60)