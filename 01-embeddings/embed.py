from sentence_transformers import SentenceTransformer
from sentences import sentences
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(sentences)

print(f"\nTotal sentences  : {len(sentences)}")
print(f"Vector dimensions: {len(embeddings[0])}")
print(f"\nFirst sentence   : {sentences[0]}")
print(f"First 5 numbers  : {np.round(embeddings[0][:5], 4)}")
print(f"\nLast sentence    : {sentences[-1]}")
print(f"First 5 numbers  : {np.round(embeddings[-1][:5], 4)}")
print("\nDone — embeddings generated successfully.")