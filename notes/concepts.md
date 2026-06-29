# RAG Learning Notes

## Embeddings
- Text converted to vectors of numbers
- Similar meaning = vectors close in space
- Model used: all-MiniLM-L6-v2 (384 dimensions)

## Cosine Similarity
- Score 1.0 = identical meaning
- Score 0.0 = unrelated
- Formula: dot(a,b) / (norm(a) * norm(b))

## Chroma
- Local vector database, no account needed
- PersistentClient saves to disk
- collection.query() returns top N similar docs
