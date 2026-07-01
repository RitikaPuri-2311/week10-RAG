import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

SIMILARITY_THRESHOLD = 0.2  # filter chunks below this score


def retrieve_chunks(
    question: str,
    collection_name: str = "resume",
    n_results: int = 5,
    threshold: float = SIMILARITY_THRESHOLD
) -> list[dict]:
    """
    Embed the question, search Chroma, filter by threshold,
    return only chunks above the quality bar.
    """
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - dist, 4)

        # Only include chunks above the similarity threshold
        if score >= threshold:
            chunks.append({
                "text": doc,
                "page": meta.get("page", 0),
                "source": meta.get("source", ""),
                "score": score
            })

    return chunks


def index_resume(file_path: str, collection_name: str = "resume") -> dict:
    """
    Index the resume PDF into Chroma.
    Reuses the exact pipeline from Day 2.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts).tolist()

    # Delete existing collection to avoid duplicate IDs
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source": file_path,
            "page": c.metadata.get("page", 0),
            "chunk_index": i
        } for i, c in enumerate(chunks)],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return {"chunks_indexed": len(chunks), "status": "indexed"}


if __name__ == "__main__":
    print("Indexing resume...")
    result = index_resume("sample.pdf")
    print(result)

    print("\nTesting retrieval...")
    question = "What Python projects has Ritika built?"
    chunks = retrieve_chunks(question)

    print(f"\nQuestion: {question}")
    print(f"Chunks retrieved above threshold: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n{i+1}. [score: {chunk['score']}] page {chunk['page']}")
        print(f"   {chunk['text'][:150]}...")