import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Load the embedding model once, outside any function —
# loading it inside a function would reload it on every call, very slow
model = SentenceTransformer("all-MiniLM-L6-v2")

# Same with the Chroma client — one connection, reused across calls
client = chromadb.PersistentClient(path="./chroma_db")


def index_document(file_path: str, collection_name: str = "documents") -> dict:
    """
    The complete pipeline: load PDF -> chunk -> embed -> store.

    This is the function the FastAPI endpoint calls when someone
    uploads a PDF through the API.
    """
    # STEP 1 — Load and extract text (LangChain handles this)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # STEP 2 — Chunk the text (RecursiveCharacterTextSplitter)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # STEP 3 — Generate embeddings for every chunk at once
    # (batch encoding is much faster than encoding one chunk at a time)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts).tolist()

    # STEP 4 — Store in Chroma WITH metadata
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source": chunk.metadata.get("source", file_path),
            "page": chunk.metadata.get("page", 0),
            "chunk_index": i
        } for i, chunk in enumerate(chunks)],
        ids=[f"{file_path}_chunk_{i}" for i in range(len(chunks))]
    )

    return {
        "file_path": file_path,
        "chunks_created": len(chunks),
        "status": "indexed",
        "collection": collection_name,
        "total_in_collection": collection.count()
    }


def query_document(question: str, collection_name: str = "documents", n_results: int = 3) -> list[dict]:
    """
    Search the indexed document for chunks relevant to a question.
    Searches across ALL documents in the collection — no filtering.
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

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "text": doc,
            "page": meta.get("page"),
            "score": round(1 - dist, 4)
        })
    return output


def query_document_filtered(question: str, source_file: str, collection_name: str = "documents", n_results: int = 3) -> list[dict]:
    """
    Same as query_document, but scoped to only one document using
    the 'where' metadata filter. This is what production RAG uses
    when multiple PDFs share the same collection — prevents results
    from one document leaking into another document's search.
    """
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where={"source": source_file}
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "text": doc,
            "page": meta.get("page"),
            "score": round(1 - dist, 4)
        })
    return output


if __name__ == "__main__":
    print("=" * 60)
    print("INDEXING")
    print("=" * 60)
    result = index_document("test-files/sample.pdf")
    print(result)

    print("\n" + "=" * 60)
    print("FILTERED QUERY — scoped to one document")
    print("=" * 60)
    question = "What experience does this person have with Python and cloud computing?"
    print(f"Question: {question}\n")

    matches = query_document_filtered(question, source_file="test-files/sample.pdf")
    for i, match in enumerate(matches):
        print(f"{i+1}. [score: {match['score']}] (page {match['page']})")
        print(f"   {match['text'][:150]}...")
        print()