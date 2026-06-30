from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_with_langchain(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    LangChain's approach — loader handles extraction,
    splitter handles chunking, automatic metadata included.
    """
    # Step 1: Load — replaces your manual fitz.open() extraction
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"LangChain loaded {len(docs)} page-documents")

    # Step 2: Chunk — tries paragraph, then sentence, then word boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    return chunks


def print_chunk_summary(chunks):
    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"\n--- First chunk ---")
    print(f"Content ({len(chunks[0].page_content)} chars):")
    print(chunks[0].page_content[:300])
    print(f"\nMetadata (automatic!): {chunks[0].metadata}")

    print(f"\n--- Chunk size distribution ---")
    sizes = [len(c.page_content) for c in chunks]
    print(f"Smallest chunk: {min(sizes)} chars")
    print(f"Largest chunk : {max(sizes)} chars")
    print(f"Average chunk : {sum(sizes)//len(sizes)} chars")


if __name__ == "__main__":
    chunks = load_and_chunk_with_langchain("test-files/sample.pdf")
    print_chunk_summary(chunks)