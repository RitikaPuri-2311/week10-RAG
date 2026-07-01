import os
import time
from dotenv import load_dotenv
from google import genai
from retrieval import retrieve_chunks, index_resume
from prompt_builder import build_rag_prompt

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-3.1-flash-lite"


def ask_rag(question: str, collection_name: str = "resume") -> dict:
    """
    The complete RAG chain:
    1. Retrieve relevant chunks from Chroma
    2. Build grounded prompt with context
    3. Call Gemini with retry on rate limit
    4. Return answer + sources for citation
    """
    # STEP 1 — Retrieve
    chunks = retrieve_chunks(question, collection_name=collection_name)

    # STEP 2 — Handle no results
    if not chunks:
        return {
            "question": question,
            "answer": "This information is not mentioned in the document.",
            "sources": [],
            "chunks_used": 0
        }

    # STEP 3 — Build prompt
    prompt_data = build_rag_prompt(question, chunks)

    # STEP 4 — Call Gemini with retry on rate limit
    max_retries = 3
    response = None

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_data["user_message"],
                config={
                    "system_instruction": prompt_data["system_instruction"],
                    "temperature": 0.1,
                    "max_output_tokens": 500,
                }
            )
            break  # success — exit retry loop

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"Rate limit hit — waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                raise

    # STEP 5 — Build sources for citation
    sources = [{
        "page": chunk["page"],
        "text_snippet": chunk["text"][:150],
        "score": chunk["score"]
    } for chunk in chunks]

    return {
        "question": question,
        "answer": response.text,
        "sources": sources,
        "chunks_used": len(chunks)
    }


if __name__ == "__main__":
    print("Indexing resume...")
    index_resume("sample.pdf")

    print("\nRAG CHAIN TEST")
    print("=" * 60)

    test_questions = [
        "What is Ritika's educational background?",
        "What Python projects has she built?",
        "What cloud certifications does she have?",
        "What is her favourite movie?",
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        result = ask_rag(question)
        print(f"A: {result['answer']}")
        print(f"Sources used: {len(result['sources'])}")
        if result['sources']:
            print(f"Top source score: {result['sources'][0]['score']}")
        print("-" * 40)
        time.sleep(5)  # small pause between questions to avoid rate limits