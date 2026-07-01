def build_context_string(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single readable context string.
    Each chunk is numbered and labelled with its page number
    so the LLM can reference sources in its answer.
    """
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} — Page {chunk['page']}]\n{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def build_rag_prompt(question: str, chunks: list[dict]) -> dict:
    """
    Build the complete prompt for Gemini.
    Returns system instruction and user message separately.

    The system instruction is the anti-hallucination rule —
    the single most important line in any RAG system.
    """
    context = build_context_string(chunks)

    system_instruction = """You are a helpful assistant that answers questions 
based strictly on the provided context from a resume document.

Rules you must follow:
1. Answer ONLY using information from the provided context.
2. If the answer is not in the context, say exactly: "This information is not mentioned in the document."
3. Always mention which source number supports your answer.
4. Be concise and factual — do not add information from your training data.
5. Never guess or make assumptions about missing information."""

    user_message = f"""Context:
{context}

Question: {question}

Answer based only on the context above, and cite which source number supports your answer."""

    return {
        "system_instruction": system_instruction,
        "user_message": user_message,
        "context_used": context,
        "chunks_count": len(chunks)
    }


if __name__ == "__main__":
    # Test with dummy chunks to verify formatting
    dummy_chunks = [
        {
            "text": "Ritika Puri — Results-driven engineering graduate with strong skills in Python and SQL.",
            "page": 0,
            "score": 0.58
        },
        {
            "text": "Face Recognition Attendance System — Achieved 95%+ accuracy using machine learning.",
            "page": 0,
            "score": 0.51
        }
    ]

    prompt = build_rag_prompt("What are Ritika's skills?", dummy_chunks)

    print("SYSTEM INSTRUCTION:")
    print(prompt["system_instruction"])
    print("\nUSER MESSAGE:")
    print(prompt["user_message"])