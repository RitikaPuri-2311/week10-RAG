from rag_chain import ask_rag


# 10 evaluation questions specific to Ritika's resume
EVAL_QUESTIONS = [
    {
        "question": "Who is this resume about? What is the person's name?",
        "expected_keyword": "Ritika",
        "category": "basic info"
    },
    {
        "question": "What degree does Ritika have?",
        "expected_keyword": "Computer Science",
        "category": "education"
    },
    {
        "question": "What is her CGPA?",
        "expected_keyword": "8.02",
        "category": "education"
    },
    {
        "question": "What programming languages does she know?",
        "expected_keyword": "Python",
        "category": "skills"
    },
    {
        "question": "What cloud platforms has she worked with?",
        "expected_keyword": "AWS",
        "category": "skills"
    },
    {
        "question": "Describe the Face Recognition Attendance System project.",
        "expected_keyword": "attendance",
        "category": "projects"
    },
    {
        "question": "What was the accuracy of the face recognition system?",
        "expected_keyword": "95",
        "category": "projects"
    },
    {
        "question": "What AWS certification does she have?",
        "expected_keyword": "AWS Academy",
        "category": "certifications"
    },
    {
        "question": "What university did she attend?",
        "expected_keyword": "Medi-Caps",
        "category": "education"
    },
    {
        "question": "What is her favourite food?",
        "expected_keyword": "not mentioned",
        "category": "not found — should return no info"
    },
]


def evaluate_rag():
    print("=" * 70)
    print("RAG QUALITY EVALUATION — Ritika's Resume")
    print("=" * 70)

    scores = []

    for i, item in enumerate(EVAL_QUESTIONS):
        question = item["question"]
        expected = item["expected_keyword"].lower()
        category = item["category"]

        result = ask_rag(question)
        answer = result["answer"].lower()

        # Simple keyword check — did the answer contain the expected content?
        passed = expected in answer

        score = 5 if passed else 1
        scores.append(score)

        print(f"\n{i+1}. [{category}]")
        print(f"   Q: {question}")
        print(f"   A: {result['answer'][:150]}")
        print(f"   Expected keyword: '{item['expected_keyword']}'")
        print(f"   Score: {score}/5 — {'PASS' if passed else 'FAIL'}")
        print(f"   Sources: {len(result['sources'])} chunks used")

    avg = sum(scores) / len(scores)
    passed_count = sum(1 for s in scores if s == 5)

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_count}/10 passed")
    print(f"Average score: {avg:.1f}/5")
    print(f"Overall quality: {'GOOD' if avg >= 4 else 'NEEDS IMPROVEMENT'}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_rag()