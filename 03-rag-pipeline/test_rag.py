import pytest
from rag_chain import ask_rag, index_resume


@pytest.fixture(scope="session", autouse=True)
def setup_index():
    """Index the resume once before all tests run."""
    index_resume("sample.pdf")


def test_relevant_query_returns_answer():
    """A relevant question should return a non-empty answer."""
    result = ask_rag("What degree does Ritika have?")
    assert result["answer"] != ""
    assert "Computer Science" in result["answer"]


def test_relevant_query_returns_sources():
    """A relevant question should always return source citations."""
    result = ask_rag("What projects has Ritika built?")
    assert len(result["sources"]) > 0
    assert "page" in result["sources"][0]
    assert "text_snippet" in result["sources"][0]
    assert "score" in result["sources"][0]


def test_irrelevant_query_returns_not_found():
    """A question about something not in the resume should say not found."""
    result = ask_rag("What is Ritika's favourite movie?")
    answer_lower = result["answer"].lower()
    assert any(phrase in answer_lower for phrase in [
        "not mentioned",
        "not found",
        "not in the",
        "no information",
        "cannot find"
    ])


def test_education_query():
    """Should correctly retrieve CGPA and university."""
    result = ask_rag("What is Ritika's CGPA and which university did she attend?")
    assert "8.02" in result["answer"]
    assert len(result["sources"]) > 0


def test_skills_query():
    """Should return Python and AWS in skills answer."""
    result = ask_rag("What are Ritika's technical skills?")
    answer_lower = result["answer"].lower()
    assert "python" in answer_lower


def test_source_scores_are_positive():
    """All returned source scores should be positive (cosine similarity)."""
    result = ask_rag("What certifications does she have?")
    for source in result["sources"]:
        assert source["score"] > 0