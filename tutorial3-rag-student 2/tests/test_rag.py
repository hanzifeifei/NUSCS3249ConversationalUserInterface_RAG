"""Activity 1 checks: vector RAG with LangChain (split, store, search).

No LLM and no network: the embedding model is plain Python, and the LLM is FAKE.
"""
import pytest
from conftest import run, say
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from backend import orchestrator as orch
from backend import rag

DOC = Document(page_content="""# Title of the document

## Cost
The first consultation is free.

## Hours
Open Monday to Friday, 9am to 5pm.
""", metadata={"source": "x.md"})


@pytest.fixture(scope="module")
def store():
    return rag.build_vector_store(rag.split_docs(rag.load_docs()))


def test_split_makes_one_chunk_per_section_with_source_and_id():
    chunks = rag.split_docs([DOC])
    assert [c.metadata["id"] for c in chunks] == ["x.md#1", "x.md#2"]
    assert all(c.metadata["source"] == "x.md" for c in chunks)
    assert "## Cost" in chunks[0].page_content and "free" in chunks[0].page_content
    assert chunks[1].metadata["section"] == "Hours"


def test_long_sections_are_split_by_size():
    long = Document(page_content="## Long\n" + " ".join(f"word{i}" for i in range(300)),
                    metadata={"source": "long.md"})
    chunks = rag.split_docs([long], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_every_document_is_split():
    chunks = rag.split_docs(rag.load_docs())
    assert {c.metadata["source"] for c in chunks} == {d.metadata["source"] for d in rag.load_docs()}
    assert len(chunks) >= 20


def test_vector_store_holds_every_chunk(store):
    assert isinstance(store, InMemoryVectorStore)
    assert len(store.store) == len(rag.split_docs(rag.load_docs()))


@pytest.mark.parametrize("question, document", [
    ("Can I cancel my appointment?", "policies.md"),
    ("Is there a workshop for exam stress?", "services.md"),
    ("What happens in the first session?", "first_session.md"),
    ("Is this confidential?", "confidentiality.md"),
])
def test_vector_search_finds_the_right_document(store, question, document):
    assert rag.vector_search(question, store)[0]["source"] == document


def test_vector_search_is_ranked_and_thresholded(store):
    found = rag.vector_search("Can I cancel my appointment?", store, k=2)
    assert 1 <= len(found) <= 2
    assert found == sorted(found, key=lambda p: p["score"], reverse=True)
    assert set(found[0]) >= {"id", "source", "text", "score"}
    assert rag.vector_search("Do you have a gym?", store) == []


def test_passages_reach_the_llm_with_the_message(fake, store):
    f = fake(say(action="answer_faq", reply="Yes, up to 24 hours before (policies.md)."))
    state = orch.new_state()
    run(orch.handle("Can I cancel my appointment?", state, [], None, store))
    sent = f.calls[0]["messages"][-1]["content"]
    assert "[policies.md#" in sent and "24 hours" in sent
    assert state["retrieved"][0]["id"].startswith("policies.md#")
