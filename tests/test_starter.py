"""[starter] checks: these pass BEFORE you change anything.

They confirm the chatbot already works: it talks through the LLM, keeps its
Tutorial 1 state, starts the clinic MCP server and connects to it.
"""
from conftest import run, say

from backend import orchestrator as orch
from backend import rag
from backend.content import NO_LLM_MESSAGE


def test_starter_chat_still_works_without_tools(fake):
    fake(say(updates={"reason_for_visit": "exams"}, reply="How long have you felt this way?"))
    state = orch.new_state()
    assert run(orch.handle("Stressed about exams", state)) == "How long have you felt this way?"
    assert state["current_step"] == "duration"


def test_starter_state_has_tool_fields():
    state = orch.new_state()
    assert state["last_tool"] is None and state["booking"] is None


def test_starter_no_llm_configured():
    assert run(orch.handle("hi", orch.new_state())) == NO_LLM_MESSAGE


def test_starter_app_starts_and_connects_to_the_server():
    """Starts the real server as a subprocess, over stdio, like `uvicorn` does."""
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        body = client.post("/api/reset", json={"session_id": "s"}).json()
    assert body["mcp"]["connected"] is True


def test_starter_documents_graph_and_prompt_are_there():
    from backend import graph_rag
    from backend.prompts import SYSTEM_PROMPT
    docs = rag.load_docs()
    assert len(docs) >= 6 and all(d.page_content.strip() for d in docs)
    assert len(graph_rag.load_graph_data()["triples"]) >= 20
    assert "ONLY clinic facts" in SYSTEM_PROMPT and "knowledge graph" in SYSTEM_PROMPT


def test_starter_embedding_model_is_a_langchain_embeddings():
    from langchain_core.embeddings import Embeddings
    e = rag.KeywordEmbeddings.fit(["exam stress workshop", "cancel appointment"])
    assert isinstance(e, Embeddings)
    assert len(e.embed_query("exam workshop")) == len(e.embed_documents(["cancel"])[0])


def test_starter_find_entities_uses_names_and_aliases():
    from backend import graph_rag
    data = graph_rag.load_graph_data()
    graph = graph_rag.NetworkxEntityGraph()
    for s, r, o in data["triples"]:
        graph.add_triple(graph_rag.KnowledgeTriple(s, r, o))
    assert graph_rag.find_entities("When is Dr Tan's workshop?", graph, data["aliases"]) == ["Dr Mei Lin Tan"]
