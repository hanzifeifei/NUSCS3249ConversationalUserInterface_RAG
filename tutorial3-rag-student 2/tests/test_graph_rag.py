"""Activity 2 checks: graph RAG with LangChain (build, expand, search), then hybrid.

No LLM and no network: the graph is a LangChain NetworkxEntityGraph, the LLM is FAKE.
"""
import pytest
from conftest import run, say

from backend import graph_rag as g
from backend import orchestrator as orch
from backend import rag

TRIPLES = [["Ann", "runs", "Yoga class"], ["Yoga class", "runs on", "Mondays"],
           ["Ann", "speaks", "French"], ["Ben", "speaks", "French"]]


@pytest.fixture(scope="module")
def clinic():
    data = g.load_graph_data()
    return g.build_graph(data["triples"]), data["aliases"]


def test_build_graph_stores_every_triple():
    graph = g.build_graph(TRIPLES)
    assert sorted(graph.get_triples()) == sorted((s, o, r) for s, r, o in TRIPLES)
    assert graph.get_number_of_nodes() == 5


def test_expand_follows_edges_forward():
    facts = g.expand(g.build_graph(TRIPLES), "Ann", hops=2)
    assert facts[:2] == ["Ann runs Yoga class", "Ann speaks French"]
    assert "Yoga class runs on Mondays" in facts


def test_expand_first_step_also_goes_backward():
    facts = g.expand(g.build_graph(TRIPLES), "French", hops=2)
    assert "Ann speaks French" in facts and "Ben speaks French" in facts
    assert "Ann runs Yoga class" in facts                       # then forward from Ann


def test_expand_respects_hops():
    assert g.expand(g.build_graph(TRIPLES), "Ann", hops=1) == ["Ann runs Yoga class", "Ann speaks French"]


def test_graph_search_answers_a_multi_hop_question(clinic):
    graph, aliases = clinic
    facts = g.graph_search("When is Dr Tan's workshop?", graph, aliases)
    assert "Dr Mei Lin Tan runs Exam Stress workshop" in facts
    assert "Exam Stress workshop runs on Thursdays at 4pm, weeks 9 to 12" in facts
    assert len(facts) <= g.MAX_FACTS and len(facts) == len(set(facts))


def test_graph_search_finds_who_speaks_a_language(clinic):
    graph, aliases = clinic
    assert "Arjun Pillai speaks Tamil" in g.graph_search("Do you have counsellors who speak Tamil?", graph, aliases)
    assert g.graph_search("Do you have a gym?", graph, aliases) == []


def test_hybrid_sends_passages_and_facts_together(fake, clinic):
    graph, aliases = clinic
    store = rag.build_vector_store(rag.split_docs(rag.load_docs()))
    f = fake(say(action="answer_faq", reply="Thursdays at 4pm (knowledge graph)."))
    state = orch.new_state()
    run(orch.handle("When is Dr Tan's workshop?", state, [], None, store, graph, aliases))
    sent = f.calls[0]["messages"][-1]["content"]
    assert "(vector search)" in sent and "[counsellors.md#" in sent
    assert "- Exam Stress workshop runs on Thursdays at 4pm" in sent
    assert state["graph_facts"] and state["retrieved"]
