"""Graph RAG with LangChain: answer questions about how things connect.

Vector search finds text that LOOKS like the question. It struggles when the
answer is spread over several documents, e.g. "When is the workshop run by the
counsellor who speaks Mandarin?" needs three facts from two files:

    Mandarin  <-speaks-  Dr Mei Lin Tan  -runs->  Exam Stress workshop  -runs on->  Thursdays at 4pm

A knowledge graph stores those facts as (subject, relation, object) triples,
so the program can walk from one entity to the next.

  data/clinic_graph.json  ->  build_graph()        a LangChain NetworkxEntityGraph
  question                ->  find_entities()      which nodes does it mention?
                          ->  expand()             walk the graph, collect facts
                          ->  graph_search()       the facts, nearest first

LangChain parts used here:
  NetworkxEntityGraph, KnowledgeTriple      langchain_community.graphs.networkx_graph
"""
import json
import re
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.graphs.networkx_graph import KnowledgeTriple, NetworkxEntityGraph

GRAPH_FILE = Path(__file__).resolve().parent.parent / "data" / "clinic_graph.json"
HOPS = 3            # how far to walk from each entity in the question
MAX_FACTS = 12      # keep the prompt short


# ==========================================================================
# Given: loading the triples, and spotting entities in a question
# ==========================================================================
def load_graph_data(path=GRAPH_FILE):
    """{"triples": [[subject, relation, object], ...], "aliases": {node: [other names]}}"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_entities(question, graph, aliases=None):
    """Nodes the question mentions, by name or alias (whole words, any case)."""
    text = " " + " ".join(re.findall(r"[a-z0-9]+", question.lower())) + " "
    nodes = {n for s, o, _ in graph.get_triples() for n in (s, o)}
    found = []
    for node in sorted(nodes):
        names = [node] + list((aliases or {}).get(node, []))
        for name in names:
            words = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
            if words and f" {words} " in text:
                found.append(node)
                break
    return found


# ==========================================================================
# Activity 2: graph RAG
# ==========================================================================
def build_graph(triples):
    """2.1  Put every [subject, relation, object] into a LangChain NetworkxEntityGraph.

    - graph = NetworkxEntityGraph()
    - for each triple: graph.add_triple(KnowledgeTriple(subject, relation, obj))
    """
    return NetworkxEntityGraph()                         # TODO: add the triples


def expand(graph, entity, hops=HOPS):
    """2.2  Facts near the entity, as sentences like "Dr Mei Lin Tan runs Exam Stress workshop".

    graph.get_triples() gives [(subject, object, relation), ...]   (note the order)
    - start with frontier = {entity}
    - repeat `hops` times: for every triple,
        forward  = subject is in the frontier
        backward = it is the FIRST step and object is in the frontier
        if either: add f"{subject} {relation} {obj}" (once), and put the OTHER end
        in the next frontier
    - return the facts, nearest first
    """
    return []                                            # TODO


def graph_search(question, graph, aliases=None, hops=HOPS, max_facts=MAX_FACTS):
    """2.3  For every entity find_entities() spots in the question, expand() it.

    Collect the facts without repeats, and return at most max_facts.
    """
    return []                                            # TODO


# ==========================================================================
# Given: turning facts into text for the LLM
# ==========================================================================
def format_facts(facts):
    if not facts:
        return "none matched"
    return "\n".join(f"- {f}" for f in facts)
