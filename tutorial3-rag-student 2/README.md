# Tutorial 3 Exercise: Vector RAG and Graph RAG with LangChain

The chatbot remembers the intake (Tutorial 1) and books through MCP (Tutorial 2). Now it should answer clinic questions from the clinic's own knowledge, as in Lecture 7:

- **Vector RAG** over seven documents in `docs/` (Activity 1, `backend/rag.py`)
- **Graph RAG** over a knowledge graph in `data/clinic_graph.json` (Activity 2, `backend/graph_rag.py`)

The orchestrator already sends both to the LLM with every message (**hybrid RAG**). You build the two retrievers.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt                       # now includes LangChain
```

Add your API key at the top of `backend/llm.py`, then run the app and open **http://localhost:8000**:

```bash
python -m uvicorn backend.app:app --reload
```

Click the **Test script** buttons. The RAG panel shows **0 chunks** and **0 facts**, and every clinic question gets "I don't have that information".

## Activity 1: Vector RAG (8 min) · `backend/rag.py`

1. **`split_docs(docs)`:** split each document by Markdown heading (`MarkdownHeaderTextSplitter`), then by size (`RecursiveCharacterTextSplitter`). Give each chunk a `source` and an `id` such as `policies.md#3`.
2. **`build_vector_store(chunks)`:** fit `KeywordEmbeddings` on the chunk texts, then add the chunks to an `InMemoryVectorStore`.
3. **`vector_search(query, store)`:** `similarity_search_with_score`, keep scores ≥ `min_score`, and return dicts.

**Check it:** the terminal and the RAG panel say **26 chunks from 7 documents**. `pytest` shows 50 passed.

## Activity 2: Graph RAG (12 min) · `backend/graph_rag.py`

1. **`build_graph(triples)`:** add every `[subject, relation, object]` to a `NetworkxEntityGraph` as a `KnowledgeTriple`.
2. **`expand(graph, entity, hops)`:** walk out from the entity. The first step goes either way along an edge; later steps go forward only. Return facts like `Dr Mei Lin Tan runs Exam Stress workshop`, nearest first.
3. **`graph_search(question, graph, aliases)`:** expand every entity that `find_entities()` spots in the question, without repeats, and return at most `max_facts`.

**Done when** "When is Dr Tan's workshop?" shows the graph path `Dr Mei Lin Tan runs Exam Stress workshop` → `runs on Thursdays at 4pm`, and `pytest` shows 57 passed.

## Check

```bash
pytest
```

You start at 40 passing. After Activity 1, 50 pass. When both activities are done, all 57 pass.

## Try this (Lecture 7)

- Compare the two panels for "Can my mum see my notes?". Which retriever found something, and why?
- Swap `KeywordEmbeddings` for a neural model, such as `OpenAIEmbeddings` from `langchain-openai`. Only one line in `build_vector_store()` changes.
- Rewrite-Retrieve-Read: before `vector_search`, ask the LLM to rewrite the question as a search query. Does retrieval improve?

## If something breaks

- **`No module named langchain_…`:** activate `.venv`, then run `pip install -r requirements.txt`.
- **Still 0 chunks or 0 facts:** check the terminal; a save with an error stops the app reloading.
- **`bad interpreter`:** delete `.venv` and redo the setup.
- **Errors in the chat:** read the terminal running uvicorn; the real error is there.

> This is a teaching prototype. The clinic, its staff and its documents are made up. Use only made-up messages.
