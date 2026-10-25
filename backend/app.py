"""Web server: connects the chat frontend to the orchestrator.

Run it with:
    python -m uvicorn backend.app:app --reload
Then open http://localhost:8000

At startup the app indexes docs/ (vector RAG) and loads the knowledge graph
(graph RAG), then (as the
MCP HOST) starts the clinic's MCP server, connects its client and discovers the tools.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import graph_rag, llm, rag
from backend import orchestrator as orch
from backend.content import GREETING, NO_LLM_MESSAGE
from backend.mcp_client import ClinicClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
log = logging.getLogger("app")
log.info("LLM: %s", llm.describe())

CLINIC = ClinicClient()
STORE = None          # vector store of docs/ (rag.py), built at startup
GRAPH = None          # knowledge graph of data/clinic_graph.json (graph_rag.py)
ALIASES = {}


@asynccontextmanager
async def lifespan(app):
    global STORE, GRAPH, ALIASES
    try:   # vector RAG: index the clinic documents once
        docs = rag.load_docs()
        chunks = rag.split_docs(docs)
        STORE = rag.build_vector_store(chunks) if chunks else None
        log.info("Vector RAG: %d chunks from %d documents", len(chunks), len(docs))
    except Exception as e:  # the chat still works without retrieval
        STORE = None
        log.warning("Vector store not built: %s: %s", type(e).__name__, e)
    try:   # graph RAG: load the knowledge graph once
        data = graph_rag.load_graph_data()
        GRAPH, ALIASES = graph_rag.build_graph(data["triples"]), data.get("aliases", {})
        log.info("Graph RAG: %d facts, %d entities", len(GRAPH.get_triples()), GRAPH.get_number_of_nodes())
    except Exception as e:
        GRAPH = None
        log.warning("Knowledge graph not built: %s: %s", type(e).__name__, e)
    try:
        await CLINIC.connect()
        await CLINIC.discover()     # clinic facts now come from docs/ via RAG, not clinic://policies
        log.info("MCP: %d tools %s", len(CLINIC.tools), [t["name"] for t in CLINIC.tools])
    except Exception as e:  # the chat still works without the server
        log.warning("MCP setup incomplete: %s: %s", type(e).__name__, e)
    yield
    try:
        await CLINIC.close()
    except Exception:
        pass


app = FastAPI(title="Pre-consultation Chatbot", lifespan=lifespan)
# session_id -> {"state": conversation state, "history": turns sent to the LLM}
SESSIONS: dict[str, dict] = {}


class ChatIn(BaseModel):
    session_id: str
    message: str


class ResetIn(BaseModel):
    session_id: str


def new_session():
    return {"state": orch.new_state(), "history": []}


def mcp_info():
    return {
        "connected": CLINIC.connected,
        "tools": [{"name": t["name"], "description": t.get("description") or ""} for t in CLINIC.tools],
    }


def rag_info():
    docs = list(STORE.store.values()) if STORE is not None else []
    return {"chunks": len(docs),
            "documents": len({d["metadata"].get("source") for d in docs}),
            "facts": len(GRAPH.get_triples()) if GRAPH is not None else 0,
            "entities": GRAPH.get_number_of_nodes() if GRAPH is not None else 0}


@app.post("/api/reset")
def reset(body: ResetIn):
    SESSIONS[body.session_id] = new_session()
    greeting = GREETING if llm.enabled() else NO_LLM_MESSAGE
    return {"reply": greeting, "state": SESSIONS[body.session_id]["state"],
            "action": None, "decided_by": None, "llm": llm.describe(), "mcp": mcp_info(),
            "rag": rag_info()}


@app.post("/api/chat")
async def chat(body: ChatIn):
    session = SESSIONS.setdefault(body.session_id, new_session())
    state = session["state"]
    tool_before = state.get("last_tool")
    try:
        reply = await orch.handle(body.message, state, session["history"], CLINIC, STORE, GRAPH, ALIASES)
        action = state.get("last_action")
    except Exception as e:  # show errors in the chat instead of crashing
        logging.exception("Orchestrator error")
        reply, action = f"[Error in orchestrator: {type(e).__name__}: {e}]", "error"
    decided_by = "llm" if action not in ("no_llm", "unavailable", "error") else None
    tool = state.get("last_tool") if state.get("last_tool") is not tool_before else None
    return {"reply": reply, "state": state, "action": action, "decided_by": decided_by,
            "tool": tool, "mcp": mcp_info(), "rag": rag_info()}


# Serve the frontend (must come after the API routes)
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
