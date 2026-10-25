"""Tutorial 3: RAG  —  STUDENT VERSION. Your work today is in backend/rag.py and backend/graph_rag.py; this file is given.

New in Tutorial 3: before asking the LLM, the program retrieves knowledge two
ways and sends both with the message (hybrid RAG):
  vector search over the clinic's documents   (backend/rag.py, LangChain vector store)
  graph search over the clinic's knowledge    (backend/graph_rag.py, LangChain graph)
The LLM answers clinic questions only from what was retrieved.

From Tutorial 2:

Builds on Tutorial 1. The LLM still handles the conversation content and picks
one action per message. New today: the action can be "use_tool". Then the
program runs that tool on the clinic's MCP server and gives the result back to
the LLM, which writes the reply:

    Thought (LLM picks a tool) -> Action (MCP tools/call) -> Observation (result back to the LLM)

This file keeps what the LLM should NOT own:
  - the conversation STATE (Tutorial 1), shown live in the browser
  - VALIDATION of every LLM decision, including every tool call
  - the fixed crisis message when the LLM escalates

The LLM proposes. This code decides what is allowed, what runs and what is remembered.
"""
import asyncio
import json
import logging

from backend import graph_rag, llm, rag
from backend.content import (
    CRISIS_MESSAGE, DONE_MESSAGE, NO_LLM_MESSAGE, QUESTIONS, STEPS,
    UNAVAILABLE_MESSAGE,
)
from backend.prompts import TOOL_RESULT_TEMPLATE, TURN_TEMPLATE, build_system_prompt, tool_args

log = logging.getLogger("orchestrator")

ACTIONS = {"fill_slot", "answer_faq", "update_info", "skip", "escalate", "use_tool", "other"}
DECLINED = "(prefers not to say)"
MAX_VALUE_CHARS = 200
MAX_REPLY_CHARS = 800
HISTORY_TURNS = 12          # messages of history sent to the LLM
TOOL_TIMEOUT = 10           # seconds before we give up on a tool call
NEEDS_INTAKE_DONE = {"book_appointment"}   # tools with side effects


# ==========================================================================
# Conversation state (Tutorial 1, plus two fields for today)
# ==========================================================================
def new_state():
    """The bot's memory for one conversation. A plain, JSON-serializable dict."""
    return {
        # slots: what the counsellor needs
        "reason_for_visit": None,
        "duration": None,
        "severity": None,
        "previous_support": None,
        # progress: which question is pending
        "current_step": "reason_for_visit",
        # flags: conditions that change the flow
        "risk_flag": False,
        "skipped": [],
        # debugging: what the LLM decided on the last turn
        "last_action": None,
        # new in Tutorial 2
        "last_tool": None,      # {"name", "arguments", "ok", "result"} of the last tool call
        "booking": None,        # the booking confirmation, once booked
        # new in Tutorial 3
        "retrieved": [],        # [{"id", "score", "text"}] passages sent with the last message
        "graph_facts": [],      # knowledge-graph facts sent with the last message
    }


def next_step(state):
    """First slot that is still empty, or 'done'."""
    for slot in STEPS:
        if state[slot] is None:
            return slot
    return "done"


def turn_message(message, state, passages=(), facts=()):
    """The user turn sent to the LLM: current state + retrieved passages + the student's message."""
    collected = {s: state[s] for s in STEPS if state[s] is not None}
    missing = [s for s in STEPS if state[s] is None]
    return TURN_TEMPLATE.format(
        collected=json.dumps(collected) if collected else "nothing yet",
        missing=", ".join(missing) or "nothing, the intake is complete",
        skipped=", ".join(state["skipped"]) or "none",
        booking=state["booking"] or "none yet",
        passages=rag.format_passages(passages),
        facts=graph_rag.format_facts(facts),
        message=message,
    )


def validate(raw):
    """Never trust the LLM blindly: keep only known actions, item names, sizes and tool shapes."""
    action = raw.get("action") if raw.get("action") in ACTIONS else "other"
    updates = raw.get("updates") if isinstance(raw.get("updates"), dict) else {}
    skipped = raw.get("skipped") if isinstance(raw.get("skipped"), list) else []
    reply = raw.get("reply") if isinstance(raw.get("reply"), str) else ""
    tool = raw.get("tool")
    if isinstance(tool, dict) and isinstance(tool.get("name"), str):
        args = tool.get("arguments") if isinstance(tool.get("arguments"), dict) else {}
        tool = {"name": tool["name"], "arguments": args}
    else:
        tool = None
    if action == "use_tool" and tool is None:
        action = "other"
    return {
        "action": action,
        "updates": {
            k: str(v).strip()[:MAX_VALUE_CHARS]
            for k, v in updates.items()
            if k in STEPS and v is not None and str(v).strip()
        },
        "skipped": [s for s in skipped if s in STEPS],
        "tool": tool if action == "use_tool" else None,
        "reply": reply.strip()[:MAX_REPLY_CHARS],
    }


def check_tool(tool, state, client):
    """Code-side rules for tool calls. Return an error message, or None if allowed."""
    if client is None or not client.connected:
        return "NOT_CONNECTED: the clinic server is not connected."
    if tool["name"] not in [t["name"] for t in client.tools]:
        return f"UNKNOWN_TOOL: {tool['name']} is not offered by the clinic server."
    if tool["name"] in NEEDS_INTAKE_DONE and state["current_step"] != "done":
        return "NOT_ALLOWED: finish the intake questions before booking."
    return None


async def ask_llm(system, messages):
    """One LLM decision. The LLM call blocks, so run it in a thread."""
    raw = await asyncio.to_thread(llm.chat_json, system, messages)
    return validate(raw)


# ==========================================================================
# Activity 2 (part 2): run the tool, pass the result back to the LLM
# ==========================================================================
async def run_tool(tool, state, client):
    """Check the call, run it on the MCP server, remember it. Return (ok, result)."""
    error = check_tool(tool, state, client)
    if error:
        ok, result = False, error
    else:
        try:
            ok, result = await asyncio.wait_for(
                client.call(tool["name"], tool["arguments"]), TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            ok, result = False, "TIMEOUT: the clinic server took too long."
        except Exception as e:  # server crashed, connection lost...
            log.warning("Tool call failed: %s", e)
            ok, result = False, f"ERROR: {type(e).__name__}"
    state["last_tool"] = {"name": tool["name"], "arguments": tool["arguments"], "ok": ok, "result": result}
    if ok and tool["name"] == "book_appointment":
        state["booking"] = result
    return ok, result


async def handle(message, state, history=None, client=None, store=None, graph=None, aliases=None):
    """Run one turn and return the bot's reply.

    state   : the conversation state (updated in place)
    history : list of earlier {"role", "content"} turns (updated in place)
    client  : the ClinicClient (MCP), or None if no server is connected
    store   : the vector store of the clinic documents (rag.py), or None
    graph   : the clinic knowledge graph (graph_rag.py), or None; aliases: other names of its nodes
    """
    history = history if history is not None else []

    if not llm.enabled():
        state["last_action"] = "no_llm"
        return NO_LLM_MESSAGE

    system = build_system_prompt(client.tools if client else [])
    # Retrieval (local, no LLM call): vector search + graph search = hybrid RAG.
    passages = rag.vector_search(message, store) if store is not None else []
    facts = graph_rag.graph_search(message, graph, aliases) if graph is not None else []
    state["retrieved"] = [{"id": p["id"], "score": p["score"], "text": p["text"]} for p in passages]
    state["graph_facts"] = facts
    turn = [{"role": "user", "content": turn_message(message, state, passages, facts)}]
    try:
        d = await ask_llm(system, history[-HISTORY_TURNS:] + turn)
        state["last_action"] = d["action"]

        if d["action"] == "escalate":
            state["risk_flag"] = True
            reply = CRISIS_MESSAGE                 # fixed text, never generated

        elif d["action"] == "use_tool":
            ok, result = await run_tool(d["tool"], state, client)
            # Observation: give the result back to the LLM, which writes the reply.
            turn += [
                {"role": "assistant", "content": json.dumps(d)},
                {"role": "user", "content": TOOL_RESULT_TEMPLATE.format(
                    name=d["tool"]["name"], arguments=tool_args(d["tool"]["arguments"]),
                    ok=ok, result=result)},
            ]
            d = await ask_llm(system, history[-HISTORY_TURNS:] + turn)
            reply = d["reply"] or result

        else:
            for slot, value in d["updates"].items():
                state[slot] = value
            for slot in d["skipped"]:
                if state[slot] is None:
                    state[slot] = DECLINED
                    state["skipped"].append(slot)
            state["current_step"] = next_step(state)
            nxt = state["current_step"]
            reply = d["reply"] or (QUESTIONS[nxt] if nxt != "done" else DONE_MESSAGE)
    except llm.LLMError as e:
        log.warning("LLM call failed: %s", e)
        state["last_action"] = "unavailable"
        return UNAVAILABLE_MESSAGE                 # history left unchanged

    # Remember the turn. Assistant turns are stored as the JSON the LLM
    # returned, so the model keeps answering in the same format.
    history.extend(turn)
    history.append({"role": "assistant", "content": json.dumps({**d, "reply": reply})})
    return reply
