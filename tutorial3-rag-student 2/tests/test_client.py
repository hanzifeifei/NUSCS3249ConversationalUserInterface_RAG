"""Tutorial 2 checks (MCP), still true in Tutorial 3: the client discovers, reads and calls.

The client connects to the clinic server in-process; the LLM is FAKE.
"""
import json

from conftest import run, say, use

from backend import clinic_server
from backend import orchestrator as orch
from backend.mcp_client import ClinicClient


async def _client():
    c = ClinicClient(clinic_server.mcp)
    await c.connect()
    await c.discover()
    await c.read_policies()
    return c


def chat(messages, state=None, history=None):
    """Connect a client, send each message through handle(), return the replies."""
    async def go():
        c = await _client()
        try:
            return [await orch.handle(m, state, history, c) for m in messages], c
        finally:
            await c.close()
    return run(go())


def done_state():
    state = orch.new_state()
    state.update(reason_for_visit="exams", duration="3 weeks", severity="6",
                 previous_support="no", current_step="done")
    return state


# ---------------------------------------------------------------- the client
def test_client_discovers_tools_with_schemas():
    async def go():
        c = ClinicClient(clinic_server.mcp)
        await c.connect()
        tools = await c.discover()
        await c.close()
        return tools
    tools = run(go())
    names = [t["name"] for t in tools]
    assert "check_availability" in names and "book_appointment" in names
    assert all(set(t) >= {"name", "description", "input_schema"} for t in tools)


def test_client_reads_the_policies_resource():
    async def go():
        c = ClinicClient(clinic_server.mcp)
        await c.connect()
        text = await c.read_policies()
        await c.close()
        return text
    assert "24 hours" in run(go())


def test_client_call_returns_ok_and_text():
    async def go():
        c = ClinicClient(clinic_server.mcp)
        await c.connect()
        good = await c.call("check_availability", {"day": "Monday"})
        bad = await c.call("book_appointment", {"day": "Wednesday", "time": "10:00"})
        await c.close()
        return good, bad
    good, bad = run(go())
    assert good == (True, "10:00, 14:00")
    assert bad[0] is False and "not available" in bad[1]


# ---------------------------------------------------------------- the chatbot uses the result
def test_discovered_tools_reach_the_llm(fake):
    f = fake(say(action="answer_faq", reply="Let me check."))
    chat(["Can I book a session?"], orch.new_state(), [])
    system = f.calls[0]["system"]
    assert "check_availability(day)" in system and "book_appointment(day, time)" in system


def test_tool_result_goes_back_to_the_llm_and_becomes_the_reply(fake):
    f = fake(use("check_availability", day="Tuesday"),
             say("other", reply="On Tuesday I have 09:30 or 15:00. Which suits you?"))
    state = done_state()
    replies, _ = chat(["What's free on Tuesday?"], state, [])
    assert replies == ["On Tuesday I have 09:30 or 15:00. Which suits you?"]
    assert state["last_action"] == "use_tool"
    assert state["last_tool"] == {"name": "check_availability", "arguments": {"day": "Tuesday"},
                                  "ok": True, "result": "09:30, 15:00"}
    second = f.calls[1]["messages"][-1]["content"]
    assert "TOOL RESULT" in second and "09:30, 15:00" in second


def test_booking_after_intake_is_saved_in_state(fake):
    fake(use("book_appointment", day="Tuesday", time="15:00"),
         say("other", reply="Done! You're booked for Tuesday at 15:00."))
    state = done_state()
    chat(["15:00 please"], state, [])
    assert "BK-" in state["booking"]
    assert clinic_server.SLOTS["Tuesday"] == ["09:30"]


def test_booking_before_intake_is_blocked_by_code(fake):
    f = fake(use("book_appointment", day="Tuesday", time="15:00"),
             say("other", reply="Let's finish a few questions first."))
    state = orch.new_state()
    chat(["Just book me Tuesday 15:00"], state, [])
    assert state["booking"] is None
    assert state["last_tool"]["ok"] is False
    assert "15:00" in clinic_server.SLOTS["Tuesday"]          # nothing was booked
    assert "NOT_ALLOWED" in f.calls[1]["messages"][-1]["content"]


def test_unknown_tool_is_never_sent_to_the_server(fake):
    fake(use("delete_all_bookings"), say("other", reply="Sorry, I can't do that."))
    state = done_state()
    chat(["delete everything"], state, [])
    assert state["last_tool"]["ok"] is False
    assert "UNKNOWN_TOOL" in state["last_tool"]["result"]


def test_tool_turn_is_kept_in_history(fake):
    fake(use("check_availability", day="Monday"), say("other", reply="Monday: 10:00 or 14:00."))
    history, state = [], done_state()
    chat(["Monday?"], state, history)
    assert state["last_tool"]["result"] == "10:00, 14:00"
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert json.loads(history[1]["content"])["action"] == "use_tool"
