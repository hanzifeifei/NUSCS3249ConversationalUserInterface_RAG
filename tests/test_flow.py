"""Tutorial 1 checks, still true in Tutorial 3. They use a FAKE LLM (no network, no key).

Each test scripts what the "LLM" returns, then checks that the orchestrator
applies good decisions, rejects bad ones and never crashes.
"""
import json

import pytest
from conftest import run, say

from backend import llm
from backend import orchestrator as orch
from backend.content import (
    CRISIS_MESSAGE, NO_LLM_MESSAGE, QUESTIONS, STEPS, UNAVAILABLE_MESSAGE,
)
from backend.prompts import SYSTEM_PROMPT


# ---------------------------------------------------------------- state
def test_state_is_plain_dict_with_extra_fields():
    state = orch.new_state()
    json.dumps(state)
    for slot in STEPS:
        assert state[slot] is None
    assert state["current_step"] == "reason_for_visit"
    assert len(state) >= 7


def test_next_step_returns_first_empty_slot():
    state = orch.new_state()
    state.update(reason_for_visit="exams", severity="6")
    assert orch.next_step(state) == "duration"
    state.update(duration="3 weeks", previous_support="no")
    assert orch.next_step(state) == "done"


# ---------------------------------------------------------------- LLM decisions
def test_answer_fills_slot_and_uses_llm_reply(fake):
    fake(say(updates={"reason_for_visit": "exam stress"}, reply="Thanks for sharing. How long has this been going on?"))
    state = orch.new_state()
    reply = run(orch.handle("I've been stressed about exams", state))
    assert state["reason_for_visit"] == "exam stress"
    assert state["current_step"] == "duration"
    assert reply == "Thanks for sharing. How long has this been going on?"


def test_one_message_can_fill_two_slots(fake):
    fake(say(updates={"reason_for_visit": "exam stress", "duration": "about a month"}))
    state = orch.new_state()
    run(orch.handle("Stressed about exams for about a month", state))
    assert state["duration"] == "about a month"
    assert state["current_step"] == "severity"


def test_faq_keeps_the_pending_question(fake):
    fake(say(action="answer_faq", reply="Yes, it stays with the counselling team. How long have you felt this way?"))
    state = orch.new_state()
    state.update(reason_for_visit="exams", current_step="duration")
    run(orch.handle("Is this confidential?", state))
    assert state["duration"] is None
    assert state["current_step"] == "duration"
    assert state["last_action"] == "answer_faq"


def test_correction_updates_slot_without_advancing(fake):
    fake(say(action="update_info", updates={"duration": "2 months"}))
    state = orch.new_state()
    state.update(reason_for_visit="exams", duration="3 weeks", severity="6", current_step="previous_support")
    run(orch.handle("Actually it's been 2 months", state))
    assert state["duration"] == "2 months"
    assert state["current_step"] == "previous_support"


def test_skip_marks_slot_and_moves_on(fake):
    fake(say(action="skip", skipped=["duration"]))
    state = orch.new_state()
    state.update(reason_for_visit="exams", current_step="duration")
    run(orch.handle("I'd rather not say", state))
    assert state["duration"] == orch.DECLINED
    assert state["skipped"] == ["duration"]
    assert state["current_step"] == "severity"


def test_escalate_sends_fixed_crisis_message_and_changes_nothing_else(fake):
    fake(say(action="escalate", updates={"duration": "forever"}, reply="generated text"))
    state = orch.new_state()
    reply = run(orch.handle("I don't want to be here anymore", state))
    assert reply == CRISIS_MESSAGE
    assert state["risk_flag"] is True
    assert state["duration"] is None


def test_full_test_script(fake):
    fake(
        say(updates={"reason_for_visit": "stressed about exams"}),
        say(updates={"duration": "about 3 weeks"}),
        say(action="answer_faq"),
        say(updates={"severity": "6"}),
        say(action="update_info", updates={"duration": "2 months"}),
        say(action="answer_faq"),
        say(updates={"previous_support": "no, never"}, reply="Thank you, I've shared a summary."),
    )
    state, history = orch.new_state(), []
    for m in ["I've been stressed about exams", "About 3 weeks", "Is this confidential?",
              "Maybe a 6", "Actually it's 2 months", "What are your hours?", "No, never"]:
        run(orch.handle(m, state, history))
    assert state["duration"] == "2 months"
    assert state["previous_support"] == "no, never"
    assert state["current_step"] == "done"
    assert len(history) == 14


# ---------------------------------------------------------------- guards
def test_unknown_items_and_actions_are_ignored(fake):
    fake(say(action="delete_records", updates={"reason_for_visit": "exams", "home_address": "12 Main St"}))
    state = orch.new_state()
    run(orch.handle("exams, I live at 12 Main St", state))
    assert "home_address" not in state
    assert state["reason_for_visit"] == "exams"
    assert state["last_action"] == "other"


def test_empty_reply_falls_back_to_next_question(fake):
    fake(say(updates={"reason_for_visit": "exams"}, reply=""))
    state = orch.new_state()
    assert run(orch.handle("exams", state)) == QUESTIONS["duration"]


def test_llm_failure_leaves_state_and_history_unchanged(fake):
    fake(llm.LLMError("timeout"))
    state, history = orch.new_state(), []
    before = json.dumps(state)
    assert run(orch.handle("exams", state, history)) == UNAVAILABLE_MESSAGE
    state["last_action"] = None
    assert json.dumps(state) == before
    assert history == []


def test_no_llm_configured():
    state = orch.new_state()
    assert run(orch.handle("hello", state)) == NO_LLM_MESSAGE


def test_llm_receives_system_prompt_state_and_history(fake):
    f = fake(say(updates={"reason_for_visit": "exams"}), say(action="answer_faq"))
    state, history = orch.new_state(), []
    run(orch.handle("exams", state, history))
    run(orch.handle("is it free?", state, history))
    second = f.calls[1]
    assert second["system"] == SYSTEM_PROMPT
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert '"reason_for_visit": "exams"' in second["messages"][-1]["content"]
    assert json.loads(second["messages"][1]["content"])["action"] == "fill_slot"


def test_system_prompt_covers_the_essentials():
    for text in ["Safety comes first", "escalate", "Clinic knowledge", "use_tool", "JSON"] + STEPS:
        assert text in SYSTEM_PROMPT


def test_parse_json_tolerates_code_fences():
    assert llm.parse_json('```json\n{"action": "other"}\n```') == {"action": "other"}
    with pytest.raises(llm.LLMError):
        llm.parse_json("Sure! The action is fill_slot.")


# ---------------------------------------------------------------- web API
def test_api_round_trip(fake):
    fake(say(updates={"reason_for_visit": "exams"}, reply="How long have you felt this way?"))
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    assert client.post("/api/reset", json={"session_id": "t"}).status_code == 200
    body = client.post("/api/chat", json={"session_id": "t", "message": "exams"}).json()
    assert body["state"]["reason_for_visit"] == "exams"
    assert body["decided_by"] == "llm"
    assert body["reply"] == "How long have you felt this way?"
