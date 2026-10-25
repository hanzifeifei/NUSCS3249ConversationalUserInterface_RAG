import asyncio
import copy

import pytest

from backend import clinic_server, llm


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Tests never call a real LLM, even if your .env or llm.py has an API key."""
    monkeypatch.setenv("LLM_PROVIDER", "none")


@pytest.fixture(autouse=True)
def fresh_clinic():
    """Every test starts with the same open slots and no bookings."""
    slots, bookings = copy.deepcopy(clinic_server.SLOTS), list(clinic_server.BOOKINGS)
    yield
    clinic_server.SLOTS.clear()
    clinic_server.SLOTS.update(slots)
    clinic_server.BOOKINGS[:] = bookings


class FakeLLM:
    """Stands in for backend.llm.chat_json. Returns scripted decisions in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, system, messages, max_tokens=500):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake(monkeypatch):
    def install(*responses):
        f = FakeLLM(*responses)
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")   # pretend an LLM is configured
        monkeypatch.setattr(llm, "chat_json", f.chat_json)
        return f
    return install


def say(action="fill_slot", updates=None, skipped=None, reply="OK.", tool=None):
    return {"action": action, "updates": updates or {}, "skipped": skipped or [],
            "tool": tool, "reply": reply}


def use(name, **arguments):
    return say("use_tool", reply="", tool={"name": name, "arguments": arguments})


def run(coro):
    return asyncio.run(coro)
