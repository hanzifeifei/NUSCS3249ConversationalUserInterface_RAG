"""Tutorial 2 checks (MCP server), still true in Tutorial 3: 2 tools and 1 resource.

The tests talk to the server through a real MCP client, in-process (no subprocess).
"""
from conftest import run
from mcp import Client

from backend import clinic_server


async def _with_server(fn):
    async with Client(clinic_server.mcp) as c:
        return await fn(c)


def tools():
    async def go(c):
        return {t.name: t for t in (await c.list_tools()).tools}
    return run(_with_server(go))


def call(name, args):
    async def go(c):
        r = await c.call_tool(name, args)
        return r.is_error, r.content[0].text
    return run(_with_server(go))


def test_server_lists_both_tools():
    assert {"check_availability", "book_appointment"} <= set(tools())


def test_tools_have_descriptions_and_typed_arguments():
    t = tools()
    for name in ["check_availability", "book_appointment"]:
        assert t[name].description and len(t[name].description) > 20, f"{name}: write a docstring"
    assert set(t["check_availability"].input_schema["properties"]) == {"day"}
    assert set(t["book_appointment"].input_schema["properties"]) == {"day", "time"}


def test_check_availability_returns_open_slots():
    assert call("check_availability", {"day": "Tuesday"}) == (False, "09:30, 15:00")


def test_check_availability_full_day_says_no_slots():
    assert call("check_availability", {"day": "Wednesday"}) == (False, "NO_SLOTS")


def test_booking_removes_the_slot_and_returns_a_reference():
    is_error, text = call("book_appointment", {"day": "Tuesday", "time": "15:00"})
    assert not is_error and "BK-" in text
    assert call("check_availability", {"day": "Tuesday"}) == (False, "09:30")


def test_booking_a_taken_slot_is_an_error():
    is_error, text = call("book_appointment", {"day": "Wednesday", "time": "10:00"})
    assert is_error and "not available" in text


def test_policies_resource_can_be_read():
    async def go(c):
        uris = [str(r.uri) for r in (await c.list_resources()).resources]
        text = (await c.read_resource("clinic://policies")).contents[0].text
        return uris, text
    uris, text = run(_with_server(go))
    assert "clinic://policies" in uris
    assert "24 hours" in text
