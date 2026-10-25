"""The clinic's MCP server: a separate program the chatbot talks to over MCP.

It offers two TOOLS and one RESOURCE:
    check_availability(day)        tool      read-only: open slots on a weekday
    book_appointment(day, time)    tool      changes data: books one slot
    clinic://policies              resource  read-only text the app loads

Run it on its own, or test it with the MCP Inspector (needs Node.js):
    mcp dev backend/clinic_server.py
The chatbot starts it for you as a subprocess (see backend/mcp_client.py).

This file imports nothing from the chatbot: it stands for a service that the
clinic's IT team would run, which any MCP client could use.
"""
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("clinic-server")

# Pretend booking system: open counselling slots for this week.
SLOTS = {
    "Monday": ["10:00", "14:00"],
    "Tuesday": ["09:30", "15:00"],
    "Wednesday": [],
    "Thursday": ["11:00"],
    "Friday": ["13:00", "16:00"],
}
BOOKINGS = []

POLICIES = """Clinic policies
- Confidentiality: what students share stays with the counselling team, unless someone is at serious risk of harm.
- Cost: the first consultation is free for students.
- Hours: Monday to Friday, 9am to 5pm.
- Cancelling: cancel or reschedule at least 24 hours before your appointment.
- Emergencies: the clinic is not an emergency service. In a crisis, call your local crisis line or emergency services."""


# ==========================================================================
# Activity 1: tools and a resource
# ==========================================================================
@mcp.tool()
def check_availability(day: str) -> str:
    """List the open counselling slots on a weekday (Monday to Friday).

    Returns times like "09:30, 15:00", or NO_SLOTS if that day is full.
    """
    day = day.strip().capitalize()
    if day not in SLOTS:
        raise ToolError(f"Unknown day {day!r}. Use Monday to Friday.")
    return ", ".join(SLOTS[day]) or "NO_SLOTS"


@mcp.tool()
def book_appointment(day: str, time: str) -> str:
    """Book one counselling slot. Only call this after the student has said
    yes to this exact day and time. Returns a booking reference."""
    day = day.strip().capitalize()
    if time not in SLOTS.get(day, []):
        raise ToolError(f"{day} {time} is not available. Check availability first.")
    SLOTS[day].remove(time)
    ref = f"BK-{1001 + len(BOOKINGS)}"
    BOOKINGS.append({"ref": ref, "day": day, "time": time})
    return f"Booked {day} at {time}. Reference {ref}."


@mcp.resource("clinic://policies")
def policies() -> str:
    """Clinic policies: confidentiality, cost, hours, cancelling, emergencies."""
    return POLICIES


if __name__ == "__main__":
    mcp.run()          # stdio: messages travel over stdin and stdout
