"""The chatbot's MCP client: one connection to the clinic's MCP server.

The chatbot (the HOST) holds one ClinicClient. The client starts the server as
a subprocess, then:
    discover()        asks the server which tools it offers     (tools/list)
    read_policies()   reads the clinic://policies resource      (resources/read)
    call()            runs one tool and returns its result      (tools/call)

The client does not know the tool names in advance. It asks.
"""
import logging
import sys
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters

log = logging.getLogger("mcp_client")

# How to start the server: `python -m backend.clinic_server`, talking over stdio.
CLINIC_SERVER = StdioServerParameters(command=sys.executable, args=["-m", "backend.clinic_server"])
POLICIES_URI = "clinic://policies"
MAX_RESULT_CHARS = 1000


class ClinicClient:
    def __init__(self, server=CLINIC_SERVER):
        self.server = server          # StdioServerParameters, or an MCPServer object in tests
        self.tools = []               # [{"name", "description", "input_schema"}], filled by discover()
        self.policies = ""            # text of clinic://policies, filled by read_policies()
        self.connected = False
        self._client = None
        self._stack = None

    async def connect(self):
        """Start the server and do the MCP handshake. (Given.)"""
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(self.server))
        self.connected = True

    async def close(self):
        if self._stack:
            await self._stack.aclose()
        self.connected, self.tools, self.policies = False, [], ""

    # ======================================================================
    # Activity 2: discover, read, call
    # ======================================================================
    async def discover(self):
        """Ask the server for its tools and keep name, description and input schema."""
        result = await self._client.list_tools()
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in result.tools
        ]
        return self.tools

    async def read_policies(self):
        """Read the clinic://policies resource into self.policies."""
        result = await self._client.read_resource(POLICIES_URI)
        self.policies = result.contents[0].text
        return self.policies

    async def call(self, name, arguments):
        """Run one tool. Return (ok, text): ok is False if the server reported an error."""
        result = await self._client.call_tool(name, arguments)
        text = result.content[0].text if result.content else ""
        return (not result.is_error), text[:MAX_RESULT_CHARS]
