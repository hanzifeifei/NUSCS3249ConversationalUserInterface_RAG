"""Small wrapper around the LLM APIs. All API calls go through this file.

Put your key in the block below, or in a .env file (see .env.example).
Every call can fail (no network, bad key, rate limit, timeout); callers must
catch LLMError.
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API keys: paste yours here. Leave a value as "" to use .env instead.
# Do NOT share this file with your key in it.
# ---------------------------------------------------------------------------
LLM_PROVIDER = ""          # "anthropic" or "openai" ("" = whichever key is set)
ANTHROPIC_API_KEY = ""     # your Claude key
OPENAI_API_KEY = ""        # your OpenAI key
OPENAI_BASE_URL = ""       # optional: OpenAI-compatible server (Gemini, Ollama)
ANTHROPIC_WORKSPACE_ID = ""  # only if Anthropic asks for an anthropic-workspace-id header
LLM_MODEL = ""             # optional: override the default model

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-6-luna",
}
TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT", "20"))


class LLMError(Exception):
    """Anything went wrong talking to the LLM."""


def provider():
    # An LLM_PROVIDER environment variable wins, so tests can switch the LLM off.
    chosen = (os.environ.get("LLM_PROVIDER") or LLM_PROVIDER).strip().lower()
    if chosen:
        return chosen
    if ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def enabled():
    return provider() in DEFAULT_MODELS


def model():
    return LLM_MODEL or os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider(), "")


def describe():
    return f"{provider()} · {model()}" if enabled() else "none (add an API key)"


_clients = {}


def _client(name):
    """Create the SDK client once and reuse it."""
    if name not in _clients:
        if name == "anthropic":
            import anthropic
            _clients[name] = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"),
                timeout=TIMEOUT_SECONDS, max_retries=1,
                default_headers={"anthropic-workspace-id": ws} if (
                    ws := ANTHROPIC_WORKSPACE_ID or os.environ.get("ANTHROPIC_WORKSPACE_ID")) else None,
            )
        elif name == "openai":
            import openai
            _clients[name] = openai.OpenAI(
                api_key=OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"),
                base_url=OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL") or None,
                timeout=TIMEOUT_SECONDS, max_retries=1,
            )
        else:
            raise LLMError(f"Unknown LLM_PROVIDER: {name!r}")
    return _clients[name]


def chat(system, messages, max_tokens=500, json_mode=False):
    """Send a system prompt plus a list of {"role", "content"} turns; return the reply text.

    messages alternate "user" / "assistant" and end with a "user" turn.
    """
    name = provider()
    if name not in DEFAULT_MODELS:
        raise LLMError("No LLM configured")
    try:
        if name == "anthropic":
            resp = _client(name).messages.create(
                model=model(), max_tokens=max_tokens, system=system, messages=messages,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        # openai (or OpenAI-compatible)
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = _client(name).chat.completions.create(
            model=model(), max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
            **kwargs,
        )
        return resp.choices[0].message.content or ""
    except LLMError:
        raise
    except Exception as e:  # network, auth, rate limit, timeout, bad model name...
        raise LLMError(f"{type(e).__name__}: {e}") from e


def chat_json(system, messages, max_tokens=500):
    """Like chat(), but parse the reply as a JSON object."""
    return parse_json(chat(system, messages, max_tokens, json_mode=True))


def parse_json(text):
    """Pull the first {...} object out of a reply, tolerating ```json fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise LLMError(f"No JSON object in reply: {text[:120]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise LLMError("JSON reply is not an object")
    return data
