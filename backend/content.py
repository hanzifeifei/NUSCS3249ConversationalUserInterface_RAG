"""Fixed content for the pre-consultation chatbot.

The LLM reads this through the system prompt (backend/prompts.py).
Edit it to change what the bot asks. Clinic policies now come from the MCP
server (the clinic://policies resource), not from this file.
"""

# The information we collect, in the order we ask for it.
STEPS = ["reason_for_visit", "duration", "severity", "previous_support"]

QUESTIONS = {
    "reason_for_visit": "What brings you here today?",
    "duration": "How long have you been feeling this way?",
    "severity": "On a scale of 1 to 10, how much is it affecting your daily life?",
    "previous_support": "Have you had any counselling or support before?",
}

# What each slot means, for the LLM.
SLOT_DESCRIPTIONS = {
    "reason_for_visit": "why the student wants counselling",
    "duration": "how long they have felt this way",
    "severity": "how much it affects daily life, ideally a number from 1 to 10",
    "previous_support": "any previous counselling or support",
}

GREETING = (
    "Hi, I'm the pre-consultation assistant. I'll ask a few short questions "
    "so your counsellor can prepare. " + QUESTIONS["reason_for_visit"]
)

DONE_MESSAGE = (
    "Thank you. I've shared a summary with your counsellor. "
    "Would you like me to find you an appointment time?"
)

# Always sent, word for word, when the LLM decides to escalate.
CRISIS_MESSAGE = (
    "Thank you for telling me. It sounds like you're going through something "
    "really hard, and you deserve support right now. Please contact your local "
    "crisis line or emergency services. I'm also flagging this so a counsellor "
    "can reach out to you as soon as possible."
)

NO_LLM_MESSAGE = (
    "This chatbot needs an LLM to run. Add an API key in backend/llm.py "
    "(or in a .env file), then restart the server."
)

UNAVAILABLE_MESSAGE = (
    "Sorry, I'm having trouble connecting right now. "
    "Please send your message again in a moment."
)
