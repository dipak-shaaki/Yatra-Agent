"""
Agent-level tunables (model name, thresholds, limits). Files import these
instead of hardcoding their own copies.
"""

# Groq model used for every agent-related call (scope check, stage 2
# classification, tool loop, loop-break message generation).
AGENT_MODEL = "openai/gpt-oss-20b"

# In-scope destinations — single source of truth. Both the system prompt and
# the shared UNCLEAR_REPLY are built from this, so adding a destination
# updates every copy.
DESTINATION_NAMES = (
    "Manaslu Circuit",
    "Annapurna Base Camp",
    "Mardi Himal",
    "Kori",
    "Badimalika",
    "Bandipur",
    "Panauti",
    "Gorkha",
    "Rara Lake",
    "Tansen/Palpa",
)

# Agent loop
MAX_TOOL_ITERATIONS = 5

# Conversation memory
CONVERSATION_HISTORY_LIMIT = 12  # messages loaded into context per turn
MAX_STORED_HISTORY_MESSAGES = 20  # messages kept in Redis before trimming
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours of inactivity before a session expires

# Turn Handler
FILLER_LOOP_THRESHOLD = 3

FINAL_ANSWER_MAX_TOKENS = 400
