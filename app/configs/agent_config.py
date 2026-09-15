"""
Agent-level tunable parameters — thresholds, model names, limits.
Single source of truth: any file needing these imports from here instead
of hardcoding its own copy, so changing a model or threshold means editing
one line, not hunting across every file that happens to use it.
"""

# Groq model used for every agent-related call (scope check, stage 2
# classification, tool-calling loop, loop-break message generation).
# One name here — change once if you swap models.
AGENT_MODEL = "openai/gpt-oss-20b"

# Agent loop
MAX_TOOL_ITERATIONS = 5

# Conversation memory
CONVERSATION_HISTORY_LIMIT = 12  # messages loaded into context per turn
MAX_STORED_HISTORY_MESSAGES = 20  # messages kept in Redis before trimming
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours of inactivity before a session expires

# Turn Handler
FILLER_LOOP_THRESHOLD = 3

TOOL_SELECTION_MAX_TOKENS = 200
FINAL_ANSWER_MAX_TOKENS = 400
