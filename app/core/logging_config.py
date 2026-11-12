"""Structured JSON logging with request + session correlation ids.

Every log line is one JSON object on stdout: deployment log collectors can
parse it without regexes, and `request_id` / `session_id` (a short hash, never
the token itself) tie a turn's lines together. `log_event()` adds an `event`
name plus arbitrary fields; ordinary library log records still work.
"""

import json
import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
            payload.update(getattr(record, "event_data", {}))
        else:
            payload["message"] = record.getMessage()
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging() -> None:
    from app.core.config import get_settings

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(get_settings().log_level.upper())
    # uvicorn installs its own handlers; route them through ours too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
