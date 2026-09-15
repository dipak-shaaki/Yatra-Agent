"""
Structured logging helper: logs as
[timestamp] [LEVEL] [name] event_name | key=value | key=value
"""

import logging

logger = logging.getLogger("yatra")


def log_event(event: str, **fields) -> None:
    parts = " | ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"{event} | {parts}" if parts else event)
