"""Log structured key=value events via the shared 'yatra' logger."""

import logging

logger = logging.getLogger("yatra")


def log_event(event: str, **fields) -> None:
    parts = " | ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"{event} | {parts}" if parts else event)
