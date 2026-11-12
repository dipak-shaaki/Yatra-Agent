"""Log structured events through the shared 'yatra' logger.

`log_event("tool_called", tool="search_destinations", time_secs="0.12")`
becomes one JSON line via JsonFormatter, with request/session ids attached.
"""

import logging

logger = logging.getLogger("yatra")


def log_event(event: str, **fields) -> None:
    logger.info(event, extra={"event": event, "event_data": fields})
