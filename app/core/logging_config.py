import logging
import sys

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO


def configure_logging() -> None:
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
