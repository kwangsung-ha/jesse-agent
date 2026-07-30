"""Application-owned Loguru configuration for diagnostic CLI output."""

import sys

from loguru import logger

__all__ = ["configure_debug_logging", "logger"]

_FORMAT = (
    "<dim>{time:HH:mm:ss}</dim> | <level>{level:<5}</level> | "
    "<cyan>{extra[event]}</cyan> | {message}"
)

logger.disable("jesseagent")


def configure_debug_logging(*, debug: bool, verbose: bool) -> None:
    """Enable JesseAgent diagnostics for one CLI invocation when requested."""
    logger.remove()
    logger.disable("jesseagent")
    if not debug:
        return
    logger.enable("jesseagent")
    logger.configure(extra={"event": "-"})
    logger.add(
        sys.stderr,
        level="TRACE" if verbose else "DEBUG",
        format=_FORMAT,
        colorize=True,
    )
