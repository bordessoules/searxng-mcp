"""Centralized logging configuration for MCP Gateway."""

import logging
import os
import sys

# Log level from environment (default: INFO)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Create the main logger for the gateway
logger = logging.getLogger("mcp_gateway")
logger.setLevel(getattr(logging, _log_level, logging.INFO))

# Avoid duplicate handlers if module is reloaded
if not logger.handlers:
    # Console handler with formatted output
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, _log_level, logging.INFO))

    # Format: timestamp - module - level - message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (e.g., 'browser', 'http', 'search')

    Returns:
        A logger instance with the gateway's configuration
    """
    return logger.getChild(name)
