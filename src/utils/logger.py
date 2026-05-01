"""Logging utilities."""

import logging


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up logger with consistent format."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)
