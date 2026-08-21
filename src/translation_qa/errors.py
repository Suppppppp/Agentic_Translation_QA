"""Application-specific exceptions with user-safe messages."""

from __future__ import annotations


class TranslationQAError(RuntimeError):
    """Base class for expected application failures."""


class ComponentUnavailableError(TranslationQAError):
    """Raised when an optional local model or runtime is unavailable."""


class ComponentExecutionError(TranslationQAError):
    """Raised when a configured component fails while processing a request."""


class ConstraintApplicationError(ComponentExecutionError):
    """Raised when lexical constraints produce an unusable translation."""


class DatasetNotFoundError(TranslationQAError):
    """Raised when a registered benchmark dataset does not exist."""
