"""Per-request context (request ID, session ID) shared by logging and error responses."""

from __future__ import annotations

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")
