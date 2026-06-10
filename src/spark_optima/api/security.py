# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Optional API-key authentication and rate limiting for the REST API.

Both mechanisms are opt-in via environment variables and read their
configuration per request (no import-time capture), so they can be enabled
or disabled without restarting and are easy to control from tests:

- ``SPARK_OPTIMA_API_KEYS``: comma-separated list of accepted API keys.
  When set, every ``/api/v1/*`` request must carry a matching ``X-API-Key``
  header. When unset, the API is fully open (previous behavior).
- ``SPARK_OPTIMA_RATE_LIMIT``: allowed requests per minute for ``/api/v1/*``
  endpoints, keyed by API key when authentication is enabled, otherwise by
  client IP. Unset, empty, or ``0`` disables rate limiting (the default).

Health endpoints are never subject to either mechanism.
"""

from __future__ import annotations

import logging
import math
import os
import secrets
import threading
import time

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

#: Header clients must send when API-key authentication is enabled.
API_KEY_HEADER_NAME = "X-API-Key"

#: Environment variable holding the comma-separated list of accepted keys.
API_KEYS_ENV_VAR = "SPARK_OPTIMA_API_KEYS"

#: Environment variable holding the requests-per-minute budget.
RATE_LIMIT_ENV_VAR = "SPARK_OPTIMA_RATE_LIMIT"

#: Fixed-window length used by the rate limiter, in seconds.
RATE_LIMIT_WINDOW_SECONDS = 60.0

#: Soft cap on tracked rate-limit windows before stale entries are pruned.
_MAX_TRACKED_WINDOWS = 1024

_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def get_configured_api_keys() -> list[str]:
    """Read the accepted API keys from the environment.

    The environment is read on every call so configuration changes and
    test monkeypatching take effect per request.

    Returns:
        List of non-empty API keys; empty list when auth is disabled.
    """
    raw = os.environ.get(API_KEYS_ENV_VAR, "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def get_configured_rate_limit() -> int:
    """Read the requests-per-minute budget from the environment.

    Returns:
        Allowed requests per minute, or 0 when rate limiting is disabled
        (unset, empty, non-numeric, or non-positive values).
    """
    raw = os.environ.get(RATE_LIMIT_ENV_VAR, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"Ignoring invalid {RATE_LIMIT_ENV_VAR} value: {raw!r}")
        return 0
    return max(value, 0)


def _api_key_is_valid(candidate: str, configured_keys: list[str]) -> bool:
    """Check a candidate key against all configured keys in constant time.

    Args:
        candidate: The key supplied by the client.
        configured_keys: Accepted keys from the environment.

    Returns:
        True if the candidate matches any configured key.
    """
    candidate_bytes = candidate.encode("utf-8")
    matched = False
    for key in configured_keys:
        # Compare every key to avoid early-exit timing differences
        if secrets.compare_digest(candidate_bytes, key.encode("utf-8")):
            matched = True
    return matched


class FixedWindowRateLimiter:
    """Simple thread-safe fixed-window request counter.

    Each client key gets a counter that resets when its window (60 seconds)
    expires. The implementation is intentionally minimal: in-memory,
    process-local, and accurate enough for basic abuse protection.

    Attributes:
        _window_seconds: Length of the fixed window in seconds.
        _windows: Mapping of client key to (window_start, request_count).
        _lock: Lock guarding the window mapping.
    """

    def __init__(self, window_seconds: float = RATE_LIMIT_WINDOW_SECONDS) -> None:
        """Initialize the limiter.

        Args:
            window_seconds: Fixed window length in seconds.
        """
        self._window_seconds = window_seconds
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> tuple[bool, int]:
        """Record a request and check it against the limit.

        Args:
            key: Client identity (API key or client IP).
            limit: Maximum allowed requests per window.

        Returns:
            Tuple of (allowed, retry_after_seconds). retry_after_seconds is
            0 when the request is allowed.
        """
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            window_start, count = self._windows.get(key, (now, 0))
            if now - window_start >= self._window_seconds:
                window_start, count = now, 0
            count += 1
            self._windows[key] = (window_start, count)
            if count > limit:
                retry_after = max(1, math.ceil(window_start + self._window_seconds - now))
                return False, retry_after
            return True, 0

    def reset(self) -> None:
        """Clear all tracked windows (intended for tests)."""
        with self._lock:
            self._windows.clear()

    def _prune_locked(self, now: float) -> None:
        """Drop expired windows once the mapping grows large.

        Must be called with the lock held.

        Args:
            now: Current epoch time in seconds.
        """
        if len(self._windows) <= _MAX_TRACKED_WINDOWS:
            return
        expired = [key for key, (start, _) in self._windows.items() if now - start >= self._window_seconds]
        for key in expired:
            del self._windows[key]


#: Shared limiter instance used by the security dependency.
_rate_limiter = FixedWindowRateLimiter()


def get_rate_limiter() -> FixedWindowRateLimiter:
    """Get the shared rate limiter instance.

    Returns:
        The module-level FixedWindowRateLimiter.
    """
    return _rate_limiter


def _client_identity(request: Request, api_key: str | None, auth_enabled: bool) -> str:
    """Determine the rate-limit key for a request.

    Args:
        request: The incoming request.
        api_key: The validated API key, if any.
        auth_enabled: Whether API-key auth is currently enabled.

    Returns:
        The API key when auth is enabled, otherwise the client IP.
    """
    if auth_enabled and api_key:
        return f"key:{api_key}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


async def enforce_api_security(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    """FastAPI dependency enforcing opt-in API-key auth and rate limiting.

    Applied to all ``/api/v1/*`` routers. Authentication is checked first
    (401 before any rate accounting), then the fixed-window rate limiter
    (429 with a Retry-After header). Both are no-ops when their environment
    variables are unset.

    Args:
        request: The incoming request.
        api_key: Value of the X-API-Key header, if present.

    Raises:
        HTTPException: 401 for a missing/invalid API key when auth is
            enabled; 429 when the rate limit is exceeded.
    """
    configured_keys = get_configured_api_keys()
    auth_enabled = bool(configured_keys)

    if auth_enabled and (api_key is None or not _api_key_is_valid(api_key, configured_keys)):
        # Same message for missing and wrong keys — do not leak which case it is
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    limit = get_configured_rate_limit()
    if limit > 0:
        identity = _client_identity(request, api_key, auth_enabled)
        allowed, retry_after = _rate_limiter.check(identity, limit)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(retry_after)},
            )
