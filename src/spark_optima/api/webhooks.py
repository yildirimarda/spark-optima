# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Webhook callbacks for asynchronous optimization jobs.

When a client passes ``webhook_url`` to ``POST /api/v1/optimize/async``,
the API delivers a JSON notification to that URL once the job finishes
(completed *or* failed). Delivery runs on the job worker thread, after the
job state has been persisted, so webhook failures can never affect job
state — they are logged and recorded as ``webhook_status`` on the job.

Payload shape::

    {
        "job_id": "...",
        "status": "completed" | "failed",
        "submitted_at": "<UTC ISO>",
        "finished_at": "<UTC ISO>",
        "result": {...},   # only when completed
        "error": "..."     # only when failed
    }

Delivery uses httpx with a 10-second timeout and up to 3 attempts with
exponential backoff (1s, then 2s, doubling for any further attempt). The
sleeps run on the worker thread, which is already off the event loop.

SSRF guard: ``validate_webhook_url`` rejects non-http(s) schemes and URLs
whose hostname is an obvious internal target (localhost, loopback and
link-local addresses such as 127.0.0.0/8 and 169.254.169.254, the
unspecified addresses 0.0.0.0 / ::, and ``*.internal`` names). This is a
**best-effort, hostname-level** check — it does not resolve DNS, so a
public hostname that resolves to an internal address is not caught.
Deploy egress controls if that matters in your environment.
"""

from __future__ import annotations

import ipaddress
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from spark_optima.api.jobs import Job

logger = logging.getLogger(__name__)

#: Per-request timeout for webhook deliveries, in seconds.
WEBHOOK_TIMEOUT_SECONDS = 10.0

#: Maximum number of delivery attempts per webhook.
WEBHOOK_MAX_ATTEMPTS = 3

#: Allowed URL schemes for webhook targets.
ALLOWED_WEBHOOK_SCHEMES = ("http", "https")

#: Hostnames rejected outright by the SSRF guard (lowercase). IP literals
#: such as 0.0.0.0, ::1, and 127.0.0.0/8 are caught by the ipaddress check.
BLOCKED_WEBHOOK_HOSTNAMES = frozenset({"localhost"})

#: Hostname suffixes rejected by the SSRF guard (lowercase).
BLOCKED_WEBHOOK_HOST_SUFFIXES = (".internal", ".localhost")

#: Indirection over time.sleep so tests can avoid real delays.
_sleep = time.sleep


def _build_client() -> httpx.Client:
    """Build the HTTP client used for webhook deliveries.

    Kept as a separate factory so tests can monkeypatch it with a client
    backed by ``httpx.MockTransport`` (no real network).

    Returns:
        An httpx client with the webhook timeout applied.
    """
    return httpx.Client(timeout=WEBHOOK_TIMEOUT_SECONDS)


def validate_webhook_url(url: str) -> str:
    """Validate a webhook URL and apply the best-effort SSRF guard.

    The guard works on the URL hostname only — it rejects non-http(s)
    schemes and obvious internal targets (localhost, loopback/link-local
    IP literals, unspecified addresses, and ``*.internal`` names). It does
    **not** resolve DNS, so it cannot catch public names pointing at
    internal addresses.

    Args:
        url: The candidate webhook URL.

    Returns:
        The URL unchanged when it passes validation.

    Raises:
        ValueError: If the URL is malformed, uses a disallowed scheme, or
            targets a blocked host.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_WEBHOOK_SCHEMES:
        raise ValueError("webhook_url must use the http or https scheme")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("webhook_url must include a hostname")

    host = hostname.lower()
    if host in BLOCKED_WEBHOOK_HOSTNAMES or host.endswith(BLOCKED_WEBHOOK_HOST_SUFFIXES):
        raise ValueError(f"webhook_url host {hostname!r} is not allowed (internal target)")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url  # not an IP literal; hostname checks above already passed
    if address.is_loopback or address.is_link_local or address.is_unspecified:
        raise ValueError(f"webhook_url host {hostname!r} is not allowed (internal target)")
    return url


def build_webhook_payload(job: Job) -> dict[str, Any]:
    """Build the JSON payload delivered to the webhook URL.

    Args:
        job: The finished job record.

    Returns:
        Payload with job identity, status, and timestamps; ``result`` is
        included only when the job completed and ``error`` only when it
        failed.
    """
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "submitted_at": job.submitted_at,
        "finished_at": job.finished_at,
    }
    if job.result is not None:
        payload["result"] = job.result
    if job.error is not None:
        payload["error"] = job.error
    return payload


def deliver_webhook(url: str, payload: dict[str, Any]) -> bool:
    """POST a webhook payload with retries.

    Performs up to ``WEBHOOK_MAX_ATTEMPTS`` attempts with exponential
    backoff (1s, 2s, ...) between them, using plain ``time.sleep`` —
    delivery runs on the job worker thread, already off the event loop.
    Any 2xx response counts as delivered; other statuses and transport
    errors are retried.

    Args:
        url: The validated webhook URL.
        payload: JSON-serializable notification body.

    Returns:
        True when a 2xx response was received, False after all attempts
        failed. Never raises — failures only affect the recorded
        ``webhook_status``, not the job state.
    """
    for attempt in range(1, WEBHOOK_MAX_ATTEMPTS + 1):
        try:
            with _build_client() as client:
                response = client.post(url, json=payload)
            if response.is_success:
                logger.info(f"Webhook for job {payload.get('job_id')} delivered to {url} (attempt {attempt})")
                return True
            logger.warning(
                f"Webhook for job {payload.get('job_id')} got HTTP {response.status_code} from {url} "
                f"(attempt {attempt}/{WEBHOOK_MAX_ATTEMPTS})"
            )
        except httpx.HTTPError as exc:
            logger.warning(
                f"Webhook for job {payload.get('job_id')} failed to reach {url}: {exc} "
                f"(attempt {attempt}/{WEBHOOK_MAX_ATTEMPTS})"
            )
        if attempt < WEBHOOK_MAX_ATTEMPTS:
            _sleep(float(2 ** (attempt - 1)))  # 1s, 2s, 4s, ...
    logger.error(f"Webhook for job {payload.get('job_id')} undeliverable after {WEBHOOK_MAX_ATTEMPTS} attempts: {url}")
    return False
