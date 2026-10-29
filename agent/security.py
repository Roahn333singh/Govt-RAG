"""Client identity, rate limiting and caller authentication."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

from fastapi import HTTPException, Request

SESSION_TOKEN_TTL_SECONDS = int(os.getenv("SESSION_TOKEN_TTL_SECONDS", "1800"))

_signing_key = os.getenv("SESSION_SIGNING_KEY") or ""
if not _signing_key:
    # Per-process key: tokens will not validate across instances.
    print("⚠️  SESSION_SIGNING_KEY not set — using an ephemeral key")
    _signing_key = secrets.token_urlsafe(32)
_SIGNING_KEY = _signing_key.encode()


def real_client_ip(request: Request) -> str:
    """Rightmost X-Forwarded-For hop: the address Cloud Run's front end saw.

    Clients can write anything into the left of XFF; only the last entry is
    appended by Google's infrastructure. If a Google HTTPS load balancer is
    ever placed in front of Cloud Run, it appends its own hop and the trusted
    client IP moves to ips[-2].
    """
    xff = request.headers.get("x-forwarded-for") or ""
    ips = [ip.strip() for ip in xff.split(",") if ip.strip()]
    if ips:
        return ips[-1]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _sign(payload: str) -> str:
    return hmac.new(_SIGNING_KEY, payload.encode(), hashlib.sha256).hexdigest()


def issue_session_token(client_ip: str, now: float | None = None) -> str:
    """`<expiry>.<hmac>` bound to the caller IP so a leaked token is less useful."""
    exp = int((now or time.time()) + SESSION_TOKEN_TTL_SECONDS)
    return f"{exp}.{_sign(f'{exp}|{client_ip}')}"


def verify_session_token(token: str, client_ip: str, now: float | None = None) -> bool:
    try:
        exp_s, sig = token.split(".", 1)
        exp = int(exp_s)
    except (ValueError, AttributeError):
        return False
    if exp < (now or time.time()):
        return False
    return hmac.compare_digest(sig, _sign(f"{exp}|{client_ip}"))


def _configured_api_keys() -> list[tuple[str, str]]:
    """(label, secret) pairs. Labels are safe to log; the secrets are not."""
    keys: list[tuple[str, str]] = []
    main = os.getenv("APP_API_KEY") or ""
    if main:
        keys.append(("main", main))
    qa = os.getenv("QA_API_KEY") or ""
    if qa:
        keys.append(("qa", qa))
    for part in (os.getenv("API_KEYS") or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, value = part.split(":", 1)
        label, value = label.strip(), value.strip()
        if label and value:
            keys.append((label, value))
    return keys


def api_key_label(provided: str | None) -> str | None:
    """Return the label of the matching key, or None. Uses hmac.compare_digest."""
    if not provided:
        return None
    provided_b = provided.encode()
    for label, expected in _configured_api_keys():
        expected_b = expected.encode()
        if len(provided_b) != len(expected_b):
            continue
        if hmac.compare_digest(provided_b, expected_b):
            return label
    return None


async def require_client(request: Request) -> None:
    """Accept X-API-Key (scripts) or a UI-issued X-Session-Token."""
    label = api_key_label(request.headers.get("x-api-key"))
    if label:
        request.state.key_id = label
        return
    token = request.headers.get("x-session-token")
    if token and verify_session_token(token, real_client_ip(request)):
        request.state.key_id = "session"
        return
    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key. Send X-API-Key, or use the web UI.",
    )


def admin_token_ok(provided: str | None) -> bool:
    expected = os.getenv("ADMIN_TOKEN") or ""
    if not expected or not provided:
        return False
    return secrets.compare_digest(provided.encode(), expected.encode())
