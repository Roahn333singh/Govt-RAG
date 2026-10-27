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


def _api_key_ok(provided: str | None) -> bool:
    expected = os.getenv("APP_API_KEY") or ""
    if not expected or not provided:
        return False
    return secrets.compare_digest(provided.encode(), expected.encode())


async def require_client(request: Request) -> None:
    """Accept X-API-Key (scripts) or a UI-issued X-Session-Token."""
    if _api_key_ok(request.headers.get("x-api-key")):
        return
    token = request.headers.get("x-session-token")
    if token and verify_session_token(token, real_client_ip(request)):
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
