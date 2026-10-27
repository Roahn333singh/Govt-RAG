"""Client identity, rate limiting and caller authentication."""

from __future__ import annotations

from fastapi import Request


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
