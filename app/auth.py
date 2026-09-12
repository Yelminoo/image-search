"""
Single-user login gate.

Deliberately simple — one hardcoded account (ADMIN_EMAIL/ADMIN_PASSWORD
in .env), not a user table, since this app has exactly one operator.
Closes the "no auth on any endpoint" gap flagged since the start of this
project (see README's "Known gaps").

Sessions are an in-memory token -> created_at map (same pattern as
app/jobs.py's job registry) — not persisted, not shared across processes.
A server restart logs everyone out; fine for a single local operator, not
appropriate if this ever runs as multiple replicas behind a load balancer.

Credential comparison uses hmac.compare_digest (constant-time) rather than
== , to avoid leaking how much of the input matched via response-timing —
cheap to do right even though the practical risk is low for a local app.
"""
import hmac
import secrets
import time
from typing import Dict, Optional

from app.config import settings

SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 hours

_sessions: Dict[str, float] = {}  # token -> created_at (unix time)


def is_configured() -> bool:
    return bool(settings.admin_email and settings.admin_password)


def verify_credentials(email: str, password: str) -> bool:
    if not is_configured():
        return False
    email_ok = hmac.compare_digest(email.strip().lower(), settings.admin_email.strip().lower())
    password_ok = hmac.compare_digest(password, settings.admin_password)
    return email_ok and password_ok


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time()
    return token


def is_valid_session(token: Optional[str]) -> bool:
    if not token:
        return False
    created = _sessions.get(token)
    if created is None:
        return False
    if time.time() - created > SESSION_TTL_SECONDS:
        del _sessions[token]
        return False
    return True


def destroy_session(token: Optional[str]) -> None:
    if token:
        _sessions.pop(token, None)
