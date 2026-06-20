"""
Pure invite-access decision logic (Phase 11). No DB, fully unit-testable.

Conventions (no schema change):
- An invite with an email set is a TARGETED invite, locked to that email.
- An invite with no email is a REUSABLE link (any candidate may use it).
- expires_at doubles as the completion deadline.
- max_attempts caps the number of COMPLETED sessions.
"""
from datetime import datetime
from typing import Optional, Tuple


def mask_email(email: Optional[str]) -> str:
    if not email or "@" not in email:
        return ""
    name, domain = email.split("@", 1)
    head = name[0] if name else ""
    return f"{head}***@{domain}"


def evaluate_access(
    *,
    status: str,
    invite_email: Optional[str],
    expires_at: Optional[datetime],
    max_attempts: int,
    completed_count: int,
    now: Optional[datetime] = None,
    candidate_email: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Returns (allowed, code, message).
    code is a stable machine string: ok | revoked | expired | email_mismatch | no_attempts | completed
    """
    now = now or datetime.utcnow()

    if status == "revoked":
        return False, "revoked", "This invite has been revoked."
    if expires_at and now > expires_at:
        return False, "expired", "This invite has expired."

    targeted = bool(invite_email)
    if targeted and candidate_email is not None:
        if (candidate_email or "").strip().lower() != invite_email.strip().lower():
            return False, "email_mismatch", f"This invite is for {mask_email(invite_email)}. Please use that email address."

    if completed_count >= max(max_attempts, 1):
        return False, "no_attempts", "No interview attempts remain for this invite."

    return True, "ok", "ok"


def attempts_remaining(max_attempts: int, completed_count: int) -> int:
    return max(max(max_attempts, 1) - completed_count, 0)
