"""Token generation + jurisdiction-aware default consent text (Phase 2)."""
import secrets

CONSENT_VERSION = "1.0"

_DEFAULT_CONSENT = {
    "uk": (
        "This interview is conducted by an AI system. Your responses (and, if enabled, an "
        "audio recording) will be processed to assess your suitability for the role and stored "
        "by the hiring organisation. Processing is carried out under UK GDPR. You can request "
        "access to or deletion of your data. By continuing you confirm you are speaking with an "
        "AI, not a human, and consent to this processing."
    ),
    "eu": (
        "This interview is conducted by an AI system. Your responses (and, if enabled, an audio "
        "recording) will be processed under the EU GDPR to assess your suitability for the role "
        "and stored by the hiring organisation. This is an automated assessment; you have the "
        "right to request human review, access, or deletion of your data. By continuing you "
        "confirm you are speaking with an AI and consent to this processing."
    ),
    "us": (
        "This interview is conducted by an AI system. Your responses (and, if enabled, an audio "
        "recording) will be processed to assess your suitability for the role and stored by the "
        "hiring organisation. By continuing you confirm you are speaking with an AI, not a human, "
        "and consent to this processing."
    ),
    "other": (
        "This interview is conducted by an AI system. Your responses (and, if enabled, a "
        "recording) will be processed to assess your suitability for the role and stored by the "
        "hiring organisation. By continuing you confirm you are speaking with an AI and consent "
        "to this processing."
    ),
}


def new_token(nbytes: int = 24) -> str:
    return secrets.token_urlsafe(nbytes)


def default_consent_text(jurisdiction: str) -> str:
    return _DEFAULT_CONSENT.get((jurisdiction or "other").lower(), _DEFAULT_CONSENT["other"])
