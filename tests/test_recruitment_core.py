"""
Phase 11 - automated tests for the pure recruitment logic (no DB required).
Run: pytest backend/tests -q
"""
from datetime import datetime, timedelta

from app.recruitment.services.invite_access import evaluate_access, attempts_remaining, mask_email
from app.recruitment.services.session_flow import next_index_after
from app.recruitment.services.scoring import _risk, _recommendation

NOW = datetime(2026, 6, 18, 12, 0, 0)


# ---------- invite access ----------
def test_access_ok_targeted_matching_email():
    ok, code, _ = evaluate_access(status="pending", invite_email="a@x.com", expires_at=None,
                                  max_attempts=1, completed_count=0, now=NOW, candidate_email="A@X.com")
    assert ok and code == "ok"


def test_access_email_mismatch():
    ok, code, _ = evaluate_access(status="pending", invite_email="a@x.com", expires_at=None,
                                  max_attempts=1, completed_count=0, now=NOW, candidate_email="b@x.com")
    assert not ok and code == "email_mismatch"


def test_access_expired():
    ok, code, _ = evaluate_access(status="pending", invite_email=None, expires_at=NOW - timedelta(days=1),
                                  max_attempts=1, completed_count=0, now=NOW)
    assert not ok and code == "expired"


def test_access_no_attempts():
    ok, code, _ = evaluate_access(status="pending", invite_email=None, expires_at=None,
                                  max_attempts=2, completed_count=2, now=NOW)
    assert not ok and code == "no_attempts"


def test_access_revoked():
    ok, code, _ = evaluate_access(status="revoked", invite_email=None, expires_at=None,
                                  max_attempts=1, completed_count=0, now=NOW)
    assert not ok and code == "revoked"


def test_access_reusable_allows_any_email():
    ok, _, _ = evaluate_access(status="pending", invite_email=None, expires_at=None,
                               max_attempts=5, completed_count=1, now=NOW, candidate_email="anyone@x.com")
    assert ok


def test_attempts_remaining_and_mask():
    assert attempts_remaining(3, 1) == 2
    assert attempts_remaining(1, 5) == 0
    assert mask_email("john@example.com") == "j***@example.com"
    assert mask_email("") == ""


# ---------- session flow / branching ----------
class _Q:
    def __init__(self, i): self.id = i


class _Rule:
    def __init__(self, cond, act): self.condition = cond; self.action = act


def test_flow_linear():
    qs = [_Q("a"), _Q("b"), _Q("c")]
    assert next_index_after(qs, 0, "answer", {}) == (1, False, False)


def test_flow_last_ends():
    qs = [_Q("a"), _Q("b")]
    assert next_index_after(qs, 1, "done", {}) == (None, True, False)


def test_flow_knockout():
    qs = [_Q("a"), _Q("b")]
    rules = {"a": [_Rule({"type": "knockout"}, {"action": "knockout"})]}
    nxt, ended, knockout = next_index_after(qs, 0, "x", rules)
    assert ended and knockout and nxt is None


def test_flow_skip_to():
    qs = [_Q("a"), _Q("b"), _Q("c")]
    rules = {"a": [_Rule({"type": "contains", "value": "senior"}, {"action": "skip_to", "target_question_id": "c"})]}
    assert next_index_after(qs, 0, "I am senior", rules) == (2, False, False)


# ---------- scoring helpers ----------
def test_recommendation_bands():
    assert _recommendation(85) == "strong"
    assert _recommendation(55) == "moderate"
    assert _recommendation(20) == "weak"
    assert _recommendation(0) == "insufficient"


class _A:
    def __init__(self, t, d=30): self.transcript_text = t; self.duration_seconds = d


def test_risk_clean_low():
    r = _risk([_A("a detailed and substantive answer with plenty of words to look genuine")], {})
    assert r["level"] == "low"


def test_risk_behavioural_high():
    r = _risk([_A("a detailed and substantive answer with plenty of words here")], {"tab_switches": 4})
    assert r["level"] == "high" and r["signals"].get("tab_switches") == 4


def test_risk_no_answers_high():
    assert _risk([], {})["level"] == "high"


# ---------- answer idempotency / sequence resolver (Phase 11 voice hardening) ----------
from app.recruitment.services.session_flow import resolve_question_index


def test_resolve_by_submitted_id():
    qs = [_Q("a"), _Q("b"), _Q("c")]
    assert resolve_question_index(qs, "b", 0) == 1
    assert resolve_question_index(qs, "c", 0) == 2


def test_resolve_falls_back_when_unknown():
    qs = [_Q("a"), _Q("b")]
    assert resolve_question_index(qs, "zzz", 1) == 1
    assert resolve_question_index(qs, None, 0) == 0


# ---------- pre-screening evaluation (Phase 11) ----------
from app.recruitment.services.prescreen import evaluate_prescreen

_PS = [
    {"id": "q1", "qtype": "yes_no", "knockout": {"op": "equals", "value": "no"}, "required": True},
    {"id": "q2", "qtype": "number", "knockout": {"op": "min", "value": 3}, "required": True},
    {"id": "q3", "qtype": "single_select", "knockout": {"op": "in", "value": ["intern"]}, "required": False},
]


def test_prescreen_eligible():
    ok, failed = evaluate_prescreen(_PS, {"q1": "yes", "q2": "5", "q3": "senior"})
    assert ok and failed == []


def test_prescreen_knockout_equals():
    ok, failed = evaluate_prescreen(_PS, {"q1": "no", "q2": "5"})
    assert not ok and "q1" in failed


def test_prescreen_knockout_min():
    ok, failed = evaluate_prescreen(_PS, {"q1": "yes", "q2": "1"})
    assert not ok and "q2" in failed


def test_prescreen_required_missing():
    ok, failed = evaluate_prescreen(_PS, {"q1": "yes"})
    assert not ok and "q2" in failed


def test_prescreen_optional_blank_ok():
    ok, failed = evaluate_prescreen(_PS, {"q1": "yes", "q2": "4"})
    assert ok and failed == []


# ---------- document storage validation (Phase 12) ----------
import pytest
from app.recruitment.services import storage


def test_storage_accepts_pdf():
    storage.validate("cv.pdf", "application/pdf", 5000)  # no exception


def test_storage_rejects_bad_type():
    with pytest.raises(storage.StorageError):
        storage.validate("malware.exe", "application/octet-stream", 100)


def test_storage_rejects_oversize():
    with pytest.raises(storage.StorageError):
        storage.validate("big.pdf", "application/pdf", storage.MAX_BYTES + 1)


# ---------- insights: key strengths / concerns (Phase 13) ----------
from app.recruitment.services.insights import key_points

_CRIT = [
    {"name": "Communication", "raw_score": 82, "weight": 30, "evidence": "Clear and structured."},
    {"name": "Technical depth", "raw_score": 45, "weight": 40, "evidence": "Shallow on trade-offs."},
    {"name": "Ownership", "raw_score": 70, "weight": 30, "evidence": "Drove the project."},
]


def test_key_points_splits_strengths_and_concerns():
    kp = key_points(_CRIT)
    names_s = [s["name"] for s in kp["strengths"]]
    names_c = [c["name"] for c in kp["concerns"]]
    assert "Communication" in names_s and "Ownership" in names_s
    assert "Technical depth" in names_c


def test_key_points_empty():
    kp = key_points([])
    assert kp["strengths"] == [] and kp["concerns"] == []


def test_key_points_all_high_has_no_concerns():
    kp = key_points([{"name": "A", "raw_score": 80, "evidence": "x"}, {"name": "B", "raw_score": 75, "evidence": "y"}])
    assert len(kp["strengths"]) >= 1 and kp["concerns"] == []


# ---------- speech analytics (Phase 14) ----------
from app.recruitment.services.speech_analytics import cadence, sentiment, count_fillers


def test_cadence_wpm_and_pace():
    ans = [{"transcript_text": "one two three four five six", "duration_seconds": 6}]  # 6 words / 0.1 min = 60 wpm
    c = cadence(ans)
    assert c["total_words"] == 6 and c["words_per_minute"] == 60 and c["pace_label"] == "measured"


def test_cadence_handles_no_duration():
    c = cadence([{"transcript_text": "hello world", "duration_seconds": None}])
    assert c["words_per_minute"] is None and c["total_words"] == 2


def test_fillers_counted():
    assert count_fillers("Um, you know, like, basically yes") >= 3


def test_sentiment_positive_and_negative():
    assert sentiment([{"transcript_text": "I loved it, great success, proud"}])["sentiment_label"] == "positive"
    assert sentiment([{"transcript_text": "It was a failure, stressful and difficult"}])["sentiment_label"] == "negative"
    assert sentiment([{"transcript_text": "The meeting was on Tuesday"}])["sentiment_label"] == "neutral"
