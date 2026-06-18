"""
Explainable rubric scoring (Phase 2).

Design (per PRD): the LLM scores EACH criterion independently against its anchors and
returns evidence + confidence; the deterministic weighted TOTAL is computed here in app
code, never by the model. Quality gates flag low-confidence/thin transcripts for human
review. Degrades gracefully when no AI provider is configured.
"""
import json
import re
from typing import List, Dict, Any, Optional

from app.services.llm_service import LLMService

_SYSTEM = (
    "You are an impartial interview assessor. Score ONE criterion against its anchors using only "
    "the transcript evidence. Never infer or use protected characteristics (age, gender, race, "
    "religion, disability, nationality, accent). Return ONLY valid JSON: "
    '{"raw_score": <0-100 integer>, "confidence": <0-1 float>, "evidence": "<short quote/paraphrase>", '
    '"reasoning": "<one or two sentences>"}.'
)


def _safe_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(json)?", "", (text or "").strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e + 1]
    return json.loads(text)


def _transcript_text(answers: List[Any]) -> str:
    parts = []
    for a in answers:
        q = a.question_text_snapshot or "Question"
        parts.append(f"Q: {q}\nA: {a.transcript_text or '(no answer)'}")
    return "\n\n".join(parts)


def _recommendation(overall: float) -> str:
    if overall >= 70:
        return "strong"
    if overall >= 50:
        return "moderate"
    if overall > 0:
        return "weak"
    return "insufficient"


def _risk(answers: List[Any], behavioral: Dict[str, Any] = None) -> Dict[str, Any]:
    """Integrity heuristics combining answer patterns with client-side behavioural signals."""
    signals = {}
    answered = [a for a in answers if (a.transcript_text or "").strip()]
    if not answered:
        base = {"level": "high", "signals": {"no_answers": True}}
        # still fold in any behavioural signals
        b = behavioral or {}
        for k in ("tab_switches", "paste_count", "focus_loss", "copy_count"):
            if int(b.get(k, 0) or 0) > 0:
                base["signals"][k] = int(b[k])
        return base

    avg_words = sum(len((a.transcript_text or "").split()) for a in answered) / max(len(answered), 1)
    if avg_words < 8:
        signals["very_short_answers"] = round(avg_words, 1)
    fast = [a for a in answered if (a.duration_seconds or 0) and a.duration_seconds < 3]
    if fast:
        signals["implausibly_fast_answers"] = len(fast)

    # Behavioural signals captured by the candidate page.
    b = behavioral or {}
    tab = int(b.get("tab_switches", 0) or 0)
    paste = int(b.get("paste_count", 0) or 0)
    focus = int(b.get("focus_loss", 0) or 0)
    copy = int(b.get("copy_count", 0) or 0)
    if tab:
        signals["tab_switches"] = tab
    if paste:
        signals["paste_count"] = paste
    if focus:
        signals["focus_loss"] = focus
    if copy:
        signals["copy_count"] = copy

    level = "low"
    if signals:
        level = "medium"
    # Strong indicators escalate to high.
    if tab >= 3 or paste >= 3 or focus >= 5:
        level = "high"
    return {"level": level, "signals": signals}


async def score_criterion(llm: LLMService, criterion, anchors: Dict[str, str], transcript: str) -> Dict[str, Any]:
    anchor_text = "; ".join(f"{lvl}: {anchors.get(lvl, 'n/a')}" for lvl in ("weak", "moderate", "strong"))
    prompt = (
        f"Criterion: {criterion.name}\n"
        f"Description: {criterion.description or 'n/a'}\n"
        f"Anchors -> {anchor_text}\n"
        f"Evidence guidance: {criterion.evidence_instructions or 'n/a'}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Score this single criterion now."
    )
    raw = await llm.generate_response(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=_SYSTEM, temperature=0.1, max_tokens=300,
    )
    data = _safe_json(raw)
    score = float(data.get("raw_score", 0) or 0)
    score = max(0.0, min(100.0, score))
    return {
        "raw_score": score,
        "confidence": float(data.get("confidence", 0.5) or 0.5),
        "evidence": str(data.get("evidence", ""))[:2000],
        "reasoning": str(data.get("reasoning", ""))[:2000],
    }


async def score_session(session, criteria: List[Any], answers: List[Any]) -> Dict[str, Any]:
    """
    Returns a dict the API persists into InterviewScore + CriterionScore.
    Always returns a structured result, even on failure (status='failed').
    """
    transcript = _transcript_text(answers)
    risk = _risk(answers, getattr(session, "risk_signals", None))
    base = {
        "status": "failed", "overall_score": None, "recommendation": None,
        "summary": None, "quality_flag": None, "needs_human_review": True,
        "risk_level": risk["level"], "risk_signals": risk["signals"],
        "model_used": None, "error": None, "criterion_scores": [],
    }

    if not criteria:
        base["error"] = "No rubric criteria on this version."
        return base
    if not [a for a in answers if (a.transcript_text or "").strip()]:
        base["error"] = "No answers to score."
        return base

    try:
        llm = LLMService()
    except Exception as ex:  # pragma: no cover
        base["error"] = f"LLM unavailable: {ex}"
        return base

    crit_results = []
    total = 0.0
    confidences = []
    try:
        for idx, c in enumerate(criteria):
            anchors = {a.level: a.descriptor for a in (c.anchors or [])}
            try:
                r = await score_criterion(llm, c, anchors, transcript)
            except Exception:
                # One criterion failing shouldn't sink the whole score; flag low confidence.
                r = {"raw_score": 0.0, "confidence": 0.0, "evidence": "", "reasoning": "Scoring error for this criterion."}
            contribution = round(r["raw_score"] * (c.weight or 0) / 100.0, 2)
            total += contribution
            confidences.append(r["confidence"])
            crit_results.append({
                "criterion_id": c.id, "criterion_name": c.name, "weight": c.weight or 0,
                "order_index": idx, "raw_score": r["raw_score"], "weighted_contribution": contribution,
                "confidence": r["confidence"], "evidence": r["evidence"], "reasoning": r["reasoning"],
            })
    except Exception as ex:
        base["error"] = f"Scoring failed: {ex}"
        return base

    overall = round(total, 2)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    needs_review = avg_conf < 0.45 or risk["level"] == "high"
    quality_flag = "low_confidence" if avg_conf < 0.45 else None

    return {
        "status": "completed",
        "overall_score": overall,
        "recommendation": _recommendation(overall),
        "summary": f"Weighted score {overall}/100 across {len(criteria)} criteria (avg confidence {round(avg_conf, 2)}).",
        "quality_flag": quality_flag,
        "needs_human_review": bool(needs_review),
        "risk_level": risk["level"],
        "risk_signals": risk["signals"],
        "model_used": "llm_service_default",
        "error": None,
        "criterion_scores": crit_results,
    }
