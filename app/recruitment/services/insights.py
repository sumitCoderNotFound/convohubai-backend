"""
Phase 13 - derive a human-readable 'key strengths / areas of concern' summary
from per-criterion scores. Pure and testable; no extra LLM cost.
"""
from typing import List, Dict, Any


def key_points(criteria: List[Dict[str, Any]], n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    criteria: [{name, raw_score, weight, evidence}]
    Returns {"strengths": [...], "concerns": [...]} each up to n items, richest first.
    Strengths are the highest-scoring criteria; concerns the lowest.
    """
    scored = [c for c in criteria if c.get("raw_score") is not None]
    if not scored:
        return {"strengths": [], "concerns": []}

    by_score = sorted(scored, key=lambda c: c["raw_score"], reverse=True)

    def pack(c):
        return {"name": c.get("name") or "Criterion", "score": round(c["raw_score"]),
                "evidence": c.get("evidence") or c.get("reasoning") or ""}

    strengths = [pack(c) for c in by_score if c["raw_score"] >= 60][:n]
    concerns = [pack(c) for c in reversed(by_score) if c["raw_score"] < 60][:n]

    # Fallbacks so there is always something useful to show.
    if not strengths:
        strengths = [pack(c) for c in by_score[:n]]
    if not concerns and len(by_score) > len(strengths):
        concerns = [pack(c) for c in by_score[::-1][:n]]
    return {"strengths": strengths, "concerns": concerns}
