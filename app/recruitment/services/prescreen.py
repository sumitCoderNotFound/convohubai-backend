"""
Pure pre-screening / knockout evaluation (Phase 11). No DB, fully unit-testable.

Each question may carry a knockout rule. A question "fails" (makes the candidate
ineligible) when the answer matches the failing condition, or when a required
question is left blank.

knockout = {"op": "equals|not_equals|in|not_in|min|max", "value": ...}
"""
from typing import List, Dict, Tuple, Any


def _to_number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fails(knockout: Dict[str, Any], value: Any) -> bool:
    if not knockout or "op" not in knockout:
        return False
    op = knockout.get("op")
    target = knockout.get("value")
    if value is None or value == "":
        return False  # missing handled separately by `required`
    sval = str(value).strip().lower()
    if op == "equals":
        return sval == str(target).strip().lower()
    if op == "not_equals":
        return sval != str(target).strip().lower()
    if op == "in":
        return sval in [str(x).strip().lower() for x in (target or [])]
    if op == "not_in":
        return sval not in [str(x).strip().lower() for x in (target or [])]
    if op == "min":
        n = _to_number(value)
        return n == n and n < _to_number(target)   # fail if below minimum
    if op == "max":
        n = _to_number(value)
        return n == n and n > _to_number(target)   # fail if above maximum
    return False


def evaluate_prescreen(questions: List[Dict[str, Any]], answers: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    questions: [{id, prompt, qtype, knockout, required}]
    answers:   {question_id: value}
    Returns (eligible, failed_question_ids).
    """
    failed: List[str] = []
    for q in questions:
        qid = str(q.get("id"))
        value = answers.get(qid)
        if q.get("required") and (value is None or str(value).strip() == ""):
            failed.append(qid)
            continue
        if _fails(q.get("knockout") or {}, value):
            failed.append(qid)
    return (len(failed) == 0, failed)
