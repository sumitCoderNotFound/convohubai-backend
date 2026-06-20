"""
Determines the next question in a session, honouring branch rules.
Branch rule shape (stored on a question):
  condition: { type: "always" | "contains" | "equals" | "knockout", value: <str> }
  action:    { action: "end" | "skip_to" | "knockout" | "continue", target_question_id: <uuid?> }
"""
from typing import List, Optional


def _condition_matches(condition: dict, answer_text: Optional[str]) -> bool:
    ctype = (condition or {}).get("type", "always")
    val = str((condition or {}).get("value", "")).lower()
    text = (answer_text or "").lower()
    if ctype in ("always", "knockout"):
        return True
    if ctype == "contains":
        return val in text
    if ctype == "equals":
        return text.strip() == val.strip()
    return False


def next_index_after(questions: List, current_index: int, answer_text: Optional[str], branch_rules_by_q: dict):
    """
    Returns (next_index, ended, knockout).
    next_index is an index into `questions` (ordered) or None when the interview ends.
    """
    if current_index >= len(questions):
        return None, True, False

    q = questions[current_index]
    rules = branch_rules_by_q.get(str(q.id), [])
    for rule in rules:
        if _condition_matches(rule.condition, answer_text):
            action = (rule.action or {}).get("action", "continue")
            if action in ("end", "knockout") or (rule.condition or {}).get("type") == "knockout":
                return None, True, action == "knockout" or (rule.condition or {}).get("type") == "knockout"
            if action == "skip_to":
                target = (rule.action or {}).get("target_question_id")
                for i, qq in enumerate(questions):
                    if str(qq.id) == str(target):
                        return i, False, False

    nxt = current_index + 1
    if nxt >= len(questions):
        return None, True, False
    return nxt, False, False


def resolve_question_index(questions, question_id, fallback_idx):
    """
    Map a submitted question_id to its position in the ordered question list.
    Falls back to the session's current index if the id isn't found.
    This makes answer storage robust to retries and out-of-order posts.
    """
    if question_id is not None:
        target = str(question_id)
        for i, q in enumerate(questions):
            if str(q.id) == target:
                return i
    return fallback_idx
