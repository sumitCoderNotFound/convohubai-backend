"""
Phase 14 - delivery (cadence) and sentiment analytics. Pure, no LLM, testable.
Advisory only: never used to penalise candidates automatically.
"""
import re
from typing import List, Dict, Any

FILLERS = ["um", "uh", "er", "ah", "like", "you know", "sort of", "kind of",
           "basically", "actually", "literally", "i mean", "right"]

POSITIVE = {"good", "great", "excellent", "confident", "enjoy", "enjoyed", "love", "loved", "proud",
            "success", "successful", "achieved", "improve", "improved", "excited", "passionate",
            "strong", "happy", "positive", "win", "won", "best", "effective", "growth", "learned"}
NEGATIVE = {"bad", "difficult", "hard", "struggle", "struggled", "fail", "failed", "failure", "hate",
            "worried", "worry", "anxious", "stressed", "problem", "problems", "issue", "issues",
            "unfortunately", "never", "cant", "couldnt", "wrong", "weak", "confused", "frustrated"}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def count_fillers(text: str) -> int:
    t = " " + (text or "").lower() + " "
    n = 0
    for f in FILLERS:
        n += len(re.findall(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", t))
    return n


def cadence(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_words = 0
    total_seconds = 0.0
    fillers = 0
    for a in answers:
        txt = a.get("transcript_text") or ""
        total_words += len(_words(txt))
        fillers += count_fillers(txt)
        d = a.get("duration_seconds")
        if d:
            total_seconds += float(d)
    wpm = round(total_words / (total_seconds / 60.0)) if total_seconds > 0 else None
    filler_rate = round(fillers / total_words, 3) if total_words else 0.0
    if wpm is None:
        pace = "unknown"
    elif wpm < 110:
        pace = "measured"
    elif wpm <= 160:
        pace = "conversational"
    else:
        pace = "fast"
    return {
        "words_per_minute": wpm, "total_words": total_words,
        "total_seconds": round(total_seconds), "filler_count": fillers,
        "filler_rate": filler_rate, "pace_label": pace,
    }


def sentiment(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    pos = neg = 0
    for a in answers:
        for w in _words(a.get("transcript_text") or ""):
            if w in POSITIVE:
                pos += 1
            elif w in NEGATIVE:
                neg += 1
    total = pos + neg
    score = round((pos - neg) / total, 2) if total else 0.0
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    return {"sentiment_label": label, "sentiment_score": score, "positive_hits": pos, "negative_hits": neg}


def analyze(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {**cadence(answers), **sentiment(answers)}
