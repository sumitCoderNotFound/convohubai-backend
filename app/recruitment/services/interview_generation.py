"""
Generates draft questions + rubric criteria from job/role context.
Used by the interview creator (FR-INT-001 / generate). Degrades gracefully.
"""
import json
import re
from typing import Dict, Any, List

from app.services.llm_service import LLMService

_SYSTEM = (
    "You design structured job interviews. Given role context, return ONLY valid JSON "
    "(no prose, no markdown). Schema: "
    '{"questions": [{"prompt_text": string, "question_type": "open_response"}], '
    '"criteria": [{"name": string, "description": string, "weight": number, '
    '"anchors": {"weak": string, "moderate": string, "strong": string}}]}. '
    "Criteria weights MUST sum to 100. Never include protected characteristics "
    "(age, gender, race, religion, disability, nationality) as criteria or questions."
)


def _safe_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s : e + 1]
    return json.loads(text)


async def generate_interview_content(context: str, num_questions: int) -> Dict[str, List[Dict[str, Any]]]:
    """Returns {questions: [...], criteria: [...], source: 'ai'|'fallback'}."""
    try:
        llm = LLMService()
        raw = await llm.generate_response(
            messages=[{"role": "user", "content": f"Role context:\n{context}\n\nProduce {num_questions} questions."}],
            system_prompt=_SYSTEM,
            temperature=0.3,
            max_tokens=1400,
        )
        data = _safe_json(raw)
        return {
            "questions": data.get("questions", [])[:num_questions],
            "criteria": data.get("criteria", []),
            "source": "ai",
        }
    except Exception:
        return {"questions": [], "criteria": [], "source": "fallback"}
