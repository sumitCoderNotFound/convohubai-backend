"""
Drafts a competency profile + suggested rubric criteria from a job description.
Uses the existing LLMService. Degrades gracefully when no AI key is configured.
"""
import json
import re
from typing import Dict, Any

from app.services.llm_service import LLMService

_SYSTEM = (
    "You are a recruitment assistant. Extract a structured competency profile from a "
    "job description. Return ONLY valid JSON, no prose, no markdown fences. Schema: "
    '{"skills": [string], "responsibilities": [string], "experience": [string], '
    '"suggested_criteria": [{"name": string, "description": string, "weight": number}]}. '
    "Provide 4-6 suggested_criteria whose weights sum to 100. Do NOT include protected "
    "characteristics (age, gender, race, religion, disability, nationality) as criteria."
)


def _safe_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


async def parse_job_description(description: str) -> Dict[str, Any]:
    """Returns {competency_profile, suggested_criteria, source}."""
    fallback = {
        "competency_profile": {"skills": [], "responsibilities": [], "experience": []},
        "suggested_criteria": [],
        "source": "fallback",
    }
    try:
        llm = LLMService()
        raw = await llm.generate_response(
            messages=[{"role": "user", "content": description}],
            system_prompt=_SYSTEM,
            temperature=0.2,
            max_tokens=900,
        )
        data = _safe_json(raw)
        return {
            "competency_profile": {
                "skills": data.get("skills", []),
                "responsibilities": data.get("responsibilities", []),
                "experience": data.get("experience", []),
            },
            "suggested_criteria": data.get("suggested_criteria", []),
            "source": "ai",
        }
    except Exception:
        # No API key, malformed output, or provider error -> recruiter fills it in manually.
        return fallback
