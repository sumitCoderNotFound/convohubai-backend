"""
Generates a non-scored preview of an interview by simulating a candidate persona.
Never persists candidate data and never consumes credits (FR-QUE-005).
"""
from typing import List, Dict, Any

from app.services.llm_service import LLMService


async def simulate_interview(version, questions: List[Any], persona: str) -> List[Dict[str, Any]]:
    """Walk the questions in order; generate a plausible candidate answer for each."""
    turns: List[Dict[str, Any]] = []
    if version.introduction:
        turns.append({"role": "interviewer", "question_id": None, "text": version.introduction})

    llm = LLMService()
    history = ""
    for q in questions:
        turns.append({"role": "interviewer", "question_id": q.id, "text": q.prompt_text})
        answer = "[preview answer unavailable - no AI provider configured]"
        try:
            answer = await llm.generate_response(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Interview question: {q.prompt_text}\n"
                        f"Answer in 2-3 sentences as {persona}. Reply with the answer only."
                    ),
                }],
                system_prompt="You are role-playing a job candidate in a preview. Be concise and realistic.",
                temperature=0.6,
                max_tokens=160,
            )
            answer = (answer or "").strip()
        except Exception:
            pass
        turns.append({"role": "candidate", "question_id": q.id, "text": answer})
        history += f"Q: {q.prompt_text}\nA: {answer}\n"

    return turns
