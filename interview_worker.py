"""
ConvoHubAI - Recruitment Interview Voice Worker (LiveKit)

A separate LiveKit agent that conducts recruitment interviews by voice. It is the
candidate-side counterpart to agent_worker.py and reuses the same stack
(groq LLM + groq Whisper STT + deepgram TTS + silero VAD).

It handles rooms named `interview-{session_token}`:
  1. fetches the interview script from the backend (questions, disclosure, intro)
  2. greets the candidate, discloses it is an AI, and asks each question in turn
  3. captures the conversation transcript
  4. on end, posts each answer to the public /answers endpoint and calls /complete,
     which triggers scoring exactly like the typed flow.

Run it as a separate process (NOT inside the API container):
    pip install -r requirements-worker.txt
    python interview_worker.py dev      # connects to your LiveKit server

Requires env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, GROQ_API_KEY,
DEEPGRAM_API_KEY, and API_URL (backend base, default http://localhost:8000).
"""
import asyncio
import logging
import os
import httpx
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, RoomInputOptions
from livekit.plugins import silero, groq, deepgram

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-worker")

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_BASE = f"{API_URL}/api/v1/recruitment/public"


async def fetch_config(session_token: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{API_BASE}/sessions/{session_token}/agent-config")
        r.raise_for_status()
        return r.json()


async def post_answer(session_token: str, question_id: str, text: str):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(f"{API_BASE}/sessions/{session_token}/answers",
                              json={"question_id": question_id, "transcript_text": text})
    except Exception as e:
        logger.warning(f"post_answer failed: {e}")


async def complete_session(session_token: str):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(f"{API_BASE}/sessions/{session_token}/complete")
    except Exception as e:
        logger.warning(f"complete failed: {e}")


def build_instructions(cfg: dict) -> str:
    qs = cfg.get("questions", [])
    numbered = "\n".join(f"{i+1}. {q['prompt_text']}" for i, q in enumerate(qs))
    disclosure = cfg.get("ai_identity_disclosure") or "You are an AI interviewer, not a human."
    intro = cfg.get("introduction") or ""
    return (
        f"{disclosure}\n\n"
        "You are conducting a structured job interview by voice. Be warm, concise and professional. "
        "First greet the candidate and clearly state you are an AI interviewer. "
        f"{('Then say: ' + intro) if intro else ''}\n"
        "Ask EXACTLY these questions, ONE AT A TIME, in order. Wait for the candidate's full answer "
        "before moving on. You may ask one short natural follow-up if an answer is very thin, then move on. "
        "Do not invent extra questions. After the final question, thank the candidate and tell them the "
        "interview is complete.\n\n"
        f"Questions:\n{numbered}"
    )


async def entrypoint(ctx: JobContext):
    room_name = ctx.room.name
    logger.info(f"Joining room: {room_name}")
    await ctx.connect()

    if not room_name.startswith("interview-"):
        logger.info("Not an interview room; this worker only handles 'interview-*' rooms.")
        return

    session_token = room_name[len("interview-"):]
    try:
        cfg = await fetch_config(session_token)
    except Exception as e:
        logger.error(f"Could not fetch interview config: {e}")
        return

    questions = cfg.get("questions", [])
    instructions = build_instructions(cfg)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(model="aura-asteria-en"),
    )

    # Capture candidate (user) utterances in order so we can post them as answers.
    captured: list[str] = []
    state = {"idx": 0}  # how many answers posted live, maps to question order

    def _extract_text(obj):
        for attr in ("transcript", "text_content", "text"):
            v = getattr(obj, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        content = getattr(obj, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, (list, tuple)):
            parts = [c for c in content if isinstance(c, str)]
            if parts:
                return " ".join(parts).strip()
        return None

    def _post_live(text: str):
        """Save each candidate answer as it is spoken, while the session is still open."""
        i = state["idx"]
        if i < len(questions):
            state["idx"] = i + 1
            asyncio.create_task(post_answer(session_token, questions[i]["id"], text))

    # Primary: final speech-to-text of the candidate's turns.
    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        try:
            if getattr(ev, "is_final", True):
                text = _extract_text(ev)
                if text:
                    captured.append(text)
                    _post_live(text)
        except Exception as e:
            logger.warning(f"transcript capture failed: {e}")

    # Secondary: conversation items (only used if no transcripts were captured).
    @session.on("conversation_item_added")
    def _on_item(ev):
        try:
            item = getattr(ev, "item", ev)
            if getattr(item, "role", None) == "user":
                text = _extract_text(item)
                if text and (not captured or captured[-1] != text):
                    # Only drive live posting from transcripts; here just keep a backup copy.
                    if not captured:
                        captured.append(text)
        except Exception:
            pass

    await session.start(agent=Agent(instructions=instructions), room=ctx.room,
                        room_input_options=RoomInputOptions())

    # Kick off the interview.
    await session.generate_reply(instructions="Greet the candidate, disclose you are an AI, then ask the first question.")

    def _history_user_turns():
        turns = []
        try:
            hist = getattr(session, "history", None)
            items = getattr(hist, "items", None) or getattr(hist, "messages", None) or []
            for it in items:
                if getattr(it, "role", None) == "user":
                    t = _extract_text(it)
                    if t:
                        turns.append(t)
        except Exception:
            pass
        return turns

    async def _finalize():
        # If nothing was posted live, recover answers from the session history (or captured backup).
        if state["idx"] == 0:
            answers = _history_user_turns() or captured
            for i, q in enumerate(questions):
                if i < len(answers):
                    await post_answer(session_token, q["id"], answers[i])
            posted = min(len(answers), len(questions))
        else:
            posted = state["idx"]
        await complete_session(session_token)
        logger.info(f"Interview {session_token} finalized with {posted} answers.")

    ctx.add_shutdown_callback(_finalize)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
