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
import random
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


LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "ja": "Japanese", "zh": "Chinese",
    "ar": "Arabic", "ru": "Russian",
}

# Deepgram Aura voices are English-first; for other languages STT still works, but TTS
# voice quality varies. This is a known limitation noted in VOICE_SETUP.md.
# Male interviewer voice (Deepgram Aura). Other male options: aura-arcas-en, aura-perseus-en, aura-angus-en, aura-zeus-en.
TTS_VOICE_BY_LANG = {"en": "aura-orion-en"}


def language_name(code: str) -> str:
    return LANG_NAMES.get((code or "en").lower(), code or "English")


def build_agent_instructions(cfg: dict) -> str:
    """Baseline persona for the agent (used by greeting + follow-ups)."""
    disclosure = cfg.get("ai_identity_disclosure") or "You are an AI interviewer, not a human."
    lang = language_name(cfg.get("language", "en"))
    return (
        f"{disclosure} You are a warm, concise, professional job interviewer. "
        f"Conduct the entire interview in {lang}. Keep every turn short. "
        "Speak only what you are asked to speak; do not volunteer extra questions."
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
    lang_code = (cfg.get("language") or "en").lower()
    lang = language_name(lang_code)
    disclosure = cfg.get("ai_identity_disclosure") or "I am an AI interviewer, not a human."
    intro = cfg.get("introduction") or ""

    # STT language hint (Whisper auto-detects, but a hint helps for non-English).
    try:
        stt = groq.STT(model="whisper-large-v3-turbo", language=lang_code) if lang_code != "en" else groq.STT(model="whisper-large-v3-turbo")
    except Exception:
        stt = groq.STT(model="whisper-large-v3-turbo")
    tts_voice = TTS_VOICE_BY_LANG.get(lang_code, "aura-orion-en")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(model=tts_voice),
    )

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

    # Queue of final candidate transcripts, used to drive a precise question-by-question loop.
    answer_q: "asyncio.Queue[str]" = asyncio.Queue()
    all_user_turns: list[str] = []
    ai_speaking = {"v": False}  # drop transcripts captured while the AI talks (echo)

    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        try:
            if ai_speaking["v"]:
                return  # ignore our own audio bleeding into the mic
            if getattr(ev, "is_final", True):
                text = _extract_text(ev)
                if text:
                    all_user_turns.append(text)
                    answer_q.put_nowait(text)
        except Exception as e:
            logger.warning(f"transcript capture failed: {e}")

    await session.start(agent=Agent(instructions=build_agent_instructions(cfg)), room=ctx.room,
                        room_input_options=RoomInputOptions())

    async def collect_answer(first_timeout: float, grace: float = 4.0) -> str:
        """Wait for the candidate to speak, then gather continuation until a longer pause.
        A 4s grace lets people pause to think mid-answer without being cut off."""
        try:
            first = await asyncio.wait_for(answer_q.get(), timeout=first_timeout)
        except asyncio.TimeoutError:
            return ""
        parts = [first]
        while True:
            try:
                nxt = await asyncio.wait_for(answer_q.get(), timeout=grace)
                parts.append(nxt)
            except asyncio.TimeoutError:
                break
        return " ".join(parts).strip()

    async def say_safe(text: str, settle: float = 0.4):
        """Speak, while muting candidate transcript capture to avoid echo."""
        ai_speaking["v"] = True
        try:
            await session.say(text)
        finally:
            await asyncio.sleep(settle)
            # drain anything captured during/just after our speech, then re-open
            while not answer_q.empty():
                try: answer_q.get_nowait()
                except Exception: break
            ai_speaking["v"] = False

    # quick keyword heuristics as a safety net if the LLM classifier is unavailable
    def _heuristic_type(text: str) -> str:
        t = (text or "").lower().strip()
        if not t or len(t) < 2:
            return "noise_or_invalid"
        if any(k in t for k in ["repeat", "come again", "say again", "didn't hear", "didnt hear", "once more"]):
            return "repeat_request"
        if any(k in t for k in ["what do you mean", "don't understand", "dont understand", "not getting you", "explain", "what are you asking", "can't understand", "cant understand", "come back"]):
            return "clarification_request"
        if any(k in t for k in ["audible", "mic", "microphone", "can't hear", "cant hear", "hear me", "disturb", "breaking up", "connection"]):
            return "technical_issue"
        if t in ("hi", "hello", "hey", "yes", "no", "okay", "ok", "yeah", "sure", "ready"):
            return "small_talk"
        return "answer"

    async def classify_candidate_turn(text: str, question: str) -> dict:
        """Classify a candidate turn. Returns {type, is_valid_answer, reason, suggested_reply}."""
        fallback = {"type": _heuristic_type(text), "is_valid_answer": _heuristic_type(text) == "answer" and len(text.split()) >= 4, "reason": "heuristic", "suggested_reply": ""}
        key = os.environ.get("GROQ_API_KEY")
        if not key or not (text or "").strip():
            return fallback
        sys = (
            "You classify a candidate's spoken turn during a job interview. "
            "Reply ONLY with strict JSON: {\"type\": one of "
            "[answer, clarification_request, repeat_request, not_understood, technical_issue, small_talk, noise_or_invalid, off_topic], "
            "\"is_valid_answer\": boolean, \"reason\": string}. "
            "answer = a genuine attempt to answer the question (even if short). "
            "clarification_request = asks what the question means. repeat_request = asks to repeat. "
            "technical_issue = audio/mic problems. small_talk = greetings/yes/ok with no substance. "
            "noise_or_invalid = gibberish, song lyrics, or unrelated background speech. off_topic = unrelated."
        )
        usr = f"Question: {question}\nCandidate said: \"{text}\"\nClassify it."
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": "llama-3.3-70b-versatile", "temperature": 0,
                          "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}]},
                )
                r.raise_for_status()
                import json as _json
                data = _json.loads(r.json()["choices"][0]["message"]["content"])
                data.setdefault("type", fallback["type"])
                data["is_valid_answer"] = bool(data.get("is_valid_answer", data["type"] == "answer"))
                return data
        except Exception as e:
            logger.warning(f"classify failed ({e}); using heuristic.")
            return fallback

    async def explain_question(question: str):
        """Briefly explain a question in plain terms, then re-ask it."""
        key = os.environ.get("GROQ_API_KEY")
        explanation = None
        if key:
            try:
                async with httpx.AsyncClient(timeout=12) as client:
                    r = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": "llama-3.3-70b-versatile", "temperature": 0.3,
                              "messages": [
                                  {"role": "system", "content": f"You are a warm interviewer. In {lang}, explain what the question is asking in one or two plain sentences with a small example. Do not answer it for them."},
                                  {"role": "user", "content": question}]},
                    )
                    r.raise_for_status()
                    explanation = r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"explain failed ({e}).")
        if explanation:
            await say_safe(f"Of course. {explanation}")
        else:
            await say_safe("Of course. Let me put it more simply.")
        await say_safe(f"So, {question}")

    posted = 0
    deterministic_ok = True
    ACKS = ["Thanks, that gives me a clearer picture.", "That's helpful, thank you.",
            "Good, I understand.", "Thanks for explaining that.", "Appreciate that."]
    LEADS = ["Let me ask you this.", "Here's the next one.", "Let's move to the next question.", "Next question for you."]
    try:
        # Warm greeting + format + readiness check.
        cand_name = (getattr(cfg, "candidate_name", None) or "").strip() if hasattr(cfg, "candidate_name") else ""
        role_name = (getattr(cfg, "role_title", None) or getattr(cfg, "job_title", None) or "").strip() if cfg else ""
        hello = f"Hi{(' ' + cand_name) if cand_name else ''}, welcome to this AI interview."
        fmt = (f"I'm your AI interviewer today{(' for the ' + role_name + ' role') if role_name else ''}. "
               "I'll ask you a few questions about your experience. Please answer naturally, with examples where you can. "
               "If you don't understand a question, just ask me to repeat or explain it.")
        await say_safe(f"{disclosure} {hello} {fmt}".strip())
        await say_safe("Are you ready to begin?")
        ready = await collect_answer(first_timeout=40)
        rtype = await classify_candidate_turn(ready, "Are you ready to begin?") if ready else {"type": "small_talk"}
        if rtype.get("type") == "technical_issue":
            await say_safe("No problem. Take a moment to check your microphone and audio, and let me know when you're ready.")
            await collect_answer(first_timeout=40)
        await say_safe("Great, let's begin.")
        await asyncio.sleep(0.4)

        for idx, q in enumerate(questions):
            question_text = q["prompt_text"]
            if idx > 0:
                await say_safe(random.choice(LEADS))
                await asyncio.sleep(0.3)
            await say_safe(question_text)

            answer = ""
            attempts = 0  # clarification/repeat/invalid retries (max 2)
            while attempts <= 2:
                turn = await collect_answer(first_timeout=75)
                if not turn:  # silence
                    if attempts < 2:
                        await say_safe("Take your time.")
                        attempts += 1
                        continue
                    else:
                        break  # move on, skipped
                cls = await classify_candidate_turn(turn, question_text)
                ttype = cls.get("type", "answer")

                if ttype in ("answer", "off_topic") and cls.get("is_valid_answer", True) and ttype != "off_topic":
                    answer = turn
                    break
                elif ttype == "repeat_request":
                    await say_safe("No problem, I'll repeat the question.")
                    await say_safe(question_text)
                    attempts += 1
                elif ttype in ("clarification_request", "not_understood"):
                    await explain_question(question_text)
                    attempts += 1
                elif ttype == "technical_issue":
                    await say_safe("No problem. Please check your microphone, and answer again when you're ready.")
                    attempts += 1
                elif ttype in ("noise_or_invalid", "small_talk", "off_topic"):
                    if attempts < 2:
                        await say_safe("I didn't catch that clearly. Could you please answer the question?")
                        attempts += 1
                    else:
                        break
                else:
                    answer = turn
                    break

            # If valid but thin, ask ONE natural follow-up and combine.
            if answer and len(answer.split()) < 20:
                try:
                    await say_safe("Thanks. Could you give one concrete example from a real project?")
                    more = await collect_answer(first_timeout=60)
                    if more:
                        mcls = await classify_candidate_turn(more, question_text)
                        if mcls.get("type") == "answer":
                            answer = f"{answer} {more}".strip()
                except Exception as e:
                    logger.warning(f"follow-up failed: {e}")

            if answer:
                await post_answer(session_token, q["id"], answer)
                posted += 1
                await say_safe(random.choice(ACKS))
            else:
                await say_safe("No problem, we'll move to the next question.")

        await say_safe("That's the end of the interview. Thank you so much for your time today. You'll receive a follow-up regarding next steps.")
    except Exception as e:
        # If the controlled loop hits a version/API issue, fall back to free-form conducting.
        deterministic_ok = False
        logger.warning(f"Controlled loop failed ({e}); falling back to free-form interview.")
        numbered = "\n".join(f"{i+1}. {q['prompt_text']}" for i, q in enumerate(questions))
        try:
            await session.generate_reply(instructions=(
                f"{disclosure} Conduct the interview in {lang}. Ask these questions one at a time, "
                f"waiting for each answer, then conclude:\n{numbered}"
            ))
        except Exception:
            pass

    async def _finalize():
        nonlocal posted
        # Fallback path: if the controlled loop didn't post answers, recover from captured turns.
        if posted == 0 and all_user_turns:
            for i, q in enumerate(questions):
                if i < len(all_user_turns):
                    await post_answer(session_token, q["id"], all_user_turns[i])
                    posted += 1
        await complete_session(session_token)
        logger.info(f"Interview {session_token} finalized with {posted} answers (deterministic={deterministic_ok}).")

    ctx.add_shutdown_callback(_finalize)

    # In the controlled path the interview is already done; finalize now too (idempotent-ish).
    if deterministic_ok:
        await _finalize()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
