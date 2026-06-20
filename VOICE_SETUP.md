# Voice / Avatar Interview Setup (Phase 3)

The typed interview already works end to end. Voice adds a spoken conversation with the
AI interviewer, running on your existing LiveKit stack. It needs external services that
are NOT part of the API container, so it requires a few setup steps and CANNOT be tested
without them. Typed mode remains the default and keeps working regardless.

## What was added
- Backend (testable now): two token-gated public endpoints
  - POST /recruitment/public/sessions/{session_token}/voice-token   -> LiveKit join token
  - GET  /recruitment/public/sessions/{session_token}/agent-config  -> interview script
- interview_worker.py : a LiveKit agent (separate process) that conducts the interview by
  voice and posts answers + completes the session (which triggers scoring).
- requirements-worker.txt : the worker's dependencies (kept out of the API image).
- Frontend: the candidate page now offers Voice or Type. Voice connects to the LiveKit room
  and streams audio. If it can't connect, the candidate can switch to typing.

## Prerequisites (what makes it actually talk)
1. A LiveKit server. Easiest is a free LiveKit Cloud project; or self-host.
   Set in backend .env AND the worker env:
     LIVEKIT_URL=wss://YOUR.livekit.cloud
     LIVEKIT_API_KEY=...
     LIVEKIT_API_SECRET=...
   And in the frontend .env:
     VITE_LIVEKIT_URL=wss://YOUR.livekit.cloud
2. Provider keys for the worker:
     GROQ_API_KEY=...        (LLM + Whisper STT, free tier)
     DEEPGRAM_API_KEY=...    (Aura TTS)
3. Run the worker as its own process (NOT in the API container):
     cd backend
     pip install -r requirements-worker.txt
     API_URL=http://localhost:8000 python interview_worker.py dev

## Flow
Candidate picks Voice -> browser asks the backend for a voice-token -> joins room
`interview-{session_token}` -> the worker is dispatched into that room, fetches the script
via agent-config, greets + discloses it's an AI, asks each question by voice, captures the
answers, then posts them and completes the session. Scoring runs exactly as in typed mode.

## Honest notes
- This could not be validated in the build sandbox (no LiveKit server / audio). The backend
  token + config endpoints ARE validated (the app boots and mints tokens).
- interview_worker.py mirrors agent_worker.py and targets the same livekit.agents stack.
  Depending on your installed livekit-agents version, the event/method names
  (conversation_item_added, generate_reply, setMicrophoneEnabled) may need small tweaks.
  Treat it as a working starting point, not a guaranteed drop-in.
- Per-question answer capture is best-effort (candidate turns mapped to questions in order).
  The full transcript is the source of truth for scoring.

---

## Phase 7 + 8 updates (voice polish + multi-language)

Phase 7 — precise mapping + follow-ups:
- interview_worker.py now drives the interview question-by-question: it asks one question,
  waits for the candidate's full answer (gathering continuation until a pause), posts that
  answer to THAT exact question (precise mapping), then moves on. If an answer is short it
  asks one adaptive follow-up and folds the reply into the same answer.
- If the controlled loop hits a version/API issue (e.g. session.say differences across
  livekit-agents versions), it automatically falls back to the previous free-form flow, and
  a history-based fallback still recovers answers on shutdown. So it degrades, it doesn't break.
- Possible tweak: if you hear the agent double-talking (speaking the scripted question AND an
  auto-generated reply), your livekit-agents version is auto-replying to user turns. Tell me
  and I'll disable auto-reply for the controlled loop.

Phase 8 — multi-language:
- Recruiters pick a language per interview in the builder. AI question generation now writes
  the questions and rubric in that language.
- The worker sets the STT language hint and instructs the agent to conduct the interview in
  the chosen language. Scoring is language-agnostic (the model assesses content directly).
- Known limitation: Deepgram Aura TTS voices are English-first. Speech-to-text works across
  languages, but spoken AI output is best in English. For full non-English voice, swap the
  TTS voice/provider (TTS_VOICE_BY_LANG in interview_worker.py) for one that supports it.
  Text-mode interviews work fully in any language regardless.
