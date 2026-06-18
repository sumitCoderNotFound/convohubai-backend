# ConvoHubAI — Recruitment Backend (Phase 1)

All 6 Phase-1 features. Self-contained bounded context under `app/recruitment/`.

## What's included
- **Feature 1 — Jobs**: CRUD, duplicate, close, archive, AI `parse-description`
- **Feature 2 — Candidates + Applications**: candidate CRUD (email upsert), applications, audited stage decisions + history
- **Feature 3 — Interviews + versioning**: template + draft/published versions, immutability, publish validation, clone-to-new-draft, AI generate
- **Feature 4 — Questions + branching**: question CRUD, reorder, branch rules (draft-only)
- **Feature 5 — Rubrics + anchors**: weighted criteria, weak/moderate/strong anchors, weight + sensitive-trait validation
- **Feature 6 — Simulation**: non-scored interview preview (no candidate data, no credits)

## Install (3 steps)

### 1. Copy the package
Copy the `app/recruitment/` folder into your project at `backend/app/recruitment/`.

### 2. Apply the two wiring edits
Both edited files are included — drop them in, or apply by hand:

**`app/models/__init__.py`** — add after the existing model imports:
```python
from app.recruitment.models import (
    JobPosition, Candidate, Application, ApplicationHistory,
    InterviewTemplate, InterviewVersion, InterviewQuestion,
    BranchRule, RubricCriterion, ScoreAnchor,
)
```
(and add the same names to `__all__`)

**`app/api/__init__.py`** — add after the last `include_router`:
```python
from app.recruitment.api import recruitment_router
api_router.include_router(recruitment_router)
```

### 3. Add the missing dependency you hit earlier
In `backend/requirements.txt`:
```
google-auth
google-auth-oauthlib
```

## Tables
On startup, `init_db()` (create_all) auto-creates the 10 new tables because they're
now imported in `app/models/__init__.py`. For production migrations, add this to
`alembic/env.py` above `target_metadata` so autogenerate sees them:
```python
import app.models  # noqa  (registers all tables incl. recruitment)
```
then: `alembic revision --autogenerate -m "add recruitment domain"` && `alembic upgrade head`

## Smoke test in Swagger (`/api/docs`)
1. Log in / authorize (Bearer token) and make sure you have a current workspace.
2. **Jobs**: `POST /recruitment/jobs/parse-description` (paste a JD) → see extracted skills.
   `POST /recruitment/jobs` → `GET /recruitment/jobs`.
3. **Candidates**: `POST /recruitment/candidates` → `POST /recruitment/applications`.
   `POST /applications/{id}/decisions` → `GET /applications/{id}/history` shows the trail.
4. **Interviews**: `POST /recruitment/interviews` (creates draft v1) →
   `GET /interviews/{id}/draft` → note the version `id`.
5. **Questions/Rubric**: `POST /versions/{vid}/questions`,
   `POST /versions/{vid}/criteria` (with anchors, weight 100).
6. **Publish**: `POST /interviews/{id}/publish` → should succeed; try editing a question
   on that version → **409 immutable**. `POST /interviews/{id}/new-draft` to edit again.
7. **Simulate**: `POST /versions/{vid}/simulate` → preview transcript.
8. **Isolation**: from a second workspace, the first workspace's jobs must NOT appear.

> Note: AI endpoints (parse-description, generate, simulate) degrade gracefully if no
> AI key is set — they return an empty/fallback result instead of erroring, so you can
> test the full flow without keys, then richer output appears once a key is configured.

---

# Phase 2 — Candidate side (invites, sessions, scoring)

Adds the candidate-facing half: invite links, a public (login-free) registration + interview
flow, answer capture, explainable rubric scoring, and recruiter result views.

## New models (6)
InterviewInvite, InterviewSession, SessionAnswer, InterviewScore, CriterionScore, RecruitmentSettings.
(create_all auto-creates them on startup; or `alembic revision --autogenerate`.)

## The three product decisions — configurable, with safe defaults
Stored per workspace in `recruitment_settings` and editable via `GET/PATCH /recruitment/settings`:
- jurisdiction = "uk"  (drives consent wording; uk/eu/us/other supported)
- default_recording_enabled = false
- candidates_see_scores = false   (candidate result screen hides scores unless true)

## Recruiter endpoints (authenticated)
- POST /recruitment/interviews/{id}/invites      create an invite for the published version (returns token + invite_url)
- GET  /recruitment/interviews/{id}/invites      list invites + statuses
- POST /recruitment/invites/{id}/revoke
- GET  /recruitment/sessions                     list sessions (filter by version/application/status)
- GET  /recruitment/sessions/{id}                session detail + transcript + answers
- GET  /recruitment/sessions/{id}/score          the score + per-criterion breakdown
- POST /recruitment/sessions/{id}/score          re-run scoring (e.g. after adding an AI key)
- GET  /recruitment/applications/{id}/result     result view for an application
- GET/PATCH /recruitment/settings                the three decisions + branding

## Public endpoints (NO login — addressed by token)
- GET  /recruitment/public/invites/{token}                 what the candidate sees (name, intro, consent text, disclosure)
- POST /recruitment/public/invites/{token}/register        name+email+consent -> creates candidate+application+session, returns first question
- GET  /recruitment/public/sessions/{session_token}        current state / current question
- POST /recruitment/public/sessions/{session_token}/answers submit an answer -> next question or finish (auto-scores on finish)
- POST /recruitment/public/sessions/{session_token}/complete finalize early
- GET  /recruitment/public/sessions/{session_token}/result  candidate result screen (respects candidates_see_scores)

## How the flow works
1. Recruiter publishes interview (Phase 1) -> creates an invite -> sends invite_url.
2. Candidate opens it, registers + consents -> an Application is created (stage "interview") and a session starts.
3. Candidate answers each question; branch rules can skip/end/knockout.
4. On completion: session -> completed, application advances to "review" (with history), and scoring runs.
5. Scoring: the LLM scores EACH criterion vs its anchors (evidence + confidence); the weighted TOTAL is
   computed in app code (not by the LLM). Low average confidence or high risk -> needs_human_review.
6. Recruiter sees the score, per-criterion evidence, transcript, and risk in the dashboard.

## Scope note — real-time voice
The live voice/avatar transport reuses your existing LiveKit engine (agent_worker.py). The backend here
owns the session state, question sequencing, transcript capture, and scoring. The voice worker/frontend
posts each turn's transcript to the /answers endpoint. Swapping text answers for live STT transcripts is
a transport detail; the scoring + flow are identical.

## Quick test in Swagger
1. PATCH /recruitment/settings -> optionally flip candidates_see_scores / jurisdiction.
2. Publish an interview (Phase 1), then POST /interviews/{id}/invites -> copy the token.
3. POST /public/invites/{token}/register {full_name, email, consent_given:true} -> returns first question.
4. Loop POST /public/sessions/{session_token}/answers {question_id, transcript_text} until finished:true.
5. GET /recruitment/sessions + /sessions/{id}/score -> see the scored result (needs an AI key for real scores;
   without a key the score row is created as 'failed'/needs-review and you can POST .../score to retry later).
