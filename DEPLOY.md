# ConvoHubAI Recruitment - Deploy Guide (Phase 10)

A practical checklist to take the recruitment product from local to production.

## 1. Services
- API (FastAPI/uvicorn), PostgreSQL, Redis: run via docker-compose.
- Interview voice worker (interview_worker.py): runs as its OWN process, not in the API
  container. See VOICE_SETUP.md.
- Frontend (Vite build) served as static files behind your web server/CDN.

## 2. Environment variables (backend .env)
Required:
- SECRET_KEY, JWT_SECRET_KEY (long random strings)
- DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST:5432/DB
- DATABASE_URL_SYNC=postgresql://USER:PASS@HOST:5432/DB   (Alembic)
AI (scoring, generation, voice brain):
- GROQ_API_KEY, and optionally OPENAI_API_KEY
Voice (optional, only for voice interviews):
- LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY
Files (candidate documents): UPLOAD_DIR (default "uploads"; mount a volume in Docker so uploads persist, or swap app/recruitment/services/storage.py for S3 in production).

Email (optional, for sending invites):
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAILS_FROM_EMAIL, EMAILS_FROM_NAME
Frontend .env:
- VITE_API_URL (your API base), VITE_LIVEKIT_URL (if using voice)

## 3. Database
- Tables auto-create on startup (init_db create_all). For controlled migrations use Alembic:
    alembic revision --autogenerate -m "recruitment"
    alembic upgrade head
- New tables across phases include: recruitment domain (16) + ats_connections, ats_mappings.

## 4. Build + run
Backend:
    cd backend && docker-compose up --build -d
Frontend:
    cd frontend && npm ci && npm run build   # serve dist/ behind nginx/CDN
Voice worker (if used):
    cd backend && pip install -r requirements-worker.txt
    API_URL=https://your-api python interview_worker.py start

## 5. Production hardening checklist
- [ ] Put the API behind HTTPS (TLS) and a reverse proxy.
- [ ] Restrict CORS to your frontend origin.
- [ ] Store ATS api_keys and LiveKit/SMTP secrets in a secret manager; the AtsConnection
      api_key column should be encrypted at rest (currently plaintext for MVP).
- [ ] Set strong SECRET_KEY/JWT_SECRET_KEY; rotate periodically.
- [ ] Back up PostgreSQL; enable point-in-time recovery.
- [ ] Rate-limit the public candidate endpoints.
- [ ] Review EU AI Act obligations: human-in-the-loop on decisions (built in), candidate
      consent (built in), no protected-trait inference (scoring prompts forbid it).

## 6. Feature flags / settings
- Per-workspace recruitment settings (jurisdiction, recording default, candidate-visible
  scores, branding) live in Settings and can be changed without redeploy.

## 7. What needs external accounts
- Voice: a LiveKit project + Deepgram key.
- Email: an SMTP provider.
- ATS sync: a Greenhouse / Lever / Workable API key (added under Settings > ATS).
