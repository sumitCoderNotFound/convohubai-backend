# ConvoHubAI Backend

AI-Powered Communication Platform for Education & Hospitality.

## 🚀 Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL + SQLAlchemy (async)
- **Cache**: Redis
- **Auth**: JWT (python-jose)
- **AI**: OpenAI, Anthropic, LangChain
- **Voice**: Twilio, Retell AI
- **Payments**: Stripe

## 📁 Project Structure

```
convohubai-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py        # Settings & env vars
│   │   ├── database.py      # Database connection
│   │   └── security.py      # JWT & password utils
│   ├── models/
│   │   ├── user.py          # User, Workspace, Member
│   │   ├── agent.py         # Agent, AgentTemplate
│   │   ├── knowledge_base.py # KnowledgeBase, Document
│   │   ├── conversation.py  # Conversation, Message
│   │   └── phone_number.py  # PhoneNumber
│   ├── schemas/
│   │   ├── auth.py          # Auth request/response
│   │   ├── user.py          # User/Workspace schemas
│   │   └── agent.py         # Agent schemas
│   └── api/
│       └── routes/
│           ├── auth.py      # /auth endpoints
│           ├── workspaces.py # /workspaces endpoints
│           └── agents.py    # /agents endpoints
├── alembic/                  # Database migrations
├── tests/                    # Test files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🛠️ Setup Instructions

### Option 1: Docker (Recommended)

```bash
# 1. Clone and navigate to project
cd convohubai-backend

# 2. Copy environment variables
cp .env.example .env

# 3. Start all services
docker-compose up -d

# 4. API is now running at http://localhost:8000
# 5. Docs available at http://localhost:8000/api/docs
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up PostgreSQL and Redis locally
# Or use Docker just for databases:
docker-compose up -d db redis

# 4. Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Run migrations (first time)
alembic upgrade head

# 6. Start the server
uvicorn app.main:app --reload

# API runs at http://localhost:8000
```

## 📚 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| GET | `/api/v1/auth/me` | Get current user |
| POST | `/api/v1/auth/logout` | Logout |
| POST | `/api/v1/auth/password/change` | Change password |
| POST | `/api/v1/auth/password/reset` | Request password reset |

### Workspaces
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/workspaces` | List user's workspaces |
| POST | `/api/v1/workspaces` | Create workspace |
| GET | `/api/v1/workspaces/{id}` | Get workspace |
| PATCH | `/api/v1/workspaces/{id}` | Update workspace |
| DELETE | `/api/v1/workspaces/{id}` | Delete workspace |
| GET | `/api/v1/workspaces/{id}/members` | List members |
| POST | `/api/v1/workspaces/{id}/members` | Invite member |

### Agents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/agents` | List agents |
| POST | `/api/v1/agents` | Create agent |
| GET | `/api/v1/agents/{id}` | Get agent |
| PATCH | `/api/v1/agents/{id}` | Update agent |
| DELETE | `/api/v1/agents/{id}` | Delete agent |
| POST | `/api/v1/agents/{id}/activate` | Activate agent |
| POST | `/api/v1/agents/{id}/pause` | Pause agent |
| POST | `/api/v1/agents/{id}/duplicate` | Duplicate agent |
| GET | `/api/v1/agents/templates/list` | List templates |
| POST | `/api/v1/agents/from-template` | Create from template |

## 🔧 Environment Variables

See `.env.example` for all required environment variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection (async) |
| `REDIS_URL` | Redis connection |
| `SECRET_KEY` | App secret key (32+ chars) |
| `JWT_SECRET_KEY` | JWT signing key |
| `OPENAI_API_KEY` | OpenAI API key |
| `TWILIO_ACCOUNT_SID` | Twilio account SID |
| `STRIPE_SECRET_KEY` | Stripe secret key |

## 🗄️ Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Run migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app

# Specific test file
pytest tests/test_auth.py -v
```

## 📝 Development Notes

### Adding a New Model

1. Create model in `app/models/`
2. Add to `app/models/__init__.py`
3. Create schema in `app/schemas/`
4. Create routes in `app/api/routes/`
5. Include router in `app/api/__init__.py`
6. Generate migration: `alembic revision --autogenerate`

### Authentication Flow

1. Register → Creates user + default workspace
2. Login → Returns access + refresh tokens
3. All protected routes require `Authorization: Bearer <token>`
4. Access token expires in 30 minutes
5. Use refresh token to get new access token

## 🔒 Security

- Passwords hashed with bcrypt
- JWT tokens for authentication
- CORS configured for allowed origins
- SQL injection protection via SQLAlchemy ORM
- Input validation via Pydantic

## 📞 Support

For issues or questions, please open a GitHub issue.

---

**ConvoHubAI** - The Future of Business Communication 🚀
# convohubai-backend
