"""
ConvoHubAI Recruitment - Shared enums.
Stored as String in the DB; validated at the schema layer.
"""
import enum


class JobStatus(str, enum.Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    ARCHIVED = "archived"


class EmploymentType(str, enum.Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"


class CandidateSource(str, enum.Enum):
    MANUAL = "manual"
    INVITATION = "invitation"
    SELF_REGISTRATION = "self_registration"
    API = "api"
    IMPORT = "import"


class ApplicationStage(str, enum.Enum):
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    REVIEW = "review"
    ADVANCED = "advanced"
    REJECTED = "rejected"
    ON_HOLD = "on_hold"
    WITHDRAWN = "withdrawn"


class ApplicationStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"


class InterviewMode(str, enum.Enum):
    VOICE_ONLY = "voice_only"
    AVATAR_NON_INTERACTIVE = "avatar_non_interactive"
    AVATAR_INTERACTIVE = "avatar_interactive"
    TEXT_PRACTICE = "text_practice"


class InterviewVersionStatus(str, enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class QuestionType(str, enum.Enum):
    OPEN_RESPONSE = "open_response"
    SCRIPTED = "scripted"
    ADAPTIVE = "adaptive"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    KNOCKOUT = "knockout"
    CONSENT = "consent"
    INFORMATION = "information"


class CriterionLevel(str, enum.Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


# ---------------- Phase 2 enums ----------------

class InviteStatus(str, enum.Enum):
    PENDING = "pending"
    REGISTERED = "registered"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    REVOKED = "revoked"


class SessionStatus(str, enum.Enum):
    CREATED = "created"
    CONSENTED = "consented"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ScoreStatus(str, enum.Enum):
    PENDING = "pending"
    SCORING = "scoring"
    COMPLETED = "completed"
    FAILED = "failed"


class RecommendationCategory(str, enum.Enum):
    STRONG = "strong"          # advance
    MODERATE = "moderate"      # review
    WEAK = "weak"              # likely reject
    INSUFFICIENT = "insufficient"  # not enough signal


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Jurisdiction(str, enum.Enum):
    UK = "uk"
    EU = "eu"
    US = "us"
    OTHER = "other"
