"""Recruitment domain models."""
from app.recruitment.models.job import JobPosition
from app.recruitment.models.candidate import Candidate, Application, ApplicationHistory
from app.recruitment.models.interview import InterviewTemplate, InterviewVersion
from app.recruitment.models.question import InterviewQuestion, BranchRule
from app.recruitment.models.rubric import RubricCriterion, ScoreAnchor
from app.recruitment.models.invite import InterviewInvite
from app.recruitment.models.session import InterviewSession, SessionAnswer
from app.recruitment.models.score import InterviewScore, CriterionScore
from app.recruitment.models.settings import RecruitmentSettings

__all__ = [
    # Phase 1
    "JobPosition",
    "Candidate",
    "Application",
    "ApplicationHistory",
    "InterviewTemplate",
    "InterviewVersion",
    "InterviewQuestion",
    "BranchRule",
    "RubricCriterion",
    "ScoreAnchor",
    # Phase 2
    "InterviewInvite",
    "InterviewSession",
    "SessionAnswer",
    "InterviewScore",
    "CriterionScore",
    "RecruitmentSettings",
]
