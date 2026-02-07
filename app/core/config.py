"""
ConvoHubAI - Application Configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "ConvoHubAI"
    app_env: str = "development"
    debug: bool = True
    api_version: str = "v1"
    secret_key: str = Field(..., min_length=32)

    # Database
    database_url: str
    database_url_sync: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""

    # AI Services
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "convohubai-kb"

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Retell AI
    retell_api_key: str = ""

    # Stripe
    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    # Email
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    emails_from_email: str = ""
    emails_from_name: str = "ConvoHubAI"

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    
    # Deepgram
    deepgram_api_key: str = ""

    # LiveKit Video
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_url: str = "ws://localhost:7880"

    # Groq (FREE AI)
    groq_api_key: str = ""

    # App URL
    app_base_url: str = "http://localhost:8000"


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()