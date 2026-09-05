"""
Application configuration.

All configurable values are read from environment variables (with sane
local-dev defaults) instead of being hardcoded, so nothing sensitive ever
ends up committed to source control. Copy `.env.example` to `.env` and
adjust as needed.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Deepfake Detection API"
    app_version: str = "2.0.0"
    environment: str = "development"  # development | production
    debug: bool = True

    # --- Security ---
    # MUST be overridden in production via the JWT_SECRET_KEY env var.
    # No default secret is used in production mode (see validation below).
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 12  # 12 hours

    # --- CORS ---
    # Comma-separated list of allowed origins, e.g. "http://localhost:3000,https://myapp.com"
    cors_origins: str = "http://localhost:3000"

    # --- Database ---
    database_url: str = "sqlite:///./deepshield.db"

    # --- Uploads ---
    max_image_size_mb: int = 15
    max_video_size_mb: int = 200
    upload_temp_dir: str = "/tmp/deepshield_uploads"

    # --- Rate limiting ---
    rate_limit_detect: str = "10/minute"
    rate_limit_auth: str = "5/minute"

    # --- Models ---
    # Directory to look for real trained checkpoints (*.pt). If empty/missing,
    # models run with randomly-initialized weights and the API reports
    # "demo_mode": true so clients can warn users predictions aren't meaningful.
    model_weights_dir: str = "model_weights"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.is_production and settings.jwt_secret_key == "dev-only-insecure-secret-change-me":
        raise RuntimeError(
            "JWT_SECRET_KEY must be set to a strong random value when ENVIRONMENT=production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return settings
