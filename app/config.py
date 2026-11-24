"""Validated application settings, read from environment variables and `.env`.

Nothing here runs at import time: call `get_settings()` (cached) or construct
`Settings(...)` directly (tests do this). Invalid configuration fails fast with a
message naming the variable.

Variable names are the upper-case field names (e.g. `llm_model` <- `LLM_MODEL`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

MIN_ADMIN_KEY_CHARS = 32


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ---------------------------------------------------------------- runtime
    environment: Literal["development", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # -------------------------------------------------------------- providers
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Comma-separated fallback list, tried in order.
    llm_model: str = "openai/gpt-4o-mini"
    stt_mode: Literal["browser", "openrouter"] = "browser"
    stt_model: str = "google/gemini-2.5-flash"
    embedding_model: str = "openai/text-embedding-3-small"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_tool_rounds: int = Field(default=5, ge=1, le=20)

    # ---------------------------------------------------------------- persona
    agent_name: str = ""
    company_name: str = Field(default="Demo Wallet", min_length=1)
    company_description: str = "a digital money-management app"
    kb_topics: str = "plans, fees, limits, cards, savings, bills, support, security and how-tos"

    # --------------------------------------------------------------- security
    admin_api_key: SecretStr = SecretStr("")
    mcp_tool_allowlist: str = "get_current_time,calculate,get_weather"
    tool_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_text_chars: int = Field(default=2000, ge=1, le=20_000)
    max_audio_bytes: int = Field(default=2_000_000, ge=10_000, le=25_000_000)
    max_json_body_bytes: int = Field(default=64 * 1024, ge=1024, le=10_000_000)
    rate_limit_per_minute: int = Field(default=30, ge=1)
    rate_limit_burst: int = Field(default=10, ge=1)

    # ---------------------------------------------------------------- logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # json (one object per line, for log pipelines) or text (human-readable).
    # Default: text in development, json in production.
    log_format: Literal["json", "text"] | None = None
    # Optional rotating log file. Unset: stdout only (containers, systemd, etc.).
    log_file: Path | None = None
    # Transcripts and tool arguments are user data. Keep them out of logs unless debugging.
    log_transcripts: bool = False

    # ------------------------------------------------------------------ paths
    kb_dir: Path = ROOT / "knowledge_base"
    data_dir: Path = ROOT / "data"
    static_dir: Path = ROOT / "app" / "static"

    # ------------------------------------------------------------- validators
    @field_validator("environment", "stt_mode", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("log_format", "log_file", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_model")
    @classmethod
    def _at_least_one_model(cls, value: str) -> str:
        if not _csv(value):
            raise ValueError("must name at least one model")
        return value

    @field_validator("openrouter_api_key", "admin_api_key", mode="before")
    @classmethod
    def _strip_secret(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        admin_key = self.admin_api_key.get_secret_value()
        if admin_key and len(admin_key) < MIN_ADMIN_KEY_CHARS:
            raise ValueError(f"ADMIN_API_KEY must be at least {MIN_ADMIN_KEY_CHARS} characters")
        if self.is_production and not self.llm_configured:
            raise ValueError("OPENROUTER_API_KEY is required when ENVIRONMENT=production")
        return self

    # ---------------------------------------------------------- derived values
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def llm_configured(self) -> bool:
        return bool(self.openrouter_api_key.get_secret_value())

    @property
    def llm_models(self) -> list[str]:
        return _csv(self.llm_model)

    @property
    def tool_allowlist(self) -> list[str]:
        return _csv(self.mcp_tool_allowlist)

    @property
    def stt_max_body_bytes(self) -> int:
        """Upload limit for /api/stt: the audio itself plus multipart framing."""
        return self.max_audio_bytes + 64 * 1024

    @property
    def effective_log_format(self) -> Literal["json", "text"]:
        if self.log_format:
            return self.log_format
        return "json" if self.is_production else "text"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once on first use (not at import)."""
    return Settings()
