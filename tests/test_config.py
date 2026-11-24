"""Settings validation (Phase 3): parsing, defaults, fail-fast errors and secret hygiene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.main import describe_config_errors


def _settings(**env: str) -> Settings:
    """Build Settings from environment variables only (no .env file)."""
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults_are_valid_and_safe() -> None:
    s = _settings()
    assert s.environment == "development"
    assert s.llm_models == ["openai/gpt-4o-mini"]
    assert s.tool_allowlist == ["get_current_time", "calculate", "get_weather"]
    assert not s.llm_configured
    assert s.admin_api_key.get_secret_value() == ""
    assert s.log_transcripts is False
    assert s.log_file is None
    assert s.effective_log_format == "text"


def test_values_are_read_from_environment() -> None:
    s = _settings(
        LLM_MODEL="a/one, b/two,,c/three",
        MCP_TOOL_ALLOWLIST="calculate",
        ENVIRONMENT="PRODUCTION",
        OPENROUTER_API_KEY="  sk-test  ",
        STT_MODE="OpenRouter",
        LOG_LEVEL="debug",
        LOG_FILE="",
        PORT="9000",
    )
    assert s.llm_models == ["a/one", "b/two", "c/three"]
    assert s.tool_allowlist == ["calculate"]
    assert s.is_production
    assert s.openrouter_api_key.get_secret_value() == "sk-test"
    assert s.stt_mode == "openrouter"
    assert s.log_level == "DEBUG"
    assert s.log_file is None
    assert s.port == 9000
    assert s.effective_log_format == "json"


def test_values_are_read_from_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("COMPANY_NAME=Acme Bank\nRATE_LIMIT_BURST=3\n# comment\n", encoding="utf-8")
    s = Settings(_env_file=env_file)  # type: ignore[call-arg]
    assert s.company_name == "Acme Bank"
    assert s.rate_limit_burst == 3


@pytest.mark.parametrize(
    ("env", "field"),
    [
        ({"STT_MODE": "carrier-pigeon"}, "stt_mode"),
        ({"ENVIRONMENT": "staging"}, "environment"),
        ({"PORT": "70000"}, "port"),
        ({"TOOL_TIMEOUT_SECONDS": "0"}, "tool_timeout_seconds"),
        ({"MAX_TEXT_CHARS": "0"}, "max_text_chars"),
        ({"RATE_LIMIT_PER_MINUTE": "-1"}, "rate_limit_per_minute"),
        ({"LLM_MODEL": " , "}, "llm_model"),
        ({"LOG_FORMAT": "xml"}, "log_format"),
    ],
)
def test_invalid_values_fail_fast(env: dict[str, str], field: str) -> None:
    with pytest.raises(ValidationError) as info:
        _settings(**env)
    assert field in {str(loc) for err in info.value.errors() for loc in err["loc"]}


def test_production_requires_llm_key() -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY is required"):
        _settings(ENVIRONMENT="production")
    assert _settings(ENVIRONMENT="production", OPENROUTER_API_KEY="sk-x").is_production


def test_short_admin_key_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _settings(ADMIN_API_KEY="password123")
    assert _settings(ADMIN_API_KEY="x" * 32).admin_api_key.get_secret_value() == "x" * 32


def test_secrets_are_masked_in_repr_and_dumps() -> None:
    s = _settings(OPENROUTER_API_KEY="sk-or-v1-supersecret", ADMIN_API_KEY="z" * 40)
    for rendered in (repr(s), str(s), s.model_dump_json()):
        assert "supersecret" not in rendered
        assert "z" * 40 not in rendered


def test_config_error_message_names_variable_without_value() -> None:
    with pytest.raises(ValidationError) as info:
        _settings(PORT="70000", STT_MODE="sk-or-v1-looks-like-a-secret")
    message = describe_config_errors(info.value)
    assert "PORT" in message and "STT_MODE" in message
    assert "looks-like-a-secret" not in message
    assert "70000" not in message


def test_construction_does_not_touch_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: calls.append(a))
    _settings(DATA_DIR=str(tmp_path / "nope"))
    assert calls == []
    assert not (tmp_path / "nope").exists()


# ------------------------------------------------- documentation consistency
ROOT = Path(__file__).resolve().parent.parent
KNOWN = {name.upper() for name in Settings.model_fields}


def test_env_example_is_valid_and_only_uses_known_settings() -> None:
    import re

    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assigned = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, flags=re.M))
    assert assigned, "no variables found"
    assert assigned - KNOWN == set(), "unknown variables in .env.example"
    Settings(_env_file=ROOT / ".env.example")  # type: ignore[call-arg]  # must validate


def test_readme_documents_only_known_settings() -> None:
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme[readme.index("## Environment variables") : readme.index("## Try these queries")]
    first_cells = re.findall(r"^\| ([^|]+)\|", table, flags=re.M)
    documented = {name for cell in first_cells for name in re.findall(r"`([A-Z][A-Z0-9_]+)`", cell)}
    assert documented - KNOWN == set(), "README documents settings that do not exist"
    assert {"OPENROUTER_API_KEY", "ADMIN_API_KEY", "ENVIRONMENT", "LOG_FORMAT"} <= documented
