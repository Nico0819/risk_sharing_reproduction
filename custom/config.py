"""Project-local configuration loading with secret-safe DeepSeek mapping.

The loader reads the project-root ``.env`` explicitly and never prints its
contents.  It maps the three project variables to the names consumed by the
installed AgentSociety2 dispatcher before importing that dispatcher.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
_PLACEHOLDER_VALUES = {"", "changeme", "your_key_here", "your-api-key", "<your-api-key>"}


class RuntimeConfigError(RuntimeError):
    """Raised before a real backend can start with invalid configuration."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved non-secret runtime configuration.

    ``api_key`` is excluded from repr to avoid accidental logging.
    """

    backend: str
    base_url: str
    model: str
    env_path: Path
    api_key: str = field(repr=False, default="")
    thinking_disabled: bool = True

    @property
    def api_key_configured(self) -> bool:
        return _is_real_value(self.api_key)

    def safe_status(self) -> dict[str, object]:
        """Return status without a key or secret-derived value."""

        return {
            "backend": self.backend,
            "env_path": str(self.env_path),
            "api_key": "configured" if self.api_key_configured else "not configured",
            "base_url": "configured" if bool(self.base_url.strip()) else "not configured",
            "model": self.model if _is_real_value(self.model) else "not configured",
            "thinking": "disabled" if self.thinking_disabled else "not disabled",
        }


def _is_real_value(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in _PLACEHOLDER_VALUES and not (
        normalized.startswith("<") and normalized.endswith(">")
    )


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse simple dotenv assignments without exposing values."""

    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "a").isalnum():
            continue
        raw_value = raw_value.strip()
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = raw_value.strip("\"'")
        values[key] = value
    return values


def _resolved_value(name: str, dotenv: Mapping[str, str]) -> str:
    """Process environment wins over the explicit project dotenv file."""

    current = os.environ.get(name)
    if current is not None and current.strip():
        return current.strip()
    return str(dotenv.get(name, "")).strip()


def _set_consistent(name: str, value: str) -> None:
    """Set a framework alias unless an existing process value conflicts."""

    existing = os.environ.get(name)
    if existing is not None and existing.strip() and existing.strip() != value:
        raise RuntimeConfigError(f"conflicting existing runtime setting: {name}")
    os.environ.setdefault(name, value)


def _apply_framework_settings(*, api_key: str, base_url: str, model: str) -> None:
    """Map one DeepSeek configuration to both AgentSociety2 LLM roles."""

    for name, value in (
        ("AGENTSOCIETY_LLM_API_KEY", api_key),
        ("AGENTSOCIETY_LLM_API_BASE", base_url),
        ("AGENTSOCIETY_LLM_MODEL", model),
        ("AGENTSOCIETY_CODER_LLM_API_KEY", api_key),
        ("AGENTSOCIETY_CODER_LLM_API_BASE", base_url),
        ("AGENTSOCIETY_CODER_LLM_MODEL", model),
    ):
        _set_consistent(name, value)

    # DeepSeek documents this OpenAI-compatible extra_body switch for Chat
    # Completions.  Refuse a conflicting pre-existing policy instead of
    # silently enabling thinking.
    for name in ("AGENTSOCIETY_LLM_THINKING", "AGENTSOCIETY_CODER_LLM_THINKING"):
        existing = os.environ.get(name, "").strip().lower()
        if existing and existing not in {"off", "0", "false", "disable", "disabled", "no"}:
            raise RuntimeConfigError(f"thinking must be disabled before real backend: {name}")
        os.environ.setdefault(name, "off")
    disabled_body = '{"thinking":{"type":"disabled"}}'
    for name in ("AGENTSOCIETY_LLM_EXTRA_BODY", "AGENTSOCIETY_CODER_LLM_EXTRA_BODY"):
        existing = os.environ.get(name, "").strip()
        if existing and existing != disabled_body:
            raise RuntimeConfigError(f"conflicting thinking payload: {name}")
        os.environ.setdefault(name, disabled_body)


def _apply_scripted_framework_settings() -> None:
    """Provide constructor-only defaults without overriding process settings.

    The scripted backend replaces both model clients before any request is
    made. Existing AgentSociety2 variables may belong to the user's shell,
    so they are retained; missing values are filled only to let the platform
    construct its dispatcher objects.
    """

    for name, value in (
        ("AGENTSOCIETY_LLM_API_KEY", "scripted-local-only"),
        ("AGENTSOCIETY_LLM_API_BASE", "https://scripted.invalid"),
        ("AGENTSOCIETY_LLM_MODEL", "scripted-model"),
        ("AGENTSOCIETY_CODER_LLM_API_KEY", "scripted-local-only"),
        ("AGENTSOCIETY_CODER_LLM_API_BASE", "https://scripted.invalid"),
        ("AGENTSOCIETY_CODER_LLM_MODEL", "scripted-model"),
    ):
        os.environ.setdefault(name, value)


def load_runtime_config(
    backend: str,
    *,
    env_path: Path | None = None,
    require_credentials: bool | None = None,
    request_timeout: float = 60.0,
) -> RuntimeConfig:
    """Load scripted or DeepSeek mode before importing AgentSociety2 config."""

    backend = backend.strip().lower()
    if backend not in {"scripted", "deepseek"}:
        raise RuntimeConfigError("backend must be scripted or deepseek")
    path = (env_path or DEFAULT_ENV_PATH).expanduser().resolve()
    dotenv = _parse_dotenv(path)

    if backend == "scripted":
        # These values only allow AgentSociety2 dispatcher objects to be
        # constructed; the runner replaces their call boundary with a local
        # scripted client and performs no network request.
        _apply_scripted_framework_settings()
        os.environ.setdefault("AGENTSOCIETY_LLM_REQUEST_TIMEOUT", str(request_timeout))
        return RuntimeConfig(
            backend=backend,
            base_url="https://scripted.invalid",
            model="scripted-model",
            env_path=path,
            api_key="",
        )

    api_key = _resolved_value("DEEPSEEK_API_KEY", dotenv)
    base_url = _resolved_value("DEEPSEEK_BASE_URL", dotenv) or DEFAULT_BASE_URL
    model = _resolved_value("DEEPSEEK_MODEL", dotenv)
    if require_credentials is not False:
        missing: list[str] = []
        if not _is_real_value(api_key):
            missing.append("DEEPSEEK_API_KEY")
        if not _is_real_value(base_url):
            missing.append("DEEPSEEK_BASE_URL")
        if not _is_real_value(model):
            missing.append("DEEPSEEK_MODEL")
        if missing:
            raise RuntimeConfigError("missing or placeholder DeepSeek configuration: " + ", ".join(missing))
    _apply_framework_settings(api_key=api_key, base_url=base_url, model=model)
    os.environ.setdefault("AGENTSOCIETY_LLM_REQUEST_TIMEOUT", str(request_timeout))
    return RuntimeConfig(
        backend=backend,
        base_url=base_url,
        model=model,
        env_path=path,
        api_key=api_key,
    )


def inspect_config_status(*, env_path: Path | None = None) -> dict[str, str]:
    """Report only configured/not-configured states for the project .env."""

    path = (env_path or DEFAULT_ENV_PATH).expanduser().resolve()
    dotenv = _parse_dotenv(path)
    return {
        "env_path": str(path),
        "api_key": "configured" if _is_real_value(_resolved_value("DEEPSEEK_API_KEY", dotenv)) else "not configured",
        "base_url": "configured" if _is_real_value(_resolved_value("DEEPSEEK_BASE_URL", dotenv) or DEFAULT_BASE_URL) else "not configured",
        "model": "configured" if _is_real_value(_resolved_value("DEEPSEEK_MODEL", dotenv)) else "not configured",
    }
