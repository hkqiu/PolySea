"""Process-wide mutable settings (API keys are kept in memory only)."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from threading import Lock
from typing import Any


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class LLMSettings:
    # Configure these through the web settings page or environment variables.
    base_url: str = field(default_factory=lambda: os.getenv("POLYSEA_BASE_URL", "").strip())
    model: str = field(default_factory=lambda: os.getenv("POLYSEA_MODEL", "").strip())
    api_key: str = field(default_factory=lambda: os.getenv("POLYSEA_API_KEY", "").strip())
    timeout_s: float = 120.0
    # Optional PolyOpus endpoint. Empty by default: no weights or credentials are bundled.
    polyopus_base_url: str = field(
        default_factory=lambda: os.getenv("POLYOPUS_BASE_URL", "").strip()
    )
    polyopus_model: str = field(
        default_factory=lambda: os.getenv("POLYOPUS_MODEL", "polyopus").strip()
    )
    polyopus_api_key: str = field(
        default_factory=lambda: os.getenv("POLYOPUS_API_KEY", "").strip()
    )
    polyopus_timeout_s: float = field(
        default_factory=lambda: _env_float("POLYOPUS_TIMEOUT_S", 120.0)
    )

    def polyopus_configured(self) -> bool:
        return bool(self.polyopus_base_url and self.polyopus_model)

    def polyopus_api_key_effective(self) -> str:
        # Local OpenAI-compatible servers may not require a real key.
        return self.polyopus_api_key or "polyopus-local"

    def api_key_effective(self) -> str:
        return (self.api_key or "").strip()


@dataclass
class AppState:
    llm: LLMSettings = field(default_factory=LLMSettings)
    lock: Lock = field(default_factory=Lock)


state = AppState()

# In-memory job store for long-running tasks (single-user local MVP)
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()
