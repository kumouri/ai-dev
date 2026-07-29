"""Configuration — every environment-specific value in one place.

Nothing in this package hardcodes a machine-specific path or URL. Defaults are documented here and
in ``.env.example``; the environment overrides them. A ``.env`` at the repo root (or beside this
member's ``pyproject.toml``) is loaded if present, and never overrides values already exported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: ai-dev/ — the workspace root (…/coryphaeus/src/coryphaeus/config.py → up four).
REPO_ROOT = Path(__file__).resolve().parents[3]
#: ai-dev/coryphaeus/
MEMBER_ROOT = Path(__file__).resolve().parents[2]

#: Ollama's standard port. Override with OLLAMA_BASE_URL when yours differs — some machines cannot
#: bind 11434 (on Windows, WinNAT's dynamic port range can claim it), and the fix belongs in a local
#: .env rather than in this default.
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"


def load_dotenv(*paths: Path) -> None:
    """Load ``KEY=value`` lines from the given files, without overriding existing env vars."""
    for path in paths or (REPO_ROOT / ".env", MEMBER_ROOT / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str
    ollama_unit_budget: int
    featherless_api_key: str | None
    featherless_base_url: str
    featherless_unit_budget: int
    data_dir: Path
    runs_dir: Path

    @property
    def has_featherless(self) -> bool:
        return bool(self.featherless_api_key)


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings, loaded once."""
    load_dotenv()
    return Settings(
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL,
        ollama_unit_budget=_int_env("OLLAMA_UNIT_BUDGET", 2),
        featherless_api_key=os.environ.get("FEATHERLESS_API_KEY", "").strip() or None,
        featherless_base_url=(
            os.environ.get("FEATHERLESS_BASE_URL", "").strip() or DEFAULT_FEATHERLESS_BASE_URL
        ),
        featherless_unit_budget=_int_env("FEATHERLESS_UNIT_BUDGET", 4),
        data_dir=_path_env("CORYPHAEUS_DATA_DIR", MEMBER_ROOT / "data"),
        runs_dir=_path_env("CORYPHAEUS_RUNS_DIR", MEMBER_ROOT / "runs"),
    )
