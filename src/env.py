"""Load .env once for entrypoints (dashboard, API, scripts)."""
from __future__ import annotations

from src.paths import ROOT

_LOADED = False


def load_env() -> None:
    """Read ROOT/.env into os.environ. Existing env vars win."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)
