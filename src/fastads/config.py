import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTADS_DATA_DIR = PROJECT_ROOT / "data" / "jobs"


def _read_setting(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            secret_value = secrets.get(name, default)
            if secret_value is None:
                return default
            return str(secret_value).strip()
    except Exception:
        pass
    return default


FASTADS_TRANSCRIBER = _read_setting("FASTADS_TRANSCRIBER", "local")
WHISPER_API_URL = _read_setting("WHISPER_API_URL")
WHISPER_AUTH_TOKEN = _read_setting("WHISPER_AUTH_TOKEN")
WHISPER_MODEL = _read_setting("WHISPER_MODEL", "whisper-1")
WHISPER_RESPONSE_FORMAT = _read_setting("WHISPER_RESPONSE_FORMAT", "verbose_json")
FASTADS_LLM_PROVIDER = _read_setting("FASTADS_LLM_PROVIDER", "openai")
AZURE_OPENAI_ENDPOINT = _read_setting("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _read_setting("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = _read_setting("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = _read_setting("AZURE_OPENAI_DEPLOYMENT")

if FASTADS_LLM_PROVIDER == "azure_openai":
    FASTADS_LLM_API_BASE = ""
    FASTADS_LLM_API_KEY = AZURE_OPENAI_API_KEY
    FASTADS_LLM_MODEL = AZURE_OPENAI_DEPLOYMENT
elif FASTADS_LLM_PROVIDER == "openrouter":
    FASTADS_LLM_API_BASE = _read_setting(
        "FASTADS_LLM_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    FASTADS_LLM_API_KEY = _read_setting("OPENROUTER_API_KEY")
    FASTADS_LLM_MODEL = _read_setting("FASTADS_LLM_MODEL", "openrouter/free")
else:
    FASTADS_LLM_API_BASE = _read_setting(
        "FASTADS_LLM_API_BASE",
        "https://api.openai.com/v1",
    )
    FASTADS_LLM_API_KEY = _read_setting("OPENAI_API_KEY")
    FASTADS_LLM_MODEL = _read_setting("FASTADS_LLM_MODEL", "gpt-4o-mini")
