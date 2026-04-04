import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


FASTADS_DATA_DIR = Path("data/jobs")
FASTADS_TRANSCRIBER = os.getenv("FASTADS_TRANSCRIBER", "local")
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "")
WHISPER_AUTH_TOKEN = os.getenv("WHISPER_AUTH_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
WHISPER_RESPONSE_FORMAT = os.getenv("WHISPER_RESPONSE_FORMAT", "verbose_json")
FASTADS_LLM_PROVIDER = os.getenv("FASTADS_LLM_PROVIDER", "openai")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

if FASTADS_LLM_PROVIDER == "azure_openai":
    FASTADS_LLM_API_BASE = ""
    FASTADS_LLM_API_KEY = AZURE_OPENAI_API_KEY
    FASTADS_LLM_MODEL = AZURE_OPENAI_DEPLOYMENT
elif FASTADS_LLM_PROVIDER == "openrouter":
    FASTADS_LLM_API_BASE = os.getenv(
        "FASTADS_LLM_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    FASTADS_LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    FASTADS_LLM_MODEL = os.getenv("FASTADS_LLM_MODEL", "openrouter/free")
else:
    FASTADS_LLM_API_BASE = os.getenv(
        "FASTADS_LLM_API_BASE",
        "https://api.openai.com/v1",
    )
    FASTADS_LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    FASTADS_LLM_MODEL = os.getenv("FASTADS_LLM_MODEL", "gpt-4o-mini")
