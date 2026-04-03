import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


FASTADS_DATA_DIR = Path("data/jobs")
FASTADS_LLM_PROVIDER = os.getenv("FASTADS_LLM_PROVIDER", "openai")

if FASTADS_LLM_PROVIDER == "openrouter":
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
