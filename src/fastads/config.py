import os
from pathlib import Path


FASTADS_DATA_DIR = Path("data/jobs")
FASTADS_LLM_API_BASE = os.getenv("FASTADS_LLM_API_BASE", "https://api.openai.com/v1")
FASTADS_LLM_MODEL = os.getenv("FASTADS_LLM_MODEL", "gpt-4o-mini")
FASTADS_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
