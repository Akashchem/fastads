import json
from pathlib import Path
from typing import Any

from fastads.config import FASTADS_DATA_DIR


def create_job_dir(job_id: str) -> Path:
    job_dir = FASTADS_DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
