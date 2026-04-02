from fastads.models import JobConfig
from fastads.storage import write_json


def run_pipeline(job_config: JobConfig) -> None:
    """Placeholder pipeline entry point."""
    output_path = job_config_output_path(job_config.job_id)
    write_json(
        output_path,
        {
            "status": "placeholder",
            "job_id": job_config.job_id,
            "competitor": job_config.competitor,
            "market": job_config.market,
            "input_path": job_config.input_path,
        },
    )


def job_config_output_path(job_id: str):
    from fastads.config import FASTADS_DATA_DIR

    return FASTADS_DATA_DIR / job_id / "pipeline_output.json"
