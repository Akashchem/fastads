import json
from pathlib import Path

import typer

from fastads.models import JobConfig, NormalizedAd
from fastads.services.media import (
    analyze_transcript,
    download_media,
    extract_media,
    prepare_media,
    transcribe_media,
)
from fastads.storage import write_json


def run_pipeline(job_config: JobConfig) -> None:
    """Placeholder pipeline entry point."""
    job_dir = job_dir_path(job_config.job_id)
    ads = ingest_ads(job_config.input_path, str(job_dir))
    media_prepared_ads = prepare_media(str(job_dir))
    media_downloaded_ads, media_failed_ads = download_media(str(job_dir))
    extract_media(str(job_dir))
    transcribe_media(str(job_dir))
    analyze_transcript(str(job_dir))
    output_path = job_dir / "pipeline_output.json"
    write_json(
        output_path,
        {
            "status": "placeholder",
            "job_id": job_config.job_id,
            "competitor": job_config.competitor,
            "market": job_config.market,
            "input_path": job_config.input_path,
            "ingested_ads": len(ads),
            "media_prepared_ads": media_prepared_ads,
            "media_downloaded_ads": media_downloaded_ads,
            "media_failed_ads": media_failed_ads,
        },
    )


def ingest_ads(input_path: str, job_dir: str) -> list[dict]:
    source_path = Path(input_path)

    if not source_path.exists():
        typer.echo(f"Error: input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: invalid JSON in {input_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not isinstance(payload, list):
        typer.echo(f"Error: expected a JSON array in {input_path}", err=True)
        raise typer.Exit(code=1)

    normalized_ads: list[dict] = []
    required_fields = ("ad_id", "page_name", "ad_copy", "video_url")

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            typer.echo(f"Error: ad #{index} must be a JSON object", err=True)
            raise typer.Exit(code=1)

        missing_fields = [field for field in required_fields if not item.get(field)]
        if missing_fields:
            typer.echo(
                f"Error: ad #{index} is missing required fields: {', '.join(missing_fields)}",
                err=True,
            )
            raise typer.Exit(code=1)

        normalized_ad = NormalizedAd(
            ad_id=str(item["ad_id"]),
            page_name=str(item["page_name"]),
            ad_copy=str(item["ad_copy"]),
            video_url=str(item["video_url"]),
        )
        normalized_ads.append(normalized_ad.model_dump())

    write_json(Path(job_dir) / "normalized_ads.json", normalized_ads)
    typer.echo(f"Ingested {len(normalized_ads)} ads")
    return normalized_ads


def job_dir_path(job_id: str) -> Path:
    from fastads.config import FASTADS_DATA_DIR

    return FASTADS_DATA_DIR / job_id
