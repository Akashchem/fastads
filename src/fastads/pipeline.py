import json
from pathlib import Path

import typer

from fastads.models import JobConfig, NormalizedAd
from fastads.services.media import (
    analyze_transcript,
    download_media,
    extract_media,
    extract_ocr,
    prepare_media,
    transcribe_media,
)
from fastads.storage import write_json


def run_pipeline(job_config: JobConfig) -> None:
    """Run the existing pipeline and aggregate final outputs."""
    job_dir = job_dir_path(job_config.job_id)
    ads = ingest_ads(job_config.input_path, str(job_dir))
    media_prepared_ads = prepare_media(str(job_dir))
    media_downloaded_ads, media_failed_ads = download_media(str(job_dir))
    extract_media(str(job_dir))
    extract_ocr(str(job_dir))
    transcribe_media(str(job_dir))
    analyze_transcript(str(job_dir))
    typer.echo("Pipeline completed successfully")
    aggregate_pipeline_output(
        job_dir=job_dir,
        job_config=job_config,
        ads=ads,
        media_prepared_ads=media_prepared_ads,
        media_downloaded_ads=media_downloaded_ads,
        media_failed_ads=media_failed_ads,
    )
    typer.echo("Aggregated outputs written to pipeline_output.json")


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
            local_path=str(item.get("local_path")) if item.get("local_path") else None,
        )
        normalized_ads.append(normalized_ad.model_dump())

    write_json(Path(job_dir) / "normalized_ads.json", normalized_ads)
    typer.echo(f"Ingested {len(normalized_ads)} ads")
    return normalized_ads


def job_dir_path(job_id: str) -> Path:
    from fastads.config import FASTADS_DATA_DIR

    return FASTADS_DATA_DIR / job_id


def aggregate_pipeline_output(
    *,
    job_dir: Path,
    job_config: JobConfig,
    ads: list[dict],
    media_prepared_ads: int,
    media_downloaded_ads: int,
    media_failed_ads: int,
) -> None:
    aggregated_ads: list[dict] = []

    for ad in ads:
        ad_id = str(ad["ad_id"])
        media_dir = job_dir / "media" / ad_id
        media_meta_path = media_dir / "media_meta.json"
        insights_path = media_dir / "insights.json"
        transcript_path = media_dir / "transcript.txt"
        ocr_path = media_dir / "ocr.json"

        if not media_meta_path.exists():
            raise RuntimeError(f"Missing media metadata for {ad_id}: {media_meta_path}")
        if not transcript_path.exists():
            raise RuntimeError(f"Missing transcript output for {ad_id}: {transcript_path}")
        if not ocr_path.exists():
            raise RuntimeError(f"Missing OCR output for {ad_id}: {ocr_path}")

        try:
            media_meta = json.loads(media_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid media metadata JSON for {ad_id}: {media_meta_path}") from exc

        analysis_status = str(media_meta.get("analysis_status", "completed"))
        analysis_mode = str(media_meta.get("analysis_mode", ""))
        confidence = str(media_meta.get("confidence", ""))

        if not insights_path.exists() and analysis_status != "no_signal":
            raise RuntimeError(f"Missing insights output for {ad_id}: {insights_path}")

        if insights_path.exists():
            try:
                insights = json.loads(insights_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid insights JSON for {ad_id}: {insights_path}") from exc
        else:
            insights = {}

        try:
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid OCR JSON for {ad_id}: {ocr_path}") from exc

        aggregated_ads.append(
            {
                "ad_id": ad_id,
                "analysis_status": analysis_status,
                "analysis_mode": analysis_mode,
                "confidence": confidence,
                "insights": insights,
                "transcript": transcript_path.read_text(encoding="utf-8"),
                "ocr": ocr,
            }
        )

    output_path = job_dir / "pipeline_output.json"
    temp_output_path = job_dir / "pipeline_output.tmp.json"
    write_json(
        temp_output_path,
        {
            "status": "completed",
            "job_id": job_config.job_id,
            "competitor": job_config.competitor,
            "market": job_config.market,
            "input_path": job_config.input_path,
            "ingested_ads": len(ads),
            "media_prepared_ads": media_prepared_ads,
            "media_downloaded_ads": media_downloaded_ads,
            "media_failed_ads": media_failed_ads,
            "ads": aggregated_ads,
        },
    )
    temp_output_path.replace(output_path)
