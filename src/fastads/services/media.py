from pathlib import Path

import typer

from fastads.storage import write_json


def prepare_media(job_dir: str) -> int:
    job_path = Path(job_dir)
    normalized_ads_path = job_path / "normalized_ads.json"

    if not normalized_ads_path.exists():
        typer.echo(
            f"Error: normalized ads file not found: {normalized_ads_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    normalized_ads = read_json(normalized_ads_path)
    if not isinstance(normalized_ads, list):
        typer.echo(
            f"Error: expected a JSON array in {normalized_ads_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    prepared_count = 0

    for item in normalized_ads:
        if not isinstance(item, dict):
            continue

        ad_id = item.get("ad_id")
        video_url = item.get("video_url")
        if not ad_id or not video_url:
            continue

        media_dir = job_path / "media" / str(ad_id)
        media_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            media_dir / "media_meta.json",
            {
                "ad_id": str(ad_id),
                "video_url": str(video_url),
                "status": "pending_download",
            },
        )
        prepared_count += 1

    typer.echo(f"Prepared media for {prepared_count} ads")
    return prepared_count


def read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
