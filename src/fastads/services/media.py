from pathlib import Path
from urllib import error, request

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


def download_media(job_dir: str) -> tuple[int, int]:
    job_path = Path(job_dir)
    media_root = job_path / "media"

    if not media_root.exists():
        typer.echo("Downloaded media for 0 ads, failed for 0 ads")
        return 0, 0

    downloaded_count = 0
    failed_count = 0

    for media_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        media_meta_path = media_dir / "media_meta.json"
        if not media_meta_path.exists():
            continue

        media_meta = read_json(media_meta_path)
        if not isinstance(media_meta, dict):
            failed_count += 1
            continue

        video_url = media_meta.get("video_url")

        if not video_url:
            media_meta["status"] = "download_failed"
            media_meta["error"] = "Missing video_url"
            write_json(media_meta_path, media_meta)
            failed_count += 1
            continue

        local_video_path = media_dir / "source.mp4"

        try:
            with request.urlopen(str(video_url), timeout=30) as response:
                local_video_path.write_bytes(response.read())
        except (error.URLError, OSError, ValueError) as exc:
            media_meta["status"] = "download_failed"
            media_meta["error"] = str(exc)
            media_meta.pop("local_video_path", None)
            write_json(media_meta_path, media_meta)
            failed_count += 1
            continue

        media_meta["status"] = "downloaded"
        media_meta["local_video_path"] = str(local_video_path)
        media_meta.pop("error", None)
        write_json(media_meta_path, media_meta)
        downloaded_count += 1

    typer.echo(
        f"Downloaded media for {downloaded_count} ads, failed for {failed_count} ads"
    )
    return downloaded_count, failed_count


def read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
