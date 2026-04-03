from pathlib import Path
import json
import shutil
import subprocess
from urllib import error, request

import typer
from faster_whisper import WhisperModel

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
    fallback_source = repo_root() / "assets" / "sample.mp4"

    if not media_root.exists():
        typer.echo("Downloaded media for 0 ads, failed for 0 ads")
        typer.echo("Used fallback copy for 0 ads")
        return 0, 0

    downloaded_count = 0
    failed_count = 0
    fallback_count = 0

    for media_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        media_meta_path = media_dir / "media_meta.json"
        if not media_meta_path.exists():
            continue

        media_meta = read_json(media_meta_path)
        if not isinstance(media_meta, dict):
            failed_count += 1
            continue

        video_url = media_meta.get("video_url")
        local_video_path = media_dir / "source.mp4"
        media_dir.mkdir(parents=True, exist_ok=True)
        download_error: str | None = None

        if video_url:
            try:
                with request.urlopen(str(video_url), timeout=30) as response:
                    local_video_path.write_bytes(response.read())
            except (error.URLError, OSError, ValueError) as exc:
                download_error = str(exc)
                failed_count += 1
        else:
            download_error = "Missing video_url"
            failed_count += 1

        if not local_video_path.exists():
            if not fallback_source.exists():
                typer.echo(
                    f"Error: fallback media file not found: {fallback_source}",
                    err=True,
                )
                raise typer.Exit(code=1)

            shutil.copy(fallback_source, local_video_path)
            fallback_count += 1

        media_meta["status"] = "downloaded"
        media_meta["local_video_path"] = "source.mp4"
        if download_error:
            media_meta["error"] = download_error
        else:
            media_meta.pop("error", None)
        write_json(media_meta_path, media_meta)
        if download_error is None:
            downloaded_count += 1

    typer.echo(
        f"Downloaded media for {downloaded_count} ads, failed for {failed_count} ads"
    )
    typer.echo(f"Used fallback copy for {fallback_count} ads")
    return downloaded_count, failed_count


def extract_media(job_dir: str) -> int:
    job_path = Path(job_dir)
    media_root = job_path / "media"

    if not media_root.exists():
        typer.echo("Processed media for 0 ads")
        return 0

    processed_count = 0

    for media_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        source_path = media_dir / "source.mp4"
        media_meta_path = media_dir / "media_meta.json"

        if not source_path.exists() or not media_meta_path.exists():
            continue

        frames_dir = media_dir / "frames"
        audio_path = media_dir / "audio.wav"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source_path),
                    "-vf",
                    "fps=1",
                    str(frames_dir / "frame_%03d.jpg"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source_path),
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue

        media_meta = read_json(media_meta_path)
        if not isinstance(media_meta, dict):
            continue

        media_meta["status"] = "processed"
        media_meta["frames_dir"] = "frames/"
        media_meta["audio_path"] = "audio.wav"
        write_json(media_meta_path, media_meta)
        processed_count += 1

    typer.echo(f"Processed media for {processed_count} ads")
    return processed_count


def transcribe_media(job_dir: str) -> int:
    job_path = Path(job_dir)
    media_root = job_path / "media"

    if not media_root.exists():
        typer.echo("Transcribed 0 ads")
        typer.echo("Failed transcription for 0 ads")
        return 0

    transcribed_count = 0
    failed_count = 0
    model: WhisperModel | None = None
    model_error: str | None = None

    try:
        model = WhisperModel("small")
    except Exception as exc:
        model_error = str(exc)

    for media_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        audio_path = media_dir / "audio.wav"
        media_meta_path = media_dir / "media_meta.json"
        transcript_path = media_dir / "transcript.txt"
        transcript_segments_path = media_dir / "transcript_segments.json"

        if not audio_path.exists() or not media_meta_path.exists():
            continue

        media_meta = read_json(media_meta_path)
        if not isinstance(media_meta, dict):
            continue

        if model is None:
            media_meta["transcription_error"] = model_error or "Failed to load model"
            write_json(media_meta_path, media_meta)
            failed_count += 1
            continue

        try:
            segments, _info = model.transcribe(str(audio_path))
            segment_payload = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                for segment in segments
            ]
            transcript_text = " ".join(
                segment["text"] for segment in segment_payload if segment["text"]
            ).strip()
        except Exception as exc:
            media_meta["transcription_error"] = str(exc)
            write_json(media_meta_path, media_meta)
            failed_count += 1
            continue

        transcript_path.write_text(transcript_text, encoding="utf-8")
        transcript_segments_path.write_text(
            json.dumps(segment_payload, indent=2),
            encoding="utf-8",
        )
        media_meta["transcript_path"] = "transcript.txt"
        media_meta["transcript_segments_path"] = "transcript_segments.json"
        media_meta.pop("transcription_error", None)
        write_json(media_meta_path, media_meta)
        transcribed_count += 1

    typer.echo(f"Transcribed {transcribed_count} ads")
    typer.echo(f"Failed transcription for {failed_count} ads")
    return transcribed_count


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
