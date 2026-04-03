from pathlib import Path
import json
import shutil
import subprocess
from urllib import error, request

import typer
from faster_whisper import WhisperModel

from fastads.providers.llm import classify_segment_with_llm
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


def analyze_transcript(job_dir: str) -> int:
    job_path = Path(job_dir)
    media_root = job_path / "media"

    if not media_root.exists():
        typer.echo("Analyzed transcripts for 0 ads")
        return 0

    analyzed_count = 0

    for media_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        transcript_segments_path = media_dir / "transcript_segments.json"
        media_meta_path = media_dir / "media_meta.json"
        insights_path = media_dir / "insights.json"

        if not transcript_segments_path.exists() or not media_meta_path.exists():
            continue

        media_meta = read_json(media_meta_path)
        if not isinstance(media_meta, dict):
            continue

        ad_id = str(media_meta.get("ad_id", media_dir.name))
        segments = read_json(transcript_segments_path)
        if not isinstance(segments, list):
            continue

        normalized_segments = [
            normalize_segment(segment)
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("text", "")).strip()
        ]
        if not normalized_segments:
            continue

        full_text = " ".join(segment["text"] for segment in normalized_segments)
        language = "hindi_or_hinglish" if has_devanagari(full_text) else "english"

        pain_points: list[dict[str, str | float]] = []
        value_props: list[dict[str, str | float]] = []
        offers: list[dict[str, str | float]] = []
        ctas: list[dict[str, str | float]] = []
        proof_points: list[dict[str, str | float]] = []
        ad_flow: list[dict[str, str | float]] = []
        hook_segment: dict[str, str | float] | None = None

        for index, segment in enumerate(normalized_segments):
            stage = classify_segment_with_llm(str(segment["text"])) or classify_segment(
                str(segment["text"]),
                is_first=index == 0,
            )
            block = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            }

            if stage == "hook":
                hook_segment = hook_segment or block
            elif stage == "pain_point":
                pain_points.append(block)
            elif stage == "value_prop":
                value_props.append(block)
            elif stage == "offer":
                offers.append(block)
            elif stage == "cta":
                ctas.append(block)
            elif stage == "proof":
                proof_points.append(block)

            ad_flow.append(
                {
                    "stage": stage,
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                }
            )

        if hook_segment is None:
            first_segment = normalized_segments[0]
            hook_segment = {
                "text": first_segment["text"],
                "type": "generic",
                "start": first_segment["start"],
                "end": first_segment["end"],
            }
        else:
            hook_segment = {
                "text": hook_segment["text"],
                "type": classify_hook_type(str(hook_segment["text"])),
                "start": hook_segment["start"],
                "end": hook_segment["end"],
            }

        summary_segments = [
            segment["text"]
            for segment in normalized_segments
            if is_meaningful_text(segment["text"])
        ][:3]
        summary = " ".join(summary_segments).strip()
        first_pain_time = first_stage_time(pain_points)
        first_value_time = first_stage_time(value_props)
        first_proof_time = first_stage_time(proof_points)
        first_offer_time = first_stage_time(offers)
        first_cta_time = first_stage_time(ctas)
        structure_labels = {
            "pain_before_value": compare_stage_times(first_pain_time, first_value_time),
            "proof_before_cta": compare_stage_times(first_proof_time, first_cta_time),
            "cta_after_value": compare_stage_times(first_value_time, first_cta_time),
            "cta_before_value": compare_stage_times(first_cta_time, first_value_time),
            "offer_before_cta": compare_stage_times(first_offer_time, first_cta_time),
        }

        insights = {
            "ad_id": ad_id,
            "language": language,
            "pain_points": pain_points,
            "summary": summary,
            "hook": hook_segment,
            "value_props": value_props,
            "offers": offers,
            "ctas": ctas,
            "proof_points": proof_points,
            "ad_flow": ad_flow,
            "structure_labels": structure_labels,
            "primary_structure": build_primary_structure(ad_flow),
            "first_cta_time": first_cta_time,
            "first_value_time": first_value_time,
            "first_proof_time": first_proof_time,
            "first_offer_time": first_offer_time,
            "cta_count": len(ctas),
        }

        write_json(insights_path, insights)
        media_meta["insights_path"] = "insights.json"
        write_json(media_meta_path, media_meta)
        analyzed_count += 1

    typer.echo(f"Analyzed transcripts for {analyzed_count} ads")
    return analyzed_count


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    for marker in ("?", "!", "।"):
        normalized = normalized.replace(marker, ".")
    return [segment.strip() for segment in normalized.split(".") if segment.strip()]


def first_meaningful_sentence(sentences: list[str]) -> str:
    for sentence in sentences:
        if len(sentence.split()) >= 3:
            return sentence
    return sentences[0] if sentences else ""


def contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def first_sentence_with_keyword(sentences: list[str], keywords: tuple[str, ...]) -> str:
    for sentence in sentences:
        if contains_keyword(sentence, keywords):
            return sentence
    return ""


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def normalize_segment(segment: dict) -> dict[str, str | float]:
    return {
        "text": str(segment.get("text", "")).strip(),
        "start": float(segment.get("start", 0.0)),
        "end": float(segment.get("end", 0.0)),
    }


def classify_segment(text: str, *, is_first: bool = False) -> str:
    pain_keywords = (
        "problem",
        "issue",
        "pain",
        "struggle",
        "solution",
        "समस्या",
        "परेशानी",
        "ठीक",
        "इलाज",
    )
    offer_keywords = (
        "%",
        "off",
        "discount",
        "free",
        "sale",
        "offer",
        "₹",
        "rs",
        "रुपये",
    )
    cta_keywords = (
        "buy",
        "shop",
        "order",
        "join",
        "click",
        "खरीदो",
        "ऑर्डर",
        "जॉइन",
        "अभी",
    )
    proof_keywords = (
        "result",
        "trusted",
        "review",
        "experience",
        "users",
        "guarantee",
        "proven",
        "testimonial",
    )
    value_keywords = (
        "improve",
        "save",
        "learn",
        "fast",
        "better",
        "easy",
        "effective",
        "benefit",
        "help",
        "solve",
        "transform",
    )

    if contains_keyword(text, pain_keywords):
        return "pain_point"
    if contains_keyword(text, offer_keywords):
        return "offer"
    if contains_keyword(text, cta_keywords):
        return "cta"
    if contains_keyword(text, proof_keywords):
        return "proof"
    if contains_keyword(text, value_keywords):
        return "value_prop"
    if is_first and is_meaningful_text(text):
        return "hook"
    return "filler"


def classify_hook_type(text: str) -> str:
    if contains_keyword(
        text,
        (
            "problem",
            "issue",
            "pain",
            "struggle",
            "solution",
            "समस्या",
            "परेशानी",
        ),
    ):
        return "problem"
    if contains_keyword(text, ("discount", "%", "off", "free", "sale", "offer", "₹", "rs")):
        return "offer"
    return "generic"


def is_meaningful_text(text: str) -> bool:
    return len(text.split()) >= 3


def compare_stage_order(
    first_stage_items: list[dict[str, str | float]],
    second_stage_items: list[dict[str, str | float]],
    *,
    before: bool,
) -> bool:
    if not first_stage_items or not second_stage_items:
        return False

    first_start = float(first_stage_items[0]["start"])
    second_start = float(second_stage_items[0]["start"])
    if before:
        return first_start < second_start
    return first_start > second_start


def first_stage_time(items: list[dict[str, str | float]]) -> float | None:
    if not items:
        return None
    return float(items[0]["start"])


def compare_stage_times(first_time: float | None, second_time: float | None) -> bool:
    if first_time is None or second_time is None:
        return False
    return first_time < second_time


def build_primary_structure(ad_flow: list[dict[str, str | float]]) -> str:
    included_stages = {"hook", "pain_point", "proof", "value_prop", "offer", "cta"}
    ordered: list[str] = []
    for block in ad_flow:
        stage = str(block["stage"])
        if stage in included_stages and stage not in ordered:
            ordered.append(stage)
    return " -> ".join(ordered)
