from pathlib import Path
import json
import re
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
            stage = choose_segment_stage(
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
        issues, recommendations = build_issues_and_recommendations(
            hook=hook_segment,
            offers=offers,
            proof_points=proof_points,
            first_cta_time=first_cta_time,
            first_value_time=first_value_time,
            cta_count=len(ctas),
        )
        improvements = build_improvements(
            hook=hook_segment,
            offers=offers,
            value_props=value_props,
            ctas=ctas,
            proof_points=proof_points,
            cta_before_value=structure_labels["cta_before_value"],
        )
        score_payload = build_ad_scores(
            hook=hook_segment,
            proof_points=proof_points,
            value_props=value_props,
            ctas=ctas,
            offers=offers,
            cta_before_value=structure_labels["cta_before_value"],
        )

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
            "issues": issues,
            "recommendations": recommendations,
            "improvements": improvements,
            "hook_score": score_payload["hook_score"],
            "proof_score": score_payload["proof_score"],
            "value_score": score_payload["value_score"],
            "cta_score": score_payload["cta_score"],
            "offer_score": score_payload["offer_score"],
            "ad_score": score_payload["ad_score"],
        }

        write_json(insights_path, insights)
        media_meta["insights_path"] = "insights.json"
        write_json(media_meta_path, media_meta)
        print_ad_summary(
            ad_id=ad_id,
            ad_score=score_payload["ad_score"],
            primary_structure=insights["primary_structure"],
            hook=hook_segment,
            proof_points=proof_points,
            value_props=value_props,
            ctas=ctas,
            offers=offers,
            first_cta_time=first_cta_time,
            improvements=improvements,
        )
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
        "join",
        "click",
        "register",
        "enroll",
        "apply",
        "book now",
        "book",
        "खरीदो",
        "ऑर्डर",
        "जॉइन",
        "रजिस्टर",
        "क्लिक",
    )
    proof_keywords = (
        "result",
        "results",
        "trusted",
        "review",
        "experience",
        "users",
        "guarantee",
        "proven",
        "testimonial",
        "transform",
        "transformation",
        "achieved",
        "already",
        "log kar chuke",
        "solve ho gaya",
        "सॉल्व हो गया",
        "हो गया",
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
        "step by step",
        "live",
        "recording",
        "what you get",
        "get access",
        "feature",
        "features",
    )

    if contains_keyword(text, pain_keywords):
        return "pain_point"
    if contains_keyword(text, offer_keywords):
        return "offer"
    if contains_keyword(text, proof_keywords):
        return "proof"
    if contains_keyword(text, value_keywords):
        return "value_prop"
    if contains_keyword(text, cta_keywords):
        return "cta"
    if is_first and is_meaningful_text(text):
        return "hook"
    return "filler"


def choose_segment_stage(text: str, *, is_first: bool = False) -> str:
    llm_label = classify_segment_with_llm(text)
    fallback_label = classify_segment(text, is_first=is_first)
    stage = llm_label or fallback_label

    if is_first and is_strong_hook(text):
        return "hook"
    if is_proof_text(text):
        return "proof"
    if has_cta_intent(text) and not is_proof_like_action_text(text):
        return "cta"
    if is_value_prop_text(text):
        return "value_prop"
    if stage == "cta" and not has_explicit_cta_action(text):
        return fallback_label if fallback_label != "cta" else "filler"
    return stage


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


def build_issues_and_recommendations(
    *,
    hook: dict[str, str | float],
    offers: list[dict[str, str | float]],
    proof_points: list[dict[str, str | float]],
    first_cta_time: float | None,
    first_value_time: float | None,
    cta_count: int,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    recommendations: list[str] = []

    if (
        first_cta_time is not None
        and first_value_time is not None
        and first_cta_time < first_value_time
    ):
        issues.append("CTA appears before value delivery")
        recommendations.append("Move CTA after explaining value")

    if not offers:
        issues.append("No clear offer or incentive present")
        recommendations.append("Add pricing, discount, or bonus to improve conversion")

    hook_text = str(hook.get("text", "")).strip()
    hook_type = str(hook.get("type", "")).strip().lower()
    if (len(hook_text.split()) < 5 or hook_type == "generic") and not is_strong_hook(hook_text):
        issues.append("Hook is weak or not attention-grabbing")
        recommendations.append("Start with a stronger, disruptive or emotional hook")

    if cta_count > 2:
        issues.append("Too many CTAs may confuse the user")
        recommendations.append("Limit to 1–2 strong CTAs")

    if not proof_points:
        issues.append("No proof or credibility signals present")
        recommendations.append("Add testimonials, results, or social proof")

    return issues, recommendations


def build_ad_scores(
    *,
    hook: dict[str, str | float],
    proof_points: list[dict[str, str | float]],
    value_props: list[dict[str, str | float]],
    ctas: list[dict[str, str | float]],
    offers: list[dict[str, str | float]],
    cta_before_value: bool,
) -> dict[str, int]:
    hook_text = str(hook.get("text", "")).strip()

    hook_score = 20 if hook_text else 0
    if hook_text and is_strong_hook(hook_text):
        hook_score += 10

    proof_score = 20 if proof_points else 0
    value_score = 20 if value_props else 0

    cta_score = 20 if ctas else 0
    if cta_score and cta_before_value:
        cta_score -= 10

    offer_score = 10 if offers else 0

    ad_score = min(
        hook_score + proof_score + value_score + cta_score + offer_score,
        100,
    )

    return {
        "hook_score": hook_score,
        "proof_score": proof_score,
        "value_score": value_score,
        "cta_score": cta_score,
        "offer_score": offer_score,
        "ad_score": ad_score,
    }


def build_improvements(
    *,
    hook: dict[str, str | float],
    offers: list[dict[str, str | float]],
    value_props: list[dict[str, str | float]],
    ctas: list[dict[str, str | float]],
    proof_points: list[dict[str, str | float]],
    cta_before_value: bool,
) -> list[str]:
    improvements: list[str] = []

    if cta_before_value:
        improvements.append("Move CTA after explaining the value proposition")

    if not offers:
        improvements.append("Add a price hook, discount, bonus, or urgency offer")

    if not value_props:
        improvements.append("Explain the user benefit more clearly before asking for action")

    cta_count = len(ctas)
    if cta_count == 0:
        improvements.append("Add a clear CTA like join, click, register, or buy now")
    if cta_count > 2:
        improvements.append("Reduce the number of CTAs to 1 or 2 strong action moments")

    if not proof_points:
        improvements.append("Add social proof, outcomes, or transformation evidence")

    hook_type = str(hook.get("type", "")).strip().lower()
    hook_text = str(hook.get("text", "")).strip()
    if hook_text and hook_type == "generic":
        improvements.append("Make the opening hook more disruptive, emotional, or curiosity-driven")

    return improvements


def print_ad_summary(
    *,
    ad_id: str,
    ad_score: int,
    primary_structure: str,
    hook: dict[str, str | float],
    proof_points: list[dict[str, str | float]],
    value_props: list[dict[str, str | float]],
    ctas: list[dict[str, str | float]],
    offers: list[dict[str, str | float]],
    first_cta_time: float | None,
    improvements: list[str],
) -> None:
    hook_status = "Strong" if str(hook.get("text", "")).strip() else "Missing"
    proof_status = "Present" if proof_points else "Missing"
    value_status = "Present" if value_props else "Missing"
    offer_status = "Present" if offers else "Missing"
    if ctas and first_cta_time is not None:
        cta_status = f"Present ({first_cta_time:.2f}s)"
    elif ctas:
        cta_status = "Present"
    else:
        cta_status = "Missing"

    typer.echo("")
    typer.echo(f"Ad: {ad_id}")
    typer.echo(f"Score: {ad_score}/100")
    typer.echo(f"Flow: {primary_structure}")
    typer.echo(f"Hook: {hook_status}")
    typer.echo(f"Proof: {proof_status}")
    typer.echo(f"Value: {value_status}")
    typer.echo(f"CTA: {cta_status}")
    typer.echo(f"Offer: {offer_status}")
    typer.echo("Top Improvements:")
    if improvements:
        for index, improvement in enumerate(improvements[:3], start=1):
            typer.echo(f"{index}. {improvement}")
    else:
        typer.echo("1. None")


def is_strong_hook(text: str) -> bool:
    strong_hook_keywords = (
        "bakwas",
        "bakwass",
        "waste",
        "scam",
        "wrong",
        "bad",
        "stop",
        "don't",
        "never",
        "useless",
        "बकवास",
        "बकुवास",
        "गलत",
        "मत",
        "नहीं",
    )
    return contains_keyword(text, strong_hook_keywords)


def has_explicit_cta_action(text: str) -> bool:
    cta_action_keywords = (
        "join",
        "click",
        "register",
        "buy",
        "enroll",
        "apply",
        "book now",
        "book",
        "जॉइन",
        "क्लिक",
        "रजिस्टर",
        "खरीदो",
        "ऑर्डर",
        "kar lo",
        "कर लो",
        "ले लो",
    )
    return contains_keyword(text, cta_action_keywords)


def has_cta_intent(text: str) -> bool:
    lowered = text.lower()
    cta_patterns = (
        r"\bjoin\b",
        r"\bclick\b",
        r"\bregister\b",
        r"\bbook\b",
        r"\bapply\b",
        r"\benroll\b",
        r"\bjoin\s+kar\s+lo\b",
        r"\bclick\s+kar(?:o|e)?\b",
        r"\bregister\s+kar(?:o|e)?\b",
        r"\bbook\s+now\b",
        r"\bkar\s*lo\b",
        r"\bkarle\b",
        r"\babhi\b.*\b(join|click|register|book|apply|enroll)\b",
        r"\b(book|click|register)\b.*\bkar\b",
        r"\bbook\b.*\bna\b.*\bpe\b",
        r"\bclick\b.*\bkar\b.*\b(register|book)\b",
        r"जॉइन",
        r"क्लिक",
        r"रजिस्टर",
        r"कर लो",
        r"बुक",
    )
    return any(re.search(pattern, lowered) for pattern in cta_patterns)


def is_proof_like_action_text(text: str) -> bool:
    proof_override_keywords = (
        "kar chuke hain",
        "kar chuke hai",
        "already",
        "people have done",
        "log join kar chuke",
        "लोग",
        "कर चुके",
    )
    return contains_keyword(text, proof_override_keywords)


def is_proof_text(text: str) -> bool:
    proof_keywords = (
        "result",
        "results",
        "transform",
        "transformation",
        "achieved",
        "already",
        "users",
        "proven",
        "testimonial",
        "log kar chuke",
        "kar chuke",
        "कर चुके",
        "लोग",
        "सारे लोग",
        "solve ho gaya",
        "हो गया",
    )
    return contains_keyword(text, proof_keywords)


def is_value_prop_text(text: str) -> bool:
    value_keywords = (
        "step by step",
        "learn",
        "training",
        "course details",
        "live",
        "recording",
        "what you get",
        "get access",
        "benefit",
        "features",
        "feature",
        "easy",
        "effective",
        "save",
        "better",
        "मिलेगी",
        "लाइव",
        "रिकॉर्डिंग",
        "सीख",
        "स्टेप बाय स्टेप",
        "मिलने वाली",
        "मिलने वाले",
        "ट्रेनिंग",
        "कोर्स",
    )
    return contains_keyword(text, value_keywords)
