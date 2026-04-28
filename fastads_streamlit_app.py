import io
import json
import os
import re
import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import time
from datetime import datetime, timezone

import edge_tts
import httpx
import streamlit as st
from dotenv import load_dotenv
from fastads.providers.llm import call_ad_strategy_llm
from googleapiclient.discovery import build
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yt_dlp

st.set_page_config(page_title="FastAds", layout="wide")
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"
UPLOADS_DIR = PROJECT_ROOT / "data" / "ui_uploads"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "").strip()
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "").strip()
MINIMAX_VOICE_ID = "English_expressive_narrator"
EDGE_TTS_EN_VOICE = "en-IN-NeerjaNeural"
EDGE_TTS_HI_VOICE = "hi-IN-SwaraNeural"
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()


def run_pipeline(input_json_path: Path, competitor: str, market: str = "IN") -> tuple[bool, str]:
    """Run the existing FastAds pipeline via CLI and return success flag + logs."""
    cmd = [
        "uv",
        "run",
        "fastads",
        "run",
        "--competitor",
        competitor,
        "--market",
        market,
        "--input",
        str(input_json_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, output.strip()
    except Exception as exc:
        return False, f"Failed to run pipeline: {exc}"



def list_candidate_jobs(start_ts: float) -> List[Tuple[Path, float]]:
    if not DATA_JOBS_DIR.exists():
        return []
    candidates: List[Tuple[Path, float]] = []
    for job in DATA_JOBS_DIR.iterdir():
        if not job.is_dir():
            continue
        mtime = job.stat().st_mtime
        if mtime >= start_ts - 2:
            candidates.append((job, mtime))
    return sorted(candidates, key=lambda pair: pair[1], reverse=True)


def job_has_expected_output(job_dir: Path, ad_ids: List[str], start_ts: float) -> bool:
    pipeline_output_path = job_dir / "pipeline_output.json"
    if not pipeline_output_path.exists():
        return False
    if pipeline_output_path.stat().st_mtime < start_ts:
        return False

    payload = load_json(pipeline_output_path)
    if not isinstance(payload, dict):
        return False

    ads = payload.get("ads", [])
    if not isinstance(ads, list):
        return False

    found_ad_ids = {str(item.get("ad_id", "")) for item in ads if isinstance(item, dict)}
    return all(ad_id in found_ad_ids for ad_id in ad_ids)


def find_latest_completed_job() -> Optional[Path]:
    if not DATA_JOBS_DIR.exists():
        return None

    candidates: List[Tuple[Path, float]] = []
    for job_dir in DATA_JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        output_path = job_dir / "pipeline_output.json"
        if not output_path.exists():
            continue
        payload = load_json(output_path)
        if isinstance(payload, dict) and payload.get("status") == "completed":
            candidates.append((job_dir, output_path.stat().st_mtime))

    if not candidates:
        return None
    return sorted(candidates, key=lambda pair: pair[1], reverse=True)[0][0]


def prepare_upload_dir() -> None:
    if UPLOADS_DIR.exists():
        for child in UPLOADS_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", value).strip()
    return cleaned[:120] or "video"


def parse_youtube_duration_to_seconds(duration: str) -> int:
    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        duration.strip(),
    )
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int) -> str:
    minutes, remaining_seconds = divmod(max(seconds, 0), 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{remaining_minutes:02d}:{remaining_seconds:02d}"
    return f"{remaining_minutes}:{remaining_seconds:02d}"


def score_video_result(video: Dict[str, Any], competitor_name: str) -> Tuple[int, str]:
    title = str(video.get("title", "")).lower()
    competitor_terms = [term for term in re.split(r"\s+", competitor_name.lower().strip()) if term]
    ad_terms = (" ad", "advertisement", "promo", "commercial")
    duration_seconds = int(video.get("duration_seconds", 0) or 0)
    title_has_ad_term = any(term in title for term in ad_terms)
    title_has_brand = any(term in title for term in competitor_terms)

    score = 0
    if duration_seconds <= 60:
        score += 3
    elif duration_seconds <= 120:
        score += 1
    if title_has_ad_term:
        score += 2
    if title_has_brand:
        score += 1

    confidence = "Likely Ad" if duration_seconds <= 60 and title_has_ad_term else "Low Confidence"
    return score, confidence


def search_competitor_videos(competitor_name: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is missing. Add it to your .env or environment.")

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    queries = [
        f"{competitor_name} ad",
        f"{competitor_name} advertisement",
        f"{competitor_name} promo",
        f"{competitor_name} commercial",
    ]
    deduped_results: Dict[str, Dict[str, Any]] = {}

    for query in queries:
        response = (
            youtube.search()
            .list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="relevance",
            )
            .execute()
        )
        items = response.get("items", [])
        for item in items:
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            if not video_id or video_id in deduped_results:
                continue
            thumbnails = snippet.get("thumbnails", {})
            thumbnail = (
                thumbnails.get("medium", {}).get("url")
                or thumbnails.get("high", {}).get("url")
                or thumbnails.get("default", {}).get("url")
                or ""
            )
            deduped_results[video_id] = {
                "video_id": video_id,
                "title": snippet.get("title", "Untitled"),
                "channel": snippet.get("channelTitle", "Unknown Channel"),
                "thumbnail": thumbnail,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "query": query,
            }

    if not deduped_results:
        return []

    metadata_response = (
        youtube.videos()
        .list(
            part="contentDetails",
            id=",".join(deduped_results.keys()),
        )
        .execute()
    )
    metadata_items = metadata_response.get("items", [])
    duration_by_id = {
        str(item.get("id", "")): parse_youtube_duration_to_seconds(
            item.get("contentDetails", {}).get("duration", "PT0S")
        )
        for item in metadata_items
        if item.get("id")
    }

    filtered_results: List[Dict[str, Any]] = []
    for video_id, video in deduped_results.items():
        duration_seconds = duration_by_id.get(video_id, 0)
        if duration_seconds > 120:
            continue
        score, confidence = score_video_result(video, competitor_name)
        filtered_results.append(
            {
                **video,
                "duration_seconds": duration_seconds,
                "duration_label": format_duration(duration_seconds),
                "confidence": confidence,
                "relevance_score": score,
            }
        )

    filtered_results.sort(
        key=lambda item: (
            item.get("confidence") != "Likely Ad",
            -int(item.get("relevance_score", 0)),
            int(item.get("duration_seconds", 0)),
        )
    )
    return filtered_results[:max_results]


def download_selected_video(video: Dict[str, str], competitor_name: str) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_competitor = sanitize_filename(competitor_name or "competitor")
    safe_title = sanitize_filename(video.get("title", "youtube_video"))
    output_template = str(UPLOADS_DIR / f"{safe_competitor}_{safe_title}.%(ext)s")

    options = {
        "format": "mp4/bestvideo+bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(video["url"], download=True)
        downloaded_path = Path(downloader.prepare_filename(info))

    if downloaded_path.suffix.lower() != ".mp4":
        mp4_candidate = downloaded_path.with_suffix(".mp4")
        if mp4_candidate.exists():
            downloaded_path = mp4_candidate

    if not downloaded_path.exists():
        raise RuntimeError("Video download failed. No file was saved.")

    return downloaded_path.resolve()


def sanitize_meta_error_message(message: str) -> str:
    sanitized = message.replace(META_ACCESS_TOKEN, "[REDACTED]") if META_ACCESS_TOKEN else message
    sanitized = re.sub(r"(access_token=)[^&\s]+", r"\1[REDACTED]", sanitized)
    return sanitized


def build_narration_text(script: Dict[str, Any]) -> str:
    parts = [
        _get_script_value(script, "hook"),
        _get_script_value(script, "proof"),
        _get_script_value(script, "value"),
        _get_script_value(script, "cta"),
    ]
    cleaned_parts = [part.strip() for part in parts if part and part.strip() and part.strip() != "—"]
    return " ".join(cleaned_parts).strip()


def select_edge_tts_voice(narration_text: str) -> str:
    if re.search(r"[\u0900-\u097F]", narration_text):
        return EDGE_TTS_HI_VOICE
    return EDGE_TTS_EN_VOICE


async def generate_voiceover_edge_tts(narration_text: str, output_path: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(narration_text, voice=voice)
    await communicate.save(str(output_path))


def ensure_voiceover(job_dir: Path, script: Dict[str, Any]) -> tuple[Path | None, Dict[str, Any] | None]:
    voiceover_path = job_dir / "voiceover.mp3"
    voice_meta_path = job_dir / "voiceover.voice.txt"

    narration_text = build_narration_text(script)
    if not narration_text:
        return None, {
            "provider": "",
            "message": "No script text available for voice generation.",
            "exception": "",
            "status_code": "",
            "response_body": "",
        }

    try:
        selected_voice = select_edge_tts_voice(narration_text)
        def _valid_audio_file(path: Path) -> bool:
            return path.exists() and path.stat().st_size > 10 * 1024

        cached_voice = voice_meta_path.read_text(encoding="utf-8").strip() if voice_meta_path.exists() else ""
        if _valid_audio_file(voiceover_path) and cached_voice == selected_voice:
            return voiceover_path, {"provider": f"Edge TTS:{selected_voice}"}

        if voiceover_path.exists():
            voiceover_path.unlink(missing_ok=True)
        if voice_meta_path.exists():
            voice_meta_path.unlink(missing_ok=True)

        asyncio.run(generate_voiceover_edge_tts(narration_text, voiceover_path, selected_voice))
        if _valid_audio_file(voiceover_path):
            voice_meta_path.write_text(selected_voice, encoding="utf-8")
            return voiceover_path, {"provider": f"Edge TTS:{selected_voice}"}
        if voiceover_path.exists():
            voiceover_path.unlink(missing_ok=True)
        return None, {
            "provider": "Edge TTS",
            "message": "Edge TTS generation failed",
            "exception": "",
            "status_code": "",
            "response_body": "",
        }
    except Exception as exc:
        if voiceover_path.exists():
            voiceover_path.unlink(missing_ok=True)
        return None, {
            "provider": "Edge TTS",
            "message": "Edge TTS generation failed",
            "exception": str(exc),
            "status_code": "",
            "response_body": "",
        }


SCENE_ORDER = ["Hook", "Proof", "Value", "CTA"]
SCENE_DURATIONS = {"Hook": 4, "Proof": 6, "Value": 8, "CTA": 5}
STORYBOARD_RENDER_VERSION = "ai_backgrounds_v2"
SCENE_BACKGROUNDS = {
    "Hook": (16, 18, 24),
    "Proof": (14, 22, 19),
    "Value": (24, 18, 15),
    "CTA": (20, 16, 26),
}
SCENE_ACCENTS = {
    "Hook": (102, 175, 255),
    "Proof": (72, 201, 163),
    "Value": (255, 174, 66),
    "CTA": (255, 111, 145),
}


def _load_storyboard_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Noto Sans Devanagari Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font, spacing=8)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return ""

    words = cleaned.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        test_line = word if not current else f"{current} {word}"
        width, _ = _measure_text(draw, test_line, font)
        if width <= max_width:
            current = test_line
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _draw_centered_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    max_width: int,
    line_spacing: int = 12,
) -> None:
    wrapped = _wrap_text(draw, text, font, max_width)
    if not wrapped:
        return
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=line_spacing, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x1, y1, x2, y2 = box
    x = x1 + max(0, (x2 - x1 - text_width) // 2)
    y = y1 + max(0, (y2 - y1 - text_height) // 2)
    draw.multiline_text((x, y), wrapped, font=font, fill=fill, spacing=line_spacing, align="center")


def _build_pollinations_prompt(scene: Dict[str, str]) -> str:
    stage = str(scene.get("label") or scene.get("stage", "Scene")).strip().title()
    show_text = str(scene.get("show", "")).strip() or str(scene.get("say", "")).strip()
    prompt = (
        f"{stage} scene for a direct-response ad. "
        f"{show_text}. "
        "vertical 9:16 ad visual, cinematic, high quality, realistic, no text, no watermark"
    )
    words = prompt.split()
    if len(words) > 95:
        prompt = " ".join(words[:95])
    return prompt.strip()


def clean_overlay_text(say_text: str) -> str:
    raw_text = str(say_text or "").strip()
    if not raw_text:
        return ""

    filler_words = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "click",
        "for",
        "from",
        "go",
        "here",
        "in",
        "is",
        "it",
        "just",
        "link",
        "now",
        "of",
        "on",
        "please",
        "register",
        "seat",
        "tap",
        "the",
        "this",
        "to",
        "today",
        "try",
        "us",
        "was",
        "we",
        "with",
        "you",
        "your",
    }

    clauses = [part.strip() for part in re.split(r"[.!?;\n]+", raw_text) if part.strip()]
    if not clauses:
        clauses = [raw_text]

    scored_clauses: List[Tuple[int, List[str]]] = []
    for clause in clauses:
        clause = re.sub(r"[\(\)\[\]{}\"“”'’,:|—–-]+", " ", clause)
        clause = re.sub(r"\s+", " ", clause).strip()
        tokens = [token for token in clause.split() if token]
        tokens = [token for token in tokens if token.lower() not in filler_words]
        if tokens:
            scored_clauses.append((len(tokens), tokens))

    if not scored_clauses:
        fallback_tokens = re.sub(r"[\(\)\[\]{}\"“”'’,:|—–-]+", " ", raw_text)
        fallback_tokens = [token for token in re.split(r"\s+", fallback_tokens) if token]
        fallback_tokens = [token for token in fallback_tokens if token.lower() not in filler_words]
        if not fallback_tokens:
            fallback_tokens = [token for token in raw_text.split() if token][:6]
        return " ".join(fallback_tokens[:6]).title().strip()

    scored_clauses.sort(key=lambda item: (item[0], len(" ".join(item[1]))), reverse=True)
    best_tokens = scored_clauses[0][1][:6]
    if len(best_tokens) < 2 and len(scored_clauses) > 1:
        for _score, tokens in scored_clauses[1:]:
            candidate = (best_tokens + tokens)[:6]
            if len(candidate) >= 2:
                best_tokens = candidate
                break

    return " ".join(best_tokens[:6]).title().strip()


def _short_bottom_subtitle(say_text: str) -> str:
    text = clean_overlay_text(say_text)
    if not text:
        return ""
    tokens = [token for token in text.split() if token]
    if len(tokens) <= 4:
        return text
    return " ".join(tokens[:4]).title().strip()


def _draw_subtitle_pill(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Tuple[int, int, int, int],
) -> None:
    if not text:
        return
    x1, y1, x2, y2 = box
    padded_box = (x1 - 24, y1 - 20, x2 + 24, y2 + 20)
    try:
        draw.rounded_rectangle(padded_box, radius=26, fill=(0, 0, 0, 140))
    except Exception:
        draw.rectangle(padded_box, fill=(0, 0, 0, 140))
    draw.multiline_text(
        ((x1 + x2) // 2, (y1 + y2) // 2),
        text,
        font=font,
        fill=(255, 255, 255, 218),
        anchor="mm",
        align="center",
        spacing=10,
    )


def _highlight_ad_copy(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> List[Tuple[str, Tuple[int, int, int]]]:
    words = [word for word in str(text or "").split() if word]
    highlight_map = {
        "Pain": (255, 227, 111),
        "BP": (255, 227, 111),
        "Stress": (255, 227, 111),
        "Fat": (255, 227, 111),
        "Free": (143, 245, 171),
        "14": (143, 245, 171),
        "Day": (143, 245, 171),
        "Plan": (143, 245, 171),
        "Book": (255, 174, 66),
        "Start": (255, 174, 66),
        "Join": (255, 174, 66),
    }
    return [(word, highlight_map.get(word.rstrip(".,!?"), (248, 248, 248))) for word in words]


def _draw_fancy_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    highlight_fill: Tuple[int, int, int],
    max_width: int,
    line_spacing: int = 16,
) -> None:
    wrapped = _wrap_text(draw, text, font, max_width)
    if not wrapped:
        return

    lines = wrapped.splitlines()
    x1, y1, x2, y2 = box
    current_y = y1
    for line in lines:
        words = line.split()
        if not words:
            continue
        line_tokens = _highlight_ad_copy(draw, line, font)
        line_text = " ".join(word for word, _ in line_tokens)
        bbox = draw.textbbox((0, 0), line_text, font=font, stroke_width=3)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        current_x = x1 + max(0, (x2 - x1 - line_width) // 2)
        for token_index, (word, color) in enumerate(line_tokens):
            token_text = word + (" " if token_index < len(line_tokens) - 1 else "")
            token_bbox = draw.textbbox((0, 0), token_text, font=font, stroke_width=3)
            token_width = token_bbox[2] - token_bbox[0]
            token_color = color if color != fill else fill
            draw.text(
                (current_x, current_y),
                token_text,
                font=font,
                fill=token_color,
                stroke_width=3,
                stroke_fill=(0, 0, 0),
            )
            current_x += token_width
        current_y += line_height + line_spacing


def _minimal_show_keywords(show_text: str) -> str:
    raw_text = str(show_text or "").strip()
    if not raw_text:
        return ""

    quoted_phrases = re.findall(r"[\"“”']([^\"“”']+)[\"“”']", raw_text)
    if quoted_phrases:
        candidate = quoted_phrases[0]
    else:
        candidate = re.split(r"[.!?;\n]+", raw_text, maxsplit=1)[0]

    candidate = re.sub(r"[\(\)\[\]{}:|]+", " ", candidate)
    candidate = re.sub(r"\b(button|arrow|overlay|montage|graphic|badge|style|style:|quick|big)\b", " ", candidate, flags=re.I)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    tokens = [token for token in candidate.split() if token]
    if not tokens:
        return ""

    shortlist = tokens[:6]
    result = " ".join(shortlist).strip()
    return result.title()


def _scene_text_layout(stage: str) -> Dict[str, Any]:
    stage_name = str(stage or "").strip().title()
    layouts = {
        "Hook": {
            "label_y": 120,
            "main_box": (90, 160, 990, 620),
            "subtitle_box": (120, 1680, 960, 1825),
            "main_font": 92,
            "sub_font": 38,
            "cta": False,
        },
        "Proof": {
            "label_y": 120,
            "main_box": (90, 260, 990, 940),
            "subtitle_box": (120, 1680, 960, 1825),
            "main_font": 78,
            "sub_font": 38,
            "cta": False,
        },
        "Value": {
            "label_y": 120,
            "main_box": (90, 380, 990, 1120),
            "subtitle_box": (120, 1680, 960, 1825),
            "main_font": 76,
            "sub_font": 38,
            "cta": False,
        },
        "CTA": {
            "label_y": 120,
            "main_box": (180, 1450, 900, 1640),
            "subtitle_box": (120, 1680, 960, 1825),
            "main_font": 80,
            "sub_font": 38,
            "cta": True,
        },
    }
    return layouts.get(
        stage_name,
        {
            "label_y": 120,
            "main_box": (90, 160, 990, 620),
            "subtitle_box": (120, 1680, 960, 1825),
            "main_font": 84,
            "sub_font": 38,
            "cta": False,
        },
    )


def _pollinations_request_scene_image(prompt: str, output_path: Path) -> tuple[bool, Dict[str, Any]]:
    encoded_prompt = quote_plus(prompt)
    url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
    headers = {}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    debug: Dict[str, Any] = {
        "provider": "pollinations",
        "url": url,
        "status_code": "",
        "response_body": "",
        "exception": "",
    }

    try:
        response = httpx.get(
            url,
            params={"width": 1080, "height": 1920, "nologo": "true", "model": "flux"},
            headers=headers,
            timeout=90.0,
        )
    except httpx.HTTPError:
        debug["exception"] = "Request failed"
        return False, debug

    if response.status_code in (401, 402):
        debug["status_code"] = response.status_code
        debug["response_body"] = response.text[:2000]
        return False, debug
    debug["status_code"] = response.status_code
    if response.status_code != 200:
        debug["response_body"] = response.text[:2000]
        return False, debug

    content_type = response.headers.get("content-type", "").lower()
    if "image" not in content_type and not response.content.startswith((b"\xff\xd8\xff", b"\x89PNG")):
        debug["response_body"] = f"Non-image content-type: {content_type or 'unknown'}"
        return False, debug

    try:
        with Image.open(io.BytesIO(response.content)) as generated_image:
            generated_image.convert("RGB").save(output_path)
        ok = output_path.exists() and output_path.stat().st_size > 10 * 1024
        if not ok:
            debug["response_body"] = "Image saved but file too small"
        return ok, debug
    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        debug["exception"] = "Failed to decode image"
        return False, debug


def generate_scene_image_pollinations(prompt: str, output_path: Path) -> bool:
    success, _debug = _pollinations_request_scene_image(prompt, output_path)
    return success


def _storyboard_scene_entries(strategy_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    script = strategy_payload.get("script", {}) or {}
    steps = strategy_payload.get("your_recipe_steps", []) or []
    stage_map: Dict[str, Dict[str, str]] = {}

    def _normalize_stage_name(raw_stage: Any) -> str:
        stage_name = str(raw_stage or "").strip().title()
        if stage_name in SCENE_ORDER:
            return stage_name
        if stage_name.lower().startswith("hook"):
            return "Hook"
        if stage_name.lower().startswith("proof"):
            return "Proof"
        if stage_name.lower().startswith("value"):
            return "Value"
        if stage_name.lower().startswith("cta"):
            return "CTA"
        return stage_name

    def _show_text_from_step(step: Dict[str, Any]) -> str:
        for key in ("show", "visual", "what", "copy", "text", "headline", "screen_text"):
            value = str(step.get(key, "")).strip()
            if value:
                return value
        return str(step.get("say", "")).strip()

    def _say_text_from_step(step: Dict[str, Any], fallback: str) -> str:
        for key in ("say", "voiceover", "narration", "audio"):
            value = str(step.get(key, "")).strip()
            if value:
                return value
        return fallback

    for step in steps:
        if not isinstance(step, dict):
            continue
        stage_name = _normalize_stage_name(step.get("stage", ""))
        if stage_name not in SCENE_ORDER:
            continue
        show_text = _show_text_from_step(step)
        say_text = _say_text_from_step(step, show_text)
        stage_map[stage_name] = {
            "stage": stage_name,
            "show": show_text,
            "say": say_text,
        }

    for stage_name in SCENE_ORDER:
        if stage_name in stage_map:
            continue
        script_value = script.get(stage_name.lower(), "")
        if isinstance(script_value, dict):
            script_show = (
                str(script_value.get("show", "")).strip()
                or str(script_value.get("visual", "")).strip()
                or str(script_value.get("what", "")).strip()
                or str(script_value.get("copy", "")).strip()
                or str(script_value.get("text", "")).strip()
                or str(script_value.get("say", "")).strip()
            )
            script_say = (
                str(script_value.get("say", "")).strip()
                or str(script_value.get("voiceover", "")).strip()
                or script_show
            )
            if script_show or script_say:
                stage_map[stage_name] = {
                    "stage": stage_name,
                    "show": script_show or script_say,
                    "say": script_say or script_show,
                }
            continue

        script_text = str(script_value).strip()
        if not script_text:
            continue
        stage_map[stage_name] = {
            "stage": stage_name,
            "show": script_text,
            "say": script_text,
        }

    ordered_entries: List[Dict[str, str]] = []
    for stage_name in SCENE_ORDER:
        entry = stage_map.get(stage_name)
        if not entry:
            continue
        if not entry.get("show"):
            entry["show"] = entry.get("say", "")
        if not entry.get("say"):
            entry["say"] = entry.get("show", "")
        entry["label"] = stage_name
        ordered_entries.append(entry)
    return ordered_entries


def _render_storyboard_scene_image(
    scene: Dict[str, str],
    output_path: Path,
    background_path: Path | None = None,
) -> None:
    size = (1080, 1920)
    background = SCENE_BACKGROUNDS.get(scene.get("stage", ""), (16, 18, 24))
    accent = SCENE_ACCENTS.get(scene.get("stage", ""), (120, 120, 120))
    image = Image.new("RGB", size, background)
    has_background = bool(background_path and background_path.exists())
    if background_path and background_path.exists():
        try:
            with Image.open(background_path) as base_image:
                base_image = base_image.convert("RGB")
                image = ImageOps.fit(base_image, size, method=Image.LANCZOS, centering=(0.5, 0.5))
        except Exception:
            image = Image.new("RGB", size, background)
    if has_background:
        image = image.convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        for top, alpha in ((0, 72), (150, 96), (300, 116)):
            draw.rectangle((0, top, 1080, top + 280), fill=(0, 0, 0, alpha))
    else:
        draw = ImageDraw.Draw(image)
        panel = (70, 250, 1010, 1610)
        try:
            draw.rounded_rectangle(panel, radius=42, fill=(24, 26, 34), outline=accent, width=4)
        except Exception:
            draw.rectangle(panel, fill=(24, 26, 34), outline=accent, width=4)

    layout = _scene_text_layout(scene.get("stage", ""))
    label_font = _load_storyboard_font(58, bold=True)
    main_font = _load_storyboard_font(layout["main_font"], bold=True)
    sub_font = _load_storyboard_font(layout["sub_font"], bold=False)
    watermark_font = _load_storyboard_font(30, bold=True)

    stage_label = str(scene.get("label") or scene.get("stage", "Scene")).strip().title()
    if has_background:
        draw.text((120, layout["label_y"]), stage_label, font=label_font, fill=accent)
        draw.rectangle((120, 210, 320, 218), fill=accent)
    else:
        label_box = (120, 135, 960, 210)
        draw.text((label_box[0], label_box[1]), stage_label, font=label_font, fill=accent)
        draw.rectangle((120, 220, 320, 228), fill=accent)

    show_text = str(scene.get("show", "")).strip()
    say_text = str(scene.get("say", "")).strip()
    overlay_text = clean_overlay_text(say_text)
    if not overlay_text:
        overlay_text = _minimal_show_keywords(show_text)
    if not overlay_text:
        overlay_text = stage_label

    if layout.get("cta"):
        cta_text = clean_overlay_text(say_text) or overlay_text
        cta_text = cta_text or "Book Free Seat"
        cta_box = (210, 1460, 870, 1580)
        try:
            draw.rounded_rectangle((180, 1440, 900, 1595), radius=48, fill=(41, 172, 91, 235), outline=(255, 255, 255, 60), width=2)
        except Exception:
            draw.rectangle((180, 1440, 900, 1595), fill=(41, 172, 91, 235), outline=(255, 255, 255, 60), width=2)
        draw.text(
            ((cta_box[0] + cta_box[2]) // 2, (cta_box[1] + cta_box[3]) // 2),
            cta_text.title(),
            font=_load_storyboard_font(70, bold=True),
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            anchor="mm",
            align="center",
        )
    else:
        if has_background:
            _draw_fancy_text(
                draw,
                overlay_text.upper(),
                main_font,
                layout["main_box"],
                (248, 248, 248),
                (255, 227, 111),
                800,
                line_spacing=14,
            )
        else:
            _draw_fancy_text(draw, overlay_text.upper(), main_font, layout["main_box"], (248, 248, 248), (255, 227, 111), 800, line_spacing=14)

    if has_background:
        draw.text((80, 1810), "FastAds", font=watermark_font, fill=(240, 240, 240, 210))
        draw.text((790, 1810), "Storyboard preview", font=watermark_font, fill=(240, 240, 240, 170))
    else:
        draw.text((80, 1810), "FastAds", font=watermark_font, fill=(170, 176, 184))
        draw.text((830, 1810), "Storyboard preview", font=watermark_font, fill=(120, 126, 134))
    image.convert("RGB").save(output_path)


def _run_ffmpeg_command(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _generate_storyboard_preview_with_ffmpeg(
    scene_paths: List[Path],
    durations: List[float],
    preview_path: Path,
    voiceover_path: Path | None,
) -> None:
    preview_path.unlink(missing_ok=True)
    concat_list_path = preview_path.with_name("concat_list.txt")
    scene_video_paths: List[Path] = []

    try:
        for scene_path, duration in zip(scene_paths, durations):
            scene_video_path = scene_path.with_suffix(".mp4")
            scene_video_path.unlink(missing_ok=True)
            scene_cmd = [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(scene_path),
                "-c:v",
                "libx264",
                "-t",
                str(duration),
                "-vf",
                "scale=1080:1920,format=yuv420p",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "24",
                "-an",
                str(scene_video_path),
            ]
            result = _run_ffmpeg_command(scene_cmd)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Failed to create scene clip.")
            scene_video_paths.append(scene_video_path)

        filter_inputs = "".join(f"[{index}:v]" for index in range(len(scene_video_paths)))
        filter_complex = f"{filter_inputs}concat=n={len(scene_video_paths)}:v=1:a=0[v]"
        concat_cmd = ["ffmpeg", "-y"]
        for scene_video_path in scene_video_paths:
            concat_cmd.extend(["-i", str(scene_video_path)])
        if voiceover_path and voiceover_path.exists() and voiceover_path.stat().st_size > 10 * 1024:
            concat_cmd.extend(["-i", str(voiceover_path)])
        concat_cmd.extend(["-filter_complex", filter_complex, "-map", "[v]"])
        if voiceover_path and voiceover_path.exists() and voiceover_path.stat().st_size > 10 * 1024:
            concat_cmd.extend(["-map", f"{len(scene_video_paths)}:a", "-shortest"])
        concat_cmd.extend([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
        ])
        if voiceover_path and voiceover_path.exists() and voiceover_path.stat().st_size > 10 * 1024:
            concat_cmd.extend(["-c:a", "aac"])
        else:
            concat_cmd.extend(["-an"])
        concat_cmd.append(str(preview_path))

        result = _run_ffmpeg_command(concat_cmd)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Failed to build storyboard preview.")

        if not preview_path.exists() or preview_path.stat().st_size <= 50 * 1024:
            raise RuntimeError("Generated storyboard preview was empty or too small.")
    finally:
        for scene_video_path in scene_video_paths:
            scene_video_path.unlink(missing_ok=True)
        concat_list_path.unlink(missing_ok=True)


def ensure_generated_ad_preview(
    job_dir: Path,
    strategy_payload: Dict[str, Any],
    voiceover_path: Path | None,
) -> tuple[Path | None, Dict[str, Any] | None]:
    preview_path = job_dir / "generated_ad_preview.mp4"
    scenes_dir = job_dir / "scenes"
    scene_manifest_path = job_dir / "generated_ad_preview.scenes.json"

    def _valid_preview(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 50 * 1024

    if _valid_preview(preview_path) and scene_manifest_path.exists():
        try:
            manifest = load_json(scene_manifest_path)
            if (
                isinstance(manifest, dict)
                and manifest.get("version") == STORYBOARD_RENDER_VERSION
                and isinstance(manifest.get("scenes"), list)
                and any(
                    isinstance(item, dict) and item.get("source") == "AI image"
                    for item in manifest.get("scenes", [])
                )
            ):
                return preview_path, {"provider": "cached", "scene_sources": manifest.get("scenes", [])}
        except Exception:
            pass

    scenes = _storyboard_scene_entries(strategy_payload)
    if not scenes:
        return None, {
            "provider": "ffmpeg",
            "message": "No storyboard stages available for preview generation.",
            "exception": "",
            "status_code": "",
            "response_body": "",
        }

    scenes_dir.mkdir(parents=True, exist_ok=True)
    for child in scenes_dir.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)

    scene_paths: List[Path] = []
    scene_sources: List[Dict[str, str]] = []
    for index, scene in enumerate(scenes, start=1):
        scene_path = scenes_dir / f"scene_{index:02d}_{scene['stage'].lower()}.png"
        background_path = scenes_dir / f"scene_{index:02d}_{scene['stage'].lower()}_ai.png"
        prompt_path = scenes_dir / f"scene_{index:02d}_{scene['stage'].lower()}_ai.prompt.txt"
        used_source = "fallback card"
        ai_prompt = _build_pollinations_prompt(scene)
        cached_prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
        if background_path.exists() and (
            background_path.stat().st_size <= 10 * 1024 or cached_prompt != ai_prompt
        ):
            background_path.unlink(missing_ok=True)
            prompt_path.unlink(missing_ok=True)
        pollinations_debug: Dict[str, Any] = {}
        if not background_path.exists():
            try:
                ai_success, pollinations_debug = _pollinations_request_scene_image(ai_prompt, background_path)
                if ai_success:
                    prompt_path.write_text(ai_prompt, encoding="utf-8")
                    used_source = "AI image"
                else:
                    background_path.unlink(missing_ok=True)
                    prompt_path.unlink(missing_ok=True)
            except Exception:
                background_path.unlink(missing_ok=True)
                prompt_path.unlink(missing_ok=True)
                pollinations_debug = {
                    "provider": "pollinations",
                    "status_code": "",
                    "response_body": "",
                    "exception": "Unexpected generation error",
                }
        if background_path.exists() and background_path.stat().st_size > 10 * 1024:
            _render_storyboard_scene_image(scene, scene_path, background_path)
            used_source = "AI image"
        else:
            _render_storyboard_scene_image(scene, scene_path, None)
        scene_paths.append(scene_path)
        scene_sources.append(
            {
                "stage": str(scene.get("label") or scene.get("stage", "")).strip().title(),
                "source": used_source,
                "image_path": str(background_path if used_source == "AI image" else scene_path),
                "pollinations_debug": pollinations_debug if used_source == "AI image" or pollinations_debug else {},
                "scene_path": str(scene_path),
            }
        )

    try:
        durations = [float(SCENE_DURATIONS.get(scene["stage"], 4)) for scene in scenes]
        _generate_storyboard_preview_with_ffmpeg(scene_paths, durations, preview_path, voiceover_path)
        if _valid_preview(preview_path):
            try:
                scene_manifest_path.write_text(
                    json.dumps(
                        {
                            "version": STORYBOARD_RENDER_VERSION,
                            "scenes": scene_sources,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return preview_path, {"provider": "ffmpeg", "scene_sources": scene_sources}
        if preview_path.exists():
            preview_path.unlink(missing_ok=True)
        return None, {
            "provider": "ffmpeg",
            "message": "Generated storyboard preview was empty or too small.",
            "scene_sources": scene_sources,
            "exception": "",
            "status_code": "",
            "response_body": "",
        }
    except Exception as exc:
        if preview_path.exists():
            preview_path.unlink(missing_ok=True)
        return None, {
            "provider": "ffmpeg",
            "message": "Storyboard preview generation failed.",
            "scene_sources": scene_sources,
            "exception": str(exc),
            "status_code": "",
            "response_body": "",
        }


def search_meta_ads(competitor_name: str, country: str = "IN", limit: int = 6) -> List[Dict[str, Any]]:
    if not META_ACCESS_TOKEN:
        return []

    params = {
        "access_token": META_ACCESS_TOKEN,
        "search_terms": competitor_name,
        "ad_reached_countries": json.dumps([country]),
        "ad_type": "ALL",
        "ad_active_status": "ALL",
        "limit": str(limit),
        "fields": ",".join(
            [
                "id",
                "page_name",
                "ad_creative_bodies",
                "ad_snapshot_url",
                "ad_delivery_start_time",
                "ad_delivery_stop_time",
                "page_id",
            ]
        ),
    }

    with httpx.Client(timeout=20.0) as client:
        response = client.get("https://graph.facebook.com/v20.0/ads_archive", params=params)
        if response.status_code != 200:
            error_body = response.text
            sanitized_error_body = sanitize_meta_error_message(error_body)
            print(f"Meta Ads API error: {sanitized_error_body}")
            raise RuntimeError(sanitized_error_body)

    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []

    return [normalize_meta_ad_result(item, country) for item in data if isinstance(item, dict)]


def normalize_meta_ad_result(item: Dict[str, Any], country: str) -> Dict[str, Any]:
    ad_snapshot_url = str(item.get("ad_snapshot_url", "")).strip()
    page_name = str(item.get("page_name", "")).strip() or "Unknown advertiser"
    bodies = item.get("ad_creative_bodies", [])
    if isinstance(bodies, list):
        body_text = " ".join(str(entry).strip() for entry in bodies if str(entry).strip())
    else:
        body_text = str(bodies).strip()

    start_date = str(item.get("ad_delivery_start_time", "")).strip()
    end_date = str(item.get("ad_delivery_stop_time", "")).strip()
    is_active = not end_date

    return {
        "id": str(item.get("id", "")).strip(),
        "page_name": page_name,
        "body_text": body_text,
        "preview_url": "",
        "status": "Active" if is_active else "Inactive",
        "start_date": format_meta_date(start_date),
        "end_date": format_meta_date(end_date) if end_date else "Running",
        "open_url": ad_snapshot_url or build_meta_ads_library_url(page_name, country),
    }


def format_meta_date(value: str) -> str:
    if not value:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def build_meta_ads_library_url(competitor_name: str, country: str = "IN") -> str:
    meta_query = quote_plus(competitor_name.strip())
    return (
        "https://www.facebook.com/ads/library/"
        f"?active_status=active&ad_type=all&country={country}&q={meta_query}&search_type=keyword_unordered"
    )


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_pipeline_output(job_dir: Path) -> Optional[Dict[str, Any]]:
    payload = load_json(job_dir / "pipeline_output.json")
    return payload if isinstance(payload, dict) else None


def load_strategy_payload(ad_dir: Path) -> Optional[Dict[str, Any]]:
    strategy_path = ad_dir / "strategy.json"
    payload = load_json(strategy_path)
    def _has_meaningful_strategy(data: Dict[str, Any]) -> bool:
        competitor = data.get("competitor_recipe")
        your_recipe = data.get("your_recipe")
        script = data.get("script")
        has_competitor = isinstance(competitor, list) and bool(competitor)
        has_your_recipe = isinstance(your_recipe, list) and bool(your_recipe)
        has_script = isinstance(script, dict) and any(
            str(script.get(key, "")).strip() for key in ("hook", "proof", "value", "cta")
        )
        return has_competitor or has_your_recipe or has_script

    if isinstance(payload, dict) and _has_meaningful_strategy(payload):
        return payload

    raw_path = ad_dir / "strategy_raw_response.json"
    raw_payload = load_json(raw_path)
    if not isinstance(raw_payload, dict):
        return None

    raw_response = raw_payload.get("raw_response", "")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return None

    try:
        parsed = json.loads(raw_response)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None
    if not parsed.get("script"):
        parsed["script"] = {}
    if not parsed.get("competitor_recipe"):
        parsed["competitor_recipe"] = []
    if not parsed.get("your_recipe"):
        parsed["your_recipe"] = []
    if not parsed.get("keep"):
        parsed["keep"] = []
    if not parsed.get("avoid"):
        parsed["avoid"] = []
    if not parsed.get("test"):
        parsed["test"] = []

    if not parsed.get("competitor_recipe") and not parsed.get("your_recipe") and not any(
        str(parsed.get("script", {}).get(key, "")).strip() for key in ("hook", "proof", "value", "cta")
    ):
        return None
    return parsed


def collect_ad_results(job_dir: Path) -> List[Dict[str, Any]]:
    payload = load_pipeline_output(job_dir)
    results: List[Dict[str, Any]] = []
    if not payload:
        return results

    ads = payload.get("ads", [])
    if not isinstance(ads, list):
        return results

    for item in ads:
        if not isinstance(item, dict):
            continue
        insights = item.get("insights", {})
        transcript = item.get("transcript", "")
        ocr_data = item.get("ocr", {})
        ad_id = str(item.get("ad_id", "")).strip() or "unknown_ad"
        results.append(
            {
                "ad_id": ad_id,
                "dir": job_dir / "media" / ad_id,
                "analysis_status": item.get("analysis_status", "completed"),
                "analysis_mode": item.get("analysis_mode", ""),
                "confidence": item.get("confidence", ""),
                "insights": insights,
                "transcript": transcript,
                "ocr": ocr_data,
            }
        )
    return results


def flatten_text_items(items: List[Dict[str, Any]]) -> List[str]:
    return [str(x.get("text", "")).strip() for x in items if str(x.get("text", "")).strip()]


def has_low_ad_signal(insights: Dict[str, Any]) -> bool:
    raw_score = insights.get("ad_score", 0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    primary_structure = str(insights.get("primary_structure", "")).strip().lower()
    has_cta = bool(insights.get("ctas"))
    has_proof = bool(insights.get("proof_points"))
    has_pain = bool(insights.get("pain_points"))
    ad_flow = insights.get("ad_flow", [])
    meaningful_stages = {
        str(stage.get("stage", "")).strip().lower()
        for stage in ad_flow
        if isinstance(stage, dict) and str(stage.get("stage", "")).strip()
    }

    return (
        score <= 50
        or primary_structure in {"", "unknown"}
        or (not has_cta and not has_proof and not has_pain)
        or len(meaningful_stages) < 2
    )


def build_pattern_summary(ad_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "proof_before_cta": 0,
        "cta_present": 0,
        "offer_present": 0,
        "flows": {},
        "pain_points": {},
        "value_props": {},
    }

    for item in ad_results:
        insights = item.get("insights", {})
        structure = insights.get("structure_labels", {})
        if structure.get("proof_before_cta") is True:
            summary["proof_before_cta"] += 1
        if insights.get("ctas"):
            summary["cta_present"] += 1
        if insights.get("offers"):
            summary["offer_present"] += 1

        flow = insights.get("primary_structure") or "unknown"
        summary["flows"][flow] = summary["flows"].get(flow, 0) + 1

        for text in flatten_text_items(insights.get("pain_points", [])):
            summary["pain_points"][text] = summary["pain_points"].get(text, 0) + 1
        for text in flatten_text_items(insights.get("value_props", [])):
            summary["value_props"][text] = summary["value_props"].get(text, 0) + 1

    return summary


def format_flow_segments_for_prompt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments or []:
        stage = str(seg.get("stage", "")).strip().capitalize()
        text = str(seg.get("text", "")).strip()
        start = seg.get("start")
        end = seg.get("end")
        timeframe = ""
        try:
            if start is not None and end is not None:
                timeframe = f"{float(start):.2f}-{float(end):.2f}s"
            elif start is not None:
                timeframe = f"{float(start):.2f}s"
        except (TypeError, ValueError):
            timeframe = ""
        entry = f"{stage} {timeframe}: {text}".strip()
        if entry:
            lines.append(entry)
    return " | ".join(lines) if lines else "None"


def join_items_for_prompt(items: List[str]) -> str:
    return "; ".join(filter(None, (it.strip() for it in items))) or "None"


def safe_script_value(value: Any) -> str:
    if not value:
        return "—"
    return str(value)


def _get_script_value(script: Dict[str, Any], key: str) -> str:
    for candidate in (key, key.capitalize(), key.upper()):
        value = script.get(candidate)
        if value:
            return safe_script_value(value)
    return "—"


def _render_competitor_recipe(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        st.info("No competitor recipe steps were detected.")
        return

    for step in steps:
        cols = st.columns([1, 1, 2, 2])
        cols[0].write(safe_script_value(step.get("timestamp")))
        cols[1].write(safe_script_value(step.get("stage")))
        cols[2].markdown(
            f"**What:** {safe_script_value(step.get('what'))}\n"
            f"**Why:** {safe_script_value(step.get('why'))}"
        )
        formula_value = step.get("formula")
        formula_display = str(formula_value).strip() if formula_value else ""
        if not formula_display:
            formula_display = "N/A"
        cols[3].markdown(
            f"**Formula:** {formula_display}"
        )


def _get_step_field(step: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = step.get(key)
        if value:
            return str(value).strip()
    return ""


def _render_your_recipe(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        return

    st.write("**📋 Shot-by-Shot Breakdown**")
    for step in steps:
        cols = st.columns([1, 1, 3, 3])
        cols[0].write(safe_script_value(_get_step_field(step, "timestamp", "time")))
        cols[1].write(safe_script_value(_get_step_field(step, "stage", "Stage")))
        cols[2].markdown(
            f"**SAY:** {safe_script_value(_get_step_field(step, 'say', 'SAY'))}"
        )
        cols[3].markdown(
            f"**SHOW:** {safe_script_value(_get_step_field(step, 'show', 'SHOW'))}\n"
            f"**WHY:** {safe_script_value(_get_step_field(step, 'why', 'WHY'))}"
        )



def build_strategy_display_payload(normalized: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = normalized or {}
    raw_your_recipe = normalized.get("your_recipe", {})
    your_recipe_steps: List[Dict[str, Any]] = []
    raw_script_source = normalized.get("script", {}) or {}
    script_source: Dict[str, Any] = raw_script_source if isinstance(raw_script_source, dict) else {}

    if isinstance(raw_your_recipe, dict):
        stages = raw_your_recipe.get("stages")
        steps = raw_your_recipe.get("steps")
        if isinstance(stages, list):
            your_recipe_steps = stages
        elif isinstance(steps, list):
            your_recipe_steps = steps
    elif isinstance(raw_your_recipe, list):
        your_recipe_steps = raw_your_recipe

    if not your_recipe_steps and isinstance(script_source, dict):
        script_stages = script_source.get("stages")
        script_steps = script_source.get("steps")
        if isinstance(script_stages, list):
            your_recipe_steps = script_stages
        elif isinstance(script_steps, list):
            your_recipe_steps = script_steps

    raw_competitor_recipe = normalized.get("competitor_recipe", [])
    competitor_steps: List[Dict[str, Any]] = []
    if isinstance(raw_competitor_recipe, list):
        competitor_steps = raw_competitor_recipe
    if not isinstance(competitor_steps, list):
        competitor_steps = []

    def _list_from_sources(key: str) -> List[Any]:
        sources: List[Any] = []
        if isinstance(raw_your_recipe, dict):
            sources.append(raw_your_recipe)
            plan_block = raw_your_recipe.get("plan")
            if isinstance(plan_block, dict):
                sources.append(plan_block)
        sources.append(normalized)
        sources.append({})
        for source in sources:
            if not isinstance(source, dict):
                continue
            value = source.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        return []

    if isinstance(script_source, dict):
        stage_to_key = {
            "hook": "hook",
            "proof": "proof",
            "value": "value",
            "cta": "cta",
        }
        for step in your_recipe_steps:
            if not isinstance(step, dict):
                continue
            stage_name = str(step.get("stage", "")).strip().lower()
            script_key = stage_to_key.get(stage_name)
            if not script_key or str(script_source.get(script_key, "")).strip():
                continue
            derived_value = _get_step_field(step, "say", "text", "what", "copy")
            if derived_value:
                script_source[script_key] = derived_value

    return {
        "competitor_recipe": competitor_steps,
        "your_recipe_steps": your_recipe_steps,
        "keep": _list_from_sources("keep"),
        "avoid": _list_from_sources("avoid"),
        "test": _list_from_sources("test"),
        "script": script_source,
    }


def _copy_script_line(ad_name: str, label: str, text: str) -> None:
    if not text:
        st.warning(f"No {label} copyable content.")
        return
    st.session_state[f"clipboard-{ad_name}-{label}"] = text
    st.success(f"Copied {label} for {ad_name}.")


def _render_script_block(script: Dict[str, Any], ad_name: str) -> None:
    st.write("**Script**")
    script_columns = st.columns(4)
    script_labels = [
        ("hook", "Hook"),
        ("proof", "Proof"),
        ("value", "Value"),
        ("cta", "CTA"),
    ]
    for column, (key, label) in zip(script_columns, script_labels):
        text = _get_script_value(script, key)
        column.write(f"**{label}**")
        column.code(text)
        column.button(
            f"Copy {label}",
            key=f"copy-{ad_name}-{label}",
            on_click=_copy_script_line,
            args=(ad_name, label, text),
        )

def render_strategy_card(ad_name: str, payload: Dict[str, Any]) -> None:
    st.markdown(f"#### {ad_name}")
    st.subheader("Competitor Recipe")
    _render_competitor_recipe(payload.get("competitor_recipe", []))
    st.subheader("🎬 Your Recipe (Based on Your Goal)")
    script = payload.get("script", {})
    _render_script_block(script, ad_name)

    strategy_sections = [
        ("Keep", payload.get("keep", [])),
        ("Avoid", payload.get("avoid", [])),
        ("Test", payload.get("test", [])),
    ]
    columns = st.columns(3)
    for (label, items), column in zip(strategy_sections, columns):
        column.write(f"**{label}**")
        if items:
            for entry in items:
                column.write(f"- {entry}")
        else:
            column.write("None")

    _render_your_recipe(payload.get("your_recipe_steps", []))


def render_voiceover_output(voiceover_path: Path | None, voiceover_debug: Dict[str, Any] | None) -> None:
    if voiceover_debug:
        if voiceover_path is None:
            st.warning(voiceover_debug.get("message", "Voice generation failed"))
        provider_errors = voiceover_debug.get("providers")
        if (
            provider_errors
            or voiceover_debug.get("exception")
            or voiceover_debug.get("response_body")
            or voiceover_debug.get("status_code")
        ):
            with st.expander("Voice generation debug details"):
                st.json(
                    {
                        "provider": voiceover_debug.get("provider", ""),
                        "status_code": voiceover_debug.get("status_code", ""),
                        "response_body": voiceover_debug.get("response_body", ""),
                        "exception": voiceover_debug.get("exception", ""),
                        "providers": provider_errors or [],
                    }
                )

    if not voiceover_path or not voiceover_path.exists():
        return

    selected_path = voiceover_path if voiceover_path.suffix.lower() == ".mp3" else voiceover_path.with_suffix(".mp3")
    if not selected_path.exists() or selected_path.stat().st_size <= 10 * 1024:
        st.error("Voice file invalid or empty")
        st.caption(
            f"Voice path: {selected_path} ({selected_path.stat().st_size / 1024:.1f} KB)"
            if selected_path.exists()
            else f"Voice path: {selected_path} (missing)"
        )
        return

    provider_label = str(voiceover_debug.get("provider", "")).strip() if voiceover_debug else ""
    if provider_label.startswith("Edge TTS:"):
        selected_voice = provider_label.split(":", 1)[1].strip()
        st.write(f"**Generated Voiceover using {selected_voice}**")
    else:
        st.write("**Generated Voiceover (Edge TTS)**")
    st.caption(f"Selected audio path: {selected_path}")
    st.caption(f"File size: {selected_path.stat().st_size / 1024:.1f} KB")
    st.audio(str(selected_path), format="audio/mp3")


def render_generated_ad_preview(
    job_dir: Path,
    strategy_payload: Dict[str, Any],
    voiceover_path: Path | None,
) -> None:
    st.subheader("Generated Storyboard Video Preview")
    preview_path, preview_debug = ensure_generated_ad_preview(job_dir, strategy_payload, voiceover_path)
    if preview_debug and preview_path is None:
        st.warning(preview_debug.get("message", "Generated storyboard preview failed."))
        if preview_debug.get("exception"):
            with st.expander("Preview debug details"):
                st.json(preview_debug)
        return

    if not preview_path or not preview_path.exists():
        st.info("Generated storyboard video preview is not available yet.")
        return

    st.caption("Generated storyboard video preview")
    scene_sources = preview_debug.get("scene_sources", []) if preview_debug else []
    if scene_sources:
        with st.expander("Scene rendering details"):
            for scene in scene_sources:
                st.write(f"- {scene.get('stage', 'Scene')}: {scene.get('source', 'fallback card')}")
                pollinations_debug = scene.get("pollinations_debug", {}) or {}
                if pollinations_debug:
                    st.caption(
                        f"Pollinations status: {pollinations_debug.get('status_code', '') or 'n/a'}"
                    )
                    if pollinations_debug.get("response_body") or pollinations_debug.get("exception"):
                        st.code(
                            json.dumps(pollinations_debug, ensure_ascii=False, indent=2),
                            language="json",
                        )
    st.video(str(preview_path))
    st.download_button(
        "Download Preview",
        data=preview_path.read_bytes(),
        file_name=preview_path.name,
        mime="video/mp4",
        key=f"download-preview-{job_dir.name}",
    )


def render_pipeline_results(job_dir: Path, logs: str = "", from_saved: bool = False) -> None:
    pipeline_output = load_pipeline_output(job_dir)
    if pipeline_output and pipeline_output.get("status") == "placeholder":
        st.error("Analysis incomplete. Please retry.")
        return
    if not pipeline_output:
        st.error("No aggregated pipeline output found for this run.")
        return
    if pipeline_output.get("status") != "completed":
        st.error("Analysis incomplete. Please retry.")
        return

    ad_results = collect_ad_results(job_dir)
    if not ad_results:
        st.error("No ad results found in latest job.")
        return

    if from_saved:
        st.caption(f"Showing latest saved analysis: {job_dir.name}")

    st.subheader("Common Patterns Across Ads")
    patterns = build_pattern_summary(ad_results)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ads analyzed", len(ad_results))
    c2.metric("Proof before CTA", f"{patterns['proof_before_cta']}/{len(ad_results)}")
    c3.metric("CTA present", f"{patterns['cta_present']}/{len(ad_results)}")

    if patterns["flows"]:
        common_flow = max(patterns["flows"].items(), key=lambda x: x[1])[0]
        st.write(f"**Most common flow:** {common_flow}")
        if patterns["proof_before_cta"]:
            st.info("Why it matters: competitors are using trust-building proof before asking for action.")

    if patterns["pain_points"]:
        st.write("**Repeated pain points**")
        st.write(", ".join([k for k, _ in sorted(patterns["pain_points"].items(), key=lambda x: x[1], reverse=True)[:5]]))
    if patterns["value_props"]:
        st.write("**Repeated value propositions**")
        st.write(", ".join([k for k, _ in sorted(patterns["value_props"].items(), key=lambda x: x[1], reverse=True)[:5]]))

    st.subheader("Per-Ad Insights")
    for item in ad_results:
        insights = item["insights"]
        ad_name = insights.get("ad_id", item["ad_id"])
        with st.container(border=True):
            top1, top2 = st.columns([1, 2])
            with top1:
                st.markdown(f"### {ad_name}")
                st.metric("Score", insights.get("ad_score", "N/A"))
            with top2:
                st.write(f"**Flow:** {insights.get('primary_structure', 'N/A')}")
                hook = insights.get("hook", {}) or {}
                st.write(f"**Hook:** {hook.get('text', 'N/A')}")

            if item.get("analysis_mode") == "visual_only":
                st.info("Analysis based on visual signals (no voiceover detected)")
            if item.get("analysis_status") == "no_signal":
                st.warning("No strong ad signals detected. This video may not be a structured ad.")
            if item.get("analysis_status") != "no_signal" and has_low_ad_signal(insights):
                st.warning(
                    "Low ad signal detected. This video may not be a structured ad. Try selecting another result."
                )

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pain points**")
                for txt in flatten_text_items(insights.get("pain_points", [])) or ["None detected"]:
                    st.write(f"- {txt}")

                st.write("**Proof points**")
                for txt in flatten_text_items(insights.get("proof_points", [])) or ["None detected"]:
                    st.write(f"- {txt}")

            with col2:
                st.write("**Value propositions**")
                for txt in flatten_text_items(insights.get("value_props", [])) or ["None detected"]:
                    st.write(f"- {txt}")

                st.write("**CTA**")
                for txt in flatten_text_items(insights.get("ctas", [])) or ["None detected"]:
                    st.write(f"- {txt}")

            st.write("**Issues**")
            for issue in insights.get("issues", []) or ["No major issues detected"]:
                st.write(f"- {issue}")

            st.write("**Recommendations**")
            for rec in insights.get("recommendations", []) or ["No recommendations available"]:
                st.write(f"- {rec}")

            st.write("**Improvements**")
            for imp in insights.get("improvements", []) or ["No improvements available"]:
                st.write(f"- {imp}")

            with st.expander("Transcript"):
                st.text(item.get("transcript", ""))
            with st.expander("OCR text"):
                st.json(item.get("ocr", {}))
            with st.expander("Raw JSON"):
                st.json(insights)

    st.subheader("Ad Recipe & Strategy")
    preview_rendered = False
    for item in ad_results:
        insights = item["insights"]
        ad_name = insights.get("ad_id", item["ad_id"])
        if item.get("analysis_status") == "no_signal":
            st.warning("No strong ad signals detected. This video may not be a structured ad.")
            continue
        if has_low_ad_signal(insights):
            st.warning(
                "Low ad signal detected. This video may not be a structured ad. There may be limited creative learning here. Try selecting a shorter ad, commercial, or promo video."
            )
            continue

        normalized_payload = load_strategy_payload(item["dir"])
        if not isinstance(normalized_payload, dict):
            st.info(f"No saved recipe found for {ad_name}. Run Analyze Ads to generate strategy.")
            continue

        strategy_payload = build_strategy_display_payload(normalized_payload)
        has_script_content = any(
            str(strategy_payload.get("script", {}).get(key, "")).strip()
            for key in ("hook", "proof", "value", "cta")
        )
        has_stage_content = bool(strategy_payload.get("your_recipe_steps"))
        if not has_script_content and not has_stage_content:
            st.warning(f"Recipe generation failed for {ad_name}.")
            continue

        render_strategy_card(ad_name, strategy_payload)
        voiceover_path, voiceover_debug = ensure_voiceover(job_dir, strategy_payload.get("script", {}))
        render_voiceover_output(voiceover_path, voiceover_debug)
        if not preview_rendered:
            render_generated_ad_preview(job_dir, strategy_payload, voiceover_path)
            preview_rendered = True

    if logs:
        with st.expander("Pipeline logs"):
            st.code(logs or "No logs")


st.title("FastAds")
st.caption("Upload a competitor ad. Get the recipe to beat it.")

st.subheader("Search Competitor Videos")
youtube_tab, meta_tab = st.tabs(["YouTube Videos", "Meta Ads Library"])

if "youtube_results" not in st.session_state:
    st.session_state.youtube_results = []
if "selected_video_path" not in st.session_state:
    st.session_state.selected_video_path = None
if "selected_video_meta" not in st.session_state:
    st.session_state.selected_video_meta = None

with youtube_tab:
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_competitor_name = st.text_input(
            "Competitor name",
            key="search_competitor_name",
            placeholder="Enter competitor brand name",
        )
    with search_col2:
        search_clicked = st.button("Search Videos", use_container_width=True)

    if search_clicked:
        st.session_state.selected_video_path = None
        st.session_state.selected_video_meta = None
        if not search_competitor_name.strip():
            st.session_state.youtube_results = []
            st.error("Enter a competitor name to search videos.")
        else:
            try:
                with st.spinner("Searching YouTube videos..."):
                    st.session_state.youtube_results = search_competitor_videos(search_competitor_name.strip())
                if not st.session_state.youtube_results:
                    st.error("No videos found for this competitor.")
            except Exception as exc:
                st.session_state.youtube_results = []
                st.error(str(exc))

    if st.session_state.youtube_results:
        result_columns = st.columns(min(len(st.session_state.youtube_results), 3))
        for index, video in enumerate(st.session_state.youtube_results):
            column = result_columns[index % len(result_columns)]
            with column:
                with st.container(border=True):
                    if video.get("thumbnail"):
                        st.image(video["thumbnail"], use_container_width=True)
                    st.markdown(f"**{video['title']}**")
                    st.caption(video["channel"])
                    confidence = video.get("confidence", "Low Confidence")
                    if confidence == "Likely Ad":
                        st.success(confidence)
                    else:
                        st.warning(confidence)
                    st.caption(f"Duration: {video.get('duration_label', 'Unknown')}")
                    st.write(video["url"])
                    if st.button("Select", key=f"select-video-{video['video_id']}"):
                        try:
                            with st.spinner("Downloading selected video..."):
                                video_path = download_selected_video(
                                    video,
                                    search_competitor_name.strip() or "competitor",
                                )
                            st.session_state.selected_video_path = str(video_path)
                            st.session_state.selected_video_meta = {
                                "title": video["title"],
                                "channel": video["channel"],
                                "url": video["url"],
                                "confidence": confidence,
                                "duration_label": video.get("duration_label", "Unknown"),
                            }
                            st.success("Video selected and downloaded. You can now click Analyze Ads.")
                        except Exception as exc:
                            st.session_state.selected_video_path = None
                            st.session_state.selected_video_meta = None
                            st.error(f"Download failed: {exc}")

with meta_tab:
    meta_competitor_name = st.text_input(
        "Competitor name",
        key="meta_competitor_name",
        placeholder="Enter competitor brand name",
    )
    meta_ads_library_url = build_meta_ads_library_url(meta_competitor_name, "IN")

    st.caption(
        "Meta Ads are shown for discovery. To analyze a Meta video, open the ad, download or screen-record the creative, then upload it to FastAds."
    )

    if not META_ACCESS_TOKEN:
        st.info("Meta Ads integration not configured")
    elif not meta_competitor_name.strip():
        st.info("Enter a competitor name to search Meta ads.")
    else:
        try:
            with st.spinner("Searching Meta Ads Library..."):
                meta_results = search_meta_ads(meta_competitor_name.strip(), country="IN")
            if not meta_results:
                st.warning("No Meta ads found for this competitor.")
            else:
                meta_columns = st.columns(min(len(meta_results), 2))
                for index, ad in enumerate(meta_results):
                    column = meta_columns[index % len(meta_columns)]
                    with column:
                        with st.container(border=True):
                            if ad.get("preview_url"):
                                st.image(ad["preview_url"], use_container_width=True)
                            st.markdown(f"**{ad['page_name']}**")
                            st.caption(ad["status"])
                            st.write(ad.get("body_text") or "No ad copy available.")
                            st.caption(f"Start: {ad.get('start_date', 'Unknown')}")
                            st.caption(f"End: {ad.get('end_date', 'Unknown')}")
                            st.link_button(
                                "Open in Meta Ad Library",
                                ad["open_url"],
                                key=f"meta-open-{ad.get('id', index)}",
                                use_container_width=True,
                            )
        except Exception as exc:
            st.error(f"Meta Ads search failed: {exc}")

    with st.container(border=True):
        st.markdown("**Open Meta Ads Library**")
        st.write(meta_ads_library_url)
        st.link_button("Open Meta Ads Library", meta_ads_library_url, use_container_width=True)

if st.session_state.selected_video_meta and st.session_state.selected_video_path:
    st.info(
        f"Selected video: {st.session_state.selected_video_meta['title']} "
        f"({st.session_state.selected_video_meta['channel']})"
    )

with st.form("fastads_input_form"):
    uploads = st.file_uploader(
        "Upload competitor ads",
        type=["mp4"],
        accept_multiple_files=True,
        help="Upload 1 to 3 ad videos.",
    )
    goal = st.selectbox(
        "What is your campaign goal?",
        ["Lead Generation", "Sales", "Awareness"],
        index=0,
    )
    competitor = st.text_input("Competitor name", value="competitor")
    submitted = st.form_submit_button("Analyze Ads")

if submitted:
    selected_video_path = st.session_state.get("selected_video_path")

    if not uploads and not selected_video_path:
        st.error("Upload at least one ad video or select a competitor video above.")
    elif uploads and len(uploads) > 3:
        st.error("Upload at most 3 ad videos.")
    else:
        input_items: List[Dict[str, Any]] = []

        if uploads:
            prepare_upload_dir()
            for idx, uploaded in enumerate(uploads, start=1):
                save_path = UPLOADS_DIR / f"uploaded_ad_{idx}.mp4"
                save_path.write_bytes(uploaded.getvalue())
                absolute_path = str(save_path.resolve())
                input_items.append(
                    {
                        "ad_id": f"ad_{idx}",
                        "video_url": absolute_path,
                        "local_path": absolute_path,
                        "source": "local",
                        "page_name": competitor,
                        "ad_copy": uploaded.name,
                    }
                )
        elif selected_video_path:
            selected_meta = st.session_state.get("selected_video_meta") or {}
            input_items.append(
                {
                    "ad_id": "ad_1",
                    "video_url": selected_video_path,
                    "local_path": selected_video_path,
                    "source": "local",
                    "page_name": competitor,
                    "ad_copy": selected_meta.get("title", Path(selected_video_path).name),
                }
            )

        input_json_path = UPLOADS_DIR / "ui_input_ads.json"
        input_json_path.write_text(json.dumps(input_items, ensure_ascii=False, indent=2), encoding="utf-8")
        assigned_ad_ids = [item["ad_id"] for item in input_items]
        run_start = time.time()

        progress = st.status("Starting analysis...", expanded=True)
        progress.write("Uploading files")
        progress.write("Extracting media")
        progress.write("Transcribing ad content")
        progress.write("Analyzing ad structure")

        ok, logs = run_pipeline(input_json_path=input_json_path, competitor=competitor)

        if not ok:
            progress.update(label="Analysis failed", state="error", expanded=True)
            st.error("Pipeline failed.")
            st.code(logs or "No logs available")
        else:
            progress.update(label="Analysis complete", state="complete", expanded=False)
            candidates = list_candidate_jobs(run_start)
            candidate_ids = [job.name for job, _ in candidates]
            selected_job: Optional[Path] = None
            selected_job_mtime: Optional[float] = None
            for job, mtime in candidates:
                if job_has_expected_output(job, assigned_ad_ids, run_start):
                    selected_job = job
                    selected_job_mtime = mtime
                    break
            if not selected_job:
                st.error("No fresh job output found for this run. The analysis may have failed.")
            else:
                job_dir = selected_job
                st.session_state.latest_job_dir = str(job_dir)
                pipeline_output = load_pipeline_output(job_dir)
                if pipeline_output and pipeline_output.get("status") == "placeholder":
                    st.error("Analysis incomplete. Please retry.")
                elif not pipeline_output:
                    st.error("No aggregated pipeline output found for this run.")
                elif pipeline_output.get("status") != "completed":
                    st.error("Analysis incomplete. Please retry.")
                else:
                    ad_results = collect_ad_results(job_dir)
                    if not ad_results:
                        st.error("No ad results found in latest job.")
                    else:
                        st.subheader("Common Patterns Across Ads")
                        patterns = build_pattern_summary(ad_results)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Ads analyzed", len(ad_results))
                        c2.metric("Proof before CTA", f"{patterns['proof_before_cta']}/{len(ad_results)}")
                        c3.metric("CTA present", f"{patterns['cta_present']}/{len(ad_results)}")

                        if patterns["flows"]:
                            common_flow = max(patterns["flows"].items(), key=lambda x: x[1])[0]
                            st.write(f"**Most common flow:** {common_flow}")
                            if patterns["proof_before_cta"]:
                                st.info("Why it matters: competitors are using trust-building proof before asking for action.")

                        if patterns["pain_points"]:
                            st.write("**Repeated pain points**")
                            st.write(", ".join([k for k, _ in sorted(patterns["pain_points"].items(), key=lambda x: x[1], reverse=True)[:5]]))
                        if patterns["value_props"]:
                            st.write("**Repeated value propositions**")
                            st.write(", ".join([k for k, _ in sorted(patterns["value_props"].items(), key=lambda x: x[1], reverse=True)[:5]]))

                        st.subheader("Per-Ad Insights")
                        for item in ad_results:
                            insights = item["insights"]
                            ad_name = insights.get("ad_id", item["ad_id"])
                            with st.container(border=True):
                                top1, top2 = st.columns([1, 2])
                                with top1:
                                    st.markdown(f"### {ad_name}")
                                    st.metric("Score", insights.get("ad_score", "N/A"))
                                with top2:
                                    st.write(f"**Flow:** {insights.get('primary_structure', 'N/A')}")
                                    hook = insights.get("hook", {}) or {}
                                    st.write(f"**Hook:** {hook.get('text', 'N/A')}")

                                if item.get("analysis_mode") == "visual_only":
                                    st.info("Analysis based on visual signals (no voiceover detected)")
                                if item.get("analysis_status") == "no_signal":
                                    st.warning("No strong ad signals detected. This video may not be a structured ad.")
                                if item.get("analysis_status") != "no_signal" and has_low_ad_signal(insights):
                                    st.warning(
                                        "Low ad signal detected. This video may not be a structured ad. Try selecting another result."
                                    )

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Pain points**")
                                    for txt in flatten_text_items(insights.get("pain_points", [])) or ["None detected"]:
                                        st.write(f"- {txt}")

                                    st.write("**Proof points**")
                                    for txt in flatten_text_items(insights.get("proof_points", [])) or ["None detected"]:
                                        st.write(f"- {txt}")

                                with col2:
                                    st.write("**Value propositions**")
                                    for txt in flatten_text_items(insights.get("value_props", [])) or ["None detected"]:
                                        st.write(f"- {txt}")

                                    st.write("**CTA**")
                                    for txt in flatten_text_items(insights.get("ctas", [])) or ["None detected"]:
                                        st.write(f"- {txt}")

                                st.write("**Issues**")
                                for issue in insights.get("issues", []) or ["No major issues detected"]:
                                    st.write(f"- {issue}")

                                st.write("**Recommendations**")
                                for rec in insights.get("recommendations", []) or ["No recommendations available"]:
                                    st.write(f"- {rec}")

                                st.write("**Improvements**")
                                for imp in insights.get("improvements", []) or ["No improvements available"]:
                                    st.write(f"- {imp}")

                                with st.expander("Transcript"):
                                    st.text(item.get("transcript", ""))
                                with st.expander("OCR text"):
                                    st.json(item.get("ocr", {}))
                                with st.expander("Raw JSON"):
                                    st.json(insights)

                        st.subheader("Ad Recipe & Strategy")
                        for item in ad_results:
                            insights = item["insights"]
                            ad_name = insights.get("ad_id", item["ad_id"])
                            if item.get("analysis_status") == "no_signal":
                                st.warning(
                                    "No strong ad signals detected. This video may not be a structured ad."
                                )
                                continue
                            if has_low_ad_signal(insights):
                                st.warning(
                                    "Low ad signal detected. This video may not be a structured ad. There may be limited creative learning here. Try selecting a shorter ad, commercial, or promo video."
                                )
                                continue
                            ad_flow_text = format_flow_segments_for_prompt(
                                insights.get("ad_flow", [])
                            )
                            pain_text = join_items_for_prompt(
                                flatten_text_items(insights.get("pain_points", []))
                            )
                            proof_text = join_items_for_prompt(
                                flatten_text_items(insights.get("proof_points", []))
                            )
                            value_text = join_items_for_prompt(
                                flatten_text_items(insights.get("value_props", []))
                            )
                            cta_text = join_items_for_prompt(
                                flatten_text_items(insights.get("ctas", []))
                            )

                            prompt_payload = {
                                "ad_flow": ad_flow_text,
                                "pain_points": pain_text,
                                "proof_points": proof_text,
                                "value_props": value_text,
                                "ctas": cta_text,
                                "goal": goal,
                            }
                            with st.expander(f"Strategy prompt payload for {ad_name}"):
                                st.json(prompt_payload)

                            with st.spinner(f"Crafting strategy for {ad_name}"):
                                normalized_payload = call_ad_strategy_llm(
                                    ad_flow_text,
                                    pain_text,
                                    proof_text,
                                    value_text,
                                    cta_text,
                                    goal,
                                )

                            raw_response = getattr(call_ad_strategy_llm, "last_raw_response", None)
                            parse_error = getattr(call_ad_strategy_llm, "last_parse_error", None)
                            fallback_warning = (
                                normalized_payload.get("_fallback_warning")
                                if normalized_payload
                                else None
                            )

                            ad_media_dir = item.get("dir")
                            if raw_response and normalized_payload and isinstance(ad_media_dir, Path):
                                (ad_media_dir / "strategy_raw_response.json").write_text(
                                    json.dumps({"raw_response": raw_response}, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                            if normalized_payload and isinstance(ad_media_dir, Path):
                                (ad_media_dir / "strategy.json").write_text(
                                    json.dumps(normalized_payload, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )

                            with st.expander(f"LLM raw response for {ad_name}"):
                                if raw_response:
                                    st.text(raw_response)
                                else:
                                    st.text("<no response>")
                                if parse_error:
                                    st.error(f"Parse error: {parse_error}")
                                if fallback_warning:
                                    st.info(fallback_warning)

                            strategy_payload = build_strategy_display_payload(normalized_payload)
                            has_script_content = any(
                                str(strategy_payload.get("script", {}).get(key, "")).strip()
                                for key in ("hook", "proof", "value", "cta")
                            )
                            has_stage_content = bool(strategy_payload.get("your_recipe_steps"))
                            if not has_script_content and not has_stage_content:
                                st.warning(
                                    f"Recipe generation failed for {ad_name}."
                                )
                                continue
                            render_strategy_card(ad_name, strategy_payload)

                            voiceover_path: Path | None = None
                            voiceover_debug: Dict[str, Any] | None = None
                            voiceover_path, voiceover_debug = ensure_voiceover(
                                job_dir, strategy_payload.get("script", {})
                            )
                            render_voiceover_output(voiceover_path, voiceover_debug)

                    with st.expander("Pipeline logs"):
                        st.code(logs or "No logs")
else:
    restored_job: Optional[Path] = None
    latest_job_dir = st.session_state.get("latest_job_dir")
    if latest_job_dir:
        candidate = Path(str(latest_job_dir))
        if candidate.exists() and load_pipeline_output(candidate):
            restored_job = candidate
    if restored_job is None:
        restored_job = find_latest_completed_job()

    if restored_job:
        st.divider()
        render_pipeline_results(restored_job, from_saved=True)
