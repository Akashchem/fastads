# FastAds

Marketing insight engine for video ads using transcription, OCR, and LLM-based structure analysis.

## What this does

Takes ad inputs -> processes video -> extracts frames/audio/OCR -> transcribes audio -> classifies ad structure -> outputs marketing insights.

Outputs include:
- ocr.json
- transcript.txt
- transcript_segments.json
- insights.json
- pipeline_output.json

## Pipeline

1. Ingest ad JSON
2. Download video or copy from a local path
3. Extract frames + audio (FFmpeg)
4. Extract OCR from frames (Tesseract)
5. Transcribe audio
   Local mode: faster-whisper
   Remote mode: custom JSON/base64 Whisper backend
6. Split into timestamped segments
7. Classify each segment (LLM + rule fallback)
8. Generate structured insights (insights.json)

## Example Insight

- First CTA at: 17.8s
- First Value Prop at: 24.0s
- Structure: pain -> proof -> hook -> CTA -> value

Insight:
CTA appears before value delivery -> may hurt conversion

## Status

🚧 MVP (working end-to-end)

## Run

```bash
uv run fastads run --competitor "brand" --market IN --input sample_ads.json
```

## Input

Remote video URL:

```json
[
  {
    "ad_id": "ad_1",
    "page_name": "brand",
    "ad_copy": "sample ad",
    "video_url": "https://example.com/video.mp4"
  }
]
```

Local file:

```json
[
  {
    "ad_id": "ad_2",
    "page_name": "brand",
    "ad_copy": "sample ad",
    "video_url": "local",
    "local_path": "assets/new_ad.mp4"
  }
]
```

## Transcriber

Environment variables:

```bash
FASTADS_TRANSCRIBER=local
FASTADS_TRANSCRIBER=remote
WHISPER_API_URL=...
WHISPER_AUTH_TOKEN=...
```

Remote mode currently uses a custom JSON/base64 contract with `vad_filter: true`, not an OpenAI multipart upload.
