# FastAds

Marketing insight engine for video ads that ingests MP4s, runs FFmpeg/whisper/Tesseract, and surfaces marketer-ready recipes via an LLM.

# What this does

Processes competitor ads end-to-end, from ingestion to media extraction through LLM analysis, then renders summaries in both the CLI job folder (`data/jobs/<job_id>/media/<ad_id>/…`) and the Streamlit UI.

Outputs include:
- `ocr.json` — text extracted from video frames
- `transcript.txt` / `transcript_segments.json` — audio transcription + timestamped segments
- `insights.json` — structured marketing insights, scoring, issues, recommendations, and improvements
- `pipeline_output.json` — aggregation containing counts for ingest / media / transcript / analysis

## Pipeline

1. Ingest ad JSON (remote URL or local `video_url`/`local_path`)
2. Prepare media folders (`data/jobs/<job_id>/media/<ad_id>`)
3. Download (or copy fallback video) → write `media_meta.json`
4. Extract frames/audio via FFmpeg + OCR via Tesseract
5. Transcribe audio (`faster-whisper` locally or streaming to custom Whisper API)
6. Normalize transcript segments → classify with the LLM + rule fallback
7. Generate insights, scores, issues, recommendations, improvements, and strategy payloads (`insights.json`)
8. Streamlit UI loads the latest job and displays:
   - pattern summary (common flows, CTAs, proof-before-CTA counts)
   - Per-ad cards with price/value/pain/proof breakdown + Shot-by-Shot Breakdown and script copy buttons
   - Strategy LLM output grouped as Competitor Recipe, Your Recipe, Keep/Avoid/Test, script cards, and copyable lines

## Streamlit MVP (optional)

Run `uv run streamlit run fastads_streamlit_app.py` and upload 1‑3 MP4s. Choose a goal (Lead Generation/Sales/Awareness) and let the app:

- display per-ad score/profile (flow, hook, pain points, proof, value, CTA, issues/recs/improvements)
- show Competitor Recipe steps + Your Recipe Shot-by-Shot Breakdown (SAY/SHOW/WHY) with copy-to-clipboard buttons
- expose strategy script lines (Hook/Proof/Value/CTA) and keep/avoid/test guidance derived from the latest job
- keep the CLI insights link in an expander, including prompt payload + raw LLM response for debugging

Jobs are stored at `data/jobs/<job_id>/media/<ad_id>/` for offline use.

## Status

🚀 Demo ready — CLI job + Streamlit recipe, with fallback downloads, OCR, transcription, and structured LLM insights in place.

## Run

CLI:

```bash
uv run fastads run --competitor "brand" --market IN --input sample_ads.json
```

Streamlit UI:

```bash
uv run streamlit run fastads_streamlit_app.py
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
FASTADS_TRANSCRIBER=local          # faster-whisper
FASTADS_TRANSCRIBER=remote         # custom Whisper proxy
WHISPER_API_URL=...
WHISPER_AUTH_TOKEN=...
```

Remote mode uses a custom JSON/base64 contract with `vad_filter: true` unless you configure `FASTADS_TRANSCRIBER=local`.
