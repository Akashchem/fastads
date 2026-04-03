# FastAds

Marketing insight engine for video ads using transcription + LLM-based structure analysis.

## What this does

Takes ad inputs -> processes video -> transcribes audio -> extracts timestamped segments -> classifies ad structure -> outputs marketing insights.

Outputs include:
- transcript.txt
- transcript_segments.json
- insights.json
- pipeline_output.json

## Pipeline

1. Ingest ad JSON
2. Download video (fallback to local sample if needed)
3. Extract frames + audio (FFmpeg)
4. Transcribe audio (faster-whisper)
5. Split into timestamped segments
6. Classify each segment (LLM + rule fallback)
7. Generate structured insights (insights.json)

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
