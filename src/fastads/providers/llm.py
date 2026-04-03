import re

import httpx
import typer

from fastads.config import (
    FASTADS_LLM_API_BASE,
    FASTADS_LLM_API_KEY,
    FASTADS_LLM_MODEL,
    FASTADS_LLM_PROVIDER,
)

VALID_SEGMENT_LABELS = {
    "hook",
    "pain_point",
    "value_prop",
    "proof",
    "offer",
    "cta",
    "filler",
}
_has_logged_provider = False


def classify_segment_with_llm(text: str) -> str | None:
    global _has_logged_provider

    if not FASTADS_LLM_API_KEY or not text.strip():
        return None

    if not _has_logged_provider:
        typer.echo(
            f"Using LLM classifier provider={FASTADS_LLM_PROVIDER} model={FASTADS_LLM_MODEL}"
        )
        _has_logged_provider = True

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                f"{FASTADS_LLM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {FASTADS_LLM_API_KEY}",
                    "Content-Type": "application/json",
                    **provider_headers(),
                },
                json={
                    "model": FASTADS_LLM_MODEL,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Classify this ad segment into one of "
                                "[hook, pain_point, value_prop, proof, offer, cta, filler]. "
                                "Return only label."
                            ),
                        },
                        {
                            "role": "user",
                            "content": text,
                        },
                    ],
                },
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    payload = response.json()
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
        .lower()
    )
    return extract_label(content)


def provider_headers() -> dict[str, str]:
    if FASTADS_LLM_PROVIDER != "openrouter":
        return {}
    return {
        "HTTP-Referer": "https://github.com/Akashchem/fastads",
        "X-Title": "FastAds",
    }


def extract_label(content: str) -> str | None:
    if content in VALID_SEGMENT_LABELS:
        return content

    match = re.search(
        r"\b(hook|pain_point|value_prop|proof|offer|cta|filler)\b",
        content,
    )
    if not match:
        return None

    label = match.group(1)
    return label if label in VALID_SEGMENT_LABELS else None
