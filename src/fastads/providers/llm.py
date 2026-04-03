import re

import httpx

from fastads.config import FASTADS_LLM_API_BASE, FASTADS_LLM_MODEL, FASTADS_OPENAI_API_KEY

VALID_SEGMENT_LABELS = {
    "hook",
    "pain_point",
    "value_prop",
    "proof",
    "offer",
    "cta",
    "filler",
}


def classify_segment_with_llm(text: str) -> str | None:
    if not FASTADS_OPENAI_API_KEY or not text.strip():
        return None

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                f"{FASTADS_LLM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {FASTADS_OPENAI_API_KEY}",
                    "Content-Type": "application/json",
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
