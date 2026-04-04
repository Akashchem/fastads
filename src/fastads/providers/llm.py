import re
import json

import httpx
from openai import AzureOpenAI
import typer

from fastads.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
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
        if FASTADS_LLM_PROVIDER == "azure_openai":
            typer.echo("Using LLM provider: azure_openai")
            typer.echo(f"Using deployment: {AZURE_OPENAI_DEPLOYMENT}")
        else:
            typer.echo(
                f"Using LLM classifier provider={FASTADS_LLM_PROVIDER} model={FASTADS_LLM_MODEL}"
            )
        _has_logged_provider = True

    messages = [
        {
            "role": "system",
            "content": (
                "Classify this ad segment into one of "
                "[hook, pain_point, value_prop, proof, offer, cta, filler]. "
                "CTA: only use cta if there is a clear user action instruction such as "
                "join, click, register, buy, enroll, apply, or informal phrasing like kar lo. "
                "Conversational CTA such as 'to abhi join kar lo' must be cta. "
                "Value Prop: explanations of benefits, features, learning, training, course details, "
                "live format, recording access, or what the user gets must be value_prop, not filler. "
                "Proof: if text describes results, transformation, or that people have "
                "already achieved something, classify as proof, never cta. "
                "Hook: if the first line is bold, negative, or controversial, treat it as a strong hook. "
                "Pain Point: treat fragments like dark circles, double chin, puffiness, wrinkles as pain_points. "
                "If unsure between filler and cta, prefer cta when action intent exists. "
                "If unsure between cta and proof, choose proof. "
                "Return JSON with flow_label and an extracted map."
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    if FASTADS_LLM_PROVIDER == "azure_openai":
        response = classify_segment_with_azure_openai(messages)
        return response

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
                    "messages": messages,
                },
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return parse_llm_output(content)


def classify_segment_with_azure_openai(messages: list[dict[str, str]]) -> str | None:
    if (
        not AZURE_OPENAI_API_KEY
        or not AZURE_OPENAI_ENDPOINT
        or not AZURE_OPENAI_API_VERSION
        or not AZURE_OPENAI_DEPLOYMENT
    ):
        return None

    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_completion_tokens=2048,
        )
    except Exception:
        return None

    content = response.choices[0].message.content or ""
    return parse_llm_output(content)


def parse_llm_output(content: str) -> dict[str, object] | None:
    try:
        payload = json.loads(content)
    except ValueError:
        label = extract_label(content.lower())
        return {
            "flow_label": label,
            "extracted": {
                "value_props": [],
                "offers": [],
                "pain_points": [],
                "proof_points": [],
            },
        }

    flow_label = payload.get("flow_label")
    raw_extracted = payload.get("extracted", {})
    extracted = {}
    for key in ("value_props", "offers", "pain_points", "proof_points"):
        values = raw_extracted.get(key, [])
        if isinstance(values, str):
            values = [values]
        extracted[key] = [str(item).strip() for item in values if str(item).strip()]

    return {"flow_label": flow_label, "extracted": extracted}


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
