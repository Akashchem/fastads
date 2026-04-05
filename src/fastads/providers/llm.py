import re
import json
from typing import Any, Dict, List, Optional, Tuple

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
    if not FASTADS_LLM_API_KEY or not text.strip():
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are an ad analyst. Analyze the ad segment and return ONLY valid JSON in this exact format:\n"
                "{\n"
                '  "flow_label": "one of: hook | pain_point | value_prop | proof | offer | cta | filler",\n'
                '  "extracted": {\n'
                '    "pain_points": ["problems or symptoms, including fragments like dark circles, double chin, puffiness, wrinkles"],\n'
                '    "value_props": ["benefits, features, format: live class, recording, step by step"],\n'
                '    "offers": ["price or discount: 99 rupees, free, bonus"],\n'
                '    "proof_points": ["results others achieved: pigmentation solve ho gaya"]\n'
                "  }\n"
                "}\n\n"
                "Rules:\n"
                "- flow_label must be exactly one label\n"
                "- extracted fields may contain multiple items from the same segment\n"
                "- Even if flow_label is cta, still extract value_props and offers present in the text\n"
                "- Treat single word fragments like dark circles, puffiness, wrinkles as pain_points\n"
                "- Proof: results already achieved by others\n"
                "- If unsure between cta and proof, choose proof\n"
                "- Always return all extracted keys even if empty\n"
                "- offers: extract price as an offer ONLY when paired with value signals like sirf, सिर्फ, only, limited, special, abhi, अभी, ₹, rupees, rupiye, discount, or urgency words. A bare price with no framing is not an offer\n"
                "- Return ONLY the JSON object, no explanation"
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    content = _call_chat_completion(messages, response_format={"type": "json_object"})
    if not content:
        return None

    return parse_llm_output(content)


def classify_segment_with_azure_openai(messages: list[dict[str, str]]) -> str | None:
    # kept for backward compatibility
    return None


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


def _ensure_provider_logged() -> None:
    global _has_logged_provider
    if _has_logged_provider:
        return

    if FASTADS_LLM_PROVIDER == "azure_openai":
        typer.echo("Using LLM provider: azure_openai")
        typer.echo(f"Using deployment: {AZURE_OPENAI_DEPLOYMENT}")
    else:
        typer.echo(
            f"Using LLM classifier provider={FASTADS_LLM_PROVIDER} model={FASTADS_LLM_MODEL}"
        )

    _has_logged_provider = True


def _call_chat_completion(
    messages: list[dict[str, str]], response_format: dict | None = None
) -> str | None:
    if not FASTADS_LLM_API_KEY:
        return None

    _ensure_provider_logged()

    if FASTADS_LLM_PROVIDER == "azure_openai":
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
            kwargs = {
                "model": AZURE_OPENAI_DEPLOYMENT,
                "messages": messages,
                "max_completion_tokens": 2048,
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
        except Exception:
            return None

        return response.choices[0].message.content or ""

    try:
        with httpx.Client(timeout=20.0) as client:
            payload = {
                "model": FASTADS_LLM_MODEL,
                "temperature": 0,
                "messages": messages,
            }
            if response_format:
                payload["response_format"] = response_format
            response = client.post(
                f"{FASTADS_LLM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {FASTADS_LLM_API_KEY}",
                    "Content-Type": "application/json",
                    **provider_headers(),
                },
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    return payload.get("choices", [{}])[0].get("message", {}).get("content", "")


def _coerce_steps(raw: Any, allowed_keys: Tuple[str, ...], allowed_stages: set[str]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            if len(normalized) >= 5:
                break
            if not isinstance(entry, dict):
                continue
            stage = str(entry.get("stage", "")).strip()
            if stage and stage not in allowed_stages:
                continue
            normalized.append({key: str(entry.get(key, "")).strip() for key in allowed_keys})
    return normalized


def _normalize_strategy_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    allowed = {"Hook", "Proof", "Value", "CTA"}
    competitor_keys = ("timestamp", "stage", "what", "why", "formula")
    def _extract_competitor(raw: Any) -> Optional[Any]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("steps", "what_we_observed", "observations"):
                candidate = raw.get(key)
                if isinstance(candidate, list):
                    return candidate
            # Some payloads put steps under a nested key
            for parent in ("plan", "manifest", "structure"):
                child = raw.get(parent)
                if isinstance(child, dict):
                    for key in ("steps", "what_we_observed", "observations"):
                        candidate = child.get(key)
                        if isinstance(candidate, list):
                            return candidate
        return None

    competitor_raw = _extract_competitor(payload.get("competitor_recipe")) or payload.get("competitor_recipe")
    competitor_steps = _coerce_steps(competitor_raw, competitor_keys, allowed)
    fallback_warning: Optional[str] = None
    if not competitor_steps:
        competitor_steps = _coerce_steps(payload.get("recipe"), competitor_keys, allowed)
    if not competitor_steps and isinstance(payload.get("steps"), list):
        competitor_steps = _coerce_steps(payload.get("steps"), competitor_keys, allowed)
        if competitor_steps:
            fallback_warning = "Converted legacy 'steps' schema for competitor_recipe."

    your_keys = ("timestamp", "stage", "say", "show", "why")
    your_raw = payload.get("your_recipe")
    def _extract_your(raw: Any) -> Optional[Any]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("stages", "steps", "recipe"):
                candidate = raw.get(key)
                if isinstance(candidate, list):
                    return candidate
            for parent in ("plan", "goal", "approach"):
                nested = raw.get(parent)
                if isinstance(nested, dict):
                    for key in ("stages", "steps", "recipe"):
                        candidate = nested.get(key)
                        if isinstance(candidate, list):
                            return candidate
        return None

    your_steps = _coerce_steps(_extract_your(your_raw), your_keys, allowed)
    if not your_steps and isinstance(your_raw, list):
        your_steps = _coerce_steps(your_raw, your_keys, allowed)
    if not your_steps:
        your_steps = _coerce_steps(None, your_keys, allowed)

    def _ensure_list(key: str) -> List[str]:
        source = None
        if isinstance(your_raw, dict):
            source = your_raw.get(key)
        if source is None:
            source = payload.get(key)
        if isinstance(source, list):
            return [str(item).strip() for item in source if str(item).strip()]
        if isinstance(source, str) and source.strip():
            return [source.strip()]
        return []

    script_source = None
    if isinstance(your_raw, dict):
        script_source = your_raw.get("script")
    if script_source is None:
        script_source = payload.get("script") or {}

    def _script_value(key: str) -> str:
        if not script_source:
            return ""
        for candidate in (key, key.capitalize(), key.upper()):
            value = script_source.get(candidate)
            if value:
                return str(value).strip()
        return ""

    normalized_script = {
        "hook": _script_value("hook"),
        "proof": _script_value("proof"),
        "value": _script_value("value"),
        "cta": _script_value("cta"),
    }

    return {
        "competitor_recipe": competitor_steps,
        "your_recipe": your_steps,
        "keep": _ensure_list("keep"),
        "avoid": _ensure_list("avoid"),
        "test": _ensure_list("test"),
        "script": normalized_script,
    }, fallback_warning


def call_ad_strategy_llm(
    ad_flow: str,
    pain_points: str,
    proof_points: str,
    value_props: str,
    ctas: str,
    goal: str,
) -> Optional[dict[str, Any]]:
    if not FASTADS_LLM_API_KEY:
        return None

    prompt = (
        f"Ad flow: {ad_flow}\n"
        f"Pain points: {pain_points}\n"
        f"Proof points: {proof_points}\n"
        f"Value props: {value_props}\n"
        f"CTAs: {ctas}\n"
        f"Campaign goal: {goal}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a creative strategist writing marketing playbooks. Return ONLY valid JSON. "
                "Provide both a competitor_recipe and a new your_recipe. The competitor_recipe should describe what the observed ad did (3-5 steps). "
                "The your_recipe must NOT repeat the competitor wording—it should propose new lines, visuals, and reasoning that align with the chosen campaign goal (Lead Generation/Sales/Awareness). "
                "The JSON must also include keep, avoid, test (lists) and a script object with hook/proof/value/cta strings. "
                "Use stages Hook, Proof, Value, CTA only. Each competitor step requires timestamp, stage, what, why, formula. Each your step requires timestamp, stage, say, show, why. "
                "IMPORTANT: competitor_recipe must be a direct array of steps. your_recipe must expose keep, avoid, test, script as direct keys and provide steps in an array under stages or steps, not nested under other wrappers."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    content = _call_chat_completion(messages, response_format={"type": "json_object"})
    call_ad_strategy_llm.last_raw_response = content
    call_ad_strategy_llm.last_parse_error = None
    if not content:
        call_ad_strategy_llm.last_parse_error = "LLM request failed"
        return None

    try:
        payload = json.loads(content)
    except ValueError as exc:
        call_ad_strategy_llm.last_parse_error = str(exc)
        return None

    normalized, fallback_warning = _normalize_strategy_payload(payload)
    if fallback_warning:
        normalized.setdefault("_fallback_warning", fallback_warning)
    return normalized


call_ad_strategy_llm.last_raw_response = None
call_ad_strategy_llm.last_parse_error = None


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
