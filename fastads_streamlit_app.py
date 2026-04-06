import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import time

import streamlit as st
from fastads.providers.llm import call_ad_strategy_llm

st.set_page_config(page_title="FastAds", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"
UPLOADS_DIR = PROJECT_ROOT / "data" / "ui_uploads"


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
    media_dir = job_dir / "media"
    if not media_dir.exists():
        return False
    for ad_id in ad_ids:
        ad_dir = media_dir / ad_id
        if not ad_dir.exists():
            return False
        insights_file = ad_dir / "insights.json"
        transcript_file = ad_dir / "transcript.txt"
        if not insights_file.exists() or not transcript_file.exists():
            return False
        if insights_file.stat().st_mtime < start_ts or transcript_file.stat().st_mtime < start_ts:
            return False
    return True


def prepare_upload_dir() -> None:
    if UPLOADS_DIR.exists():
        for child in UPLOADS_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
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


def collect_ad_results(job_dir: Path) -> List[Dict[str, Any]]:
    media_dir = job_dir / "media"
    results: List[Dict[str, Any]] = []
    if not media_dir.exists():
        return results

    for ad_dir in sorted([p for p in media_dir.iterdir() if p.is_dir()]):
        insights = load_json(ad_dir / "insights.json") or {}
        transcript = load_text(ad_dir / "transcript.txt")
        ocr_data = load_json(ad_dir / "ocr.json")
        results.append(
            {
                "ad_id": ad_dir.name,
                "dir": ad_dir,
                "insights": insights,
                "transcript": transcript,
                "ocr": ocr_data,
            }
        )
    return results


def flatten_text_items(items: List[Dict[str, Any]]) -> List[str]:
    return [str(x.get("text", "")).strip() for x in items if str(x.get("text", "")).strip()]


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
    script_source: Dict[str, Any] = {}
    if isinstance(raw_your_recipe, dict):
        script_source = raw_your_recipe.get("script", {}) or {}
    if not script_source:
        script_source = normalized.get("script", {}) or {}

    script_stages: List[Dict[str, Any]] = []
    for source in (script_source, raw_your_recipe if isinstance(raw_your_recipe, dict) else {}):
        for key in ("stages", "steps"):
            candidate = source.get(key) if isinstance(source, dict) else None
            if isinstance(candidate, list):
                script_stages = candidate
                break
        if script_stages:
            break

    if script_stages:
        your_recipe_steps = script_stages
    elif isinstance(raw_your_recipe, dict):
        stages = raw_your_recipe.get("stages")
        steps = raw_your_recipe.get("steps")
        if isinstance(stages, list):
            your_recipe_steps = stages
        elif isinstance(steps, list):
            your_recipe_steps = steps
        else:
            your_recipe_steps = []
    elif isinstance(raw_your_recipe, list):
        your_recipe_steps = raw_your_recipe

    raw_competitor_recipe = normalized.get("competitor_recipe", [])
    competitor_steps: List[Dict[str, Any]] = []
    if isinstance(raw_competitor_recipe, dict):
        competitor_steps = raw_competitor_recipe.get("steps", [])
    elif isinstance(raw_competitor_recipe, list):
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


st.title("FastAds")
st.caption("Upload a competitor ad. Get the recipe to beat it.")

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
    if not uploads:
        st.error("Upload at least one ad video.")
    elif len(uploads) > 3:
        st.error("Upload at most 3 ad videos.")
    else:
        prepare_upload_dir()
        input_items: List[Dict[str, Any]] = []

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

        input_json_path = UPLOADS_DIR / "ui_input_ads.json"
        input_json_path.write_text(json.dumps(input_items, ensure_ascii=False, indent=2), encoding="utf-8")
        uploaded_names = [uploaded.name for uploaded in uploads]
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
                        if not strategy_payload.get("competitor_recipe"):
                            st.warning(
                                f"Recipe generation failed for {ad_name}."
                            )
                            continue
                        render_strategy_card(ad_name, strategy_payload)

                    with st.expander("Pipeline logs"):
                        st.code(logs or "No logs")
