import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

st.set_page_config(page_title="FastAds", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"


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


def latest_job_dir() -> Optional[Path]:
    if not DATA_JOBS_DIR.exists():
        return None
    jobs = [p for p in DATA_JOBS_DIR.iterdir() if p.is_dir()]
    if not jobs:
        return None
    return max(jobs, key=lambda p: p.stat().st_mtime)


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


st.title("FastAds")
st.caption("Analyze competitor ads and turn them into campaign-ready insights.")

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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_items: List[Dict[str, Any]] = []

            for idx, uploaded in enumerate(uploads, start=1):
                save_path = tmpdir_path / f"uploaded_ad_{idx}.mp4"
                save_path.write_bytes(uploaded.getvalue())
                input_items.append(
                    {
                        "ad_id": f"ad_{idx}",
                        "video_url": "local",
                        "local_path": str(save_path),
                        "page_name": competitor,
                        "ad_copy": uploaded.name,
                    }
                )

            input_json_path = tmpdir_path / "ui_input_ads.json"
            input_json_path.write_text(json.dumps(input_items, ensure_ascii=False, indent=2), encoding="utf-8")

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
                job_dir = latest_job_dir()
                if not job_dir:
                    st.error("No job output found.")
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

                        st.caption(f"Campaign goal selected: {goal}. Goal-based strategy and script generation will be added in the next step.")

                        with st.expander("Pipeline logs"):
                            st.code(logs or "No logs")
