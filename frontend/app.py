from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import os
import time
import uuid

import pandas as pd
import requests
import streamlit as st

from backend.config import (
    database_configured,
    get_app_mode,
    get_evidence_backend,
    get_jobstore_backend,
    get_llm_backend,
    get_planner_model,
    get_synth_model,
    openai_configured,
    resolve_backend_url,
)
from backend.jobstore import read_job
from backend.worker import run_query_job
from frontend.help_content import (
    get_about_evidence_markdown,
    get_how_to_use_markdown,
    should_show_api_connection_controls,
)


FRONTEND_DIR = Path(__file__).resolve().parent


def load_local_dotenv() -> None:
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


load_local_dotenv()

APP_MODE = get_app_mode()
DEFAULT_BACKEND_URL = resolve_backend_url()


st.set_page_config(page_title="miRAssist", layout="wide")

with st.sidebar:
    st.image(
        str(FRONTEND_DIR / "assets" / "miRAssist_logo.png"),
        use_container_width=True,
    )


st.markdown(
    """
    <style>
    :root {
        --mir-green: #5DBB63;
        --mir-teal: #2CA6A4;
        --mir-dark: #0E1117;
    }

    body {
        background-color: var(--mir-dark);
    }

    h1, h2, h3 {
        color: var(--mir-teal);
    }

    .stButton>button {
        background-color: var(--mir-teal);
        color: black;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: var(--mir-green);
        color: black;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_NAME = "miRAssist"
APP_VERSION = "0.8.0"
APP_AUTHOR = "Andy Ring"


def normalize_base_url(u: str) -> str:
    u = (u or "").strip()
    if u.endswith("/"):
        u = u[:-1]
    return u


def safe_request_json(method: str, url: str, timeout=30, **kwargs):
    r = requests.request(method, url, timeout=timeout, **kwargs)
    ct = (r.headers.get("content-type") or "").lower()

    if "application/json" not in ct:
        raise RuntimeError(
            f"Non-JSON response from {url}\n"
            f"status={r.status_code} content-type={ct}\n"
            f"text_head={r.text[:1000]}"
        )

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(
            f"Could not parse JSON response from {url}\n"
            f"status={r.status_code}\n"
            f"text_head={r.text[:1000]}\n"
            f"error={e}"
        )

    if r.status_code >= 400:
        raise RuntimeError(
            f"Backend request failed: {method} {url}\n"
            f"status={r.status_code}\n"
            f"response={json.dumps(data, indent=2)[:4000]}"
        )

    return data


def is_terminal_status(status_json: dict) -> bool:
    status = status_json.get("status", "unknown")
    stage = status_json.get("stage")

    if status in {"done", "error", "failed", "not_found"}:
        return True

    if status == "unknown" and stage in {
        "queued",
        "planner",
        "planning",
        "retrieval",
        "retrieve",
        "shortlist",
        "synthesis",
        "synthesizing",
        "generation",
        "generating",
    }:
        return False

    if status in {"queued", "running"}:
        return False

    return False


def status_display_text(status_json: dict) -> str:
    status = status_json.get("status", "unknown")
    stage = status_json.get("stage")

    bits = [f"Status: **{status}**"]
    if stage:
        bits.append(f"stage: **{stage}**")

    ranking_mode = status_json.get("ranking_mode")
    if ranking_mode:
        bits.append(f"ranking: **{ranking_mode}**")

    structure_alpha = status_json.get("structure_alpha")
    if structure_alpha is not None:
        bits.append(f"structure alpha: **{structure_alpha}**")

    return " | ".join(bits)


def progress_from_status(status_json: dict, elapsed: float, max_wait_seconds: int) -> float:
    status = status_json.get("status", "unknown")
    stage = status_json.get("stage")

    if status == "done":
        return 1.0

    stage_progress = {
        "queued": 0.05,
        "planner": 0.20,
        "planning": 0.20,
        "retrieval": 0.45,
        "retrieve": 0.45,
        "shortlist": 0.55,
        "synthesis": 0.75,
        "synthesizing": 0.75,
        "generation": 0.80,
        "generating": 0.80,
    }

    if stage in stage_progress:
        base = stage_progress[stage]
    elif status == "queued":
        base = 0.05
    elif status == "running":
        base = 0.35
    else:
        base = 0.15

    time_component = min(0.20, elapsed / max(1, max_wait_seconds) * 0.20)
    return min(0.99, base + time_component)


def _get_nested_dict(payload: dict, path: tuple) -> dict:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return {}
        if key not in cur:
            return {}
        cur = cur.get(key)

    if isinstance(cur, dict) and cur:
        return cur
    return {}


def _get_nested_value(payload: dict, path: tuple):
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        if key not in cur:
            return None
        cur = cur.get(key)
    return cur


def extract_queryspec(result: dict) -> dict:
    if not isinstance(result, dict):
        return {}

    candidate_paths = [
        ("queryspec",),
        ("query_spec",),
        ("querySpec",),
        ("planner",),
        ("planner_output",),
        ("plannerOutput",),
        ("meta", "queryspec"),
        ("meta", "query_spec"),
        ("meta", "planner"),
        ("meta", "planner_output"),
        ("debug", "queryspec"),
        ("debug", "query_spec"),
        ("debug", "planner"),
        ("debug", "planner_output"),
        ("result", "queryspec"),
        ("result", "query_spec"),
        ("result", "planner"),
        ("result", "planner_output"),
        ("result", "meta", "queryspec"),
        ("result", "meta", "query_spec"),
        ("result", "debug", "queryspec"),
        ("result", "debug", "planner_output"),
        ("answer", "queryspec"),
        ("answer", "query_spec"),
        ("answer", "planner"),
        ("answer", "planner_output"),
        ("answer", "meta", "queryspec"),
        ("answer", "meta", "query_spec"),
        ("answer", "debug", "queryspec"),
        ("answer", "debug", "planner_output"),
        ("final_answer", "queryspec"),
        ("final_answer", "query_spec"),
        ("final_answer", "meta", "queryspec"),
        ("synthesis", "queryspec"),
        ("synthesis", "query_spec"),
        ("synthesis", "meta", "queryspec"),
        ("prompt_bundle", "meta", "queryspec"),
        ("prompt_bundle", "queryspec"),
        ("prompt_meta", "queryspec"),
        ("prompt_meta", "query_spec"),
    ]

    for path in candidate_paths:
        value = _get_nested_dict(result, path)
        if value:
            return value

    return {}


def extract_planner_debug_candidates(result: dict) -> dict:
    if not isinstance(result, dict):
        return {}

    candidate_paths = [
        ("queryspec",),
        ("query_spec",),
        ("querySpec",),
        ("planner",),
        ("planner_output",),
        ("plannerOutput",),
        ("meta",),
        ("meta", "queryspec"),
        ("meta", "query_spec"),
        ("meta", "planner"),
        ("meta", "planner_output"),
        ("debug",),
        ("debug", "queryspec"),
        ("debug", "query_spec"),
        ("debug", "planner"),
        ("debug", "planner_output"),
        ("result",),
        ("result", "queryspec"),
        ("result", "meta"),
        ("result", "meta", "queryspec"),
        ("answer",),
        ("answer", "meta"),
        ("answer", "meta", "queryspec"),
        ("prompt_bundle",),
        ("prompt_bundle", "meta"),
        ("prompt_bundle", "meta", "queryspec"),
        ("prompt_meta",),
        ("prompt_meta", "queryspec"),
    ]

    found = {}
    for path in candidate_paths:
        value = _get_nested_value(result, path)
        if value not in (None, {}, [], ""):
            found[".".join(path)] = value
    return found


def extract_answer_obj(result: dict):
    for key in ["answer", "final_answer", "response", "synthesis", "llm_answer"]:
        value = result.get(key)
        if value:
            return value
    return None


def pick_summary_markdown_from_answer(answer_obj) -> str | None:
    if answer_obj is None:
        return None

    if isinstance(answer_obj, str) and answer_obj.strip():
        return answer_obj.strip()
    if not isinstance(answer_obj, dict):
        return None

    for key in [
        "summary",
        "answer",
        "final_answer",
        "response",
        "text",
        "synthesis",
        "recommendation",
        "markdown",
    ]:
        value = answer_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    rt = answer_obj.get("raw_text")
    if isinstance(rt, str) and rt.strip():
        return rt.strip()

    if isinstance(rt, dict):
        for key in [
            "summary",
            "answer",
            "final_answer",
            "response",
            "text",
            "synthesis",
            "recommendation",
            "markdown",
        ]:
            value = rt.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        rt2 = rt.get("raw_text")
        if isinstance(rt2, str) and rt2.strip():
            return rt2.strip()
        if isinstance(rt2, dict):
            for key in [
                "summary",
                "answer",
                "final_answer",
                "response",
                "text",
                "synthesis",
                "recommendation",
                "markdown",
            ]:
                value = rt2.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return None


def pick_summary_markdown(result: dict) -> str | None:
    if not isinstance(result, dict):
        return None

    for key in [
        "summary",
        "answer_text",
        "final_answer",
        "response",
        "text",
        "synthesis",
        "llm_answer",
        "recommendation",
        "markdown",
    ]:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answer_obj = extract_answer_obj(result)
    return pick_summary_markdown_from_answer(answer_obj)


def pick_suggested_experiments(result: dict) -> list:
    keys = [
        "suggested_experiments",
        "experiments",
        "validation_experiments",
        "recommended_experiments",
    ]

    for key in keys:
        value = result.get(key)
        if isinstance(value, list):
            return value

    answer_obj = extract_answer_obj(result)
    if isinstance(answer_obj, dict):
        for key in keys:
            value = answer_obj.get(key)
            if isinstance(value, list):
                return value

    return []


def typewriter_markdown(md: str, container, cps: int = 60, chunk: str = "word"):
    if not md:
        return

    if chunk == "char":
        out = ""
        delay = 1.0 / max(1, cps)
        for ch in md:
            out += ch
            container.markdown(out, unsafe_allow_html=False)
            time.sleep(delay)
    else:
        words = md.split(" ")
        out_words = []
        delay = 1.0 / max(1, int(cps / 6))
        for w in words:
            out_words.append(w)
            container.markdown(" ".join(out_words), unsafe_allow_html=False)
            time.sleep(delay)


def sidebar_footer(author: str, version: str):
    st.sidebar.markdown(
        """
        <style>
        section[data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .sidebar-footer {
            margin-top: auto;
            padding-top: 1rem;
            font-size: 0.85rem;
            opacity: 0.75;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-footer">
            <hr />
            <div><strong>{APP_NAME}</strong></div>
            <div>{author}</div>
            <div>Version {version}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_app_diagnostics(api_url: str) -> dict:
    return {
        "app_mode": APP_MODE,
        "llm_backend": get_llm_backend(),
        "planner_model": get_planner_model(),
        "synth_model": get_synth_model(),
        "openai_configured": openai_configured(),
        "jobstore_backend": get_jobstore_backend(),
        "evidence_backend": get_evidence_backend(),
        "database_configured": database_configured(),
        "backend_url_in_use": bool(APP_MODE == "api" and api_url),
    }


def clear_session_outputs() -> None:
    for key in [
        "last_result",
        "last_query_id",
        "last_error",
        "last_status",
        "last_submit_response",
    ]:
        st.session_state.pop(key, None)


def run_direct_mode(submit_payload: dict) -> None:
    query_id = uuid.uuid4().hex[:16]
    st.session_state["last_query_id"] = query_id
    st.session_state["last_submit_response"] = {"query_id": query_id, "mode": "direct"}

    status_box = st.empty()
    progress = st.progress(0)
    status_box.info("Status: **queued** | mode: **direct**")
    progress.progress(0.05)

    with st.spinner("Running miRAssist directly in this Streamlit process..."):
        result = run_query_job(
            query_id=query_id,
            question=submit_payload["question"],
            k=submit_payload["k"],
            min_support=submit_payload["min_support"],
            novel=submit_payload["novel"],
            require_binding_evidence=submit_payload["require_binding_evidence"],
            require_expression=submit_payload["require_expression"],
            pathway_mode="auto",
        )

    final_result = read_job(query_id)
    if not final_result or final_result.get("status") == "unknown":
        final_result = result

    st.session_state["last_status"] = {
        "status": final_result.get("status", "unknown"),
        "stage": final_result.get("stage"),
    }
    st.session_state["last_result"] = final_result

    if final_result.get("status") == "done":
        progress.progress(1.0)
        status_box.success(f"Status: **done** | query_id: **{query_id}**")
    else:
        progress.progress(0.0)
        status_box.error(
            f"Status: **{final_result.get('status', 'error')}** | "
            f"{final_result.get('error', 'Direct mode execution failed.')}"
        )


def run_api_mode(api_url: str, submit_payload: dict) -> None:
    resp = safe_request_json(
        "POST",
        f"{api_url}/query",
        json=submit_payload,
        timeout=60,
    )
    st.session_state["last_submit_response"] = resp

    query_id = resp.get("query_id") or resp.get("job_id") or resp.get("id")
    if not query_id:
        raise RuntimeError(
            "Backend did not return a query_id/job_id/id.\n"
            f"Response: {json.dumps(resp, indent=2)[:4000]}"
        )

    st.session_state["last_query_id"] = query_id
    st.info(f"Submitted: `{query_id}`")

    status_box = st.empty()
    progress = st.progress(0)
    max_wait_seconds = 15 * 60
    poll_every = 5
    t0 = time.time()

    with st.spinner("Running..."):
        final_status_json = {}

        while True:
            elapsed = time.time() - t0
            if elapsed > max_wait_seconds:
                raise TimeoutError("Timed out waiting for miRAssist to finish.")

            s = safe_request_json("GET", f"{api_url}/status/{query_id}", timeout=30)
            st.session_state["last_status"] = s
            final_status_json = s

            progress.progress(progress_from_status(s, elapsed, max_wait_seconds))
            status_box.info(f"{status_display_text(s)} | elapsed: **{int(elapsed)}s**")

            if is_terminal_status(s):
                break
            time.sleep(poll_every)

    status = final_status_json.get("status", "unknown")
    if status in {"error", "failed", "not_found"}:
        try:
            result = safe_request_json("GET", f"{api_url}/result/{query_id}", timeout=60)
            st.session_state["last_result"] = result
        except Exception:
            pass

        err_msg = final_status_json.get("error") or f"Backend returned status: {status}"
        raise RuntimeError(err_msg)

    result = safe_request_json("GET", f"{api_url}/result/{query_id}", timeout=600)
    st.session_state["last_result"] = result
    progress.progress(1.0)


st.title("Welcome to miRAssist")
st.caption("Enter your natural language prompt below to query the miRNA-target database")

with st.sidebar:
    default_url = st.session_state.get("api_url", DEFAULT_BACKEND_URL)
    api_url = normalize_base_url(default_url)

    with st.expander("How to use miRAssist", expanded=False):
        st.markdown(get_how_to_use_markdown())

    with st.expander("About evidence", expanded=False):
        st.markdown(get_about_evidence_markdown())

    if should_show_api_connection_controls(APP_MODE):
        st.subheader("Connection")
        st.caption(f"Mode: `{APP_MODE}`")
        api_url = st.text_input(
            "Backend API base URL",
            value=default_url,
            placeholder="http://127.0.0.1:7861 or https://xxxxx.ngrok-free.app",
            help="Paste the base URL only (no trailing slash).",
        )
        api_url = normalize_base_url(api_url)
        st.session_state["api_url"] = api_url
        ping = st.button("Test Connection")

        if ping:
            if not api_url:
                st.warning("Enter an API base URL first.")
            else:
                try:
                    out = safe_request_json("GET", f"{api_url}/health", timeout=10)
                    st.success("Backend reachable.")
                    st.json(out)
                except Exception as e:
                    st.error(str(e))
    else:
        st.session_state["api_url"] = api_url

    sidebar_footer(APP_AUTHOR, APP_VERSION)


st.subheader("")
question = st.text_area(
    "Enter your prompt here",
    placeholder="Example: I overexpressed miR-21 and saw increased proliferation in colon cancer cells. What might it regulate?",
    height=120,
)

st.markdown("### Override options (optional)")
st.caption(
    "miRAssist will infer settings from your question. These controls override defaults without needing to specify in the question."
)

c1, c2, c3 = st.columns(3)
with c1:
    novel = st.checkbox(
        "Novel mode",
        value=False,
        help=(
            "If enabled, miRAssist will avoid labeling miRTarBase functional positives as 'novel'. "
            "It may still mention known targets."
        ),
    )
with c2:
    k = st.number_input(
        "Candidate pool size (k)",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
        help="k is the number of evidence cards passed to the synthesizer after backend filtering and scoring. The app prints the top 5 ranked results by default.",
    )
with c3:
    min_support = st.number_input(
        "Min support",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Minimum number of supporting evidence channels required to keep a pair.",
    )

run_disabled = not question.strip() or (APP_MODE == "api" and not st.session_state.get("api_url"))
run = st.button("Run miRAssist", type="primary", disabled=run_disabled)


if run:
    clear_session_outputs()

    try:
        submit_payload = {
            "question": question.strip(),
            "novel": bool(novel),
            "k": int(k),
            "min_support": int(min_support),
            "require_binding_evidence": False,
            "require_expression": False,
            "pathway_mode": "auto",
        }

        if APP_MODE == "direct":
            run_direct_mode(submit_payload)
        else:
            run_api_mode(st.session_state.get("api_url", ""), submit_payload)
    except Exception as e:
        st.session_state["last_error"] = str(e)


err = st.session_state.get("last_error")
if err:
    st.error(err)


result = st.session_state.get("last_result")
if result:
    result_status = result.get("status", "unknown")

    if result_status in {"error", "failed", "not_found"}:
        st.error(result.get("error", f"Backend returned status: {result_status}"))
    else:
        st.markdown("## Answer")

        summary_md = pick_summary_markdown(result)
        if summary_md:
            placeholder = st.empty()
            if st.session_state.get("animate_answer", True):
                typewriter_markdown(
                    summary_md,
                    placeholder,
                    cps=int(st.session_state.get("typing_speed", 80)),
                    chunk=st.session_state.get("typing_mode", "word"),
                )
            else:
                placeholder.markdown(summary_md, unsafe_allow_html=False)
        else:
            st.info("No summary text found in backend response.")

        experiments = pick_suggested_experiments(result)
        if experiments:
            st.markdown("## Suggested experiments")
            for exp in experiments:
                st.markdown(f"- {exp}")

        with st.expander("Planner output (QuerySpec)", expanded=False):
            queryspec = extract_queryspec(result)
            if queryspec:
                st.json(queryspec)
            else:
                st.warning(
                    "No planner/queryspec object was found at the expected response paths."
                )
                st.caption(
                    "This usually means either the backend did not include planner output "
                    "in the result payload, or it is stored under a response key not yet handled."
                )

        with st.expander("Evidence shortlist", expanded=False):
            shortlist = result.get("shortlist", [])
            if isinstance(shortlist, list) and len(shortlist) > 0:
                df = pd.DataFrame(shortlist)
                preferred_columns = [
                    "gene_symbol",
                    "mirna_name",
                    "learned_score_xgb_raw_v1",
                    "learned_score_xgb_raw_nomissing_v1",
                    "retrieval_rank_score",
                    "retrieval_score",
                    "learned_score_model_version",
                    "learned_score_feature_set",
                    "support_count",
                    "mirdb_best_score",
                    "ts_context_strength",
                    "clip_exp_sum",
                    "best_mfe",
                ]
                ordered_columns = [col for col in preferred_columns if col in df.columns]
                ordered_columns.extend([col for col in df.columns if col not in ordered_columns])
                df = df.loc[:, ordered_columns]
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Shortlist is empty.")
