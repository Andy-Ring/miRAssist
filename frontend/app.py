import json
import time

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="miRAssist", layout="wide")


# -----------------------------
# Logo
# -----------------------------
with st.sidebar:
    st.image(
        "assets/miRAssist_logo.png",
        use_container_width=True
    )


# -----------------------------
# Styling
# -----------------------------
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


# ----------------------------
# Dev Info
# ----------------------------
APP_NAME = "miRAssist"
APP_VERSION = "0.6.0"
APP_AUTHOR = "Andy Ring"


# ----------------------------
# Helpers
# ----------------------------
def normalize_base_url(u: str) -> str:
    u = (u or "").strip()
    if u.endswith("/"):
        u = u[:-1]
    return u


def safe_request_json(method: str, url: str, timeout=30, **kwargs):
    """
    Return JSON from a backend request with useful error messages.
    """
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
    """
    Backend can temporarily return:
      status='unknown', stage='synthesis'

    That should NOT be considered terminal. Only these statuses are terminal.
    """
    status = status_json.get("status", "unknown")
    stage = status_json.get("stage")

    if status in {"done", "error", "failed", "not_found"}:
        return True

    # Defensive: if the backend returns unknown while actively in a known stage,
    # keep polling instead of fetching result too early.
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

    # Unknown status with no useful stage: keep polling for a bit rather than
    # immediately treating it as terminal.
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
        bits.append(f"structure α: **{structure_alpha}**")

    return " • ".join(bits)


def progress_from_status(status_json: dict, elapsed: float, max_wait_seconds: int) -> float:
    """
    Approximate progress. This is intentionally stage-based, not exact.
    """
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


def extract_queryspec(result: dict) -> dict:
    """
    Support old and new backend schemas.
    """
    for key in ["queryspec", "query_spec", "planner", "planner_output", "querySpec"]:
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def extract_answer_obj(result: dict):
    """
    Support old and new backend schemas.
    """
    for key in ["answer", "final_answer", "response", "synthesis", "llm_answer"]:
        value = result.get(key)
        if value:
            return value
    return None


def pick_summary_markdown_from_answer(answer_obj) -> str | None:
    """
    Return the best markdown string to display from an answer object.

    Supported shapes:
      1) answer is a string
      2) answer = {"summary": "...", "raw_text": "..."}
      3) answer = {"raw_text": {"summary": "...", "raw_text": "..."}}
      4) answer = {"raw_text": {"raw_text": {"summary": "...", ...}}}
    """
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
    """
    First check top-level result fields, then nested answer-like fields.
    """
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
    """
    Support optional suggested experiments if the backend provides them.
    """
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


# ----------------------------
# UI
# ----------------------------
st.title("miRAssist")
st.caption("Ask a question to query the miRNA–Target database")

with st.sidebar:
    st.subheader("Connection")

    default_url = st.session_state.get("api_url", "http://127.0.0.1:7861")
    api_url = st.text_input(
        "Backend API base URL",
        value=default_url,
        placeholder="http://127.0.0.1:7861 or https://xxxxx.ngrok-free.app",
        help="Paste the base URL only (no trailing slash).",
    )
    api_url = normalize_base_url(api_url)
    st.session_state["api_url"] = api_url

    colA, colB = st.columns([3, 2])
    with colA:
        ping = st.button("Test Connection")
    with colB:
        clear = st.button("Clear")

    if clear:
        for k2 in [
            "last_result",
            "last_query_id",
            "last_error",
            "last_status",
            "last_submit_response",
        ]:
            st.session_state.pop(k2, None)
        st.rerun()

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

    sidebar_footer(APP_AUTHOR, APP_VERSION)


st.subheader("Ask a question")
question = st.text_area(
    "Question",
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
        "Novel mode (override)",
        value=True,
        help=(
            "If enabled, miRAssist will avoid labeling miRTarBase functional positives as 'novel'. "
            "It may still mention known targets."
        ),
    )
with c2:
    k = st.number_input(
        "k (override)",
        min_value=5,
        max_value=200,
        value=25,
        step=5,
        help="How many candidates to retrieve before synthesis.",
    )
with c3:
    min_support = st.number_input(
        "Min support (override)",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Minimum number of supporting evidence channels required to keep a pair.",
    )

c4, c5, c6 = st.columns(3)
with c4:
    require_binding = st.checkbox(
        "Require binding evidence (override)",
        value=False,
        help="If enabled, only keep pairs with binding-type evidence, e.g. CLIP/TargetScan/miRDB.",
    )
with c5:
    require_expression = st.checkbox(
        "Require expression evidence (override)",
        value=False,
        help="If enabled, require miRNA and gene expression evidence where available.",
    )
with c6:
    pathway_mode = st.selectbox(
        "Pathway mode (override)",
        options=["auto", "boost", "filter"],
        index=0,
        help=(
            "auto: use planner defaults\n"
            "boost: prefer genes with pathway hits\n"
            "filter: only return genes with pathway hits"
        ),
    )

run = st.button("Run miRAssist", type="primary", disabled=(not api_url or not question.strip()))


# ----------------------------
# Run + poll
# ----------------------------
if run:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_error", None)
    st.session_state.pop("last_status", None)
    st.session_state.pop("last_submit_response", None)

    try:
        submit_payload = {
            "question": question.strip(),
            "novel": bool(novel),
            "k": int(k),
            "min_support": int(min_support),
            "require_binding_evidence": bool(require_binding),
            "require_expression": bool(require_expression),
            "pathway_mode": str(pathway_mode),
        }

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

        with st.spinner("Running…"):
            final_status_json = {}

            while True:
                elapsed = time.time() - t0

                if elapsed > max_wait_seconds:
                    raise TimeoutError("Timed out waiting for miRAssist to finish.")

                s = safe_request_json(
                    "GET",
                    f"{api_url}/status/{query_id}",
                    timeout=30,
                )
                st.session_state["last_status"] = s
                final_status_json = s

                progress.progress(progress_from_status(s, elapsed, max_wait_seconds))
                status_box.info(
                    f"{status_display_text(s)} • elapsed: **{int(elapsed)}s**"
                )

                if is_terminal_status(s):
                    break

                time.sleep(poll_every)

        status = final_status_json.get("status", "unknown")

        if status in {"error", "failed", "not_found"}:
            # Fetch result too, because backend may include traceback/error details there.
            try:
                result = safe_request_json(
                    "GET",
                    f"{api_url}/result/{query_id}",
                    timeout=60,
                )
                st.session_state["last_result"] = result
            except Exception:
                pass

            err_msg = final_status_json.get("error") or f"Backend returned status: {status}"
            raise RuntimeError(err_msg)

        result = safe_request_json(
            "GET",
            f"{api_url}/result/{query_id}",
            timeout=600,
        )

        st.session_state["last_result"] = result
        progress.progress(1.0)

    except Exception as e:
        st.session_state["last_error"] = str(e)


# ----------------------------
# Display results
# ----------------------------
err = st.session_state.get("last_error")
if err:
    st.error(err)

    with st.expander("Debug: last status", expanded=False):
        st.json(st.session_state.get("last_status", {}))

    with st.expander("Debug: submit response", expanded=False):
        st.json(st.session_state.get("last_submit_response", {}))


result = st.session_state.get("last_result")
if result:
    result_status = result.get("status", "unknown")

    if result_status in {"error", "failed", "not_found"}:
        st.error(result.get("error", f"Backend returned status: {result_status}"))

        tb = result.get("traceback")
        if tb:
            with st.expander("Traceback"):
                st.code(tb)

        with st.expander("Debug: full result JSON", expanded=False):
            st.json(result)

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
            st.json(extract_queryspec(result))

        with st.expander("Evidence shortlist (optional)", expanded=False):
            shortlist = result.get("shortlist", [])
            if isinstance(shortlist, list) and len(shortlist) > 0:
                df = pd.DataFrame(shortlist)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Shortlist is empty.")

        with st.expander("Debug: answer JSON", expanded=False):
            answer_obj = extract_answer_obj(result)
            if answer_obj is None:
                answer_obj = {}
            st.json(answer_obj)

        with st.expander("Debug: full result JSON", expanded=False):
            st.json(result)