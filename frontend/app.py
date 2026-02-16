import json
import time

import pandas as pd
import requests
import streamlit as st


#-----------------------------
# Logo
#-----------------------------
with st.sidebar:
    st.image(
        "assets/miRAssist_logo.png",
        use_container_width=True
    )

#-----------------------------
# Styling
#-----------------------------
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



#----------------------------
# Dev Info
#----------------------------
APP_NAME = "miRAssist"
APP_VERSION = "0.4.0"
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
    r = requests.request(method, url, timeout=timeout, **kwargs)
    ct = (r.headers.get("content-type") or "").lower()
    if "application/json" not in ct:
        raise RuntimeError(
            f"Non-JSON response from {url}\n"
            f"status={r.status_code} content-type={ct}\n"
            f"text_head={r.text[:500]}"
        )
    return r.json()


def pick_summary_markdown(answer_obj) -> str | None:
    """
    Return the best markdown string to display, handling multiple backend schemas.

    Supported shapes:
      1) answer is a string -> markdown
      2) answer = {"summary": "...", "raw_text": "..."}
      3) answer = {"raw_text": {"summary": "...", "raw_text": "..."}}
      4) answer = {"raw_text": {"raw_text": {"summary": "...", ...}}}  (rare/double-wrap)
    """
    if answer_obj is None:
        return None

    # If answer is already a string, treat as markdown
    if isinstance(answer_obj, str) and answer_obj.strip():
        return answer_obj

    if not isinstance(answer_obj, dict):
        return None

    # --- Schema 1: answer["summary"] ---
    s = answer_obj.get("summary")
    if isinstance(s, str) and s.strip():
        return s

    # --- Try answer["raw_text"] ---
    rt = answer_obj.get("raw_text")

    # Schema 2: answer["raw_text"] is a string
    if isinstance(rt, str) and rt.strip():
        return rt

    # Schema 3: answer["raw_text"] is a dict with summary/raw_text
    if isinstance(rt, dict):
        s2 = rt.get("summary")
        if isinstance(s2, str) and s2.strip():
            return s2

        rt2 = rt.get("raw_text")
        if isinstance(rt2, str) and rt2.strip():
            return rt2

        # Schema 4: answer["raw_text"]["raw_text"] is a dict (double-wrapped)
        if isinstance(rt2, dict):
            s3 = rt2.get("summary")
            if isinstance(s3, str) and s3.strip():
                return s3
            rt3 = rt2.get("raw_text")
            if isinstance(rt3, str) and rt3.strip():
                return rt3

    return None


def typewriter_markdown(md: str, container, cps: int = 60, chunk: str = "word"):
    """
    Animate markdown text in a 'typing' style.

    cps = characters per second target (roughly).
    chunk = "char" or "word" (word is much faster and less flickery).
    """
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
        # word mode
        words = md.split(" ")
        out_words = []
        # approximate: average ~5 chars/word + 1 space
        delay = 1.0 / max(1, int(cps / 6))
        for w in words:
            out_words.append(w)
            container.markdown(" ".join(out_words), unsafe_allow_html=False)
            time.sleep(delay)


def sidebar_footer(author: str, version: str):
    # Push footer to bottom
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


def sidebar_spacer(n=1):
    for _ in range(n):
        st.markdown("")


# ----------------------------
# UI
# ----------------------------
st.title("miRAssist")
st.caption("Ask a question to query the miRNA-Target Database")

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

    colA, colB = st.columns([3,2])
    with colA:
        ping = st.button("Test Connection")
    with colB:
        clear = st.button("Clear")

    if clear:
        for k in ["last_result", "last_query_id", "last_error"]:
            st.session_state.pop(k, None)

    if ping:
        if not api_url:
            st.warning("Enter an API base URL first.")
        else:
            try:
                out = safe_request_json("GET", f"{api_url}/health", timeout=10)
                st.success(f"OK: {out}")
            except Exception as e:
                st.error(str(e))

    # Footer pinned to bottom of sidebar
    sidebar_footer(APP_AUTHOR, APP_VERSION)


st.subheader("Ask a question")
question = st.text_area(
    "Question",
    placeholder="Example: I overexpressed miR-21 and saw increased proliferation in colon cancer cells. What might it regulate?",
    height=120,
)

st.markdown("### Override options (optional)")
st.caption(
    "miRAssist will infer settings from your question. These controls are only if you want to override default settings without specifying in the question."
)

c1, c2, c3 = st.columns(3)
with c1:
    novel = st.checkbox(
        "Novel mode (override)",
        value=True,
        help=(
            "If enabled, miRAssist will *not* recommend miRNA–gene pairs that are already "
            "flagged as functional positives in miRTarBase as 'novel'. It may still mention them as known."
        ),
    )
with c2:
    k = st.number_input(
        "k (override)",
        min_value=5,
        max_value=200,
        value=25,
        step=5,
        help=(
            "How many candidates to retrieve before synthesis. Larger k can improve recall but may slow "
            "the run or produce longer outputs."
        ),
    )
with c3:
    min_support = st.number_input(
        "Min support (override)",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help=(
            "Minimum number of supporting evidence sources required for a pair to enter retrieval. "
            "Example: 2 means the pair must have support from at least two evidence channels."
        ),
    )

c4, c5 = st.columns(2)
with c4:
    require_binding = st.checkbox(
        "Require binding evidence (override)",
        value=False,
        help=(
            "If enabled, retrieval will only keep pairs with at least one binding-evidence source "
            "(e.g., ENCORI CLIP, TargetScan sites, or miRDB). Usually leave OFF for discovery."
        ),
    )
with c5:
    require_expression = st.checkbox(
        "Require expression evidence (override)",
        value=False,
        help=(
            "If enabled, retrieval will require the miRNA and gene to be expressed in the selected context "
            "(if TCGA context is available). Usually leave OFF unless you want to be strict."
        ),
    )

# NEW: pathway mode override (auto/boost/filter)
st.markdown("#### Pathway mode (override)")
pathway_mode = st.selectbox(
    "Pathway integration mode",
    options=["auto", "boost", "filter"],
    index=0,
    help=(
        "auto: planner decides\n"
        "boost: prefer genes in relevant pathways\n"
        "filter: ONLY return genes in relevant pathways"
    ),
)

run = st.button("Run miRAssist", type="primary", disabled=(not api_url or not question.strip()))

# ----------------------------
# Run + poll
# ----------------------------
if run:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_error", None)

    try:
        submit_payload = {
            "question": question.strip(),
            "novel": bool(novel),
            "k": int(k),
            "min_support": int(min_support),
            "require_binding_evidence": bool(require_binding),
            "require_expression": bool(require_expression),
            # NEW: pass pathway override through to backend
            "pathway_mode": str(pathway_mode),
        }

        resp = safe_request_json("POST", f"{api_url}/query", json=submit_payload, timeout=30)
        query_id = resp["query_id"]
        st.session_state["last_query_id"] = query_id
        st.info(f"Submitted: `{query_id}`")

        status_box = st.empty()
        progress = st.progress(0)

        max_wait_seconds = 15 * 60
        poll_every = 5
        t0 = time.time()

        with st.spinner("Running…"):
            while True:
                elapsed = time.time() - t0
                if elapsed > max_wait_seconds:
                    raise TimeoutError("Timed out waiting for miRAssist to finish.")

                s = safe_request_json("GET", f"{api_url}/status/{query_id}", timeout=30)
                status = s.get("status", "unknown")
                status_box.info(f"Status: **{status}** • elapsed: {int(elapsed)}s")
                progress.progress(min(0.99, (elapsed % 60) / 60))

                if status not in ("queued", "running"):
                    break

                time.sleep(poll_every)

        result = safe_request_json("GET", f"{api_url}/result/{query_id}", timeout=600)
        st.session_state["last_result"] = result

    except Exception as e:
        st.session_state["last_error"] = str(e)

# ----------------------------
# Display results
# ----------------------------
err = st.session_state.get("last_error")
if err:
    st.error(err)

result = st.session_state.get("last_result")
if result:
    if result.get("status") == "error":
        st.error(result.get("error", "Unknown error"))
        tb = result.get("traceback")
        if tb:
            with st.expander("Traceback"):
                st.code(tb)
    else:
        st.markdown("## Answer")

        ans_obj = result.get("answer")
        summary_md = pick_summary_markdown(ans_obj)

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

        with st.expander("Planner output (QuerySpec)", expanded=False):
            st.json(result.get("queryspec", {}))

        with st.expander("Evidence shortlist (optional)", expanded=False):
            shortlist = result.get("shortlist", [])
            if shortlist:
                df = pd.DataFrame(shortlist)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Shortlist is empty.")

        with st.expander("Debug: answer JSON", expanded=False):
            st.json(result.get("answer", {}))
