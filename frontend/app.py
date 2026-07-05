from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import time
import traceback
import uuid

import pandas as pd
import streamlit as st

from backend.config import (
    get_debug_max_rows,
    get_debug_ui,
)
from frontend.help_content import (
    get_about_evidence_markdown,
    get_how_to_use_markdown,
)


FRONTEND_DIR = Path(__file__).resolve().parent


def _startup_log(message: str) -> None:
    print(f"[miRAssist] {message}", flush=True)


def render_sidebar_logo() -> None:
    logo_path = FRONTEND_DIR / "assets" / "miRAssist_logo.png"
    if not logo_path.exists():
        _startup_log(f"sidebar logo missing at {logo_path}")
        return

    try:
        st.image(str(logo_path), use_container_width=True)
        _startup_log("sidebar logo rendered with use_container_width")
    except TypeError:
        # Older hosted Streamlit builds may only support the legacy keyword.
        st.image(str(logo_path), use_column_width=True)
        _startup_log("sidebar logo rendered with use_column_width fallback")
    except Exception as exc:
        _startup_log(f"sidebar logo render failed: {exc}")


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
_startup_log("local dotenv load attempted")


st.set_page_config(page_title="miRAssist", layout="wide")
_startup_log("set_page_config complete")

with st.sidebar:
    render_sidebar_logo()
_startup_log("sidebar context entered")


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
_startup_log("global stylesheet injected")


APP_NAME = "miRAssist"
APP_VERSION = "0.9.2"
APP_AUTHOR = "Andy Ring"
DEFAULT_UI_CANDIDATE_POOL = 10
DEFAULT_UI_MIN_SUPPORT = 2
CHART_SCORE_COLUMNS = [
    "mirassist_xgboost_score",
    "learned_score_used",
    "retrieval_rank_score",
    "best_backend_model_score",
    "learned_score",
    "model_score",
    "overall_evidence_support_percentile",
    "retrieval_score",
    "support_count",
]


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


def clear_session_outputs() -> None:
    for key in [
        "last_result",
        "last_query_id",
        "last_error",
        "last_error_traceback",
        "last_status",
        "last_submit_response",
    ]:
        st.session_state.pop(key, None)


def _format_candidate_label(row: pd.Series) -> str:
    gene = str(row.get("gene_symbol", "") or "").strip()
    mirna = str(row.get("mirna_name", "") or "").strip()
    if gene and mirna:
        return f"{gene} <- {mirna}"
    return gene or mirna or "Unknown candidate"


def _pick_chart_score_column(df: pd.DataFrame) -> str | None:
    for column in CHART_SCORE_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            return column
    return None


def build_shortlist_chart_frame(shortlist: list[dict]) -> tuple[pd.DataFrame, str | None]:
    if not isinstance(shortlist, list) or not shortlist:
        return pd.DataFrame(), None

    df = pd.DataFrame(shortlist).copy()
    score_column = _pick_chart_score_column(df)
    if not score_column:
        return pd.DataFrame(), None

    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df = df.loc[df[score_column].notna()].copy()
    if df.empty:
        return pd.DataFrame(), None

    df["candidate_label"] = df.apply(_format_candidate_label, axis=1)
    if "support_count" in df.columns:
        df["support_count"] = pd.to_numeric(df["support_count"], errors="coerce")

    chart_columns = [
        col
        for col in ["candidate_label", score_column, "support_count", "score_column_used"]
        if col in df.columns
    ]
    chart_df = df.sort_values(score_column, ascending=False).head(10).loc[:, chart_columns]
    return chart_df, score_column


def render_shortlist_chart(shortlist: list[dict]) -> None:
    chart_df, score_column = build_shortlist_chart_frame(shortlist)
    if chart_df.empty or not score_column:
        st.info("No chartable ranked candidates were returned.")
        return

    try:
        import altair as alt

        tooltip = [
            alt.Tooltip("candidate_label:N", title="Candidate"),
            alt.Tooltip(f"{score_column}:Q", title=score_column, format=".4f"),
        ]
        if "support_count" in chart_df.columns:
            tooltip.append(alt.Tooltip("support_count:Q", title="Support count", format=".0f"))
        if "score_column_used" in chart_df.columns:
            tooltip.append(alt.Tooltip("score_column_used:N", title="Score column used"))

        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X(f"{score_column}:Q", title=score_column),
                y=alt.Y("candidate_label:N", sort="-x", title=None),
                tooltip=tooltip,
            )
            .properties(height=max(280, 32 * len(chart_df)))
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        # Posit/hosted environments may provide Streamlit without a direct Altair import.
        fallback_df = chart_df.loc[:, ["candidate_label", score_column]].set_index("candidate_label")
        st.bar_chart(fallback_df, use_container_width=True)

    st.caption(f"Showing the top {len(chart_df)} retrieved candidates ranked by `{score_column}`.")


def run_direct_mode(submit_payload: dict) -> None:
    from backend.jobstore import read_job
    from backend.worker import run_query_job

    query_id = uuid.uuid4().hex[:16]
    st.session_state["last_query_id"] = query_id
    st.session_state["last_submit_response"] = {"query_id": query_id, "mode": "direct"}

    status_box = st.empty()
    progress = st.progress(0)
    status_box.info("Status: **queued** | mode: **direct**")
    progress.progress(0.05)

    with st.spinner("Running miRAssist query..."):
        result = run_query_job(
            query_id=query_id,
            question=submit_payload["question"],
            k=submit_payload["k"],
            min_support=submit_payload["min_support"],
            novel=submit_payload["novel"],
            disable_synthesis=submit_payload["disable_synthesis"],
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
st.title("Welcome to miRAssist")
st.caption("Enter your natural language prompt below to query the miRNA-target database")
_startup_log("title and caption rendered")

with st.sidebar:
    with st.expander("How to use miRAssist", expanded=False):
        st.markdown(get_how_to_use_markdown())

    with st.expander("About evidence", expanded=False):
        st.markdown(get_about_evidence_markdown())

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

c1 = st.columns(1)[0]
with c1:
    novel = st.checkbox(
        "Novel mode",
        value=False,
        help=(
            "If enabled, miRAssist will avoid labeling miRTarBase functional positives as 'novel'. "
            "It may still mention known targets."
        ),
    )

run_disabled = not question.strip()
run = st.button("Run miRAssist", type="primary", disabled=run_disabled)


if run:
    clear_session_outputs()

    try:
        submit_payload = {
            "question": question.strip(),
            "novel": bool(novel),
            "k": DEFAULT_UI_CANDIDATE_POOL,
            "min_support": DEFAULT_UI_MIN_SUPPORT,
            "disable_synthesis": True,
            "require_binding_evidence": False,
            "require_expression": False,
            "pathway_mode": "auto",
        }

        run_direct_mode(submit_payload)
    except Exception as e:
        st.session_state["last_error"] = "The query failed during retrieval or chart generation."
        st.session_state["last_error_traceback"] = traceback.format_exc()
        if get_debug_ui():
            st.session_state["last_error"] = f"The query failed during retrieval or chart generation.\n\n{e}"


err = st.session_state.get("last_error")
if err:
    st.error(err)
    error_traceback = st.session_state.get("last_error_traceback")
    if error_traceback and get_debug_ui():
        with st.expander("Traceback", expanded=False):
            st.code(error_traceback)


result = st.session_state.get("last_result")
if result:
    result_status = result.get("status", "unknown")

    if result_status in {"error", "failed", "not_found"}:
        st.error(result.get("error", f"Backend returned status: {result_status}"))
    else:
        shortlist = result.get("shortlist", [])
        synthesis_disabled = bool(result.get("disable_synthesis"))

        if synthesis_disabled:
            st.markdown("## Retrieved candidates")
            render_shortlist_chart(shortlist)
            summary_md = pick_summary_markdown(result)
            if summary_md and not shortlist:
                st.info(summary_md)
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
            if isinstance(shortlist, list) and len(shortlist) > 0:
                max_rows = max(1, int(get_debug_max_rows()))
                df = pd.DataFrame(shortlist[:max_rows])
                preferred_columns = [
                    "gene_symbol",
                    "mirna_name",
                    "mirassist_xgboost_score",
                    "score_column_used",
                    "learned_score_xgb_raw_v1",
                    "learned_score_xgb_raw_nomissing_v1",
                    "learned_score_used",
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
                if len(shortlist) > max_rows:
                    st.caption(f"Showing the first {max_rows} debug rows.")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Shortlist is empty.")
