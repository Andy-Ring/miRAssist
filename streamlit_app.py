from pathlib import Path
import runpy
import traceback

from backend.runtime_diagnostics import initialize_process

initialize_process()
import streamlit as st

print("[miRAssist] streamlit_app.py import started", flush=True)

FRONTEND_APP = Path(__file__).resolve().parent / "frontend" / "app.py"

try:
    runpy.run_path(str(FRONTEND_APP), run_name="__main__")
    print("[miRAssist] frontend app script executed from streamlit_app.py", flush=True)
except Exception as e:
    print(f"[miRAssist] startup error: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
    st.error("miRAssist failed during app startup.")
    st.exception(e)
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
