import traceback

import streamlit as st

print("[miRAssist] root app.py import started", flush=True)

try:
    import frontend.app  # noqa: F401
    print("[miRAssist] frontend.app import completed", flush=True)
except Exception as e:
    print(f"[miRAssist] startup error: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
    st.error("miRAssist failed during app startup.")
    st.exception(e)
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
