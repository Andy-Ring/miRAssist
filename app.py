import traceback

import streamlit as st

try:
    import frontend.app  # noqa: F401
except Exception as e:
    st.error("miRAssist failed during app startup.")
    st.exception(e)
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
