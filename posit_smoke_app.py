from __future__ import annotations

import os
import platform
import sys

import streamlit as st


st.set_page_config(page_title="miRAssist Posit Smoke Test", layout="centered")

st.title("miRAssist Posit Smoke Test")
st.write("If you can see this page, the basic Streamlit session is working on Posit Connect Cloud.")

st.subheader("Runtime")
st.code(
    "\n".join(
        [
            f"python: {sys.version}",
            f"executable: {sys.executable}",
            f"platform: {platform.platform()}",
        ]
    )
)

st.subheader("Environment")
for key in [
    "JOBSTORE_BACKEND",
    "EVIDENCE_BACKEND",
    "MIRASSIST_LLM_BACKEND",
    "MIRASSIST_DEBUG_UI",
]:
    st.write(f"{key}: {os.getenv(key, '<unset>')}")

st.success("Basic Streamlit rendering completed.")
