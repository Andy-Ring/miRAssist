# streamlit_app.py

import time
import requests
import streamlit as st
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="miRAssist", layout="wide")

st.title("🧬 miRAssist")
st.caption("GenAI-assisted miRNA–target reasoning")

question = st.text_area(
    "Ask a biological question",
    placeholder="I overexpressed miR-21 and saw increased proliferation in colon cancer cells..."
)

col1, col2, col3 = st.columns(3)
with col1:
    novel = st.checkbox("Novel mode (exclude functional miRTarBase)", value=False)
with col2:
    k = st.number_input("Shortlist size (k)", 10, 200, 50)
with col3:
    min_support = st.number_input("Min support", 1, 4, 1)

submit = st.button("Run miRAssist")

if submit and question:
    resp = requests.post(
        f"{API_URL}/query",
        json={
            "question": question,
            "novel": novel,
            "k": k,
            "min_support": min_support
        }
    ).json()

    query_id = resp["query_id"]
    st.info(f"Query submitted: {query_id}")

    with st.spinner("Running miRAssist…"):
        while True:
            time.sleep(2)
            result = requests.get(f"{API_URL}/result/{query_id}").json()
            if result["status"] != "running":
                break

    if result["status"] == "error":
        st.error(result["error"])
    else:
        st.subheader("Planner output (QuerySpec)")
        st.json(result["queryspec"])

        st.subheader("Evidence shortlist")
        df = pd.DataFrame(result["shortlist"])
        st.dataframe(df, use_container_width=True)

        st.subheader("miRAssist recommendation")
        st.markdown(result["answer"]["summary"])

        st.subheader("Suggested experiments")
        for exp in result["answer"]["suggested_experiments"]:
            st.markdown(f"- {exp}")
