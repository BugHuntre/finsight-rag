"""Streamlit UI for FinSight RAG. Calls the FastAPI backend (FR7: show source
snippets and page numbers; FR5: company/year filters).

Run with::

    streamlit run app/streamlit_app.py

Set FINSIGHT_API_URL to point at a non-default backend (e.g. a deployed one).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("FINSIGHT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # local Ollama can be slow on a cold start

st.set_page_config(page_title="FinSight RAG", page_icon="📊", layout="centered")


@st.cache_data(ttl=60)
def fetch_documents() -> list[dict]:
    resp = requests.get(f"{API_URL}/documents", timeout=10)
    resp.raise_for_status()
    return resp.json()


def ask(question: str, company: str | None, fiscal_year: str | None) -> dict:
    payload = {"question": question, "company": company, "fiscal_year": fiscal_year}
    resp = requests.post(f"{API_URL}/query", json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text)
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


st.title("📊 FinSight RAG")
st.caption(
    "Ask a question about Indian company annual reports. Answers are grounded in the "
    "documents and cite their sources."
)

try:
    documents = fetch_documents()
except requests.RequestException:
    documents = []
    st.warning(
        f"Could not reach the FinSight API at {API_URL}. Is it running "
        "(`uvicorn finsight.api:app`)? Filters are unavailable until it responds."
    )

companies = sorted({d["company"] for d in documents})
years = sorted({d["fiscal_year"] for d in documents})

col1, col2 = st.columns(2)
company = col1.selectbox("Company (optional)", ["Any"] + companies)
fiscal_year = col2.selectbox("Fiscal year (optional)", ["Any"] + years)

question = st.text_input(
    "Your question", placeholder="e.g. What was TCS's revenue in FY2026?"
)
ask_clicked = st.button("Ask", type="primary", disabled=not documents)

if ask_clicked and question.strip():
    with st.spinner("Retrieving and generating an answer..."):
        try:
            result = ask(
                question,
                None if company == "Any" else company,
                None if fiscal_year == "Any" else fiscal_year,
            )
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Query failed: {exc}")
        else:
            st.markdown("### Answer")
            st.write(result["answer"])
            st.caption(f"Model: {result['model']} · Latency: {result['latency_ms']} ms")

            if result["sources"]:
                st.markdown("### Sources")
                for s in result["sources"]:
                    label = f"[{s['id']}] {s['company']} - {s['document']}, page {s['page']}"
                    with st.expander(label):
                        st.write(s["snippet"])
elif ask_clicked:
    st.warning("Type a question first.")

with st.expander("Indexed documents"):
    if documents:
        st.table(
            [
                {
                    "Company": d["company"],
                    "Year": d["fiscal_year"],
                    "Type": d["doc_type"],
                    "Chunks": d["chunk_count"],
                }
                for d in documents
            ]
        )
    else:
        st.write("No documents indexed yet.")
