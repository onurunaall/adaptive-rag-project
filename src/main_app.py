import streamlit as st
from src.core_rag_engine import CoreRAGEngine

st.set_page_config(page_title="InsightEngine - Adaptive RAG", layout="wide")
st.title("InsightEngine – Adaptive RAG")

@st.cache_resource
def load_engine():
    return CoreRAGEngine()

engine = load_engine()

# Sidebar: Config & Ingest
st.sidebar.header("Config & Ingest")
collection_name = st.sidebar.text_input("Collection name", value=engine.default_collection_name)
recreate_collection = st.sidebar.checkbox("Recreate collection", value=False)
st.sidebar.markdown("---")
st.sidebar.subheader("Add Sources")
uploaded = st.sidebar.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
urls_text = st.sidebar.text_area("URLs (one per line)", height=150)

if st.sidebar.button("Ingest Documents"):
    sources = []
    if uploaded:
        for f in uploaded:
            sources.append({"type": "uploaded_pdf", "value": f})
    for line in urls_text.splitlines():
        u = line.strip()
        if u:
            sources.append({"type": "url", "value": u})
    if not sources:
        st.sidebar.warning("No sources to ingest.")
    else:
        try:
            engine.ingest(
                sources=sources,
                collection_name=collection_name,
                recreate_collection=recreate_collection
            )
            st.sidebar.success("✅ Documents ingested.")
        except Exception as e:
            st.sidebar.error(f"❌ Ingestion failed: {e}")

# Main: Q&A
st.header("Query the Collection")
question = st.text_input("Your question here")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating adaptive answer using full workflow..."):
            try:
                result = engine.run_full_rag_workflow(
                    question=question,
                    collection_name=collection_name
                )
            except Exception as e:
                st.error(f"An unexpected error occurred during the RAG workflow: {e}")
                result = {
                    "answer": "Failed to retrieve answer due to an internal workflow error.",
                    "sources": []
                }

        st.subheader("Answer")
        st.write(result.get("answer", ""))

        st.subheader("Sources")
        srcs = result.get("sources", [])
        if not srcs:
            st.write("No sources.")
        else:
            for s in srcs:
                st.markdown(f"**{s['source']}**: {s['preview']}")
