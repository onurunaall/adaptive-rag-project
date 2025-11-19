"""
Improved MCP-enabled main application with enhanced features.
"""
import os
from typing import List
import logging
import streamlit as st
import json
import asyncio

from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.core_rag_engine import CoreRAGEngine
from src.stock import fetch_stock_news_documents
from src.scraper import scrape_urls_as_documents
from src.loop import AgentLoopWorkflow, AgentLoopState
from mcp.rag_integration import MCPEnhancedRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Initialize session state
if "qa_chat_history" not in st.session_state:
    st.session_state.qa_chat_history: List[BaseMessage] = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "mcp_enabled" not in st.session_state:
    st.session_state.mcp_enabled = True  # Enable by default

if "use_conversation_memory" not in st.session_state:
    st.session_state.use_conversation_memory = True

if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

st.set_page_config(
    page_title="InsightEngine - Adaptive RAG with MCP",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("InsightEngine – Adaptive RAG with MCP 🚀")


@st.cache_resource
def load_engine(enable_mcp: bool = True):
    """Load RAG engine with optional MCP enhancement."""
    core_engine = CoreRAGEngine()

    if enable_mcp:
        try:
            mcp_engine = MCPEnhancedRAG(
                core_rag_engine=core_engine,
                enable_filesystem=True,
                enable_memory=True,
                enable_sql=True
            )
            # Initialize MCP servers
            asyncio.run(mcp_engine.initialize_mcp_servers())
            st.success("✅ MCP servers initialized successfully!")
            return mcp_engine
        except Exception as e:
            st.warning(f"⚠️ MCP initialization failed: {e}. Using standard RAG.")
            return core_engine

    return core_engine


engine = load_engine(enable_mcp=st.session_state.mcp_enabled)

# === SIDEBAR: Enhanced Configuration ===
st.sidebar.header("⚙️ Configuration")

# MCP Settings
with st.sidebar.expander("🔌 MCP Settings", expanded=True):
    mcp_status = "🟢 Active" if hasattr(engine, 'mcp_client') else "🔴 Disabled"
    st.markdown(f"**Status:** {mcp_status}")

    if hasattr(engine, 'tools'):
        st.markdown(f"**Tools Available:** {len(engine.tools)}")
        with st.expander("View MCP Tools"):
            for tool_name in engine.tools.keys():
                st.code(tool_name, language="text")

    st.session_state.use_conversation_memory = st.checkbox(
        "Use Conversation Memory",
        value=st.session_state.use_conversation_memory,
        help="Store and retrieve relevant past conversations"
    )

st.sidebar.markdown("---")

# Collection Settings
st.sidebar.subheader("📚 Collection Settings")
collection_name = st.sidebar.text_input(
    "Collection name",
    value=engine.rag.default_collection_name if hasattr(engine, 'rag') else engine.default_collection_name
)
recreate_collection = st.sidebar.checkbox("Recreate collection", value=False)

st.sidebar.markdown("---")

# === INGESTION SECTION ===
st.sidebar.subheader("📥 Add Sources")

# File Upload
uploaded = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload PDF documents to ingest into the knowledge base"
)

# URL Input
urls_text = st.sidebar.text_area(
    "URLs (one per line)",
    height=100,
    help="Enter URLs to scrape and ingest"
)

# Directory Monitoring (MCP Feature)
if hasattr(engine, 'tools') and engine.tools:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Auto-Ingest (MCP)")

    watch_dir = st.sidebar.text_input(
        "Watch Directory",
        value="./documents",
        help="Monitor this directory for new files"
    )

    if st.sidebar.button("🔄 Start Auto-Monitoring"):
        try:
            async def start_monitoring():
                await engine.auto_ingest_with_monitoring(
                    watch_dir=watch_dir,
                    collection_name=collection_name,
                    file_types=["pdf", "txt", "md"],
                    poll_interval=10
                )

            st.session_state.monitoring_active = True
            st.sidebar.success(f"Monitoring {watch_dir}...")
        except Exception as e:
            st.sidebar.error(f"Failed to start monitoring: {e}")

    if st.sidebar.button("⏹️ Stop Monitoring"):
        if hasattr(engine, 'monitoring_active'):
            engine.monitoring_active = False
        st.session_state.monitoring_active = False
        st.sidebar.info("Monitoring stopped")

# Ingest Button
if st.sidebar.button("📤 Ingest Documents", type="primary"):
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
        with st.spinner("Ingesting documents..."):
            try:
                # Use the appropriate engine method
                if hasattr(engine, 'rag'):
                    engine.rag.ingest(
                        sources=sources,
                        collection_name=collection_name,
                        recreate_collection=recreate_collection,
                    )
                else:
                    engine.ingest(
                        sources=sources,
                        collection_name=collection_name,
                        recreate_collection=recreate_collection,
                    )
                st.sidebar.success(f"✅ {len(sources)} document(s) ingested!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# === MAIN AREA: Chat Interface ===
st.markdown("### 💬 Chat with Your Documents")

# Display session info
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
with col2:
    st.caption(f"Collection: `{collection_name}`")
with col3:
    if st.button("🗑️ Clear History"):
        st.session_state.qa_chat_history = []
        st.rerun()

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.qa_chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.qa_chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use MCP-enhanced RAG with memory if available
                if hasattr(engine, 'rag_with_memory') and st.session_state.use_conversation_memory:
                    result = asyncio.run(engine.rag_with_memory(
                        question=prompt,
                        session_id=st.session_state.session_id,
                        collection_name=collection_name,
                        use_memory_context=True,
                        store_conversation=True,
                        use_embeddings=True
                    ))

                    # Display answer
                    st.write(result.get("answer", "No answer generated."))

                    # Show memory context if used
                    if result.get("memory_context_used"):
                        with st.expander(f"📝 Used {result.get('memories_retrieved', 0)} past conversations"):
                            st.info("This answer incorporates relevant conversation history")

                else:
                    # Standard RAG
                    rag_engine = engine.rag if hasattr(engine, 'rag') else engine
                    result = rag_engine.run_full_rag_workflow(
                        question=prompt,
                        collection_name=collection_name,
                        chat_history=st.session_state.qa_chat_history[:-1]
                    )
                    st.write(result.get("answer", "No answer generated."))

                # Display sources
                if "sources" in result and result["sources"]:
                    with st.expander(f"📚 Sources ({len(result['sources'])})"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**{i}.** {src.get('source', 'Unknown')}")
                            st.caption(src.get('preview', '')[:200] + "...")

                # Add assistant response to history
                st.session_state.qa_chat_history.append(
                    AIMessage(content=result.get("answer", ""))
                )

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# === FOOTER: Statistics ===
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Conversations", len([m for m in st.session_state.qa_chat_history if isinstance(m, HumanMessage)]))

with col2:
    if hasattr(engine, 'rag'):
        stats = engine.rag.get_cache_stats()
        st.metric("Cached Collections", stats.get("cached_collections", 0))
    else:
        stats = engine.get_cache_stats()
        st.metric("Cached Collections", stats.get("cached_collections", 0))

with col3:
    mcp_status = "Active" if hasattr(engine, 'mcp_client') else "Disabled"
    st.metric("MCP Status", mcp_status)
