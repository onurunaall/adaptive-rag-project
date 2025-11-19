# 🚀 MCP Integration Guide for Adaptive RAG

## Overview

The Model Context Protocol (MCP) extends your RAG system with **4 powerful servers** that enable:

1. **📁 Filesystem Server** - Auto-ingestion, file monitoring, metadata extraction
2. **🧠 Memory Server** - Conversation history with semantic search
3. **💾 SQL Server** - Database integration and structured data queries
4. **📊 Analytics Server** - Query analytics, performance metrics, insights

## Quick Start

### 1. Install MCP Dependencies

```bash
pip install -e ".[mcp]"  # Python 3.10+ required
```

### 2. Enable MCP in Your Application

```python
from src.core_rag_engine import CoreRAGEngine
from mcp.rag_integration import MCPEnhancedRAG

# Create core engine
core_engine = CoreRAGEngine()

# Wrap with MCP
mcp_engine = MCPEnhancedRAG(
    core_rag_engine=core_engine,
    enable_filesystem=True,
    enable_memory=True,
    enable_sql=True
)

# Initialize servers
import asyncio
asyncio.run(mcp_engine.initialize_mcp_servers())
```

### 3. Use the Enhanced Streamlit App

```bash
# Copy the MCP-enhanced version
cp src/main_app_mcp_enhanced.py src/main_app.py

# Run it
streamlit run src/main_app.py
```

---

## 🎯 Key MCP Features & Use Cases

### Feature 1: **Conversation Memory with Semantic Search**

**Why it's powerful:** Remembers context across sessions using embeddings for semantic similarity.

**Use Case:** Multi-turn conversations where users ask follow-up questions

**Example:**
```python
# User Session 1
Q1: "What is quantum computing?"
A1: [Answer with context from docs]

# User Session 2 (days later)
Q2: "How does that relate to cryptography?"
# MCP retrieves Q1/A1 automatically via semantic search!
```

**Implementation:**
```python
result = await mcp_engine.rag_with_memory(
    question="How does that relate to cryptography?",
    session_id="user_123",
    collection_name="quantum_docs",
    use_memory_context=True,  # Auto-retrieves relevant past Q&A
    store_conversation=True,   # Saves this conversation
    use_embeddings=True        # Uses semantic similarity
)
```

### Feature 2: **Auto-Ingestion with File Monitoring**

**Why it's powerful:** Automatically ingests new documents as they're added to a directory.

**Use Case:** Research teams with constantly updating document repositories

**Example:**
```python
# Start monitoring a directory
await mcp_engine.auto_ingest_with_monitoring(
    watch_dir="./research_papers",
    collection_name="research_kb",
    file_types=["pdf", "md", "txt"],
    poll_interval=10,  # Check every 10 seconds
    use_watchdog=True  # Real-time file system events
)

# Now any file added to ./research_papers is automatically ingested!
```

**Handles:**
- ✅ New files → Auto-ingested
- ✅ Modified files → Re-ingested
- ✅ Deleted files → Custom handler for cleanup

### Feature 3: **Query Analytics & Performance Tracking**

**Why it's powerful:** Understand how your RAG system is being used and optimize accordingly.

**Use Case:** Identifying popular queries for caching, monitoring response times

**Example:**
```python
from mcp.analytics_server import log_query, get_query_analytics

# Log every query
await log_query(
    query="What is photosynthesis?",
    session_id="user_456",
    collection_name="biology",
    response_time_ms=1250,
    num_documents_retrieved=5,
    grounding_check_passed=True
)

# Get analytics
analytics = await get_query_analytics(time_period="24h")
print(f"Total queries: {analytics['total_queries']}")
print(f"Avg response time: {analytics['avg_response_time_ms']}ms")
print(f"Top queries: {analytics['top_queries']}")
```

### Feature 4: **SQL Database Integration**

**Why it's powerful:** Query structured data alongside unstructured documents.

**Use Case:** Product docs + user database, customer support with CRM integration

**Example:**
```python
# Query: "Show me all orders for customer John Doe"
# MCP SQL server can:
# 1. Query customer database for John Doe's ID
# 2. Retrieve order history
# 3. Combine with product documentation
# 4. Generate comprehensive answer
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           Your Application (Streamlit)              │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│         MCPEnhancedRAG (Orchestrator)               │
│  ┌──────────────────────────────────────────────┐   │
│  │    CoreRAGEngine (Refactored)                │   │
│  │  - LLM Factory                               │   │
│  │  - Embedding Factory                         │   │
│  │  - Chain Factory                             │   │
│  │  - Graph Nodes                               │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              MCP Servers (Tools)                    │
│  ┌──────────┬──────────┬──────────┬─────────────┐  │
│  │ Files    │ Memory   │ SQL      │ Analytics   │  │
│  │ Server   │ Server   │ Server   │ Server      │  │
│  └──────────┴──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 💡 Advanced Use Cases

### Use Case 1: **Research Assistant with Memory**

```python
# Researcher queries over weeks
mcp_engine = MCPEnhancedRAG(core_engine, enable_memory=True)

# Week 1: Background research
await mcp_engine.rag_with_memory(
    question="What is CRISPR?",
    session_id="researcher_001"
)

# Week 2: Deep dive (automatically recalls Week 1 context)
await mcp_engine.rag_with_memory(
    question="What are the ethical implications?",
    session_id="researcher_001"
)
# MCP retrieves the CRISPR context automatically!
```

### Use Case 2: **Living Documentation System**

```python
# Monitor docs directory
await mcp_engine.auto_ingest_with_monitoring(
    watch_dir="/company/documentation",
    collection_name="company_kb",
    file_types=["md", "pdf", "docx"]
)

# Deleted file handler
async def handle_deleted_files(deleted_files, collection):
    for file_path in deleted_files:
        # Remove from vector store
        # Log deletion
        # Notify team
        pass

mcp_engine.set_deleted_files_handler(handle_deleted_files)
```

### Use Case 3: **Analytics-Driven Optimization**

```python
# Get popular queries
popular = await get_popular_queries(limit=20)

# Pre-generate answers for top queries (caching)
for query_info in popular['popular_queries']:
    if query_info['count'] > 100:  # High traffic query
        # Pre-compute and cache answer
        result = await mcp_engine.rag_with_memory(
            question=query_info['query'],
            session_id="cache_warmup"
        )

# Monitor performance
trends = await get_performance_trends(
    metric_type="response_time",
    time_period="7d"
)
if trends['recent_trend'] == "declining":
    # Alert: Performance degradation detected!
    send_alert_to_team()
```

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
MCP_FILESYSTEM_COMMAND=python
MCP_MEMORY_COMMAND=python
MCP_SQL_COMMAND=python

# For analytics (optional)
MCP_ANALYTICS_RETENTION_DAYS=90
MCP_ANALYTICS_AUTO_CLEANUP=true
```

### Custom MCP Configuration

```python
custom_config = {
    "filesystem": {
        "command": "python",
        "args": ["/path/to/filesystem_server.py"],
        "transport": "stdio"
    },
    "memory": {
        "command": "python",
        "args": ["/path/to/memory_server.py"],
        "transport": "stdio"
    }
}

mcp_engine = MCPEnhancedRAG(
    core_rag_engine=core_engine,
    mcp_config=custom_config
)
```

---

## 📊 Monitoring & Debugging

### Check MCP Status

```python
# In Streamlit app
if hasattr(engine, 'mcp_client'):
    st.success("✅ MCP Active")
    st.write(f"Tools available: {len(engine.tools)}")
    for tool_name in engine.tools.keys():
        st.code(tool_name)
else:
    st.warning("⚠️ MCP Disabled")
```

### View Analytics Dashboard

```python
# Get comprehensive analytics
analytics = await get_query_analytics(time_period="24h")

# Display in Streamlit
st.metric("Total Queries", analytics['total_queries'])
st.metric("Avg Response Time", f"{analytics['avg_response_time_ms']}ms")
st.bar_chart(analytics['top_queries'])
```

---

## 🚀 Next Steps

1. **Enable MCP in production** - Use `main_app_mcp_enhanced.py`
2. **Set up monitoring directories** - Auto-ingest company docs
3. **Enable conversation memory** - Better multi-turn dialogues
4. **Add analytics dashboard** - Track usage and performance
5. **Integrate with SQL** - Combine structured and unstructured data

## 📚 Additional Resources

- [MCP Documentation](https://github.com/anthropics/mcp)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Project README](/README.md)

---

## ⚠️ Requirements

- **Python 3.10+** for MCP support
- **ChromaDB** for vector storage
- **SQLite** for analytics and memory storage
- **Optional: watchdog** for real-time file monitoring

## 🐛 Troubleshooting

**Problem:** MCP servers not initializing

**Solution:**
```bash
# Check MCP installation
python -c "from mcp.server.fastmcp import FastMCP; print('OK')"

# Install if missing
pip install mcp
```

**Problem:** File monitoring not working

**Solution:**
```bash
# Install watchdog for real-time monitoring
pip install watchdog
```

**Problem:** Memory retrieval too slow

**Solution:**
```python
# Use embeddings for semantic search (faster)
use_embeddings=True

# Adjust similarity threshold
similarity_threshold=0.5  # Higher = stricter matching
```

---

🎉 **Your RAG system is now supercharged with MCP!**
