# MCP Feature Migration Guide

**Version:** 2.0
**Last Updated:** 2025-12-30

## Overview

The MCP (Model Context Protocol) integration is now an **optional feature** that can be enabled via configuration. This guide explains how to enable and use MCP features.

## What is MCP?

MCP enhances the Adaptive RAG Engine with:
- **Filesystem Monitoring**: Auto-ingest documents from watched directories
- **Conversation Memory**: Persistent memory across chat sessions
- **SQL Database Access**: Query databases directly from the RAG workflow
- **Enhanced Observability**: Additional monitoring and logging capabilities

## Enabling MCP

### Method 1: Environment Variable

Add to your `.env` file:

```bash
# Enable MCP features
MCP_ENABLE_MCP=true
```

### Method 2: Programmatic Configuration

```python
from src.config import settings

# Enable MCP
settings.mcp.enable_mcp = True
```

## Using MCP Features

### 1. Filesystem Monitoring (Auto-Ingest)

When MCP is enabled, the Streamlit UI shows an "Auto-Ingest" section:

```python
# Watch a directory for new files
watch_dir = "./documents"
file_types = ["pdf", "txt", "md"]
poll_interval = 10  # seconds

# Start monitoring (via UI button)
```

New files added to the watched directory will be automatically ingested into the specified collection.

### 2. Conversation Memory

When enabled, conversations are stored and retrieved across sessions:

```python
# In Streamlit UI, check "Use Conversation Memory"

# The engine will:
# - Store each Q&A pair
# - Retrieve relevant past conversations
# - Include context in future answers
```

### 3. SQL Database Access

Query databases directly from the RAG workflow:

```python
# Example: Query a database for context
# (Available when MCP is enabled)
```

## Graceful Fallback

If MCP initialization fails:
- ✅ Application continues running
- ✅ Falls back to standard RAG mode
- ⚠️ Warning logged for troubleshooting
- ℹ️ User sees: "MCP initialization failed. Using standard RAG."

## Disabling MCP

### Method 1: Environment Variable

```bash
# Disable MCP (or omit the variable - disabled by default)
MCP_ENABLE_MCP=false
```

### Method 2: Programmatic

```python
settings.mcp.enable_mcp = False
```

## Architecture Differences

### Standard RAG Mode (MCP Disabled)

```
User → CoreRAGEngine → Modules → LLM/VectorStore
```

### MCP-Enhanced Mode (MCP Enabled)

```
User → MCPEnhancedRAG → CoreRAGEngine → Modules → LLM/VectorStore
                  ↓
            MCP Servers
            (Filesystem, Memory, SQL)
```

## Feature Comparison

| Feature | Standard RAG | MCP-Enhanced RAG |
|---------|--------------|------------------|
| Document Ingestion | Manual upload | Manual + Auto-monitoring |
| Conversation Memory | Session-only | Persistent across sessions |
| Database Access | Not available | SQL queries supported |
| Filesystem Monitoring | Not available | Available |
| Performance | Fast | Slightly slower (due to MCP overhead) |
| Dependencies | Minimal | Requires MCP servers |

## Troubleshooting

### MCP Not Starting

**Symptom:** "MCP initialization failed" warning

**Possible Causes:**
1. MCP servers not installed/available
2. Python path issues
3. Missing dependencies

**Solution:**
```bash
# Check MCP server files exist
ls mcp/filesystem_server.py
ls mcp/memory_server.py
ls mcp/sql_server.py

# Verify Python can import MCP module
python -c "from mcp.rag_integration import MCPEnhancedRAG"
```

### MCP Features Not Showing in UI

**Symptom:** No "Auto-Ingest" or "Conversation Memory" options

**Solution:**
```bash
# Verify MCP_ENABLE_MCP is set
echo $MCP_ENABLE_MCP  # Should output: true

# Check config
python -c "from src.config import settings; print(settings.mcp.enable_mcp)"
```

### Performance Issues with MCP

**Symptom:** Slower query responses

**Solution:**
- MCP adds overhead for memory/filesystem operations
- Consider disabling if performance is critical
- Profile to identify bottlenecks

## Best Practices

### When to Enable MCP

✅ **Enable MCP when you need:**
- Automatic document monitoring
- Conversation history across sessions
- Database integration
- Enhanced observability

❌ **Disable MCP when:**
- Performance is critical
- Simpler architecture preferred
- MCP features not needed
- Running in constrained environments

### Production Deployment

```bash
# Production environment
MCP_ENABLE_MCP=true  # Only if MCP features needed

# Development environment
MCP_ENABLE_MCP=true  # Enable for testing all features

# CI/CD environment
MCP_ENABLE_MCP=false  # Disable for faster tests
```

## Migration from main_app_mcp_enhanced.py

**Old Approach (v1.0):**
- Two separate apps: `main_app.py` and `main_app_mcp_enhanced.py`
- Manual switching between apps
- Code duplication

**New Approach (v2.0):**
- Single `main_app.py` with feature flag
- Automatic fallback
- No code duplication
- Configuration-driven

### No Code Changes Required!

The API is **100% backward compatible**. Existing code will continue to work without modification.

## Examples

### Enable MCP for Development

```bash
# .env
MCP_ENABLE_MCP=true
OPENAI_API_KEY=your_key
```

### Disable MCP for Production

```bash
# .env.production
MCP_ENABLE_MCP=false
OPENAI_API_KEY=your_key
```

### Conditional MCP in Code

```python
from src.config import settings

if settings.mcp.enable_mcp:
    print("MCP features available")
else:
    print("Standard RAG mode")
```

## Support

For questions or issues:
- **Documentation**: See [Architecture.md](./Architecture.md)
- **MCP Integration Guide**: See `docs/MCP_INTEGRATION_GUIDE.md`
- **Issues**: Report at GitHub Issues

---

**Last Updated:** 2025-12-30
**Version:** 2.0
