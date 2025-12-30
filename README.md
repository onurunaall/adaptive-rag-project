# Adaptive RAG Engine

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

**A sophisticated, production-ready Retrieval-Augmented Generation (RAG) system with adaptive query processing, multi-provider LLM support, and intelligent document handling.**

[Features](#-key-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Architecture](#-architecture) •
[Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Performance](#-performance)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

The **Adaptive RAG Engine** is an enterprise-grade, production-ready system that intelligently retrieves and generates contextually relevant answers by combining the power of large language models with dynamic document retrieval. Unlike traditional RAG systems, our engine adapts its strategy based on query complexity, document relevance, and answer quality.

### Why Adaptive RAG?

Traditional RAG systems use a one-size-fits-all approach. Our engine intelligently:
- **Analyzes queries** to determine the best retrieval strategy
- **Adapts chunking** based on document type (code, academic, financial)
- **Validates answers** through multi-stage grounding checks
- **Falls back** to web search when local knowledge is insufficient
- **Optimizes performance** with intelligent caching and streaming

---

## ✨ Key Features

### 🧠 Intelligent Query Processing

- **Query Analysis**: Automatically classifies queries (factual, analytical, exploratory)
- **Query Rewriting**: Optimizes search queries for better retrieval
- **Semantic Search**: Vector-based similarity search with hybrid options
- **Multi-hop Reasoning**: Handles complex, multi-part questions

### 📚 Advanced Document Management

- **Adaptive Chunking**: Dynamic chunking strategies based on content type
  - Code documents: Syntax-aware chunking
  - Academic papers: Section-based chunking
  - Financial reports: Structure-preserving chunking
- **Document Grading**: Relevance scoring and filtering
- **Reranking**: Cross-encoder based document reranking
- **Streaming**: Efficient handling of large document collections

### 🔍 Multi-Source Retrieval

- **Vector Database**: ChromaDB for efficient vector storage
- **Web Search Integration**: Tavily API for real-time information
- **Stock Data Feed**: Financial data integration
- **Custom Scrapers**: Flexible document ingestion

### 🤖 Multi-Provider LLM Support

- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini
- **Google**: Gemini Pro and variants
- **Ollama**: Local LLM deployment
- **Extensible**: Easy integration of new providers

### ✅ Answer Quality Assurance

- **Grounding Checks**: Validates answers against source documents
- **Hallucination Detection**: Identifies and corrects ungrounded statements
- **Iterative Refinement**: Automatic answer improvement
- **Citation Support**: Tracks sources for transparency

### ⚡ Performance Optimization

- **Intelligent Caching**: Multi-level caching system
- **Batch Processing**: Efficient document processing
- **Streaming Responses**: Real-time answer generation
- **Memory Management**: Automatic cache eviction

### 🎯 LangGraph Workflow

- **State Machine**: Robust workflow orchestration
- **Error Handling**: Comprehensive error tracking and recovery
- **Parallel Processing**: Concurrent document processing
- **Extensible Nodes**: Easy workflow customization

---

## 🏗 Architecture

### Modular Facade Pattern (v2.0)

The Adaptive RAG Engine follows a **modular, facade-based architecture** with **7 specialized modules** orchestrated by a lightweight facade.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                           │
│                    (Streamlit / API / CLI)                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                CoreRAGEngine (Facade - 1,147 lines)                  │
│  Delegates to specialized modules:                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │   Document     │  │  VectorStore   │  │      Query            │ │
│  │   Manager      │  │   Manager      │  │    Processor          │ │
│  └────────────────┘  └────────────────┘  └───────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │   Document     │  │    Answer      │  │      Cache            │ │
│  │    Grader      │  │   Generator    │  │  Orchestrator         │ │
│  └────────────────┘  └────────────────┘  └───────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │            WorkflowOrchestrator (LangGraph)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼─────────┐                   ┌──────▼──────┐
│  Vector Store   │                   │ LLM Providers│
│   (ChromaDB)    │                   │ (Multi-model)│
└─────────────────┘                   └─────────────┘
```

### **Refactored Architecture Benefits**

| Metric | Before (v1.0) | After (v2.0) | Improvement |
|--------|---------------|--------------|-------------|
| **CoreRAGEngine** | 2,976 lines | 1,147 lines | **61% reduction** |
| **Modularity** | Monolithic | 7 modules | **100% modular** |
| **Test Coverage** | Limited | 184 tests | **Comprehensive** |
| **Maintainability** | Low | High | **Easy to modify** |

### Core Modules

#### 1. **DocumentManager** (`src/rag/document_manager.py`)
Document loading and splitting with adaptive strategies (342 lines, 14 tests)

#### 2. **VectorStoreManager** (`src/rag/vector_store_manager.py`)
Vector store operations and hybrid search (449 lines, 30 tests)

#### 3. **QueryProcessor** (`src/rag/query_processor.py`)
Query analysis and rewriting (228 lines, 20 tests)

#### 4. **DocumentGrader** (`src/rag/document_grader.py`)
Relevance grading and reranking (252 lines, 27 tests)

#### 5. **AnswerGenerator** (`src/rag/answer_generator.py`)
Answer generation and grounding validation (383 lines, 30 tests)

#### 6. **CacheOrchestrator** (`src/rag/cache_orchestrator.py`)
Intelligent cache management (326 lines, 35 tests)

#### 7. **WorkflowOrchestrator** (`src/rag/workflow_orchestrator.py`)
LangGraph workflow orchestration (335 lines, 28 tests)

#### 8. **CoreRAGEngine** (`src/core_rag_engine.py`)
Facade that delegates to all modules (1,147 lines)

### Additional Components

#### 9. **Configuration Management** (`src/config.py`)
Centralized configuration with Pydantic models and MCP feature flag

#### 10. **Context Manager** (`src/context_manager.py`)
Intelligent context window management

#### 11. **Data Feeds**
- `src/stock_feed.py`: Financial data integration
- `src/scraper_feed.py`: Web scraping

#### 12. **User Interface** (`src/main_app.py`)
Unified Streamlit app with optional MCP enhancement

**For detailed architecture documentation, see [Architecture.md](./Architecture.md)**

---

## 🚀 Installation

### Prerequisites

- **Python 3.10 or higher** (Python 3.9 reached EOL in October 2025)
- pip for package management
- OpenAI API key (for OpenAI models)
- (Optional) Google API key for Gemini models
- (Optional) Tavily API key for web search

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-rag-project.git
cd adaptive-rag-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using pip (project uses pyproject.toml)
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Using Poetry

```bash
# Install with poetry
poetry install

# Activate environment
poetry shell
```

### Docker Installation

```bash
# Build image
docker build -t adaptive-rag .

# Run container
docker run -p 8501:8501 --env-file .env adaptive-rag
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: LLM Provider API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=your-google-api-key  # Optional
TAVILY_API_KEY=your-tavily-key      # Optional for web search

# Optional: Model Configuration
LLM_PROVIDER=openai                  # openai, google, or ollama
LLM_MODEL_NAME=gpt-4o
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small

# Optional: Performance Tuning
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_GROUNDING_ATTEMPTS=3
CACHE_TTL_SECONDS=300
MAX_CACHE_SIZE_MB=500
```

### GitHub Codespaces

If using GitHub Codespaces:

1. Go to **Settings** → **Secrets and variables** → **Codespaces**
2. Add secrets: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`
3. Restart your Codespace

### Configuration File

Advanced configuration in `src/config.py`:

```python
from src.config import LLMSettings, EmbeddingSettings, EngineSettings

# Customize LLM settings
llm_config = LLMSettings(
    provider="openai",
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=2000
)

# Customize engine settings
engine_config = EngineSettings(
    chunk_size=1000,
    chunking_strategy="adaptive",
    enable_hybrid_search=True,
    max_grounding_attempts=5
)
```

---

## 🎯 Quick Start

### Using the Streamlit UI

```bash
# Run the application
make run

# Or directly
streamlit run src/main_app.py
```

Navigate to `http://localhost:8501` in your browser.

### Python API

```python
from src.core_rag_engine import CoreRAGEngine
from langchain_core.documents import Document

# Initialize the engine
engine = CoreRAGEngine(
    llm_provider="openai",
    embedding_provider="openai"
)

# Ingest documents
documents = [
    Document(
        page_content="Paris is the capital of France.",
        metadata={"source": "geography.txt"}
    ),
    Document(
        page_content="Python is a programming language.",
        metadata={"source": "programming.txt"}
    )
]

engine.ingest(
    direct_documents=documents,
    collection_name="my_knowledge_base",
    recreate_collection=True
)

# Query the system
result = engine.run_full_rag_workflow(
    question="What is the capital of France?",
    collection_name="my_knowledge_base"
)

print(result["answer"])
# Output: "The capital of France is Paris."
```

### Streamlit Web Interface

```bash
# Run the web UI
streamlit run src/main_app.py

# Or use the Makefile
make run

# For MCP-enhanced version with conversation memory
streamlit run src/main_app_mcp_enhanced.py
```

---

## 💡 Usage Examples

### Example 1: Basic Document Q&A

```python
from src.core_rag_engine import CoreRAGEngine

engine = CoreRAGEngine()

# Ingest from directory
engine.ingest(
    source_directory="./my_documents",
    collection_name="company_docs"
)

# Ask questions
answer = engine.run_full_rag_workflow(
    question="What is our company policy on remote work?",
    collection_name="company_docs"
)
```

### Example 2: Web-Enhanced Search

```python
# Enable web search fallback
engine = CoreRAGEngine(
    tavily_api_key="your-key"
)

# If local knowledge is insufficient, automatically searches the web
result = engine.run_full_rag_workflow(
    question="What are the latest developments in quantum computing?",
    collection_name="tech_docs"
)
```

### Example 3: Financial Data Analysis

```python
from src.stock_feed import fetch_stock_news_as_documents

# Fetch financial data
stock_docs = fetch_stock_news_as_documents(
    tickers=["AAPL", "MSFT", "GOOGL"],
    days_back=7
)

# Ingest and analyze
engine.ingest(
    direct_documents=stock_docs,
    collection_name="market_analysis"
)

result = engine.run_full_rag_workflow(
    question="What are the key trends for tech stocks this week?",
    collection_name="market_analysis"
)
```

### Example 4: Code Documentation

```python
# Ingest code files with adaptive chunking
engine.ingest(
    source_directory="./src",
    collection_name="codebase",
    chunking_strategy="adaptive"  # Automatically detects code
)

# Query about your codebase
result = engine.run_full_rag_workflow(
    question="How does the grounding check work?",
    collection_name="codebase"
)
```

### Example 5: Multi-Collection Search

```python
# Search across multiple knowledge bases
result = engine.run_full_rag_workflow(
    question="Compare our product features with industry standards",
    collection_name="product_docs"  # Primary collection
)

# Access retrieved documents
for doc in result["documents"]:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content}\n")
```

### Example 6: Streaming Responses

```python
# Stream answers in real-time
for chunk in engine.stream_answer(
    question="Explain the RAG architecture in detail",
    collection_name="tech_docs"
):
    print(chunk, end="", flush=True)
```

### Example 7: Advanced Configuration

```python
from src.config import EngineSettings

# Custom configuration
settings = EngineSettings(
    chunk_size=1000,
    chunk_overlap=200,
    chunking_strategy="adaptive",
    enable_hybrid_search=True,
    hybrid_search_alpha=0.7,
    max_grounding_attempts=5,
    enable_advanced_grounding=True,
    cache_ttl_seconds=600
)

engine = CoreRAGEngine(engine_settings=settings)
```

---

## 📚 API Reference

### CoreRAGEngine

#### `__init__(self, **kwargs)`

Initialize the RAG engine.

**Parameters:**
- `llm_provider` (str): LLM provider ("openai", "google", "ollama")
- `llm_model_name` (str): Model name
- `embedding_provider` (str): Embedding provider
- `persist_directory_base` (str): Base directory for vector storage
- `openai_api_key` (str): OpenAI API key
- `google_api_key` (str): Google API key
- `tavily_api_key` (str): Tavily API key

#### `ingest(self, **kwargs)`

Ingest documents into the vector store.

**Parameters:**
- `direct_documents` (List[Document]): Documents to ingest
- `source_directory` (str): Directory path to load documents from
- `urls` (List[str]): URLs to scrape and ingest
- `collection_name` (str): Name of the collection
- `recreate_collection` (bool): Whether to recreate the collection

**Returns:** None

**Example:**
```python
engine.ingest(
    direct_documents=docs,
    collection_name="my_collection",
    recreate_collection=True
)
```

#### `run_full_rag_workflow(self, question: str, **kwargs)`

Execute the complete RAG workflow.

**Parameters:**
- `question` (str): User's question
- `collection_name` (str): Collection to search
- `chat_history` (List): Previous conversation history
- `max_retries` (int): Maximum query rewrite attempts

**Returns:** dict with keys:
- `answer` (str): Generated answer
- `documents` (List[Document]): Retrieved documents
- `context` (str): Context used for generation
- `error_message` (str, optional): Any errors encountered

**Example:**
```python
result = engine.run_full_rag_workflow(
    question="What is RAG?",
    collection_name="ai_docs"
)
print(result["answer"])
```

#### `get_all_documents(self, collection_name: str)`

Retrieve all documents from a collection.

**Parameters:**
- `collection_name` (str): Collection name

**Returns:** List[Document]

#### `get_collection_stats(self, collection_name: str)`

Get statistics about a collection.

**Parameters:**
- `collection_name` (str): Collection name

**Returns:** dict with collection statistics

#### `get_cache_stats(self)`

Get cache statistics.

**Returns:** dict with cache metrics

#### `invalidate_all_caches(self)`

Clear all caches.

**Returns:** None

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/test_core_rag_engine.py -v

# Run specific test
pytest tests/test_core_rag_engine.py::test_rag_direct_answer -v

# Run with markers
pytest -m "not slow" -v
```

### Test Structure

```
tests/
├── test_core_rag_engine.py     # Core engine tests
├── test_config.py               # Configuration tests
├── test_context_manager.py      # Context management tests
├── test_agent_loop.py           # Agent workflow tests
├── test_stock_feed.py           # Stock feed tests
├── test_scraper_feed.py         # Scraper tests
└── test_sql_security.py         # Security tests
```

### Writing Tests

```python
import pytest
from src.core_rag_engine import CoreRAGEngine

@pytest.fixture
def rag_engine():
    return CoreRAGEngine(
        llm_provider="openai",
        persist_directory_base="test_db"
    )

def test_custom_functionality(rag_engine):
    # Your test here
    assert rag_engine is not None
```

---

## 📊 Performance

### Benchmarks

| Operation | Avg Time | Throughput |
|-----------|----------|------------|
| Document Ingestion (1000 docs) | 45s | ~22 docs/s |
| Query Processing | 2-5s | - |
| Vector Search | 100-300ms | - |
| Answer Generation | 1-3s | - |

### Optimization Tips

1. **Enable Caching**: Significant speedup for repeated queries
```python
engine = CoreRAGEngine(
    cache_ttl_seconds=600,
    max_cache_size_mb=1000
)
```

2. **Hybrid Search**: Better relevance, slightly slower
```python
settings = EngineSettings(
    enable_hybrid_search=True,
    hybrid_search_alpha=0.7
)
```

3. **Batch Processing**: For large document sets
```python
engine.ingest(
    source_directory="large_corpus",
    batch_size=1000
)
```

4. **Streaming**: For large collections
```python
for batch in engine.stream_documents_from_collection(
    "large_collection",
    batch_size=1000
):
    process_batch(batch)
```

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build production image
docker build -t adaptive-rag:latest .

# Run with docker-compose
docker-compose up -d

# Scale services
docker-compose up --scale worker=3
```

### Cloud Deployment

#### AWS

```bash
# Deploy to ECS
aws ecs create-service --cli-input-json file://ecs-service.json

# Or use CDK
cdk deploy AdaptiveRagStack
```

#### Google Cloud

```bash
# Deploy to Cloud Run
gcloud run deploy adaptive-rag \
  --image gcr.io/PROJECT_ID/adaptive-rag \
  --platform managed
```

#### Azure

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name adaptive-rag \
  --image myregistry.azurecr.io/adaptive-rag
```

### Environment Variables for Production

```bash
# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=yourdomain.com

# Scaling
MAX_WORKERS=4
WORKER_TIMEOUT=30

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "OPENAI_API_KEY is missing"

**Solution:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# Or export directly
export OPENAI_API_KEY=sk-...
```

#### Issue: "Collection not found"

**Solution:**
```python
# List available collections
engine.list_collections()

# Create new collection
engine.ingest(
    direct_documents=docs,
    collection_name="new_collection",
    recreate_collection=True
)
```

#### Issue: "Out of memory"

**Solution:**
```python
# Enable streaming for large collections
settings = EngineSettings(
    enable_streaming_for_large_collections=True,
    max_cache_size_mb=500
)

# Or reduce cache size
engine.invalidate_all_caches()
```

#### Issue: Tests failing with async errors

**Solution:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Verify pyproject.toml has:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

engine = CoreRAGEngine()
# Now all operations will show detailed logs
```

### Performance Issues

1. **Slow queries**: Enable caching, use hybrid search
2. **High memory**: Reduce cache size, enable streaming
3. **Poor results**: Adjust chunking strategy, increase retrieval_top_k

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/yourusername/adaptive-rag-project.git
cd adaptive-rag-project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
make format

# Run linters
make lint

# Type check
make type-check
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`make test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Commit Messages

Follow conventional commits:

```
feat: add new chunking strategy
fix: resolve cache invalidation issue
docs: update API documentation
test: add tests for grounding checks
refactor: simplify query rewriting logic
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Adaptive RAG Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

### Technologies Used

- **LangChain**: Framework for LLM applications
- **LangGraph**: Workflow orchestration
- **ChromaDB**: Vector database
- **Streamlit**: User interface
- **OpenAI**: LLM provider
- **Google Gemini**: Alternative LLM provider
- **Tavily**: Web search API

### Inspiration

This project was inspired by:
- [LangChain RAG Tutorials](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone's RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- Research papers on adaptive retrieval systems

### Contributors

Thanks to all our contributors! 🎉

<a href="https://github.com/yourusername/adaptive-rag-project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yourusername/adaptive-rag-project" />
</a>

---

## 📞 Support

- **Documentation**: [https://docs.yourproject.com](https://docs.yourproject.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/adaptive-rag-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/adaptive-rag-project/discussions)
- **Email**: support@yourproject.com

---

## 🗺️ Roadmap

### Version 2.0 (Q2 2025)

- [ ] Multi-modal support (images, audio)
- [ ] Graph-based retrieval
- [ ] Advanced agentic workflows
- [ ] Cloud-native deployment templates

### Version 2.1 (Q3 2025)

- [ ] Real-time collaborative features
- [ ] Custom LLM fine-tuning integration
- [ ] Enhanced monitoring and observability
- [ ] Mobile application

---

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/adaptive-rag-project?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/adaptive-rag-project?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/adaptive-rag-project?style=social)

---

<div align="center">

**[⬆ back to top](#adaptive-rag-engine)**

Made with ❤️ by the Adaptive RAG Team

</div>