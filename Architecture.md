# Adaptive RAG Engine - Architecture Documentation

**Version:** 2.0 (Post-Refactoring)
**Last Updated:** 2025-12-30
**Status:** Production-Ready

## Table of Contents

1. [System Overview](#system-overview)
2. [Refactored Architecture](#refactored-architecture)
3. [Core Modules](#core-modules)
4. [CoreRAGEngine (Facade)](#coreragengine-facade)
5. [Data Flow](#data-flow)
6. [LangGraph Workflow](#langgraph-workflow)
7. [Storage Architecture](#storage-architecture)
8. [LLM Integration](#llm-integration)
9. [Caching Strategy](#caching-strategy)
10. [MCP Integration (Optional)](#mcp-integration-optional)
11. [Error Handling](#error-handling)
12. [Performance Considerations](#performance-considerations)
13. [Security](#security)

---

## System Overview

The Adaptive RAG Engine follows a **modular, facade-based architecture** that separates concerns into specialized modules while maintaining a simple public API. The system has been refactored from a monolithic 2,976-line class into 7 specialized modules orchestrated by a lightweight facade.

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐    │
│  │   Streamlit    │  │   REST API     │  │   CLI Interface    │    │
│  │      UI        │  │   (Future)     │  │                    │    │
│  └────────┬───────┘  └────────┬───────┘  └──────────┬─────────┘    │
└───────────┼──────────────────────┼────────────────────┼──────────────┘
            │                      │                    │
┌───────────▼──────────────────────▼────────────────────▼──────────────┐
│                         Application Layer                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         CoreRAGEngine (Facade Pattern - 1,147 lines)         │   │
│  │  Delegates to specialized modules:                           │   │
│  │  • DocumentManager    • VectorStoreManager                   │   │
│  │  • QueryProcessor     • DocumentGrader                       │   │
│  │  • AnswerGenerator    • CacheOrchestrator                    │   │
│  │  • WorkflowOrchestrator                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │   Context    │  │    Agent     │  │   Configuration          │ │
│  │   Manager    │  │     Loop     │  │     Manager              │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
└────────────┬──────────────────┬──────────────────┬──────────────────┘
             │                  │                  │
┌────────────▼──────────────────▼──────────────────▼──────────────────┐
│                         Integration Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │     LLM      │  │    Vector    │  │     Data Feeds           │ │
│  │  Providers   │  │    Store     │  │  • Stock • Scraper       │ │
│  │  (Factories) │  │  (ChromaDB)  │  │  • MCP (Optional)        │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Facade Pattern**: CoreRAGEngine provides simple API, delegates to modules
3. **Modularity**: 7 specialized modules (2,315 lines) replace monolith
4. **Extensibility**: Easy to add new features or swap implementations
5. **Type Safety**: Full type hints and runtime validation with Pydantic
6. **Observability**: Comprehensive logging and error tracking
7. **Performance**: Intelligent caching and async processing
8. **Testability**: 184 unit tests across all modules

---

## Refactored Architecture

### Before Refactoring (v1.0)

```
CoreRAGEngine (2,976 lines - God Class)
├── Document Loading & Splitting (200 lines)
├── Vector Store Management (400 lines)
├── Query Processing (200 lines)
├── Document Grading (250 lines)
├── Answer Generation (400 lines)
├── Cache Management (200 lines)
├── Workflow Orchestration (500 lines)
└── Public API Methods (300 lines)
```

**Problems:**
- ❌ Single 2,976-line class (god class anti-pattern)
- ❌ Multiple responsibilities violating SRP
- ❌ Hard to test individual components
- ❌ Difficult to understand and maintain
- ❌ Tight coupling between components

### After Refactoring (v2.0)

```
CoreRAGEngine (1,147 lines - Facade)
├── Delegates to DocumentManager (342 lines)
├── Delegates to VectorStoreManager (449 lines)
├── Delegates to QueryProcessor (228 lines)
├── Delegates to DocumentGrader (252 lines)
├── Delegates to AnswerGenerator (383 lines)
├── Delegates to CacheOrchestrator (326 lines)
└── Delegates to WorkflowOrchestrator (335 lines)
```

**Benefits:**
- ✅ 61% code reduction in CoreRAGEngine (2,976 → 1,147 lines)
- ✅ Each module has single responsibility
- ✅ 184 comprehensive unit tests
- ✅ Easy to test, understand, and maintain
- ✅ Loose coupling via callback patterns
- ✅ 100% backward API compatibility

---

## Core Modules

### 1. DocumentManager (`src/rag/document_manager.py`)

**Responsibility:** Document loading and splitting operations

**Lines:** 342
**Tests:** 14 test cases

**Key Methods:**
```python
class DocumentManager:
    def load_documents(
        self,
        source_type: str,
        source_value: Any
    ) -> List[Document]:
        """Load documents from various sources (URL, PDF, text, uploaded)."""

    def split_documents(
        self,
        documents: List[Document],
        strategy: str = "default"
    ) -> List[Document]:
        """Split documents using specified strategy (adaptive, semantic, hybrid)."""
```

**Supported Sources:**
- URLs (via WebBaseLoader)
- PDF files (via PyPDFLoader)
- Text files (via TextLoader)
- Uploaded files (Streamlit UploadedFile)

**Splitting Strategies:**
- `default`: RecursiveCharacterTextSplitter
- `adaptive`: AdaptiveChunker (content-type aware)
- `semantic`: Semantic chunking based on embeddings
- `hybrid`: HybridChunker combining multiple strategies

---

### 2. VectorStoreManager (`src/rag/vector_store_manager.py`)

**Responsibility:** Vector store initialization, persistence, and retrieval

**Lines:** 449
**Tests:** 30 test cases

**Key Methods:**
```python
class VectorStoreManager:
    def init_or_load_vectorstore(
        self,
        collection_name: str,
        recreate: bool = False
    ):
        """Initialize or load vectorstore from disk."""

    def index_documents(
        self,
        documents: List[Document],
        collection_name: str,
        recreate: bool = False
    ):
        """Index documents into collection."""

    def setup_retriever_for_collection(self, collection_name: str):
        """Setup retriever (standard or hybrid) for collection."""

    def list_collections(self) -> List[str]:
        """List all available collections."""

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
```

**Features:**
- Persistent storage with ChromaDB
- Hybrid search support (vector + BM25/TF-IDF)
- Memory-efficient document streaming
- Collection management (list, delete, recreate)
- Cache invalidation callbacks

---

### 3. QueryProcessor (`src/rag/query_processor.py`)

**Responsibility:** Query analysis and rewriting operations

**Lines:** 228
**Tests:** 20 test cases

**Key Methods:**
```python
class QueryProcessor:
    def analyze_query(
        self,
        question: str,
        chat_history: Optional[List] = None
    ) -> Optional[QueryAnalysis]:
        """Analyze query to determine type, intent, key terms."""

    def rewrite_query(
        self,
        question: str,
        chat_history: Optional[List] = None
    ) -> str:
        """Rewrite query to be clear, specific, self-contained."""

    def should_use_web_search(
        self,
        query_analysis: Optional[QueryAnalysis]
    ) -> bool:
        """Determine if web search should be used."""
```

**Query Types:**
- `factual_lookup`: Specific facts or definitions
- `comparison`: Comparing multiple items
- `summary_request`: Summaries or overviews
- `complex_reasoning`: Multi-step reasoning
- `ambiguous`: Unclear questions
- `keyword_search_sufficient`: Simple lookups
- `greeting`: Casual greetings
- `not_a_question`: Statements or commands

---

### 4. DocumentGrader (`src/rag/document_grader.py`)

**Responsibility:** Document relevance grading and reranking

**Lines:** 252
**Tests:** 27 test cases

**Key Methods:**
```python
class DocumentGrader:
    def grade_documents(
        self,
        documents: List[Document],
        question: str
    ) -> List[Document]:
        """Grade documents for relevance and filter out irrelevant ones."""

    def rerank_documents(
        self,
        documents: List[Document],
        question: str
    ) -> List[Document]:
        """Rerank documents by relevance scores (highest first)."""

    def calculate_relevance_score(
        self,
        document: Document,
        question: str
    ) -> float:
        """Calculate relevance score for single document (0.0-1.0)."""

    def is_relevant(
        self,
        document: Document,
        question: str
    ) -> bool:
        """Boolean relevance check."""
```

**Features:**
- LLM-based relevance grading
- Score-based reranking
- Parallel document processing
- Relevance threshold filtering

---

### 5. AnswerGenerator (`src/rag/answer_generator.py`)

**Responsibility:** Answer generation and grounding validation

**Lines:** 383
**Tests:** 30 test cases

**Key Methods:**
```python
class AnswerGenerator:
    def generate_answer(
        self,
        question: str,
        context: str,
        chat_history: Optional[List] = None,
        regeneration_feedback: Optional[str] = None
    ) -> str:
        """Generate answer from context with chat history support."""

    def check_grounding(
        self,
        context: str,
        generation: str
    ) -> Optional[GroundingCheck]:
        """Check if generated answer is grounded in context."""

    def generate_basic_feedback(
        self,
        grounding_result: GroundingCheck,
        question: str
    ) -> str:
        """Generate feedback for basic grounding failures."""

    def generate_advanced_feedback(
        self,
        advanced_results: Dict,
        question: str
    ) -> str:
        """Generate detailed feedback from advanced grounding analysis."""

    def format_context(self, documents: List[Any]) -> str:
        """Format documents into context string."""
```

**Features:**
- RAG-based answer generation
- Chat history integration
- Grounding validation
- Hallucination detection
- Iterative refinement with feedback
- Context formatting

---

### 6. CacheOrchestrator (`src/rag/cache_orchestrator.py`)

**Responsibility:** Cache management coordination

**Lines:** 326
**Tests:** 35 test cases

**Key Methods:**
```python
class CacheOrchestrator:
    def cache_documents(
        self,
        collection_name: str,
        documents: List[Any]
    ):
        """Cache documents for collection."""

    def get_cached_documents(
        self,
        collection_name: str
    ) -> Optional[List[Any]]:
        """Get cached documents with TTL check."""

    def maintain_cache(
        self,
        max_cache_size_mb: Optional[float] = None
    ):
        """Remove old entries if memory usage exceeds limit."""

    def clear_document_cache(
        self,
        collection_name: Optional[str] = None
    ):
        """Clear cache for specific collection or all."""

    def invalidate_collection_cache(self, collection_name: str):
        """Invalidate cache for specific collection."""

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
```

**Features:**
- TTL-based cache expiration (default: 5 minutes)
- Memory-based eviction (default: 500 MB)
- Per-collection caching
- Cache statistics tracking
- Automatic garbage collection

---

### 7. WorkflowOrchestrator (`src/rag/workflow_orchestrator.py`)

**Responsibility:** LangGraph workflow orchestration for RAG pipeline

**Lines:** 335
**Tests:** 28 test cases

**Key Methods:**
```python
class WorkflowOrchestrator:
    def compile_workflow(
        self,
        entry_point: str = "analyze_query",
        edges: Optional[List] = None,
        conditional_edges: Optional[List] = None
    ):
        """Compile RAG workflow graph."""

    def compile_default_rag_workflow(self):
        """Compile default RAG workflow with standard structure."""

    def invoke_workflow(
        self,
        initial_state: CoreGraphState
    ) -> CoreGraphState:
        """Invoke workflow synchronously."""

    async def ainvoke_workflow(
        self,
        initial_state: CoreGraphState
    ) -> CoreGraphState:
        """Invoke workflow asynchronously."""
```

**Workflow Nodes:**
- `analyze_query`: Analyze query characteristics
- `rewrite_query`: Rewrite query for better retrieval
- `retrieve`: Retrieve documents from vector store
- `grade_documents`: Grade documents for relevance
- `rerank_documents`: Rerank by relevance scores
- `generate_answer`: Generate answer from context
- `grounding_check`: Validate answer grounding
- `web_search`: Fallback to web search
- `increment_retries`: Track retry attempts

**Workflow Edges:**
- Conditional routing based on relevance checks
- Conditional routing based on grounding checks
- Retry logic with max attempts
- Fallback to web search when needed

---

## CoreRAGEngine (Facade)

**File:** `src/core_rag_engine.py`
**Lines:** 1,147 (reduced from 2,976)
**Pattern:** Facade

### Role

CoreRAGEngine serves as a **facade** that provides a simple, unified interface while delegating actual work to specialized modules. It maintains 100% backward API compatibility.

### Initialization

```python
def __init__(self, **config):
    # 1. Configure settings
    self.llm_provider = llm_provider or app_settings.llm.llm_provider
    # ... configuration ...

    # 2. Initialize factories
    self.llm_factory = LLMFactory(...)
    self.embedding_factory = EmbeddingFactory(...)
    self.text_splitter_factory = TextSplitterFactory(...)

    # 3. Initialize specialized modules
    self.document_manager = DocumentManager(...)
    self.cache_orchestrator = CacheOrchestrator(...)
    self.vector_store_manager = VectorStoreManager(...)
    self.query_processor = QueryProcessor(...)
    self.document_grader = DocumentGrader(...)
    self.answer_generator = AnswerGenerator(...)

    # 4. Initialize workflow orchestrator
    self.workflow_orchestrator = WorkflowOrchestrator(
        node_functions=self._create_node_functions(),
        routing_functions=self._create_routing_functions()
    )
    self.rag_workflow = self.workflow_orchestrator.compile_default_rag_workflow()
```

### Public API

**Document Ingestion:**
```python
def ingest(
    self,
    source_type: str,
    source_value: Any,
    collection_name: Optional[str] = None,
    recreate: bool = False,
    strategy: str = "default"
) -> Dict[str, Any]:
    """Delegates to DocumentManager and VectorStoreManager."""
```

**Query Processing:**
```python
def answer_query(
    self,
    question: str,
    collection_name: Optional[str] = None,
    chat_history: Optional[List[BaseMessage]] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """Delegates to WorkflowOrchestrator."""
```

**Async Workflow:**
```python
async def run_full_rag_workflow(
    self,
    question: str,
    collection_name: Optional[str] = None,
    chat_history: Optional[List[BaseMessage]] = None
) -> CoreGraphState:
    """Delegates to WorkflowOrchestrator (async)."""
```

**Collection Management:**
```python
def list_collections(self) -> List[str]:
    """Delegates to VectorStoreManager."""

def delete_collection(self, collection_name: str) -> bool:
    """Delegates to VectorStoreManager."""
```

**Cache Management:**
```python
def clear_document_cache(self, collection_name: Optional[str] = None):
    """Delegates to CacheOrchestrator."""

def get_cache_stats(self) -> Dict[str, Any]:
    """Delegates to CacheOrchestrator."""
```

---

## Data Flow

### Ingestion Flow

```
User Input (URL/PDF/Text)
    ↓
CoreRAGEngine.ingest()
    ↓
DocumentManager.load_documents()
    ↓
DocumentManager.split_documents()
    ↓
VectorStoreManager.index_documents()
    ↓
CacheOrchestrator.invalidate_collection_cache()
    ↓
ChromaDB (Persistent Storage)
```

### Query Flow

```
User Question
    ↓
CoreRAGEngine.answer_query()
    ↓
WorkflowOrchestrator.invoke_workflow()
    ↓
┌────────────────────────────────────────┐
│     LangGraph Workflow Execution       │
│                                        │
│ 1. QueryProcessor.analyze_query()     │
│ 2. QueryProcessor.rewrite_query()     │
│ 3. VectorStoreManager (retrieve)      │
│ 4. DocumentGrader.grade_documents()   │
│ 5. DocumentGrader.rerank_documents()  │
│ 6. AnswerGenerator.generate_answer()  │
│ 7. AnswerGenerator.check_grounding()  │
│                                        │
│ Conditional Routing:                  │
│ • Relevance failed → Web search       │
│ • Grounding failed → Regenerate       │
│ • Max retries → Return answer         │
└────────────────────────────────────────┘
    ↓
Formatted Response
```

---

## LangGraph Workflow

### State Machine

```python
@dataclass
class CoreGraphState(TypedDict):
    question: str
    original_question: str
    collection_name: str
    documents: List[Document]
    context: str
    generation: str
    chat_history: List[BaseMessage]
    query_analysis: Optional[QueryAnalysis]
    retries: int
    grounding_attempts: int
    relevance_check_passed: bool
    is_grounded: bool
    regeneration_feedback: Optional[str]
```

### Workflow Graph

```
         START
           ↓
    analyze_query ────────┐
           ↓              │
    rewrite_query         │
           ↓              │
       retrieve           │
           ↓              │
   grade_documents        │
           ↓              │
    [Relevant?] ─NO→ increment_retries
      YES  ↓              ↓
   rerank_documents  [Retries<Max?]
           ↓          YES↓     NO↓
   generate_answer ←─────┘  web_search
           ↓                    ↓
   grounding_check         (rejoin)
           ↓                    ↓
    [Grounded?] ────────────────┘
      YES  ↓  NO
      END  → [Attempts<Max?]
                YES↓     NO↓
           regenerate   END
```

---

## Storage Architecture

### Vector Store (ChromaDB)

**Location:** `{persist_directory_base}/core_rag_engine_chroma/{collection_name}/`

**Collections:**
- Each document set stored in separate collection
- Persistent storage on disk
- In-memory caching for performance

**Retrieval Methods:**
- **Standard:** Pure vector similarity search
- **Hybrid:** Vector + BM25 keyword search (alpha-weighted)

---

## LLM Integration

### Factory Pattern

```python
# LLMFactory
llm = llm_factory.create_llm(use_json_format=False)
json_llm = llm_factory.create_llm(use_json_format=True)

# EmbeddingFactory
embeddings = embedding_factory.create_embedding_model()

# TextSplitterFactory
text_splitter = text_splitter_factory.create_text_splitter()
```

### Supported Providers

| Provider | Models | Use Cases |
|----------|--------|-----------|
| **OpenAI** | GPT-4, GPT-4o, GPT-4o-mini | Production, high quality |
| **Google** | Gemini Pro, Gemini 1.5 Pro | Cost-effective, multimodal |
| **Ollama** | Llama 2, Mistral, etc. | Local deployment, privacy |

---

## Caching Strategy

### Multi-Level Caching

1. **Document Cache** (CacheOrchestrator)
   - TTL: 5 minutes (configurable)
   - Max Size: 500 MB (configurable)
   - LRU eviction policy

2. **Vector Store Cache**
   - Persistent ChromaDB storage
   - In-memory index caching

3. **LLM Response Cache**
   - Framework-level caching (LangChain)

### Cache Invalidation

- Automatic invalidation on document ingestion
- Manual invalidation via `clear_document_cache()`
- TTL-based expiration
- Memory-based eviction

---

## MCP Integration (Optional)

**Feature Flag:** `MCP_ENABLE_MCP` (default: `false`)

### MCP-Enhanced Features

When enabled (`MCP_ENABLE_MCP=true`):
- Filesystem monitoring for auto-ingestion
- Conversation memory across sessions
- SQL database query capabilities
- Enhanced monitoring and observability

### Graceful Fallback

If MCP initialization fails:
- Automatically falls back to standard RAG
- No interruption to user experience
- Warning logged for troubleshooting

---

## Error Handling

### ErrorHandler Module

```python
class ErrorHandler:
    def handle_error(
        self,
        error: Exception,
        context: str
    ) -> Dict[str, Any]:
        """Centralized error handling with logging."""
```

### Error Categories

- **Ingestion Errors**: Document loading/splitting failures
- **Retrieval Errors**: Vector store query failures
- **Generation Errors**: LLM API failures
- **Validation Errors**: Configuration/input validation

### Recovery Strategies

- Automatic retries with exponential backoff
- Fallback to web search on retrieval failure
- Graceful degradation on LLM failures
- Comprehensive error logging

---

## Performance Considerations

### Optimizations

1. **Caching**
   - Multi-level caching reduces redundant operations
   - TTL-based expiration balances freshness and performance

2. **Streaming**
   - Memory-efficient document loading for large collections
   - Batch processing with configurable batch sizes

3. **Parallel Processing**
   - Concurrent document grading
   - Async workflow execution

4. **Hybrid Search**
   - Optimized for collections up to 50,000 documents
   - Automatically warned for larger collections

### Benchmarks

| Operation | Cold Start | Warm Cache |
|-----------|------------|------------|
| Document Ingestion (1000 docs) | 45s | N/A |
| Query Processing | 3-5s | 0.5-1s |
| Cache Hit | N/A | 50ms |

---

## Security

### API Key Management

- Environment variable-based configuration
- No hardcoded secrets
- Support for `.env` files

### Input Validation

- Pydantic models for type safety
- SQL injection prevention
- Path traversal protection

### Data Privacy

- Local vector store storage
- No data sent to third parties (except LLM APIs)
- Optional local LLM deployment (Ollama)

---

## Migration from v1.0 to v2.0

### Breaking Changes

**None** - 100% backward compatible!

### New Features

- Modular architecture with 7 specialized modules
- 184 comprehensive unit tests
- MCP integration (optional)
- Improved error handling
- Better performance (caching optimizations)

### Deprecated

**None** - All existing APIs preserved

---

## References

- **LangChain Documentation**: https://python.langchain.com/
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **ChromaDB Documentation**: https://docs.trychroma.com/
- **Pydantic Documentation**: https://docs.pydantic.dev/

---

**Last Updated:** 2025-12-30
**Version:** 2.0
**Maintained By:** Adaptive RAG Team
