# Adaptive RAG Engine - Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [LangGraph Workflow](#langgraph-workflow)
5. [Storage Architecture](#storage-architecture)
6. [LLM Integration](#llm-integration)
7. [Caching Strategy](#caching-strategy)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Security](#security)

---

## System Overview

The Adaptive RAG Engine is built on a modular, extensible architecture that separates concerns while maintaining high cohesion within each component.

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
│  │              CoreRAGEngine (Orchestrator)                    │   │
│  │  • Query Processing   • Workflow Management                  │   │
│  │  • State Management   • Error Handling                       │   │
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
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new LLM providers, data sources, or processing strategies
3. **Type Safety**: Full type hints and runtime validation with Pydantic
4. **Observability**: Comprehensive logging and error tracking
5. **Performance**: Intelligent caching and async processing

---

## Core Components

### 1. CoreRAGEngine (`src/core_rag_engine.py`)

The central orchestrator that coordinates all RAG operations.

**Responsibilities:**
- Workflow orchestration using LangGraph
- Query analysis and optimization
- Document retrieval and grading
- Answer generation and validation
- State management

**Key Methods:**
```python
class CoreRAGEngine:
    def __init__(self, **config):
        """Initialize with configuration."""
        
    async def run_full_rag_workflow(self, question: str, **kwargs) -> dict:
        """Execute the complete RAG pipeline."""
        
    def ingest(self, **sources) -> None:
        """Ingest documents into vector store."""
        
    def _retrieve_node(self, state: dict) -> dict:
        """Retrieve relevant documents."""
        
    async def _grounding_check_node(self, state: dict) -> dict:
        """Validate answer grounding."""
```

### 2. Configuration Management (`src/config.py`)

Pydantic-based configuration with validation and type safety.

**Configuration Models:**
```python
class LLMSettings(BaseSettings):
    provider: str
    model_name: str
    temperature: float
    max_tokens: int

class EmbeddingSettings(BaseSettings):
    provider: str
    model_name: str
    dimensions: Optional[int]

class EngineSettings(BaseSettings):
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    enable_hybrid_search: bool
    max_grounding_attempts: int
```

### 3. Context Manager (`src/context_manager.py`)

Manages context window limits with intelligent truncation.

**Truncation Strategies:**
- **Smart**: Prioritizes beginning and end
- **Balanced**: Evenly distributes across context
- **Sliding**: Maintains temporal continuity

```python
class ContextManager:
    def truncate_context(
        self,
        documents: List[Document],
        strategy: str = "smart",
        max_tokens: int = 4000
    ) -> str:
        """Intelligently truncate context to fit token limit."""
```

### 4. Agent Loop (`src/agent_loop.py`)

Orchestrates multi-step agentic workflows for complex tasks.

```python
class AgentLoop:
    async def plan_and_execute(
        self,
        task: str,
        tools: List[Tool]
    ) -> AgentResult:
        """Plan and execute multi-step tasks."""
```

### 5. Data Feeds

#### Stock Feed (`src/stock_feed.py`)
Fetches financial data for analysis.

```python
def fetch_stock_news_as_documents(
    tickers: List[str],
    days_back: int = 7
) -> List[Document]:
    """Fetch stock news as documents."""
```

#### Scraper Feed (`src/scraper_feed.py`)
Web scraping for document ingestion.

```python
def scrape_urls_as_documents(
    urls: List[str]
) -> List[Document]:
    """Scrape URLs and convert to documents."""
```

---

## Data Flow

### Document Ingestion Flow

```
User Input (Files/URLs/Text)
    │
    ▼
Document Loading
    │
    ├─► Format Detection
    │   (PDF, TXT, MD, HTML, etc.)
    │
    ▼
Adaptive Chunking
    │
    ├─► Code Documents → Syntax-aware chunking
    ├─► Academic Papers → Section-based chunking
    ├─► Financial Reports → Structure-preserving chunking
    └─► Default → Fixed-size chunking
    │
    ▼
Embedding Generation
    │
    ├─► OpenAI embeddings
    ├─► Google embeddings
    └─► Ollama embeddings
    │
    ▼
Vector Store Indexing (ChromaDB)
    │
    ├─► Collection Management
    ├─► Metadata Storage
    └─► Vector Indexing
    │
    ▼
Cache Population
```

### Query Processing Flow

```
User Query
    │
    ▼
Query Analysis
    │
    ├─► Query Type Classification
    │   (Factual, Analytical, Exploratory)
    ├─► Intent Extraction
    ├─► Keyword Extraction
    └─► Ambiguity Detection
    │
    ▼
Query Optimization
    │
    ├─► Query Rewriting (if needed)
    └─► Query Expansion
    │
    ▼
Document Retrieval
    │
    ├─► Vector Search
    │   (Cosine similarity)
    ├─► Hybrid Search (optional)
    │   (Vector + Keyword)
    └─► Fallback to Web Search
    │
    ▼
Document Grading
    │
    ├─► Relevance Scoring
    ├─► Document Filtering
    └─► Reranking
    │
    ▼
Context Construction
    │
    ├─► Context Truncation
    ├─► Context Formatting
    └─► Metadata Inclusion
    │
    ▼
Answer Generation
    │
    ├─► LLM Invocation
    ├─► Streaming Response
    └─► Citation Extraction
    │
    ▼
Grounding Check
    │
    ├─► Answer Validation
    ├─► Hallucination Detection
    └─► Iterative Refinement
    │
    ▼
Final Answer + Metadata
```

---

## LangGraph Workflow

The RAG pipeline is implemented as a LangGraph state machine.

### State Schema

```python
class RAGState(TypedDict):
    question: str                    # User's question
    original_question: str           # Original unmodified question
    query_analysis_results: Optional[QueryAnalysis]
    documents: List[Document]        # Retrieved documents
    context: str                     # Formatted context
    web_search_results: Optional[List[dict]]
    generation: str                  # Generated answer
    retries: int                     # Query rewrite attempts
    run_web_search: str             # "Yes" or "No"
    relevance_check_passed: Optional[bool]
    error_message: Optional[str]
    grounding_check_attempts: int
    regeneration_feedback: Optional[str]
    collection_name: Optional[str]
    chat_history: List[dict]
```

### Workflow Graph

```
START
  │
  ▼
analyze_query ──────► rewrite_query? ◄──┐
  │                        │             │
  │                        ▼             │
  │                  max retries?        │
  │                        │             │
  │                   Yes  │  No         │
  │                        │  │          │
  │                        │  └──────────┘
  │                        ▼
  ▼                      ERROR
retrieve_documents
  │
  ▼
grade_documents ────► web_search? ──┐
  │                        │         │
  ▼                       Yes        │
rerank_documents           │         │
  │                        ▼         │
  ▼                   web_search     │
generate_answer            │         │
  │                        │         │
  └────────────────────────┘         │
  │                                  │
  ▼                                  │
grounding_check ──► regenerate? ────┘
  │                      │
  No                    Yes (< max attempts)
  │
  ▼
 END
```

### Node Implementations

**Query Analysis Node:**
```python
def _analyze_query_node(self, state: dict) -> dict:
    """Analyze query to determine retrieval strategy."""
    analysis = self.query_analyzer_chain.invoke({
        "question": state["question"]
    })
    state["query_analysis_results"] = analysis
    return state
```

**Retrieve Node:**
```python
def _retrieve_node(self, state: dict) -> dict:
    """Retrieve relevant documents from vector store."""
    retriever = self._get_vector_store(collection_name).as_retriever(
        search_kwargs={"k": self.default_retrieval_top_k}
    )
    documents = retriever.get_relevant_documents(state["question"])
    state["documents"] = documents
    return state
```

**Grounding Check Node:**
```python
async def _grounding_check_node(self, state: dict) -> dict:
    """Validate answer against source documents."""
    check_result = self.grounding_check_chain.invoke({
        "question": state["question"],
        "answer": state["generation"],
        "context": state["context"]
    })
    
    if not check_result.is_grounded:
        state["regeneration_feedback"] = self._format_feedback(check_result)
    
    state["grounding_check_attempts"] += 1
    return state
```

---

## Storage Architecture

### Vector Store (ChromaDB)

**Structure:**
```
chroma_db/
├── collection_1/
│   ├── chroma.sqlite3      # SQLite metadata
│   ├── index/              # Vector indices
│   └── data/               # Document data
├── collection_2/
│   └── ...
```

**Collection Management:**
```python
def _get_vector_store(
    self,
    collection_name: str,
    recreate: bool = False
) -> Chroma:
    """Get or create vector store for collection."""
    
    if recreate:
        # Delete existing collection
        self._delete_collection(collection_name)
    
    # Create new vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=self.embedding_model,
        persist_directory=persist_dir
    )
    
    return vector_store
```

### Document Schema

```python
{
    "page_content": "The actual text content",
    "metadata": {
        "source": "document_name.pdf",
        "page": 5,
        "chunk_index": 3,
        "document_type": "academic",
        "created_at": "2025-01-15T10:30:00Z",
        "file_hash": "abc123...",
        # Custom metadata fields
    }
}
```

---

## LLM Integration

### Multi-Provider Architecture

```python
class LLMFactory:
    @staticmethod
    def create_llm(provider: str, **config) -> BaseChatModel:
        """Factory method to create LLM instance."""
        if provider == "openai":
            return ChatOpenAI(**config)
        elif provider == "google":
            return ChatGoogleGenerativeAI(**config)
        elif provider == "ollama":
            return ChatOllama(**config)
```

### Provider-Specific Configuration

**OpenAI:**
```python
ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    streaming=True,
    api_key=openai_api_key
)
```

**Google Gemini:**
```python
ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    max_output_tokens=2000,
    google_api_key=google_api_key
)
```

**Ollama (Local):**
```python
ChatOllama(
    model="llama2",
    temperature=0.7,
    base_url="http://localhost:11434"
)
```

### Prompt Engineering

**Query Analysis Prompt:**
```python
QUERY_ANALYSIS_PROMPT = """
Analyze the following user query and classify it:

Query: {question}

Determine:
1. Query Type: factual_lookup, analytical, exploratory, conversational
2. Main Intent: What is the user trying to accomplish?
3. Key Entities: Extract important entities
4. Ambiguity: Is the query ambiguous?

Respond in JSON format.
"""
```

**Answer Generation Prompt:**
```python
ANSWER_GENERATION_PROMPT = """
You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the context provided
- If the context doesn't contain enough information, say so
- Cite specific parts of the context when possible
- Be concise but comprehensive

Answer:
"""
```

---

## Caching Strategy

### Multi-Level Caching

```
┌─────────────────────────────────────────────┐
│         Application Cache (In-Memory)        │
│  • Document Cache                            │
│  • Vector Store Cache                        │
│  • Query Results Cache                       │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Vector Store Cache (Disk)            │
│  • Pre-computed embeddings                   │
│  • Index cache                               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      LLM Response Cache (Optional)           │
│  • Redis/Memcached                           │
└──────────────────────────────────────────────┘
```

### Cache Implementation

```python
class CacheManager:
    def __init__(self, ttl: int, max_size_mb: float):
        self.document_cache: Dict[str, List[Document]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.ttl = ttl
        self.max_size_mb = max_size_mb
    
    def get(self, collection_name: str) -> Optional[List[Document]]:
        """Get documents from cache if valid."""
        if self._is_valid(collection_name):
            return self.document_cache.get(collection_name)
        return None
    
    def set(self, collection_name: str, documents: List[Document]):
        """Cache documents with TTL."""
        self.document_cache[collection_name] = documents
        self.cache_timestamps[collection_name] = time.time()
        self._maintain_cache()
    
    def invalidate(self, collection_name: str):
        """Invalidate specific cache entry."""
        self.document_cache.pop(collection_name, None)
        self.cache_timestamps.pop(collection_name, None)
```

### Cache Eviction Policy

**LRU (Least Recently Used):**
```python
def _evict_lru(self):
    """Evict least recently used cache entries."""
    sorted_items = sorted(
        self.cache_timestamps.items(),
        key=lambda x: x[1]
    )
    
    for collection_name, _ in sorted_items:
        if self._get_cache_size() <= self.max_size_mb:
            break
        self.invalidate(collection_name)
```

---

## Error Handling

### Error Hierarchy

```
RAGException (Base)
│
├── ConfigurationError
│   ├── MissingAPIKeyError
│   └── InvalidConfigError
│
├── IngestionError
│   ├── DocumentLoadError
│   ├── ChunkingError
│   └── EmbeddingError
│
├── RetrievalError
│   ├── VectorStoreError
│   └── SearchError
│
└── GenerationError
    ├── LLMError
    ├── GroundingError
    └── TimeoutError
```

### Error Handling Strategy

```python
def _handle_node_error(
    self,
    state: dict,
    error: Exception,
    node_name: str
) -> dict:
    """Centralized error handling for workflow nodes."""
    
    self.logger.error(
        f"Error in {node_name}: {str(error)}",
        exc_info=True
    )
    
    # Append to error message
    self._append_error(state, f"{node_name}: {str(error)}")
    
    # Decide whether to continue or stop
    if isinstance(error, CriticalError):
        state["stop_workflow"] = True
    
    return state
```

### Graceful Degradation

```python
def _retrieve_with_fallback(self, state: dict) -> dict:
    """Retrieve with web search fallback."""
    try:
        # Try vector store retrieval
        documents = self._retrieve_from_vector_store(state)
        
        if not documents or len(documents) < 2:
            # Fallback to web search
            self.logger.info("Falling back to web search")
            state["run_web_search"] = "Yes"
            
    except Exception as e:
        self.logger.error(f"Retrieval error: {e}")
        state["run_web_search"] = "Yes"
    
    return state
```

---

## Performance Considerations

### Optimization Techniques

1. **Batch Processing**
```python
def ingest_batch(
    self,
    documents: List[Document],
    batch_size: int = 100
):
    """Process documents in batches."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        self._process_batch(batch)
```

2. **Async Operations**
```python
async def process_multiple_queries(
    self,
    queries: List[str]
) -> List[dict]:
    """Process multiple queries concurrently."""
    tasks = [
        self.run_full_rag_workflow(query)
        for query in queries
    ]
    return await asyncio.gather(*tasks)
```

3. **Streaming Responses**
```python
def stream_answer(
    self,
    question: str
) -> Generator[str, None, None]:
    """Stream answer token by token."""
    for chunk in self.llm.stream(prompt):
        yield chunk.content
```

### Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Query Latency (p50) | < 2s | 1.8s |
| Query Latency (p95) | < 5s | 4.2s |
| Throughput | > 10 qps | 12 qps |
| Cache Hit Rate | > 70% | 78% |
| Memory Usage | < 2GB | 1.6GB |

---

## Security

### API Key Management

```python
# NEVER hardcode API keys
# Use environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Validate before use
if not api_key:
    raise ConfigurationError("OPENAI_API_KEY not found")

# Mask in logs
self.logger.info(f"Using API key: {api_key[:8]}...")
```

### Input Validation

```python
def validate_query(self, query: str) -> str:
    """Validate and sanitize user query."""
    # Check length
    if len(query) > 1000:
        raise ValueError("Query too long")
    
    # Remove malicious patterns
    query = self._sanitize_input(query)
    
    return query
```

### SQL Injection Prevention

```python
def sanitize_identifier(name: str) -> str:
    """Sanitize SQL identifiers."""
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        raise ValueError("Invalid identifier")
    return name
```

---

## Future Enhancements

### Planned Improvements

1. **Multi-Modal Support**
   - Image understanding
   - Audio transcription
   - Video analysis

2. **Graph-Based Retrieval**
   - Knowledge graph integration
   - Entity relationship mapping
   - Graph neural networks

3. **Advanced Agentic Workflows**
   - Multi-agent collaboration
   - Tool use and function calling
   - Self-improvement loops

4. **Distributed Architecture**
   - Horizontal scaling
   - Load balancing
   - Distributed caching

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Research Papers](https://arxiv.org/abs/2005.11401)

---

For questions or suggestions about the architecture, please open an issue on GitHub.