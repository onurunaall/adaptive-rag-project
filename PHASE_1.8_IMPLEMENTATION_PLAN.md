# Phase 1.8 Implementation Plan: CoreRAGEngine Facade Refactoring

**Status**: Planning
**Date**: 2025-12-30
**Goal**: Reduce CoreRAGEngine from 2,976 lines to ~300-400 lines by converting it to a facade pattern

---

## Executive Summary

CoreRAGEngine currently contains 2,976 lines of code with many responsibilities. We have successfully extracted 7 specialized modules (2,315 lines total). Phase 1.8 will refactor CoreRAGEngine to become a lightweight facade that delegates to these modules while maintaining 100% backward compatibility.

**Key Metrics:**
- **Current**: 2,976 lines (god class anti-pattern)
- **Target**: ~300-400 lines (facade pattern)
- **Code Reduction**: ~2,600 lines (~87% reduction)
- **Modules to Integrate**: 7 specialized modules
- **Backward Compatibility**: 100% (all public APIs preserved)

---

## Phase 1.8 Modules Already Extracted

| Module | Lines | Responsibility | Tests |
|--------|-------|----------------|-------|
| DocumentManager | 342 | Document loading & splitting | 14 |
| VectorStoreManager | 449 | Vector store management | 30 |
| QueryProcessor | 228 | Query analysis & rewriting | 20 |
| DocumentGrader | 252 | Document relevance grading | 27 |
| AnswerGenerator | 383 | Answer generation & grounding | 30 |
| CacheOrchestrator | 326 | Cache management coordination | 35 |
| WorkflowOrchestrator | 335 | LangGraph workflow orchestration | 28 |
| **Total** | **2,315** | - | **184** |

---

## Current CoreRAGEngine Structure Analysis

### What CoreRAGEngine Currently Does (2,976 lines)

1. **Initialization** (~250 lines)
   - Configure all settings
   - Create LLM, embeddings, text splitters
   - Create all chains (grading, reranking, query analysis, answer generation, grounding)
   - Initialize vector stores and retrievers
   - Set up document cache
   - Compile workflow graph

2. **Document Management** (~200 lines) - **EXTRACTED to DocumentManager**
   - `load_documents()` - Load from URLs, PDFs, text files
   - `split_documents()` - Split into chunks with various strategies

3. **Vector Store Management** (~400 lines) - **EXTRACTED to VectorStoreManager**
   - `_init_or_load_vectorstore()` - Initialize or load vector stores
   - `_setup_retriever_for_collection()` - Set up retrievers
   - `_get_persist_dir()` - Get persistence directory
   - `_handle_recreate_collection()` - Handle collection recreation
   - `index_documents()` - Index documents into collections
   - `list_collections()` - List all collections
   - `delete_collection()` - Delete a collection

4. **Query Processing** (~200 lines) - **EXTRACTED to QueryProcessor**
   - `_create_query_analyzer_chain()` - Create query analysis chain
   - `_create_query_rewriter_chain()` - Create query rewriting chain
   - Query analysis and rewriting logic

5. **Document Grading** (~250 lines) - **EXTRACTED to DocumentGrader**
   - `_create_document_relevance_grader_chain()` - Create grading chain
   - `_create_document_reranker_chain()` - Create reranking chain
   - Document grading and reranking logic

6. **Answer Generation** (~400 lines) - **EXTRACTED to AnswerGenerator**
   - `_create_answer_generation_chain()` - Create answer generation chain
   - `_create_grounding_check_chain()` - Create grounding check chain
   - Answer generation and grounding validation logic

7. **Cache Management** (~200 lines) - **EXTRACTED to CacheOrchestrator**
   - `_maintain_cache()` - Cache maintenance
   - `clear_document_cache()` - Clear cache
   - `_invalidate_collection_cache()` - Invalidate specific cache
   - `get_cache_stats()` - Get cache statistics
   - Cache management methods

8. **Workflow Orchestration** (~500 lines) - **EXTRACTED to WorkflowOrchestrator**
   - `_compile_rag_workflow()` - Compile LangGraph workflow
   - Node methods (`_analyze_query_node`, `_retrieve_node`, etc.)
   - Routing methods (`_route_after_grading`, `_route_after_grounding_check`)
   - Workflow execution methods

9. **Public API Methods** (~300 lines) - **KEEP & REFACTOR**
   - `ingest()` - Public ingestion API
   - `answer_query()` - Public query API
   - `run_full_rag_workflow()` - Async workflow execution
   - `run_full_rag_workflow_sync()` - Sync workflow execution

10. **Helper/Utility Methods** (~200 lines) - **KEEP or DELEGATE**
    - `_setup_logger()` - Logger setup
    - `_append_error()` - Error handling
    - `_format_chat_history()` - Chat history formatting
    - Various helper methods

---

## Detailed Refactoring Plan

### Step 1: Update Imports (Add Extracted Modules)

**Current imports:**
```python
from src.rag import (
    GroundingCheck,
    RelevanceGrade,
    RerankScore,
    QueryAnalysis,
    CoreGraphState,
    LLMFactory,
    EmbeddingFactory,
    TextSplitterFactory,
    ChainFactory,
    ErrorHandler,
)
```

**New imports:**
```python
from src.rag import (
    # Models
    GroundingCheck,
    RelevanceGrade,
    RerankScore,
    QueryAnalysis,
    CoreGraphState,
    # Factories
    LLMFactory,
    EmbeddingFactory,
    TextSplitterFactory,
    ChainFactory,
    # Managers
    DocumentManager,
    VectorStoreManager,
    # Processors
    QueryProcessor,
    # Graders
    DocumentGrader,
    # Generators
    AnswerGenerator,
    # Orchestrators
    CacheOrchestrator,
    WorkflowOrchestrator,
    # Utilities
    ErrorHandler,
    CacheManager,
)
```

### Step 2: Refactor `__init__()` Method

**Current responsibilities:**
- Configure settings (KEEP)
- Create LLM, embeddings, text splitters via factories (KEEP)
- Create all chains via ChainFactory (REMOVE - delegate to modules)
- Initialize vector stores dict (REMOVE - delegate to VectorStoreManager)
- Initialize document cache (REMOVE - delegate to CacheOrchestrator)
- Compile workflow (REMOVE - delegate to WorkflowOrchestrator)

**New initialization approach:**

```python
def __init__(self, ...):
    # 1. Configure settings (same as before)
    self.llm_provider = llm_provider or app_settings.llm.llm_provider
    # ... all configuration ...

    # 2. Setup logger (same as before)
    self._setup_logger()

    # 3. Initialize factories (same as before)
    self.llm_factory = LLMFactory(...)
    self.llm = self.llm_factory.create_llm(use_json_format=False)
    self.json_llm = self.llm_factory.create_llm(use_json_format=True)

    self.embedding_factory = EmbeddingFactory(...)
    self.embedding_model = self.embedding_factory.create_embedding_model()

    # 4. Initialize NEW specialized modules
    self.document_manager = DocumentManager(
        text_splitter_factory=self.text_splitter_factory,
        logger=self.logger,
    )

    self.cache_orchestrator = CacheOrchestrator(
        cache_ttl=300,
        max_cache_size_mb=500.0,
        logger=self.logger,
    )

    self.vector_store_manager = VectorStoreManager(
        embedding_model=self.embedding_model,
        persist_directory_base=self.persist_directory_base,
        default_retrieval_top_k=self.default_retrieval_top_k,
        enable_hybrid_search=self.enable_hybrid_search,
        get_all_documents_callback=self._get_all_documents_from_collection,
        stream_documents_callback=self._stream_documents_from_collection,
        invalidate_cache_callback=self.cache_orchestrator.invalidate_collection_cache,
        logger=self.logger,
    )

    self.query_processor = QueryProcessor(
        llm=self.llm,
        json_llm=self.json_llm,
        logger=self.logger,
    )

    self.document_grader = DocumentGrader(
        json_llm=self.json_llm,
        logger=self.logger,
    )

    self.answer_generator = AnswerGenerator(
        llm=self.llm,
        json_llm=self.json_llm,
        logger=self.logger,
    )

    # 5. Initialize workflow orchestrator with node functions
    self.workflow_orchestrator = WorkflowOrchestrator(
        node_functions=self._create_node_functions(),
        routing_functions=self._create_routing_functions(),
        logger=self.logger,
    )
    self.rag_workflow = self.workflow_orchestrator.compile_default_rag_workflow()

    # 6. Initialize error handler and other utilities
    self.error_handler = ErrorHandler(logger=self.logger)
    self.context_manager = ContextManager(...)

    # 7. Advanced grounding (optional)
    if self.enable_advanced_grounding:
        self.advanced_grounding_checker = MultiLevelGroundingChecker(self.llm)
```

**Estimated line reduction: 250 lines → 150 lines**

### Step 3: Remove Extracted Chain Creation Methods

**Methods to REMOVE (now in extracted modules):**
- `_create_document_relevance_grader_chain()` → DocumentGrader
- `_create_document_reranker_chain()` → DocumentGrader
- `_create_query_analyzer_chain()` → QueryProcessor
- `_create_query_rewriter_chain()` → QueryProcessor
- `_create_answer_generation_chain()` → AnswerGenerator
- `_create_grounding_check_chain()` → AnswerGenerator

**Lines to remove: ~300 lines**

### Step 4: Remove Extracted Vector Store Methods

**Methods to REMOVE (now in VectorStoreManager):**
- `_get_persist_dir()` → VectorStoreManager.get_persist_dir()
- `_init_or_load_vectorstore()` → VectorStoreManager.init_or_load_vectorstore()
- `_setup_retriever_for_collection()` → VectorStoreManager.setup_retriever_for_collection()
- `_handle_recreate_collection()` → VectorStoreManager._handle_recreate_collection()
- `index_documents()` → VectorStoreManager.index_documents()
- `delete_collection()` → VectorStoreManager.delete_collection()
- `list_collections()` → VectorStoreManager.list_collections()

**Lines to remove: ~400 lines**

### Step 5: Remove Extracted Document Management Methods

**Methods to REMOVE (now in DocumentManager):**
- `load_documents()` → DocumentManager.load_documents()
- `split_documents()` → DocumentManager.split_documents()

**Lines to remove: ~200 lines**

### Step 6: Remove Extracted Cache Management Methods

**Methods to REMOVE (now in CacheOrchestrator):**
- `_maintain_cache()` → CacheOrchestrator.maintain_cache()
- `clear_document_cache()` → CacheOrchestrator.clear_document_cache()
- `_invalidate_collection_cache()` → CacheOrchestrator.invalidate_collection_cache()
- `get_cache_stats()` → CacheOrchestrator.get_cache_stats()
- `invalidate_all_caches()` → CacheOrchestrator.invalidate_all_caches()
- `set_cache_ttl()` → CacheOrchestrator.set_cache_ttl()

**Lines to remove: ~200 lines**

### Step 7: Refactor Workflow Nodes to Use Extracted Modules

**Current node methods** (500+ lines total):
- `_analyze_query_node()` - Delegates to QueryProcessor
- `_rewrite_query_node()` - Delegates to QueryProcessor
- `_retrieve_node()` - Delegates to VectorStoreManager
- `_grade_documents_node()` - Delegates to DocumentGrader
- `_rerank_documents_node()` - Delegates to DocumentGrader
- `_generate_answer_node()` - Delegates to AnswerGenerator
- `_grounding_check_node()` - Delegates to AnswerGenerator
- `_web_search_node()` - Keep (uses search_tool)
- `_increment_retries_node()` - Keep (simple state update)

**Refactoring approach:**

Create lightweight node wrapper methods that delegate to modules:

```python
def _create_node_functions(self) -> Dict[str, Callable]:
    """Create node functions for workflow orchestrator."""
    return {
        "analyze_query": self._analyze_query_node,
        "rewrite_query": self._rewrite_query_node,
        "retrieve": self._retrieve_node,
        "grade_documents": self._grade_documents_node,
        "rerank_documents": self._rerank_documents_node,
        "generate_answer": self._generate_answer_node,
        "grounding_check": self._grounding_check_node,
        "web_search": self._web_search_node,
        "increment_retries": self._increment_retries_node,
    }

def _analyze_query_node(self, state: CoreGraphState) -> CoreGraphState:
    """Analyze query node - delegates to QueryProcessor."""
    self.logger.info("NODE: Analyzing query")

    question = state["question"]
    chat_history = state.get("chat_history", [])

    query_analysis = self.query_processor.analyze_query(question, chat_history)

    if query_analysis:
        state["query_analysis"] = query_analysis

    return state

def _rewrite_query_node(self, state: CoreGraphState) -> CoreGraphState:
    """Rewrite query node - delegates to QueryProcessor."""
    self.logger.info("NODE: Rewriting query")

    question = state["question"]
    chat_history = state.get("chat_history", [])

    rewritten = self.query_processor.rewrite_query(question, chat_history)
    state["question"] = rewritten

    return state

# ... similar lightweight wrappers for other nodes ...
```

**Lines: 500 lines → ~200 lines (delegation wrappers)**

### Step 8: Remove Workflow Compilation Method

**Method to REMOVE:**
- `_compile_rag_workflow()` → WorkflowOrchestrator.compile_default_rag_workflow()

**Lines to remove: ~50 lines**

### Step 9: Refactor Public API Methods to Delegate

**Public methods to REFACTOR (not remove):**

```python
def ingest(self, source_type: str, source_value: Any, collection_name: Optional[str] = None, ...) -> Dict:
    """Public ingestion API - delegates to modules."""
    try:
        # 1. Load documents via DocumentManager
        documents = self.document_manager.load_documents(source_type, source_value)

        # 2. Split documents via DocumentManager
        split_docs = self.document_manager.split_documents(documents)

        # 3. Index via VectorStoreManager
        self.vector_store_manager.index_documents(
            split_docs,
            collection_name or self.default_collection_name,
            recreate=recreate
        )

        return {"status": "success", "documents_ingested": len(split_docs)}
    except Exception as e:
        return self.error_handler.handle_error(e, "ingestion")

def answer_query(self, question: str, collection_name: Optional[str] = None, ...) -> Dict:
    """Public query API - delegates to workflow."""
    try:
        # Build initial state
        initial_state = CoreGraphState(
            question=question,
            original_question=question,
            collection_name=collection_name or self.default_collection_name,
            ...
        )

        # Execute workflow via WorkflowOrchestrator
        final_state = self.workflow_orchestrator.invoke_workflow(initial_state)

        return self._format_response(final_state)
    except Exception as e:
        return self.error_handler.handle_error(e, "query")
```

**Lines: ~300 lines → ~150 lines (simplified delegation)**

### Step 10: Keep Essential Helper Methods

**Methods to KEEP (needed by facade):**
- `_setup_logger()` - Logger setup
- `_append_error()` - Error handling helper
- `_format_chat_history()` - Formatting helper
- `_get_all_documents_from_collection()` - Callback for VectorStoreManager
- `_stream_documents_from_collection()` - Callback for VectorStoreManager
- `_format_response()` - Response formatting

**Estimated lines: ~150 lines**

---

## New CoreRAGEngine Structure (Target: ~350 lines)

```
CoreRAGEngine (facade pattern)
├── __init__() [~150 lines]
│   ├── Configuration
│   ├── Factory initialization
│   └── Module initialization (7 modules)
│
├── Public API Methods [~150 lines]
│   ├── ingest() - Delegates to DocumentManager + VectorStoreManager
│   ├── answer_query() - Delegates to WorkflowOrchestrator
│   ├── run_full_rag_workflow() - Delegates to WorkflowOrchestrator
│   └── run_full_rag_workflow_sync() - Delegates to WorkflowOrchestrator
│
├── Node Wrapper Methods [~150 lines]
│   ├── _create_node_functions() - Creates node function dict
│   ├── _create_routing_functions() - Creates routing function dict
│   ├── _analyze_query_node() - Delegates to QueryProcessor
│   ├── _rewrite_query_node() - Delegates to QueryProcessor
│   ├── _retrieve_node() - Delegates to VectorStoreManager
│   ├── _grade_documents_node() - Delegates to DocumentGrader
│   ├── _rerank_documents_node() - Delegates to DocumentGrader
│   ├── _generate_answer_node() - Delegates to AnswerGenerator
│   ├── _grounding_check_node() - Delegates to AnswerGenerator
│   ├── _route_after_grading() - Routing logic
│   └── _route_after_grounding_check() - Routing logic
│
└── Helper Methods [~100 lines]
    ├── _setup_logger()
    ├── _append_error()
    ├── _format_chat_history()
    ├── _get_all_documents_from_collection()
    ├── _stream_documents_from_collection()
    └── _format_response()
```

**Total estimated lines: ~350-400 lines** (down from 2,976)

---

## Backward Compatibility Strategy

### Public API Preservation

**All public methods MUST remain unchanged:**
- `ingest()`
- `answer_query()`
- `run_full_rag_workflow()`
- `run_full_rag_workflow_sync()`
- `list_collections()`
- `delete_collection()`
- `get_cache_stats()`
- `invalidate_all_caches()`
- `set_cache_ttl()`
- `clear_document_cache()`

### Internal API Delegation

**Private methods that external code might call:**
- `load_documents()` → Delegate to DocumentManager
- `split_documents()` → Delegate to DocumentManager
- `index_documents()` → Delegate to VectorStoreManager

### Property Access Preservation

**Properties/attributes that external code might access:**
- `self.vectorstores` → Delegate to VectorStoreManager
- `self.retrievers` → Delegate to VectorStoreManager
- `self.document_cache` → Delegate to CacheOrchestrator
- `self.rag_workflow` → Delegate to WorkflowOrchestrator

**Implementation approach:**
```python
@property
def vectorstores(self):
    """Backward compatibility: access to vector stores."""
    return self.vector_store_manager.vectorstores

@property
def retrievers(self):
    """Backward compatibility: access to retrievers."""
    return self.vector_store_manager.retrievers

@property
def document_cache(self):
    """Backward compatibility: access to document cache."""
    return self.cache_orchestrator.document_cache
```

---

## Testing Strategy

### 1. Existing Tests Must Pass

**Critical requirement**: All existing tests for CoreRAGEngine MUST pass without modification.

**Test files to verify:**
- `tests/test_core_rag_engine.py`
- `tests/test_agent_loop.py`
- Any integration tests

### 2. New Module Integration Tests

**Create tests for module integration:**
- Test that CoreRAGEngine correctly delegates to each module
- Test that callbacks work correctly (cache invalidation, document streaming)
- Test that workflow compilation uses WorkflowOrchestrator correctly

### 3. Backward Compatibility Tests

**Test deprecated access patterns:**
- Test direct access to `self.vectorstores`
- Test direct access to `self.retrievers`
- Test direct access to `self.document_cache`

### 4. Performance Tests

**Ensure no performance regression:**
- Measure ingestion time before/after
- Measure query time before/after
- Measure memory usage before/after

---

## Implementation Checklist

### Phase 1.8.1: Preparation
- [ ] Create backup branch
- [ ] Run all existing tests (baseline)
- [ ] Document current test coverage
- [ ] Create integration test suite

### Phase 1.8.2: Module Integration
- [ ] Update imports to include all 7 modules
- [ ] Refactor `__init__()` to instantiate modules
- [ ] Create property accessors for backward compatibility
- [ ] Test: Verify initialization works

### Phase 1.8.3: Public API Delegation
- [ ] Refactor `ingest()` to delegate
- [ ] Refactor `answer_query()` to delegate
- [ ] Refactor `run_full_rag_workflow()` to delegate
- [ ] Test: Verify public API works

### Phase 1.8.4: Remove Extracted Code
- [ ] Remove chain creation methods (DocumentGrader, QueryProcessor, AnswerGenerator)
- [ ] Remove vector store methods (VectorStoreManager)
- [ ] Remove document management methods (DocumentManager)
- [ ] Remove cache management methods (CacheOrchestrator)
- [ ] Test: Verify no broken references

### Phase 1.8.5: Workflow Refactoring
- [ ] Create `_create_node_functions()` method
- [ ] Create `_create_routing_functions()` method
- [ ] Refactor all node methods to delegate
- [ ] Remove `_compile_rag_workflow()` method
- [ ] Test: Verify workflow execution works

### Phase 1.8.6: Cleanup
- [ ] Remove unused imports
- [ ] Remove unused instance variables
- [ ] Add comprehensive docstrings
- [ ] Run linters (black, isort, flake8)
- [ ] Test: Run full test suite

### Phase 1.8.7: Verification
- [ ] Verify line count reduction (target: ~350-400 lines)
- [ ] Run all existing tests
- [ ] Run new integration tests
- [ ] Performance testing
- [ ] Code review

### Phase 1.8.8: Documentation
- [ ] Update CoreRAGEngine docstring
- [ ] Document facade pattern usage
- [ ] Update architecture documentation
- [ ] Create migration guide (if needed)

---

## Risk Assessment

### High Risks
1. **Breaking Changes**: Accidental API breakage
   - **Mitigation**: Comprehensive test coverage, property accessors

2. **Performance Regression**: Additional delegation overhead
   - **Mitigation**: Performance testing, profiling

3. **Workflow Breakage**: LangGraph workflow execution issues
   - **Mitigation**: Extensive workflow testing

### Medium Risks
1. **Callback Issues**: Cache invalidation callbacks not working
   - **Mitigation**: Integration tests for callbacks

2. **State Management**: Shared state between modules
   - **Mitigation**: Clear module boundaries, immutable state where possible

### Low Risks
1. **Import Errors**: Missing module imports
   - **Mitigation**: Linting, import verification

---

## Success Criteria

### Must Have
- ✅ CoreRAGEngine reduced to ~300-400 lines
- ✅ All existing tests pass
- ✅ 100% backward compatibility
- ✅ All 7 modules properly integrated
- ✅ No performance regression

### Should Have
- ✅ Improved code readability
- ✅ Clear separation of concerns
- ✅ Comprehensive integration tests
- ✅ Updated documentation

### Nice to Have
- ✅ Performance improvements
- ✅ Reduced memory usage
- ✅ Easier maintainability

---

## Timeline Estimate

| Phase | Estimated Time | Complexity |
|-------|---------------|------------|
| 1.8.1: Preparation | 30 min | Low |
| 1.8.2: Module Integration | 1-2 hours | Medium |
| 1.8.3: Public API Delegation | 1-2 hours | Medium |
| 1.8.4: Remove Extracted Code | 1 hour | Low |
| 1.8.5: Workflow Refactoring | 2-3 hours | High |
| 1.8.6: Cleanup | 30 min | Low |
| 1.8.7: Verification | 1-2 hours | Medium |
| 1.8.8: Documentation | 30 min | Low |
| **Total** | **8-12 hours** | **High** |

---

## Rollback Plan

If issues arise during implementation:

1. **Immediate Rollback**: Revert to pre-refactoring commit
2. **Partial Rollback**: Keep module imports but restore delegated methods
3. **Gradual Migration**: Implement facade pattern incrementally

---

## Next Steps

**After this plan is approved:**

1. Create implementation branch: `claude/phase-1.8-facade-refactoring`
2. Begin with Phase 1.8.1 (Preparation)
3. Implement incrementally with frequent testing
4. Commit after each successful phase
5. Final review and merge

---

## Questions for Review

Before proceeding with implementation, please confirm:

1. **Approach**: Is the facade pattern delegation approach acceptable?
2. **Backward Compatibility**: Are the proposed property accessors sufficient?
3. **Testing**: Is the testing strategy comprehensive enough?
4. **Timeline**: Is 8-12 hours of work acceptable for this phase?
5. **Risk**: Are you comfortable with the identified risks and mitigations?

---

**End of Phase 1.8 Implementation Plan**
