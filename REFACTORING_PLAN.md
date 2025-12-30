# Adaptive RAG Project - Comprehensive Refactoring Plan

**Date:** 2025-12-30
**Status:** Ready for Execution
**Estimated Effort:** Large (Multi-phase refactoring)

---

## Executive Summary

The Adaptive RAG codebase is functional but suffers from a critical **"god object" anti-pattern** in `core_rag_engine.py` (2,976 lines, 50+ methods). This refactoring plan breaks down the monolithic class into manageable, single-responsibility modules while removing duplicates and dead code.

### Critical Issues Identified

1. **CRITICAL**: `core_rag_engine.py` - Monolithic god class (2,976 lines)
2. **HIGH**: Duplicate Streamlit apps (`main_app.py` vs `main_app_mcp_enhanced.py`)
3. **MEDIUM**: Dead commented code in `main_app.py` (lines 35-48)
4. **LOW**: Minor code duplication in factory patterns

---

## Phase 1: Core RAG Engine Refactoring

### 1.1 Analysis of Current Structure

**Current State:**
- Single class `CoreRAGEngine` with 50+ methods
- Responsibilities include:
  - Document loading/ingestion
  - Vector store management
  - Embedding initialization
  - Text splitting
  - Chain creation (query analysis, grading, rewriting, generation, grounding)
  - LangGraph workflow orchestration
  - Cache management
  - Error handling

**Target State:**
- Modular architecture with clear separation of concerns
- Each module has single responsibility
- Easier testing and maintenance

### 1.2 New Module Structure

Create the following new modules in `src/rag/`:

#### **Module 1: `document_manager.py`**
**Responsibility:** Document loading, splitting, and preparation

**Classes:**
- `DocumentManager`
  - `load_documents(source_type, source_value)` → List[Document]
  - `split_documents(docs)` → List[Document]
  - `prepare_documents_for_indexing(docs)` → List[Document]

**Methods to extract from CoreRAGEngine:**
- `load_documents()` (line ~2390)
- `split_documents()` (line ~2430)
- `_init_text_splitter()` (line ~916)
- `_create_adaptive_splitter()` (line ~984)
- `_create_semantic_splitter()` (line ~1010)
- `_create_hybrid_splitter()` (line ~1075)
- `_create_default_splitter()` (line ~1108)

**Dependencies:**
- `TextSplitterFactory` (already exists)
- `AdaptiveChunker`, `HybridChunker` (existing)

---

#### **Module 2: `vector_store_manager.py`**
**Responsibility:** Vector store initialization, persistence, and retrieval setup

**Classes:**
- `VectorStoreManager`
  - `init_or_load_vectorstore(collection_name, recreate)` → None
  - `index_documents(docs, collection_name, recreate)` → None
  - `get_vectorstore(collection_name)` → Chroma
  - `delete_collection(collection_name)` → None
  - `list_collections()` → List[str]
  - `setup_retriever(collection_name)` → None

**Methods to extract from CoreRAGEngine:**
- `_init_or_load_vectorstore()` (line ~637)
- `_handle_existing_collection_in_memory()` (line ~676)
- `_handle_load_from_disk()` (line ~726)
- `_setup_retriever_for_collection()` (line ~778)
- `index_documents()` (line ~2490)
- `_handle_recreate_collection()` (line ~2525)
- `_get_persist_dir()` (line ~599)

**Dependencies:**
- `EmbeddingFactory` (already exists)
- ChromaDB
- `AdaptiveHybridRetriever` (existing)

---

#### **Module 3: `query_processor.py`**
**Responsibility:** Query analysis, rewriting, and processing

**Classes:**
- `QueryProcessor`
  - `analyze_query(question)` → QueryAnalysis
  - `rewrite_query(question)` → str
  - `should_use_web_search(query_analysis)` → bool

**Methods to extract from CoreRAGEngine:**
- `_analyze_query_node()` (line ~2138)
- `_rewrite_query_node()` (line ~2169)
- `_create_query_analyzer_chain()` (line ~469)
- `_create_query_rewriter_chain()` (line ~1219)

**Dependencies:**
- `ChainFactory` (already exists)
- LLM provider

---

#### **Module 4: `document_grader.py`**
**Responsibility:** Document relevance grading and reranking

**Classes:**
- `DocumentGrader`
  - `grade_documents(docs, question)` → List[Document]
  - `rerank_documents(docs, question)` → List[Document]
  - `calculate_relevance_score(doc, question)` → float

**Methods to extract from CoreRAGEngine:**
- `_grade_documents_node()` (line ~2077)
- `_rerank_documents_node()` (line ~1991)
- `_create_document_relevance_grader_chain()` (line ~1180)
- `_create_document_reranker_chain()` (line ~1043)

**Dependencies:**
- `ChainFactory`
- `RelevanceGrade`, `RerankScore` models (already exist)

---

#### **Module 5: `answer_generator.py`**
**Responsibility:** Answer generation and grounding validation

**Classes:**
- `AnswerGenerator`
  - `generate_answer(context, question)` → str
  - `check_grounding(answer, documents)` → GroundingCheck
  - `perform_advanced_grounding_check(answer, documents, question)` → Dict

**Methods to extract from CoreRAGEngine:**
- `_generate_answer_node()` (line ~2268)
- `_create_answer_generation_chain()` (line ~1257)
- `_create_grounding_check_chain()` (line ~1299)
- `_perform_basic_grounding_check()` (line ~1485)
- `_generate_basic_feedback()` (line ~1549)
- `_generate_advanced_feedback()` (line ~1619)
- `_handle_grounding_check_exception()` (line ~1732)

**Dependencies:**
- `ChainFactory`
- `MultiLevelGroundingChecker` (existing)
- `GroundingCheck` model (already exists)

---

#### **Module 6: `workflow_orchestrator.py`**
**Responsibility:** LangGraph workflow orchestration and state management

**Classes:**
- `WorkflowOrchestrator`
  - `compile_workflow()` → StateGraph
  - `run_workflow_sync(question, collection_name, top_k)` → Dict
  - `run_workflow_async(question, collection_name, top_k)` → Dict

**Methods to extract from CoreRAGEngine:**
- `_compile_rag_workflow()` (line ~2341)
- `run_full_rag_workflow_sync()` (line ~2864)
- `answer_query()` (line ~2761)
- `_retrieve_node()` (line ~1866)
- `_web_search_node()` (line ~2195)
- `_increment_retries_node()` (line ~2309)
- `_route_after_grading()` (line ~2316)
- `_route_after_grounding_check()` (line ~1805)

**Dependencies:**
- All other modules (DocumentGrader, QueryProcessor, AnswerGenerator, etc.)
- LangGraph

---

#### **Module 7: `cache_orchestrator.py`**
**Responsibility:** Coordinate caching across the engine

**Classes:**
- `CacheOrchestrator`
  - `invalidate_collection_cache(collection_name)` → None
  - `clear_all_caches()` → None
  - `get_cache_stats()` → Dict
  - `set_cache_ttl(ttl_seconds)` → None
  - `maintain_cache(max_size_mb)` → None

**Methods to extract from CoreRAGEngine:**
- `_invalidate_collection_cache()` (line ~2450)
- `clear_document_cache()` (line ~518)
- `get_cache_stats()` (line ~2687)
- `invalidate_all_caches()` (line ~2725)
- `set_cache_ttl()` (line ~2745)
- `_maintain_cache()` (line ~416)
- `_stream_documents_from_collection()` (line ~252)
- `_get_all_documents_from_collection()` (line ~314)

**Dependencies:**
- `CacheManager` (already exists in `src/rag/cache_manager.py`)

---

#### **Module 8: Refactored `core_rag_engine.py`**
**Responsibility:** Facade/orchestrator that delegates to specialized modules

**New Structure:**
```python
class CoreRAGEngine:
    def __init__(self, ...):
        # Initialize all managers
        self.document_manager = DocumentManager(...)
        self.vector_store_manager = VectorStoreManager(...)
        self.query_processor = QueryProcessor(...)
        self.document_grader = DocumentGrader(...)
        self.answer_generator = AnswerGenerator(...)
        self.workflow_orchestrator = WorkflowOrchestrator(...)
        self.cache_orchestrator = CacheOrchestrator(...)

    # Public API methods delegate to managers
    def ingest(self, source_type, source_value, name, recreate=False):
        docs = self.document_manager.load_documents(source_type, source_value)
        split_docs = self.document_manager.split_documents(docs)
        self.vector_store_manager.index_documents(split_docs, name, recreate)
        self.cache_orchestrator.invalidate_collection_cache(name)

    def answer_query(self, question, collection_name=None, top_k=4):
        return self.workflow_orchestrator.run_workflow_sync(
            question, collection_name, top_k
        )

    # ... delegate other public methods
```

**Lines:** ~300-400 (down from 2,976)

---

### 1.3 Refactoring Steps (Detailed)

#### **Step 1.3.1: Create DocumentManager**
1. Create `src/rag/document_manager.py`
2. Define `DocumentManager` class
3. Extract and move document loading/splitting methods
4. Add type hints to all methods
5. Update imports to use existing factories
6. Write unit tests in `tests/test_document_manager.py`

**Verification:**
- All document loading still works
- Text splitting produces same results
- No behavior changes

---

#### **Step 1.3.2: Create VectorStoreManager**
1. Create `src/rag/vector_store_manager.py`
2. Define `VectorStoreManager` class
3. Extract vector store initialization and persistence methods
4. Handle collection management (create, delete, list)
5. Setup retrievers for each collection
6. Write unit tests in `tests/test_vector_store_manager.py`

**Verification:**
- Collections are created/loaded correctly
- Persistence to disk works
- Retrievers are properly configured

---

#### **Step 1.3.3: Create QueryProcessor**
1. Create `src/rag/query_processor.py`
2. Define `QueryProcessor` class
3. Extract query analysis and rewriting logic
4. Move chain creation for query processing
5. Write unit tests in `tests/test_query_processor.py`

**Verification:**
- Query analysis produces same results
- Query rewriting works correctly
- Chain creation is functional

---

#### **Step 1.3.4: Create DocumentGrader**
1. Create `src/rag/document_grader.py`
2. Define `DocumentGrader` class
3. Extract grading and reranking logic
4. Move relevance scoring chains
5. Write unit tests in `tests/test_document_grader.py`

**Verification:**
- Document grading produces same scores
- Reranking order is consistent
- Filtering works correctly

---

#### **Step 1.3.5: Create AnswerGenerator**
1. Create `src/rag/answer_generator.py`
2. Define `AnswerGenerator` class
3. Extract answer generation logic
4. Extract grounding check logic (basic and advanced)
5. Move feedback generation methods
6. Write unit tests in `tests/test_answer_generator.py`

**Verification:**
- Answers are generated correctly
- Grounding checks work
- Advanced grounding integration works
- Feedback generation is correct

---

#### **Step 1.3.6: Create CacheOrchestrator**
1. Create `src/rag/cache_orchestrator.py`
2. Define `CacheOrchestrator` class
3. Extract cache management methods
4. Integrate with existing `CacheManager`
5. Write unit tests in `tests/test_cache_orchestrator.py`

**Verification:**
- Cache invalidation works
- Cache stats are accurate
- TTL management works

---

#### **Step 1.3.7: Create WorkflowOrchestrator**
1. Create `src/rag/workflow_orchestrator.py`
2. Define `WorkflowOrchestrator` class
3. Extract LangGraph workflow compilation
4. Extract node functions (retrieve, grade, rewrite, generate, etc.)
5. Extract routing logic
6. Inject dependencies (DocumentGrader, QueryProcessor, AnswerGenerator, etc.)
7. Write unit tests in `tests/test_workflow_orchestrator.py`

**Verification:**
- Workflow compiles correctly
- All nodes execute properly
- Routing decisions are correct
- End-to-end workflow produces same results

---

#### **Step 1.3.8: Refactor CoreRAGEngine as Facade**
1. Update `src/core_rag_engine.py`
2. Replace method implementations with delegation to managers
3. Keep public API unchanged
4. Update `__init__` to instantiate all managers
5. Update existing tests to ensure backward compatibility

**Verification:**
- Public API remains unchanged
- All existing tests pass
- Behavior is identical to before refactoring
- Integration tests pass

---

### 1.4 Migration Strategy

**Approach:** Incremental extraction with feature flags

1. **Create new modules alongside old code** (no breaking changes)
2. **Add feature flag in config**: `USE_REFACTORED_MODULES = True/False`
3. **Test both paths** to ensure equivalence
4. **Gradually migrate** one module at a time
5. **Remove old code** only after all tests pass with new modules

**Rollback Plan:**
- Keep git commits atomic (one module per commit)
- Each commit should be independently revertable
- Tag each phase completion

---

## Phase 2: Consolidate Streamlit Apps

### 2.1 Analysis

**Current State:**
- `main_app.py` (289 lines) - Basic RAG UI
- `main_app_mcp_enhanced.py` (293 lines) - MCP-enhanced UI
- ~80% code duplication

**Target State:**
- Single `main_app.py` with conditional MCP loading
- Feature flag: `ENABLE_MCP` (from environment or config)

### 2.2 Implementation Steps

#### **Step 2.2.1: Add MCP Feature Flag**
1. Update `src/config.py` to add `enable_mcp: bool` setting
2. Read from environment variable `ENABLE_MCP` (default: False)

#### **Step 2.2.2: Merge Streamlit Apps**
1. Keep `main_app.py` as base
2. Add conditional MCP loading:
   ```python
   @st.cache_resource
   def load_engine():
       core_engine = CoreRAGEngine()
       if app_settings.enable_mcp:
           from mcp.rag_integration import MCPEnhancedRAG
           mcp_engine = MCPEnhancedRAG(core_engine, ...)
           asyncio.run(mcp_engine.initialize_mcp_servers())
           return mcp_engine
       return core_engine
   ```
3. Update UI to conditionally show MCP features
4. Test both modes (MCP enabled and disabled)

#### **Step 2.2.3: Remove Duplicate File**
1. Delete `main_app_mcp_enhanced.py`
2. Update documentation to reference new feature flag approach
3. Update README with instructions for enabling MCP

#### **Step 2.2.4: Update Docker/Compose**
1. Update `Dockerfile` to use single `main_app.py`
2. Update `docker-compose.yml` to set `ENABLE_MCP` env var if needed

**Verification:**
- App runs in both MCP and non-MCP modes
- No functionality lost
- UI works correctly in both modes

---

## Phase 3: Remove Dead Code and Clean Up

### 3.1 Remove Commented Dead Code

#### **File: `src/main_app.py` (after merge)**
- Remove lines 35-48 (commented MCP loading code)

### 3.2 Clean Up Unused Imports

Run automated cleanup:
```bash
# Use autoflake to remove unused imports
autoflake --in-place --remove-all-unused-imports src/**/*.py

# Use isort to organize imports
isort src/ tests/

# Use black to format
black src/ tests/
```

### 3.3 Remove Temporary/Test Directories

Add to `.gitignore` if not already present:
```
test_chroma_db/
agent_test_chroma_db_e2e/
*.pyc
__pycache__/
.pytest_cache/
```

Clean up:
```bash
rm -rf test_chroma_db/ agent_test_chroma_db_e2e/
```

---

## Phase 4: Configuration Consolidation (Optional - Low Priority)

### 4.1 Current State

Configuration scattered across:
- `src/config.py` - Pydantic settings
- `.env.example` - Environment template
- `pyproject.toml` - Package config

### 4.2 Improvement (Optional)

**If desired:**
1. Keep `config.py` as single source of truth
2. Ensure all settings are documented in `.env.example`
3. Add validation for required settings
4. Add `config.validate()` method to check configuration at startup

**Note:** This is low priority and can be skipped if current setup works well.

---

## Phase 5: Update Tests and Documentation

### 5.1 Test Updates

#### **New Test Files to Create:**
1. `tests/test_document_manager.py`
2. `tests/test_vector_store_manager.py`
3. `tests/test_query_processor.py`
4. `tests/test_document_grader.py`
5. `tests/test_answer_generator.py`
6. `tests/test_cache_orchestrator.py`
7. `tests/test_workflow_orchestrator.py`

#### **Existing Tests to Update:**
1. `tests/test_core_rag_engine.py` - Update to test facade pattern
2. `tests/test_agent_loop.py` - Ensure compatibility with refactored engine

#### **Test Coverage Goals:**
- Maintain or improve current coverage
- Each new module should have >80% coverage
- Integration tests for full workflow

### 5.2 Documentation Updates

#### **Update `Architecture.md`:**
1. Add new module structure diagram
2. Document responsibilities of each module
3. Update data flow diagrams
4. Add dependency graph

#### **Update `README.md`:**
1. Update installation instructions (if changed)
2. Update usage examples to reflect new structure
3. Add section on MCP feature flag
4. Update API reference with new module structure

#### **Update `QUICKSTART.md`:**
1. Update code examples to reflect new imports (if any)
2. Add note about MCP mode

#### **Create `MIGRATION_GUIDE.md`:**
1. Document changes from old to new structure
2. Provide migration examples for existing users
3. Highlight breaking changes (if any)

---

## Phase 6: Verification and Testing

### 6.1 Automated Testing

```bash
# Run full test suite
make test

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Check for coverage drops
pytest --cov=src --cov-fail-under=80
```

### 6.2 Manual Testing Checklist

- [ ] **Document Ingestion:**
  - [ ] Load from file (PDF, text)
  - [ ] Load from URL
  - [ ] Load from uploaded file (Streamlit)

- [ ] **Vector Store:**
  - [ ] Create new collection
  - [ ] Load existing collection
  - [ ] Recreate collection
  - [ ] Delete collection

- [ ] **Query Processing:**
  - [ ] Simple queries
  - [ ] Complex multi-step queries
  - [ ] Queries requiring web search

- [ ] **Answer Generation:**
  - [ ] Basic RAG workflow
  - [ ] With grounding checks
  - [ ] With advanced grounding
  - [ ] Multiple iterations (query rewriting)

- [ ] **MCP Features:**
  - [ ] App runs with MCP disabled
  - [ ] App runs with MCP enabled
  - [ ] All MCP servers work

- [ ] **Cache Management:**
  - [ ] Cache invalidation works
  - [ ] Cache stats are accurate
  - [ ] TTL enforcement works

### 6.3 Performance Testing

Compare before/after refactoring:

1. **Query Latency:**
   - Measure average response time for 100 queries
   - Should be within 5% of original

2. **Memory Usage:**
   - Check memory footprint
   - Should not increase significantly

3. **Throughput:**
   - Measure queries per second
   - Should maintain or improve

### 6.4 Integration Testing

- [ ] Test with OpenAI provider
- [ ] Test with Google provider
- [ ] Test with Ollama provider
- [ ] Test with different embedding models
- [ ] Test with agent loop integration
- [ ] Test with stock feed integration
- [ ] Test with scraper feed integration

---

## Rollout Plan

### Timeline

**Phase 1: Core Refactoring (Largest effort)**
- Step 1.3.1 (DocumentManager): 2-3 hours
- Step 1.3.2 (VectorStoreManager): 3-4 hours
- Step 1.3.3 (QueryProcessor): 2-3 hours
- Step 1.3.4 (DocumentGrader): 2-3 hours
- Step 1.3.5 (AnswerGenerator): 3-4 hours
- Step 1.3.6 (CacheOrchestrator): 2-3 hours
- Step 1.3.7 (WorkflowOrchestrator): 4-5 hours
- Step 1.3.8 (Facade Refactor): 2-3 hours
- **Total: 20-28 hours**

**Phase 2: Streamlit Consolidation**
- Merge apps: 1-2 hours
- Testing: 1 hour
- **Total: 2-3 hours**

**Phase 3: Clean Up**
- Remove dead code: 30 min
- Clean imports: 30 min
- **Total: 1 hour**

**Phase 4: Configuration (Optional)**
- **Skip or 1-2 hours**

**Phase 5: Documentation**
- Update docs: 2-3 hours
- **Total: 2-3 hours**

**Phase 6: Verification**
- Automated tests: 1 hour
- Manual testing: 2-3 hours
- Performance testing: 1-2 hours
- **Total: 4-6 hours**

**Grand Total: 29-43 hours of focused work**

### Execution Order

1. **Phase 1** (Core refactoring) - Can be done incrementally, one module at a time
2. **Phase 2** (Streamlit merge) - Can be done independently
3. **Phase 3** (Clean up) - Quick wins, can be done anytime
4. **Phase 5** (Documentation) - After Phase 1 is complete
5. **Phase 6** (Verification) - After all changes

### Git Workflow

Each phase should be developed in a separate branch:

```bash
# Phase 1: Core refactoring
git checkout -b refactor/phase1-core-engine

# Create one commit per module
git commit -m "refactor: Extract DocumentManager from CoreRAGEngine"
git commit -m "refactor: Extract VectorStoreManager from CoreRAGEngine"
# ... etc

# Phase 2: Streamlit consolidation
git checkout -b refactor/phase2-streamlit-merge
git commit -m "refactor: Merge Streamlit apps with MCP feature flag"

# Phase 3: Clean up
git checkout -b refactor/phase3-cleanup
git commit -m "chore: Remove dead code and unused imports"

# Merge to main branch when each phase is complete and tested
```

### Communication

After each phase:
1. Create a PR with detailed description
2. Include before/after metrics
3. Document any breaking changes
4. Update CHANGELOG.md

---

## Success Criteria

### Code Quality
- [ ] `core_rag_engine.py` reduced from 2,976 lines to <500 lines
- [ ] Each new module is <400 lines
- [ ] All modules have single, clear responsibility
- [ ] No code duplication between Streamlit apps
- [ ] All dead code removed
- [ ] Code follows project style guidelines (see RULES.md)

### Functionality
- [ ] All existing tests pass
- [ ] No breaking changes to public API
- [ ] Behavior is identical to pre-refactoring
- [ ] Performance within 5% of baseline

### Maintainability
- [ ] Each module can be understood independently
- [ ] Dependencies are clear and minimal
- [ ] New features can be added without touching multiple modules
- [ ] Tests are isolated and fast

### Documentation
- [ ] Architecture diagram updated
- [ ] All new modules documented
- [ ] Migration guide created
- [ ] README updated

---

## Risk Assessment

### High Risk
- **Breaking existing functionality during refactoring**
  - **Mitigation:** Incremental changes, feature flags, extensive testing

### Medium Risk
- **Performance regression**
  - **Mitigation:** Performance benchmarking before/after

- **Test failures due to refactoring**
  - **Mitigation:** Update tests incrementally, maintain backward compatibility

### Low Risk
- **Documentation gaps**
  - **Mitigation:** Update docs as part of each phase

- **Merge conflicts**
  - **Mitigation:** Work in feature branches, merge frequently

---

## Appendix A: File Size Summary

### Before Refactoring
```
core_rag_engine.py:         2976 lines
main_app.py:                 289 lines
main_app_mcp_enhanced.py:    293 lines
```

### After Refactoring (Estimated)
```
core_rag_engine.py:          ~300-400 lines (facade)
rag/document_manager.py:     ~250 lines
rag/vector_store_manager.py: ~350 lines
rag/query_processor.py:      ~200 lines
rag/document_grader.py:      ~250 lines
rag/answer_generator.py:     ~400 lines
rag/cache_orchestrator.py:   ~200 lines
rag/workflow_orchestrator.py:~500 lines

main_app.py:                 ~320 lines (merged)
main_app_mcp_enhanced.py:    REMOVED
```

**Total lines of code:** Similar, but much better organized

---

## Appendix B: Dependency Graph (After Refactoring)

```
CoreRAGEngine (Facade)
├── DocumentManager
│   ├── TextSplitterFactory
│   ├── AdaptiveChunker
│   └── HybridChunker
├── VectorStoreManager
│   ├── EmbeddingFactory
│   ├── Chroma
│   └── AdaptiveHybridRetriever
├── QueryProcessor
│   ├── ChainFactory
│   └── LLMFactory
├── DocumentGrader
│   ├── ChainFactory
│   └── LLMFactory
├── AnswerGenerator
│   ├── ChainFactory
│   ├── LLMFactory
│   └── MultiLevelGroundingChecker
├── CacheOrchestrator
│   └── CacheManager
└── WorkflowOrchestrator
    ├── DocumentGrader
    ├── QueryProcessor
    ├── AnswerGenerator
    ├── VectorStoreManager
    └── LangGraph
```

---

## Appendix C: Quick Reference - What to Keep/Remove/Refactor

### ✅ KEEP (Core Features)
- All RAG functionality
- LangGraph workflow
- Multi-provider support (OpenAI, Google, Ollama)
- Adaptive chunking
- Hybrid search
- Advanced grounding (`advanced_grounding.py` - IS USED)
- Context management
- Agent loop
- Stock feed integration
- Web scraper
- MCP integration
- All tests

### 🔄 REFACTOR (Restructure)
- `core_rag_engine.py` → Split into 7 modules
- `main_app.py` + `main_app_mcp_enhanced.py` → Merge with feature flag

### ❌ REMOVE (Dead/Unnecessary)
- Commented code in `main_app.py` (lines 35-48)
- `main_app_mcp_enhanced.py` (after merge)
- Unused test directories

### ⚠️ VERIFY (Check if actually used)
- ✅ `advanced_grounding.py` - **CONFIRMED: Used in core_rag_engine.py line 1442**
- All current dependencies in `pyproject.toml`

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Get approval** for Phase 1 execution
3. **Set up feature branch** for refactoring work
4. **Begin with Phase 1, Step 1.3.1** (DocumentManager)
5. **Track progress** using the todo list
6. **Commit frequently** with clear commit messages
7. **Test continuously** after each module extraction

---

**Document Version:** 1.0
**Last Updated:** 2025-12-30
**Author:** Claude Code
**Status:** Ready for Approval & Execution
