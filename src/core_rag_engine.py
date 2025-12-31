import logging
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END

from src.advanced_grounding import MultiLevelGroundingChecker
from src.config import settings as app_settings
from src.context_manager import ContextManager

# Import refactored RAG components
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
)

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ImportError:
    UploadedFile = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


load_dotenv()


PROMPT_TEMPLATE = (
    "Answer the question based only on the following context.\n"
    "If the context does not contain the answer, state that you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


class CoreRAGEngine:
    """
    Core RAG engine: ingestion, indexing, and adaptive querying via LangGraph.

    This class now serves as a facade, delegating to specialized modules:
    - DocumentManager: Document loading and splitting
    - VectorStoreManager: Vector store operations
    - QueryProcessor: Query analysis and rewriting
    - DocumentGrader: Document grading and reranking
    - AnswerGenerator: Answer generation and grounding
    - CacheOrchestrator: Cache management
    - WorkflowOrchestrator: LangGraph workflow orchestration

    All public APIs are preserved for backward compatibility.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        default_collection_name: Optional[str] = None,
        persist_directory_base: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        max_rewrite_retries: Optional[int] = None,
        max_grounding_attempts: Optional[int] = None,
        default_retrieval_top_k: Optional[int] = None,
        enable_hybrid_search: Optional[bool] = None,
        hybrid_search_alpha: Optional[float] = None,
        enable_advanced_grounding: Optional[bool] = None,
    ) -> None:
        """
        Initializes the CoreRAGEngine with configuration settings.

        This sets up LLMs, embeddings, and initializes all specialized modules.
        """
        # --- Core Configuration ---
        self.llm_provider = llm_provider or app_settings.llm.llm_provider
        _llm_model_name_from_config = app_settings.llm.get_model_name_for_provider(self.llm_provider)
        self.llm_model_name = llm_model_name or _llm_model_name_from_config

        self.embedding_provider = embedding_provider or app_settings.embedding.embedding_provider
        _embedding_model_name_from_config = app_settings.embedding.get_model_name_for_provider(self.embedding_provider)
        self.embedding_model_name = embedding_model_name or _embedding_model_name_from_config

        self.temperature = temperature if temperature is not None else app_settings.llm.temperature
        self.chunk_size = chunk_size or app_settings.engine.chunk_size
        self.chunk_overlap = chunk_overlap or app_settings.engine.chunk_overlap
        self.default_collection_name = default_collection_name or app_settings.engine.default_collection_name

        # --- Persistence Setup ---
        configured_persist_dir = (
            persist_directory_base if persist_directory_base is not None else app_settings.engine.persist_directory_base
        )
        base_dir = configured_persist_dir if configured_persist_dir is not None else tempfile.gettempdir()
        self.persist_directory_base = os.path.join(base_dir, "core_rag_engine_chroma")
        os.makedirs(self.persist_directory_base, exist_ok=True)

        # --- API Keys ---
        self.tavily_api_key = tavily_api_key if tavily_api_key is not None else app_settings.api.tavily_api_key
        self.openai_api_key = openai_api_key if openai_api_key is not None else app_settings.api.openai_api_key
        self.google_api_key = google_api_key if google_api_key is not None else app_settings.api.google_api_key

        # --- Workflow Limits ---
        self.max_rewrite_retries = (
            max_rewrite_retries if max_rewrite_retries is not None else app_settings.engine.max_rewrite_retries
        )
        self.max_grounding_attempts = (
            max_grounding_attempts if max_grounding_attempts is not None else app_settings.engine.max_grounding_attempts
        )
        self.default_retrieval_top_k = (
            default_retrieval_top_k if default_retrieval_top_k is not None else app_settings.engine.default_retrieval_top_k
        )

        # --- Hybrid Search Configuration ---
        self.enable_hybrid_search = (
            enable_hybrid_search
            if enable_hybrid_search is not None
            else getattr(app_settings.engine, "enable_hybrid_search", False)
        )
        self.hybrid_search_alpha = (
            hybrid_search_alpha
            if hybrid_search_alpha is not None
            else getattr(app_settings.engine, "hybrid_search_alpha", 0.7)
        )

        # --- Advanced Grounding Configuration ---
        self.enable_advanced_grounding = (
            enable_advanced_grounding
            if enable_advanced_grounding is not None
            else getattr(app_settings.engine, "enable_advanced_grounding", False)
        )

        # --- Logger Setup ---
        self._setup_logger()

        # --- Core Components Initialization using Factories ---
        # Initialize LLM Factory and create LLMs
        self.llm_factory = LLMFactory(
            llm_provider=self.llm_provider,
            llm_model_name=self.llm_model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            google_api_key=self.google_api_key,
            logger=self.logger,
        )
        self.llm = self.llm_factory.create_llm(use_json_format=False)
        self.json_llm = self.llm_factory.create_llm(use_json_format=True)

        # Initialize Embedding Factory and create embedding model
        self.embedding_factory = EmbeddingFactory(
            embedding_provider=self.embedding_provider,
            embedding_model_name=self.embedding_model_name,
            openai_api_key=self.openai_api_key,
            google_api_key=self.google_api_key,
            logger=self.logger,
        )
        self.embedding_model = self.embedding_factory.create_embedding_model()

        # Initialize Text Splitter Factory
        self.text_splitter_factory = TextSplitterFactory(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            llm_provider=self.llm_provider,
            llm_model_name=self.llm_model_name,
            openai_api_key=self.openai_api_key,
            logger=self.logger,
        )

        # Initialize Chain Factory (for search tool)
        self.chain_factory = ChainFactory(
            llm=self.llm,
            json_llm=self.json_llm,
            tavily_api_key=self.tavily_api_key,
            logger=self.logger,
        )
        self.search_tool = self.chain_factory.create_search_tool()

        # Initialize Error Handler
        self.error_handler = ErrorHandler(logger=self.logger)

        # --- Initialize Specialized Modules ---

        # Document Manager (handles document loading and splitting)
        self.document_manager = DocumentManager(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            openai_api_key=self.openai_api_key,
            llm_provider=self.llm_provider,
            llm_model_name=self.llm_model_name,
            logger=self.logger,
        )

        # Cache Orchestrator (handles caching)
        self.cache_orchestrator = CacheOrchestrator(
            cache_ttl=300,  # 5 minutes
            max_cache_size_mb=500.0,
            logger=self.logger,
        )

        # Vector Store Manager (handles vector store operations)
        self.vector_store_manager = VectorStoreManager(
            embedding_model=self.embedding_model,
            persist_directory_base=self.persist_directory_base,
            default_retrieval_top_k=self.default_retrieval_top_k,
            enable_hybrid_search=self.enable_hybrid_search,
            get_all_documents_callback=self._get_all_documents_callback,
            stream_documents_callback=self._stream_documents_callback,
            invalidate_cache_callback=self.cache_orchestrator.invalidate_collection_cache,
            logger=self.logger,
        )

        # Query Processor (handles query analysis and rewriting)
        self.query_processor = QueryProcessor(
            llm=self.llm,
            json_llm=self.json_llm,
            logger=self.logger,
        )

        # Document Grader (handles document relevance grading and reranking)
        self.document_grader = DocumentGrader(
            json_llm=self.json_llm,
            logger=self.logger,
        )

        # Answer Generator (handles answer generation and grounding)
        self.answer_generator = AnswerGenerator(
            llm=self.llm,
            json_llm=self.json_llm,
            logger=self.logger,
        )

        # Workflow Orchestrator (handles LangGraph workflow)
        self.workflow_orchestrator = WorkflowOrchestrator(
            node_functions=self._create_node_functions(),
            routing_functions=self._create_routing_functions(),
            logger=self.logger,
        )

        # Compile the RAG workflow
        self.rag_workflow = self.workflow_orchestrator.compile_default_rag_workflow()

        # Initialize Context Manager
        self.context_manager = ContextManager(model_name=self.llm_model_name, reserved_tokens=1000)

        # Initialize advanced grounding checker if enabled
        self.advanced_grounding_checker = None
        if self.enable_advanced_grounding:
            try:
                self.advanced_grounding_checker = MultiLevelGroundingChecker(self.llm)
                self.logger.info("Advanced grounding checker initialized successfully.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced grounding checker: {e}")

        self.logger.info("CoreRAGEngine initialization complete and workflow compiled.")

    # ==================== Logger Setup ====================

    def _setup_logger(self) -> None:
        """
        Set up logger for CoreRAGEngine.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # ==================== Callback Methods for VectorStoreManager ====================

    def _get_all_documents_callback(
        self,
        collection_name: str,
        use_cache: bool = True,
        max_documents: Optional[int] = None,
    ) -> List[Document]:
        """
        Callback for VectorStoreManager to retrieve all documents from a collection.

        This method uses the cache orchestrator for caching.

        Args:
            collection_name: Collection to retrieve from
            use_cache: Whether to use cache
            max_documents: Maximum documents to return

        Returns:
            List of Document objects
        """
        # Check cache first
        if use_cache:
            cached_docs = self.cache_orchestrator.get_cached_documents(collection_name)
            if cached_docs is not None:
                if max_documents is not None:
                    cached_docs = cached_docs[:max_documents]
                self.logger.debug(
                    f"Cache HIT: Using cached documents for '{collection_name}' ({len(cached_docs)} docs)"
                )
                return cached_docs

        # Cache miss - fetch from vector store
        try:
            vectorstore = self.vector_store_manager.vectorstores.get(collection_name)
            if not vectorstore:
                self.logger.warning(f"Vector store for collection '{collection_name}' not found.")
                return []

            # Get collection
            collection = vectorstore._collection

            try:
                # Get count
                count_result = collection.count()
                doc_count = count_result if isinstance(count_result, int) else None

                if doc_count is not None:
                    self.logger.info(f"Collection '{collection_name}' contains {doc_count} documents")

                    if doc_count > 10000:
                        self.logger.warning(
                            f"Large collection detected ({doc_count} docs). "
                            f"Consider using streaming for memory efficiency."
                        )
            except Exception as count_error:
                self.logger.debug(f"Could not get document count: {count_error}")
                doc_count = None

            # Fetch documents using streaming
            all_docs: List[Document] = []

            for batch in self._stream_documents_callback(collection_name, batch_size=1000):
                all_docs.extend(batch)

                if max_documents is not None and len(all_docs) >= max_documents:
                    all_docs = all_docs[:max_documents]
                    self.logger.info(f"Reached max_documents limit ({max_documents}), stopping retrieval")
                    break

                if len(all_docs) % 5000 == 0:
                    self.logger.debug(f"Retrieved {len(all_docs)} documents so far...")

            self.logger.info(f"Retrieved {len(all_docs)} documents from '{collection_name}'.")

            # Update cache
            if use_cache:
                self.cache_orchestrator.cache_documents(collection_name, all_docs)

            return all_docs

        except Exception as e:
            self.logger.error(
                f"Failed to retrieve documents from '{collection_name}': {e}",
                exc_info=True,
            )
            return []

    def _stream_documents_callback(self, collection_name: str, batch_size: int = 1000):
        """
        Callback for VectorStoreManager to stream documents from a collection.

        Args:
            collection_name: Collection to stream from
            batch_size: Number of documents per batch

        Yields:
            Batches of Document objects
        """
        try:
            vectorstore = self.vector_store_manager.vectorstores.get(collection_name)
            if not vectorstore:
                self.logger.warning(f"Vector store for collection '{collection_name}' not found.")
                return

            collection = vectorstore._collection
            offset = 0

            self.logger.info(f"Starting document stream for collection '{collection_name}'...")

            while True:
                try:
                    results = collection.get(limit=batch_size, offset=offset)

                    if not results.get("documents"):
                        break

                    contents = results.get("documents", [])
                    metadatas = results.get("metadatas") or [None] * len(contents)

                    batch_docs = []
                    for content, metadata in zip(contents, metadatas):
                        if isinstance(content, str):
                            doc_metadata = metadata if isinstance(metadata, dict) else {}
                            doc = Document(page_content=content, metadata=doc_metadata)
                            batch_docs.append(doc)

                    if batch_docs:
                        yield batch_docs

                    if len(contents) < batch_size:
                        break

                    offset += batch_size

                except Exception as batch_error:
                    self.logger.error(
                        f"Error streaming batch at offset {offset} for '{collection_name}': {batch_error}",
                        exc_info=True,
                    )
                    break

            self.logger.info(f"Finished streaming documents from '{collection_name}'.")

        except Exception as e:
            self.logger.error(f"Error initializing stream for '{collection_name}': {e}", exc_info=True)

    # ==================== Workflow Node Functions ====================

    def _create_node_functions(self) -> Dict[str, Callable]:
        """
        Create node functions for workflow orchestrator.

        Returns:
            Dictionary mapping node names to node functions
        """
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
        """
        Analyze query node - delegates to QueryProcessor.

        Args:
            state: Current workflow state

        Returns:
            Updated state with query analysis
        """
        self.logger.info("NODE: Analyzing query")

        question = state["question"]
        chat_history = state.get("chat_history", [])

        query_analysis = self.query_processor.analyze_query(question, chat_history)

        if query_analysis:
            state["query_analysis"] = query_analysis

        return state

    def _rewrite_query_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Rewrite query node - delegates to QueryProcessor.

        Args:
            state: Current workflow state

        Returns:
            Updated state with rewritten query
        """
        self.logger.info("NODE: Rewriting query")

        question = state["question"]
        chat_history = state.get("chat_history", [])

        rewritten = self.query_processor.rewrite_query(question, chat_history)
        state["question"] = rewritten

        return state

    def _retrieve_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Retrieve documents node - delegates to VectorStoreManager.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieved documents
        """
        self.logger.info("NODE: Retrieving documents")

        collection_name = state.get("collection_name", self.default_collection_name)
        question = state["question"]

        # Get retriever from VectorStoreManager
        retriever = self.vector_store_manager.retrievers.get(collection_name)

        if not retriever:
            self.logger.warning(f"No retriever found for collection '{collection_name}'")
            state["documents"] = []
            return state

        try:
            documents = retriever.invoke(question)
            state["documents"] = documents
            self.logger.info(f"Retrieved {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}", exc_info=True)
            state["documents"] = []

        return state

    def _grade_documents_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Grade documents node - delegates to DocumentGrader.

        Args:
            state: Current workflow state

        Returns:
            Updated state with graded documents
        """
        self.logger.info("NODE: Grading documents")

        documents = state.get("documents", [])
        question = state["question"]

        if not documents:
            state["relevance_check_passed"] = False
            return state

        # Grade documents using DocumentGrader
        relevant_docs = self.document_grader.grade_documents(documents, question)

        state["documents"] = relevant_docs
        state["relevance_check_passed"] = len(relevant_docs) > 0

        self.logger.info(f"Relevance check: {len(relevant_docs)} relevant documents")

        return state

    def _rerank_documents_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Rerank documents node - delegates to DocumentGrader.

        Args:
            state: Current workflow state

        Returns:
            Updated state with reranked documents
        """
        self.logger.info("NODE: Reranking documents")

        documents = state.get("documents", [])
        question = state["question"]

        if not documents:
            return state

        # Rerank documents using DocumentGrader
        reranked_docs = self.document_grader.rerank_documents(documents, question)

        state["documents"] = reranked_docs

        self.logger.info(f"Reranked {len(reranked_docs)} documents")

        return state

    def _generate_answer_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Generate answer node - delegates to AnswerGenerator.

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated answer
        """
        self.logger.info("NODE: Generating answer")

        documents = state.get("documents", [])
        question = state["question"]
        chat_history = state.get("chat_history", [])
        regeneration_feedback = state.get("regeneration_feedback")

        # Format context from documents
        context = self.answer_generator.format_context(documents)
        state["context"] = context

        # Generate answer using AnswerGenerator
        generation = self.answer_generator.generate_answer(
            question=question,
            context=context,
            chat_history=chat_history,
            regeneration_feedback=regeneration_feedback,
        )

        state["generation"] = generation

        self.logger.info("Answer generated")

        return state

    def _grounding_check_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Grounding check node - delegates to AnswerGenerator.

        Args:
            state: Current workflow state

        Returns:
            Updated state with grounding check result
        """
        self.logger.info("NODE: Performing grounding check")

        context = state.get("context", "")
        generation = state["generation"]
        question = state["question"]

        # Perform grounding check using AnswerGenerator
        grounding_result = self.answer_generator.check_grounding(context, generation)

        if not grounding_result:
            self.logger.warning("Grounding check failed")
            state["is_grounded"] = False
            return state

        is_grounded = grounding_result.is_grounded
        state["is_grounded"] = is_grounded

        if not is_grounded:
            # Generate feedback for regeneration
            if self.enable_advanced_grounding and self.advanced_grounding_checker:
                try:
                    advanced_results = self.advanced_grounding_checker.check_grounding(
                        generation, context, question
                    )
                    feedback = self.answer_generator.generate_advanced_feedback(
                        advanced_results, question
                    )
                except Exception as e:
                    self.logger.warning(f"Advanced grounding check failed: {e}")
                    feedback = self.answer_generator.generate_basic_feedback(
                        grounding_result, question
                    )
            else:
                feedback = self.answer_generator.generate_basic_feedback(
                    grounding_result, question
                )

            state["regeneration_feedback"] = feedback

        self.logger.info(f"Grounding check: {'PASSED' if is_grounded else 'FAILED'}")

        return state

    def _web_search_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Web search node - uses Tavily search tool.

        Args:
            state: Current workflow state

        Returns:
            Updated state with web search results
        """
        self.logger.info("NODE: Performing web search")

        question = state["question"]

        try:
            if self.search_tool:
                search_results = self.search_tool.invoke(question)

                # Convert search results to documents
                documents = []
                if isinstance(search_results, str):
                    doc = Document(page_content=search_results, metadata={"source": "web_search"})
                    documents.append(doc)
                elif isinstance(search_results, list):
                    for result in search_results:
                        if isinstance(result, dict):
                            content = result.get("content", str(result))
                            doc = Document(page_content=content, metadata={"source": "web_search"})
                            documents.append(doc)

                state["documents"] = documents
                self.logger.info(f"Web search returned {len(documents)} documents")
            else:
                self.logger.warning("Search tool not available")
                state["documents"] = []

        except Exception as e:
            self.logger.error(f"Web search error: {e}", exc_info=True)
            state["documents"] = []

        return state

    def _increment_retries_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Increment retries node - simple state update.

        Args:
            state: Current workflow state

        Returns:
            Updated state with incremented retries
        """
        retries = state.get("retries", 0)
        state["retries"] = retries + 1

        self.logger.info(f"Incremented retries to {state['retries']}")

        return state

    # ==================== Workflow Routing Functions ====================

    def _create_routing_functions(self) -> Dict[str, Callable]:
        """
        Create routing functions for workflow orchestrator.

        Returns:
            Dictionary mapping routing names to routing functions
        """
        return {
            "route_after_grading": self._route_after_grading,
            "route_after_grounding_check": self._route_after_grounding_check,
        }

    def _route_after_grading(self, state: CoreGraphState) -> str:
        """
        Route after document grading.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        relevance_passed = state.get("relevance_check_passed", False)
        retries = state.get("retries", 0)
        query_analysis = state.get("query_analysis")

        # If relevance check passed, generate answer
        if relevance_passed:
            return "generate_answer"

        # Check if should use web search
        if self.query_processor.should_use_web_search(query_analysis):
            self.logger.info("ROUTING: Using web search")
            return "web_search"

        # Check retry limit
        if retries >= self.max_rewrite_retries:
            self.logger.warning(f"Max retries ({self.max_rewrite_retries}) reached, generating answer anyway")
            return "generate_answer"

        # Increment retries and rewrite query
        self.logger.info("ROUTING: Incrementing retries and rewriting query")
        return "increment_retries"

    def _route_after_grounding_check(self, state: CoreGraphState) -> str:
        """
        Route after grounding check.

        Args:
            state: Current workflow state

        Returns:
            Next node name or END
        """
        is_grounded = state.get("is_grounded", False)
        grounding_attempts = state.get("grounding_attempts", 0)

        if is_grounded:
            self.logger.info("ROUTING: Grounding passed, ending workflow")
            return END

        # Increment grounding attempts
        state["grounding_attempts"] = grounding_attempts + 1

        if state["grounding_attempts"] >= self.max_grounding_attempts:
            self.logger.warning(
                f"Max grounding attempts ({self.max_grounding_attempts}) reached, ending workflow"
            )
            return END

        # Regenerate answer with feedback
        self.logger.info("ROUTING: Grounding failed, regenerating answer")
        return "generate_answer"

    # ==================== Public API Methods ====================

    def ingest(
        self,
        source_type: str,
        source_value: Any,
        collection_name: Optional[str] = None,
        recreate: bool = False,
        strategy: str = "default",
    ) -> Dict[str, Any]:
        """
        Public ingestion API - delegates to DocumentManager and VectorStoreManager.

        Args:
            source_type: Type of source (url, pdf_path, text_path, uploaded_pdf)
            source_value: Source value (URL string, file path, or uploaded file)
            collection_name: Collection name (defaults to default_collection_name)
            recreate: Whether to recreate collection
            strategy: Splitting strategy (default, adaptive, semantic, hybrid)

        Returns:
            Dictionary with status and number of documents ingested
        """
        try:
            self.logger.info(f"Ingesting from {source_type}: {source_value}")

            # Load documents via DocumentManager
            documents = self.document_manager.load_documents(source_type, source_value)

            if not documents:
                return {"status": "error", "message": "No documents loaded", "documents_ingested": 0}

            # Split documents via DocumentManager
            split_docs = self.document_manager.split_documents(documents, strategy=strategy)

            if not split_docs:
                return {"status": "error", "message": "No documents after splitting", "documents_ingested": 0}

            # Index via VectorStoreManager
            target_collection = collection_name or self.default_collection_name
            self.vector_store_manager.index_documents(split_docs, target_collection, recreate=recreate)

            self.logger.info(f"Ingestion complete: {len(split_docs)} documents indexed")

            return {
                "status": "success",
                "documents_ingested": len(split_docs),
                "collection_name": target_collection,
            }

        except Exception as e:
            error_msg = f"Ingestion failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return self.error_handler.handle_error(e, "ingestion")

    def answer_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Public query API - delegates to WorkflowOrchestrator.

        Args:
            question: User question
            collection_name: Collection to query (defaults to default_collection_name)
            chat_history: Optional chat history
            stream: Whether to stream results

        Returns:
            Dictionary with answer and metadata
        """
        try:
            self.logger.info(f"Answering query: {question}")

            # Build initial state
            initial_state = CoreGraphState(
                question=question,
                original_question=question,
                collection_name=collection_name or self.default_collection_name,
                documents=[],
                context="",
                generation="",
                chat_history=chat_history or [],
                retries=0,
                grounding_attempts=0,
                relevance_check_passed=False,
                is_grounded=False,
            )

            # Execute workflow via WorkflowOrchestrator
            if stream:
                return self._stream_workflow(initial_state)
            else:
                final_state = self.workflow_orchestrator.invoke_workflow(initial_state)
                return self._format_response(final_state)

        except Exception as e:
            error_msg = f"Query failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return self.error_handler.handle_error(e, "query")

    async def run_full_rag_workflow(
        self,
        question: str,
        collection_name: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> CoreGraphState:
        """
        Async workflow execution - delegates to WorkflowOrchestrator.

        Args:
            question: User question
            collection_name: Collection to query
            chat_history: Optional chat history

        Returns:
            Final workflow state
        """
        self.logger.info(f"Running async RAG workflow for: {question}")

        initial_state = CoreGraphState(
            question=question,
            original_question=question,
            collection_name=collection_name or self.default_collection_name,
            documents=[],
            context="",
            generation="",
            chat_history=chat_history or [],
            retries=0,
            grounding_attempts=0,
            relevance_check_passed=False,
            is_grounded=False,
        )

        final_state = await self.workflow_orchestrator.ainvoke_workflow(initial_state)

        return final_state

    def run_full_rag_workflow_sync(
        self,
        question: str,
        collection_name: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> CoreGraphState:
        """
        Sync workflow execution - delegates to WorkflowOrchestrator.

        Args:
            question: User question
            collection_name: Collection to query
            chat_history: Optional chat history

        Returns:
            Final workflow state
        """
        self.logger.info(f"Running sync RAG workflow for: {question}")

        initial_state = CoreGraphState(
            question=question,
            original_question=question,
            collection_name=collection_name or self.default_collection_name,
            documents=[],
            context="",
            generation="",
            chat_history=chat_history or [],
            retries=0,
            grounding_attempts=0,
            relevance_check_passed=False,
            is_grounded=False,
        )

        final_state = self.workflow_orchestrator.invoke_workflow(initial_state)

        return final_state

    def _format_response(self, state: CoreGraphState) -> Dict[str, Any]:
        """
        Format workflow state into response dictionary.

        Args:
            state: Final workflow state

        Returns:
            Formatted response dictionary
        """
        return {
            "answer": state.get("generation", ""),
            "sources": [doc.metadata.get("source", "unknown") for doc in state.get("documents", [])],
            "is_grounded": state.get("is_grounded", False),
            "retries": state.get("retries", 0),
            "grounding_attempts": state.get("grounding_attempts", 0),
        }

    def _stream_workflow(self, initial_state: CoreGraphState):
        """
        Stream workflow execution.

        Args:
            initial_state: Initial workflow state

        Yields:
            State updates
        """
        for update in self.workflow_orchestrator.stream_workflow(initial_state):
            yield update

    # ==================== Collection Management (Delegated) ====================

    def list_collections(self) -> List[str]:
        """
        List all collections - delegates to VectorStoreManager.

        Returns:
            List of collection names
        """
        return self.vector_store_manager.list_collections()

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection - delegates to VectorStoreManager.

        Args:
            collection_name: Name of collection to delete

        Returns:
            True if successful, False otherwise
        """
        return self.vector_store_manager.delete_collection(collection_name)

    # ==================== Cache Management (Delegated) ====================

    def clear_document_cache(self, collection_name: Optional[str] = None) -> None:
        """
        Clear document cache - delegates to CacheOrchestrator.

        Args:
            collection_name: Collection to clear (None = all)
        """
        self.cache_orchestrator.clear_document_cache(collection_name)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics - delegates to CacheOrchestrator.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache_orchestrator.get_cache_stats()

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """
        Set cache TTL - delegates to CacheOrchestrator.

        Args:
            ttl_seconds: TTL in seconds
        """
        self.cache_orchestrator.set_cache_ttl(ttl_seconds)

    def invalidate_all_caches(self) -> None:
        """
        Invalidate all caches - delegates to CacheOrchestrator.
        """
        self.cache_orchestrator.invalidate_all_caches()

    # ==================== Helper Methods ====================

    def _format_chat_history(self, chat_history: Optional[List[BaseMessage]]) -> str:
        """
        Format chat history for prompts.

        Args:
            chat_history: List of chat messages

        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for msg in chat_history:
            role = "User" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")

        return "\n".join(formatted)

    def _append_error(self, error_list: List[str], error_msg: str) -> List[str]:
        """
        Append error message to error list.

        Args:
            error_list: List of errors
            error_msg: Error message to append

        Returns:
            Updated error list
        """
        if error_list is None:
            error_list = []
        error_list.append(error_msg)
        return error_list
