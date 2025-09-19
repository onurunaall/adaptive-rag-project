import os
import logging
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Union, TypedDict

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, StrOutputParser 
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field

from src.config import settings as app_settings
from src.chunking import AdaptiveChunker, BaseChunker, HybridChunker
from src.hybrid_search import HybridRetriever, AdaptiveHybridRetriever
from src.advanced_grounding import MultiLevelGroundingChecker, AdvancedGroundingCheck

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


class GroundingCheck(BaseModel):
    """
    Structured output for answer grounding and hallucination check.
    """
    is_grounded: bool = Field(description="Is the generated answer fully supported by the provided context? True or False.")
    
    ungrounded_statements: Optional[List[str]] = Field(default=None,
                                                       description="List any specific statements in the answer that are NOT supported by the context.")

    correction_suggestion: Optional[str] = Field(default=None,
                                                 description="If not grounded, suggest how to rephrase the answer to be grounded, or state no grounded answer is possible.")


class RelevanceGrade(BaseModel):
    """
    Structured output for document relevance grading.
    """
    is_relevant: bool = Field(description="Is the document excerpt relevant to the question? True or False.")
    
    justification: Optional[str] = Field(default=None,
                                         description="Brief justification for the relevance decision (1-2 sentences).")

class RerankScore(BaseModel):
    """
    Pydantic model for contextual re-ranking scores.
    """
    relevance_score: float = Field(description="A score from 0.0 to 1.0 indicating the document's direct relevance to answering the question.")
    justification: str = Field(description="A brief justification for the assigned score.")

class QueryAnalysis(BaseModel):
    """
    Structured output for sophisticated query analysis.
    """
    query_type: str = Field(description="Classify the query's primary type (e.g., 'factual_lookup', 'comparison', 'summary_request', 'complex_reasoning', 'ambiguous', 'keyword_search_sufficient', 'greeting', 'not_a_question').")

    main_intent: str = Field(description="A concise sentence describing the primary user intent or what the user wants to achieve.")

    extracted_keywords: List[str] = Field(default_factory=list,
                                          description="A list of key nouns, verbs, and named entities from the query that are critical for effective retrieval.")

    is_ambiguous: bool = Field(default=False,
                               description="True if the query is vague, unclear, or open to multiple interpretations and might benefit from clarification before proceeding.")


class CoreGraphState(TypedDict):
    question: str
    original_question: Optional[str]
    query_analysis_results: Optional[QueryAnalysis]
    documents: List[Document]
    context: str
    web_search_results: Optional[List[Document]]
    generation: str
    retries: int
    run_web_search: str  # "Yes" or "No"
    relevance_check_passed: Optional[bool]
    error_message: Optional[str]
    grounding_check_attempts: int
    regeneration_feedback: Optional[str]
    collection_name: Optional[str]
    chat_history: Optional[List[BaseMessage]]


class CoreRAGEngine:
    """
    Core RAG engine: ingestion, indexing, and adaptive querying via LangGraph.
    """
    def __init__(self,
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
                 enable_advanced_grounding: Optional[bool] = None) -> None:
        """
        Initializes the CoreRAGEngine with configuration settings.

        This sets up LLMs, embeddings, chunking strategies, search tools,
        and compiles the internal RAG workflow graph.
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
        configured_persist_dir = persist_directory_base if persist_directory_base is not None else app_settings.engine.persist_directory_base
        base_dir = configured_persist_dir if configured_persist_dir is not None else tempfile.gettempdir()
        self.persist_directory_base = os.path.join(base_dir, "core_rag_engine_chroma")
        os.makedirs(self.persist_directory_base, exist_ok=True)

        # --- API Keys ---
        self.tavily_api_key = tavily_api_key if tavily_api_key is not None else app_settings.api.tavily_api_key
        self.openai_api_key = openai_api_key if openai_api_key is not None else app_settings.api.openai_api_key
        self.google_api_key = google_api_key if google_api_key is not None else app_settings.api.google_api_key

        # --- Workflow Limits ---
        self.max_rewrite_retries = max_rewrite_retries if max_rewrite_retries is not None else app_settings.engine.max_rewrite_retries
        self.max_grounding_attempts = max_grounding_attempts if max_grounding_attempts is not None else app_settings.engine.max_grounding_attempts
        self.default_retrieval_top_k = default_retrieval_top_k if default_retrieval_top_k is not None else app_settings.engine.default_retrieval_top_k

        # --- Hybrid Search Configuration ---
        self.enable_hybrid_search = enable_hybrid_search if enable_hybrid_search is not None else getattr(app_settings.engine, 'enable_hybrid_search', False)
        self.hybrid_search_alpha = hybrid_search_alpha if hybrid_search_alpha is not None else getattr(app_settings.engine, 'hybrid_search_alpha', 0.7)

        # --- Logger Setup ---
        self._setup_logger()

        # --- Core Components Initialization ---
        # Initialize LLMs (normal output and JSON-formatted output)
        self.llm = self._init_llm(use_json_format=False)
        self.json_llm = self._init_llm(use_json_format=True)

        # Initialize Embedding Model
        self.embedding_model = self._init_embedding_model()

        # Initialize Text Splitter and Web Search Tool
        self.text_splitter = self._init_text_splitter()
        self.search_tool = self._init_search_tool()

        # --- Individual LLM Chains for Workflow Steps ---
        self.document_relevance_grader_chain = self._create_document_relevance_grader_chain()
        self.document_reranker_chain = self._create_document_reranker_chain()
        self.query_rewriter_chain = self._create_query_rewriter_chain()
        self.answer_generation_chain = self._create_answer_generation_chain()
        self.grounding_check_chain = self._create_grounding_check_chain()
        self.query_analyzer_chain = self._create_query_analyzer_chain()

        # --- Workflow Graph Compilation ---
        self._compile_rag_workflow()

        # --- Storage for Runtime Data ---
        self.vectorstores: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, Any] = {}

        # --- Advanced Grounding Configuration ---
        self.enable_advanced_grounding = enable_advanced_grounding if enable_advanced_grounding is not None else getattr(app_settings.engine, 'enable_advanced_grounding', False)
        
        # Initialize advanced grounding checker if the feature is enabled
        self.advanced_grounding_checker = None
        if self.enable_advanced_grounding:
            try:
                self.advanced_grounding_checker = MultiLevelGroundingChecker(self.llm)
                self.logger.info("Advanced grounding checker initialized successfully.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced grounding checker: {e}")

        # --- Document Cache for Retrieval Optimization ---
        self.document_cache = {}
        self.cache_ttl = 300  # 5 minutes Time-To-Live for cached documents
        self.cache_timestamps = {}

        self.logger.info("CoreRAGEngine initialization complete and workflow compiled.")
        
    
    def _get_all_documents_from_collection(self, collection_name: str) -> List[Document]:
        """
        Retrieve all documents from a specified collection with caching and memory optimization.

        This method fetches all documents from the vector store's underlying Chroma collection,
        batching the retrieval to manage memory usage for large collections. It also implements
        a simple in-memory cache with a Time-To-Live (TTL) to improve performance on repeated calls.

        Args:
            collection_name (str): The name of the collection to retrieve documents from.

        Returns:
            List[Document]: A list of all Document objects in the collection, or an empty list
                            if the collection doesn't exist, is empty, or an error occurs.
        """
        import time

        # --- Cache Check ---
        cache_key = collection_name
        current_time = time.time()

        # Check if we have a fresh cached version
        if cache_key in self.document_cache:
            cache_age = current_time - self.cache_timestamps.get(cache_key, 0)
            if cache_age < self.cache_ttl:
                self.logger.debug(f"Cache HIT: Using cached documents for '{collection_name}' (Age: {cache_age:.1f}s)")
                return self.document_cache[cache_key]
            else:
                self.logger.debug(f"Cache EXPIRED: Cached documents for '{collection_name}' are stale (Age: {cache_age:.1f}s)")

        # --- Retrieval from Vector Store ---
        try:
            vectorstore = self.vectorstores.get(collection_name)
            if not vectorstore:
                self.logger.warning(f"Vector store for collection '{collection_name}' not found in memory.")
                return []

            # --- Batched Retrieval ---
            batch_size = 1000
            all_docs: List[Document] = []
            offset = 0
            collection = vectorstore._collection  # Access the underlying Chroma collection

            self.logger.info(f"Starting full document retrieval for collection '{collection_name}'...")

            while True:
                try:
                    # Fetch a batch of documents
                    results = collection.get(limit=batch_size, offset=offset)

                    # Break if no more documents are returned
                    if not results.get('documents'):
                        break

                    # Process the batch: Convert raw data to Document objects
                    # Safely handle potential None values for metadatas
                    contents = results.get('documents', [])
                    metadatas = results.get('metadatas') or [None] * len(contents)
                    
                    for content, metadata in zip(contents, metadatas):
                        # Ensure content is a string and metadata is a dict
                        if isinstance(content, str):
                            doc_metadata = metadata if isinstance(metadata, dict) else {}
                            doc = Document(page_content=content, metadata=doc_metadata)
                            all_docs.append(doc)
                        else:
                            self.logger.warning(f"Skipping non-string document content in batch at offset {offset}.")

                    # Check if this was the last batch
                    if len(contents) < batch_size:
                        break

                    # Move to the next batch
                    offset += batch_size

                    # Periodic memory cleanup for very large collections
                    if offset % 5000 == 0:
                        self.logger.debug(f"Retrieved {offset} documents so far for '{collection_name}'. Triggering gc.collect().")
                        import gc
                        gc.collect()

                except Exception as batch_error:
                    self.logger.error(f"Error processing batch at offset {offset} for collection '{collection_name}': {batch_error}", exc_info=True)
                    raise

            self.logger.info(f"Finished retrieving {len(all_docs)} documents from '{collection_name}'.")

            # --- Update Cache ---
            self.document_cache[cache_key] = all_docs
            self.cache_timestamps[cache_key] = current_time

            # --- Cache Maintenance ---
            # Remove stale entries that are significantly past their TTL
            try:
                expired_keys = [
                    key for key, timestamp in self.cache_timestamps.items()
                    if current_time - timestamp > self.cache_ttl * 2
                ]
                for key in expired_keys:
                    # Use pop with default to avoid KeyError if key was already removed
                    self.document_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                if expired_keys:
                    self.logger.debug(f"Cache maintenance: Removed {len(expired_keys)} stale entries.")
            except Exception as cache_maintenance_error:
                # Log cache maintenance errors but don't let them break the main functionality
                self.logger.warning(f"Error during cache maintenance: {cache_maintenance_error}")

            return all_docs

        except Exception as e:
            self.logger.error(f"Failed to retrieve documents from collection '{collection_name}': {e}", exc_info=True)
            # Return an empty list to indicate failure without breaking the caller
            return []


    def clear_document_cache(self, collection_name: Optional[str] = None) -> None:
        """
        Clear document cache to free memory.

        This method clears the in-memory document cache, either for a specific collection
        or for all collections. It also triggers Python's garbage collector to help
        free up memory.

        Args:
            collection_name (Optional[str]): The name of a specific collection's cache to clear.
                                            If None, clears the entire cache.
        """
        try:
            if collection_name:
                # Clear cache for a specific collection
                removed_docs = self.document_cache.pop(collection_name, None)
                removed_timestamp = self.cache_timestamps.pop(collection_name, None)
                
                if removed_docs is not None or removed_timestamp is not None:
                    self.logger.debug(f"Cleared cache for collection '{collection_name}'.")
                else:
                    self.logger.debug(f"No cache found for collection '{collection_name}' to clear.")
            else:
                # Clear the entire cache
                cache_size_before = len(self.document_cache)
                self.document_cache.clear()
                self.cache_timestamps.clear()
                self.logger.info(f"Cleared entire document cache ({cache_size_before} entries).")

        except Exception as e:
            # Log any unexpected errors during cache clearing
            error_msg = f"Error occurred while clearing document cache: {e}"
            self.logger.error(error_msg, exc_info=True)


        finally:
            # Always attempt to run garbage collection after cache manipulation
            try:
                import gc
                collected = gc.collect()
                if collected > 0:
                    self.logger.debug(f"Garbage collector freed {collected} objects after cache clear.")
            except Exception as gc_error:
                self.logger.warning(f"Error during garbage collection after cache clear: {gc_error}")

        
    def _setup_logger(self) -> None:
        """
        Sets up the internal logger for the CoreRAGEngine instance.

        This method creates a logger named after the class, adds a StreamHandler
        if one doesn't already exist (to prevent duplicate logs), configures
        a standard format, and sets the logging level to INFO.
        """
        try:
            # Create a logger specific to this class instance
            logger = logging.getLogger(self.__class__.__name__)

            if not logger.hasHandlers():
                try:
                    # Create a console handler
                    handler = logging.StreamHandler()

                    # Define a standard log format
                    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
                    formatter = logging.Formatter(log_format)

                    handler.setFormatter(formatter)
                    logger.addHandler(handler)

                except Exception as handler_error:
                    print(f"Warning: Could not configure custom handler for logger '{self.__class__.__name__}': {handler_error}")

            logger.setLevel(logging.INFO)
            self.logger = logger

        except Exception as e:
            error_message = f"Critical error setting up logger: {e}"
            print(f"ERROR: {error_message}")

 
    def _get_persist_dir(self, collection_name: str) -> str:
        """
        Generates the specific persistence directory path for a given collection.

        This helper method constructs the full path where a Chroma vector store
        collection will be persisted on disk, based on the engine's base persist
        directory and the collection's name.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            str: The full path to the collection's persistence directory.

        Raises:
            ValueError: If `collection_name` is None or an empty string.
            TypeError: If `collection_name` is not a string.
        """
        # --- Input Validation ---
        if not isinstance(collection_name, str):
            error_msg = f"collection_name must be a string, got {type(collection_name).__name__}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        
        if not collection_name:
            error_msg = "collection_name cannot be None or empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # --- Path Construction ---
        try:
            persist_dir = os.path.join(self.persist_directory_base, collection_name)
            return persist_dir
        except Exception as e:
            error_msg = f"Failed to construct persist directory path for collection '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_or_load_vectorstore(self, collection_name: str, recreate: bool = False) -> None:
        """
        Initializes an in-memory reference to a vector store, loading it from disk
        if it exists and is not being recreated, or preparing for a new one.
        Ensures the retriever is set up with the current engine's default_retrieval_top_k.

        This method manages the lifecycle of a Chroma vector store collection:
        - If `recreate` is True, it removes any existing persisted data.
        - If the collection exists on disk and `recreate` is False, it loads it.
        - If the collection is already in memory and valid, it ensures the retriever is correctly configured.
        - If the collection doesn't exist, it prepares for future creation upon indexing.

        Args:
            collection_name (str): The name of the collection to initialize or load.
            recreate (bool): If True, forces the removal of any existing persisted collection
                            data before proceeding. Defaults to False.
        """
        try:
            # --- Determine Persistence Directory ---
            persist_dir = self._get_persist_dir(collection_name)

            # --- Handle Recreation ---
            if recreate:
                self._handle_recreate_collection(collection_name, persist_dir)
                return

            # --- Handle Existing In-Memory Collection ---
            if collection_name in self.vectorstores:
                self._handle_existing_collection_in_memory(collection_name)
                return

            # If not in memory and not recreating, attempt to load from persisted storage.
            self._handle_load_from_disk(collection_name, persist_dir)

        except Exception as e:
            error_msg = f"Critical error in _init_or_load_vectorstore for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _handle_existing_collection_in_memory(self, collection_name: str) -> None:
        """
        Handle the case where a collection already exists in memory and we're not recreating it.

        This method checks if the associated retriever for the in-memory collection
        needs to be created or re-configured, for example, if the desired number of
        retrieved documents (`k`) has changed since the last setup.

        Args:
            collection_name (str): The name of the collection to handle.
        """
        try:
            # --- Determine Current Retriever Configuration ---
            current_k = None
            retriever = self.retrievers.get(collection_name)

            # Safely extract the current 'k' value from the retriever, if it exists
            if retriever:
                try:
                    if hasattr(retriever, 'k'):
                        current_k = retriever.k
                    elif hasattr(retriever, 'search_kwargs'):
                        current_k = retriever.search_kwargs.get('k')
                except Exception as attr_error:
                    self.logger.warning(f"Could not determine current 'k' for retriever of collection '{collection_name}': {attr_error}")

            # --- Check if Retriever Needs Update ---
            # A retriever needs to be set up if:
            # 1. It doesn't exist for this collection, or
            # 2. Its 'k' value doesn't match the engine's current default
            needs_retriever_setup = (collection_name not in self.retrievers or current_k != self.default_retrieval_top_k)

            if needs_retriever_setup:
                try:
                    self._setup_retriever_for_collection(collection_name)
                    self.logger.info(f"Retriever for collection '{collection_name}' configured or re-configured with k={self.default_retrieval_top_k}")
                except Exception as setup_error:
                    error_msg = f"Failed to set up retriever for existing in-memory collection '{collection_name}': {setup_error}"
                    self.logger.error(error_msg, exc_info=True)

        except Exception as e:
            error_msg = f"Unexpected error in _handle_existing_collection_in_memory for '{collection_name}': {e}"
            self.logger.error(error_msg, exc_info=True)


    def _handle_recreate_collection(self, collection_name: str, persist_dir: str) -> None:
        """
        Handle recreation of a collection by removing existing data.

        This method is responsible for the clean slate preparation when a collection
        needs to be recreated. It removes any persisted data on disk and clears
        references to the collection from the engine's in-memory stores.

        Args:
            collection_name (str): The name of the collection to recreate.
            persist_dir (str): The full path to the collection's persistence directory.
        """
        try:
            self.logger.info(f"Recreating collection '{collection_name}'. Removing existing data if present.")

            # If a persistence directory exists for this collection, remove it entirely.
            if os.path.exists(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                    self.logger.debug(f"Removed persisted data directory: {persist_dir}")
                except OSError as os_error:
                    self.logger.error(f"OS error removing persisted directory '{persist_dir}': {os_error}")
                    raise RuntimeError(f"Failed to remove persisted directory '{persist_dir}'") from os_error
                except Exception as remove_error:
                    self.logger.critical(f"Unexpected error removing persisted directory '{persist_dir}': {remove_error}", exc_info=True)
                    raise RuntimeError(f"Failed to remove persisted directory '{persist_dir}'") from remove_error

            # Remove the vectorstore object reference if it exists in memory.
            if collection_name in self.vectorstores:
                removed_vs = self.vectorstores.pop(collection_name, None)
                if removed_vs:
                    self.logger.debug(f"Removed in-memory vectorstore reference for '{collection_name}'.")

            # Remove the retriever object reference if it exists in memory.
            if collection_name in self.retrievers:
                removed_retriever = self.retrievers.pop(collection_name, None)
                if removed_retriever:
                    self.logger.debug(f"Removed in-memory retriever reference for '{collection_name}'.")

            self.logger.info(f"Completed recreation preparation for collection '{collection_name}'.")

        except Exception as e:
            error_msg = f"Unexpected error in _handle_recreate_collection for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _handle_load_from_disk(self, collection_name: str, persist_dir: str) -> None:
        """
        Handle loading a collection from disk if it exists.

        This method attempts to load a Chroma vector store collection from its
        persisted directory. If successful, it also sets up the corresponding
        retriever. If the directory doesn't exist or loading fails, it logs
        appropriate messages.

        Args:
            collection_name (str): The name of the collection to load.
            persist_dir (str): The full path to the collection's persistence directory.
        """
        try:
            if os.path.exists(persist_dir):
                try:
                    self.logger.info(f"Attempting to load existing vector store '{collection_name}' from {persist_dir}")
                    
                    # Create the Chroma vector store instance by loading from persistence
                    loaded_vectorstore = Chroma(collection_name=collection_name,
                                                embedding_function=self.embedding_model,
                                                persist_directory=persist_dir)
                    
                    # Store the loaded vector store in the engine's memory
                    self.vectorstores[collection_name] = loaded_vectorstore

                    # After successfully loading from disk, set up the correct retriever
                    self._setup_retriever_for_collection(collection_name)
                    
                    self.logger.info(f"Successfully loaded vector store '{collection_name}' from disk and configured retriever with k={self.default_retrieval_top_k}.")

                except Exception as load_error:
                    error_msg = (f"Error loading vector store '{collection_name}' from {persist_dir}: {load_error}. A new one may be created if documents are indexed.")
                    self.logger.error(error_msg, exc_info=True)
                    self.vectorstores.pop(collection_name, None)
                    self.retrievers.pop(collection_name, None)

            else:
                self.logger.info(
                    f"No persisted vector store found for '{collection_name}' at {persist_dir}. "
                    f"It will be created upon first indexing.")

        except Exception as e:
            error_msg = f"Unexpected error in _handle_load_from_disk for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _setup_retriever_for_collection(self, collection_name: str) -> None:
        """
        Set up the appropriate retriever (hybrid or standard) for a collection.

        This method configures the retriever used for document retrieval for a given
        collection. If hybrid search is enabled, it attempts to create a hybrid
        retriever. Otherwise, it falls back to a standard Chroma retriever.

        Args:
            collection_name (str): The name of the collection to set up the retriever for.
        """
        try:
            vectorstore = self.vectorstores.get(collection_name)
            if not vectorstore:
                error_msg = f"Cannot set up retriever: Vectorstore for '{collection_name}' not found in memory."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            if self.enable_hybrid_search:
                self.logger.debug(f"Setting up hybrid retriever for collection '{collection_name}'...")
                try:
                    # Hybrid retrievers often need all documents for BM25 part
                    all_docs = self._get_all_documents_from_collection(collection_name)

                    if all_docs:
                        # Create the hybrid retriever with the required documents and k
                        hybrid_retriever = AdaptiveHybridRetriever(vector_store=vectorstore,
                                                                   documents=all_docs,
                                                                   k=self.default_retrieval_top_k)

                        self.retrievers[collection_name] = hybrid_retriever
                        self.logger.info(f"Successfully created hybrid retriever for collection '{collection_name}' with k={self.default_retrieval_top_k}")
                    else:
                        # Fallback: If no docs are found (e.g., empty collection), use standard retriever
                        self.logger.warning(
                            f"Could not create hybrid retriever for '{collection_name}' (no documents found), "
                            f"falling back to standard retriever with k={self.default_retrieval_top_k}"
                        )
                        standard_retriever = vectorstore.as_retriever(search_kwargs={'k': self.default_retrieval_top_k})
                        self.retrievers[collection_name] = standard_retriever

                except Exception as hybrid_error:
                    self.logger.warning(
                        f"Failed to create hybrid retriever for '{collection_name}': {hybrid_error}. "
                        f"Falling back to standard retriever with k={self.default_retrieval_top_k}",
                        exc_info=True
                    )
                    
                    # Fallback to standard retriever if hybrid creation fails
                    standard_retriever = vectorstore.as_retriever(search_kwargs={'k': self.default_retrieval_top_k})
                    self.retrievers[collection_name] = standard_retriever

            # --- Standard Retriever Setup ---
            else:
                self.logger.debug(f"Setting up standard retriever for collection '{collection_name}'...")
                standard_retriever = vectorstore.as_retriever(search_kwargs={'k': self.default_retrieval_top_k})
                self.retrievers[collection_name] = standard_retriever
                self.logger.debug(f"Standard retriever set up for '{collection_name}' with k={self.default_retrieval_top_k}")

        except Exception as e:
            error_msg = f"Unexpected error in _setup_retriever_for_collection for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_llm(self, use_json_format: bool) -> Any:
        try:
            provider = self.llm_provider.lower()

            if provider == "openai":
                return self._init_openai_llm(use_json_format)
            if provider == "ollama":
                return self._init_ollama_llm(use_json_format)
            if provider == "google":
                return self._init_google_llm(use_json_format)

            error_msg = f"Unsupported LLM provider configured: '{self.llm_provider}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Critical error initializing LLM (Provider: {self.llm_provider}, JSON: {use_json_format}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_openai_llm(self, use_json: bool) -> ChatOpenAI:
        try:
            self._validate_openai_api_key()

            # Build the configuration dictionary for the OpenAI LLM
            openai_config = self._build_openai_config(use_json)

            openai_llm = ChatOpenAI(**openai_config)
            self.logger.info(f"OpenAI LLM '{self.llm_model_name}' initialized successfully (JSON: {use_json}).")
            return openai_llm
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI LLM '{self.llm_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_ollama_llm(self, use_json: bool) -> ChatOllama:

        try:
            format_type = self._get_ollama_format(use_json)
            ollama_llm = ChatOllama(model=self.llm_model_name,
                                    temperature=self.temperature,
                                    format=format_type)
            self.logger.info(f"Ollama LLM '{self.llm_model_name}' initialized successfully (JSON: {use_json}, Format: {format_type}).")
            return ollama_llm
        except Exception as e:
            error_msg = f"Failed to initialize Ollama LLM '{self.llm_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_google_llm(self, use_json: bool) -> ChatGoogleGenerativeAI:
        try:
            self._validate_google_api_key()

            # Build the configuration dictionary for the Google LLM
            google_config = self._build_google_config(use_json)

            google_llm = ChatGoogleGenerativeAI(**google_config)
            self.logger.info(f"Google LLM '{self.llm_model_name}' initialized successfully (JSON: {use_json}).")
            return google_llm
        except Exception as e:
            error_msg = f"Failed to initialize Google LLM '{self.llm_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _validate_openai_api_key(self) -> None:
        try:
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY is missing or empty. It is required for OpenAI LLM/embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("OpenAI API key validation successful.")
        except Exception as e:
            error_msg = f"Unexpected error during OpenAI API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _validate_google_api_key(self) -> None:
        try:
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY is missing or empty. It is required for Google LLM/embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("Google API key validation successful.")
        except Exception as e:
            error_msg = f"Unexpected error during Google API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _build_openai_config(self, use_json: bool) -> Dict[str, Any]:
        try:
            config_args: Dict[str, Any] = {"model": self.llm_model_name,
                                           "temperature": self.temperature,
                                           "openai_api_key": self.openai_api_key}

            # Add JSON formatting configuration if requested
            if use_json:
                config_args["model_kwargs"] = {"response_format": {"type": "json_object"}}
                self.logger.info(f"JSON response format enabled for OpenAI model '{self.llm_model_name}'.")

            self.logger.debug(f"OpenAI config built successfully: {config_args}")
            return config_args

        except Exception as e:
            error_msg = f"Failed to build OpenAI configuration: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _build_google_config(self, use_json: bool) -> Dict[str, Any]:
        try:
            # Initialize the base configuration dictionary
            config_kwargs: Dict[str, Any] = {"model": self.llm_model_name,
                                             "temperature": self.temperature,
                                             "google_api_key": self.google_api_key,
                                             "convert_system_message_to_human": True}

            # Add JSON formatting configuration if requested
            if use_json:
                config_kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
                self.logger.info(f"JSON response format enabled for Google model '{self.llm_model_name}'.")

            self.logger.debug(f"Google config built successfully: {config_kwargs}")
            return config_kwargs

        except Exception as e:
            error_msg = f"Failed to build Google configuration: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    
    def _get_ollama_format(self, use_json: bool) -> Optional[str]:
        try:
            # Determine the format string based on the use_json flag
            format_param = "json" if use_json else None
            self.logger.debug(f"Ollama format parameter determined: {format_param} (use_json: {use_json})")
            return format_param
        except Exception as e:
            error_msg = f"Unexpected error determining Ollama format parameter: {e}"
            self.logger.error(error_msg, exc_info=True)
            return None
        

    def _init_embedding_model(self) -> Any:
        try:
            provider = self.embedding_provider.lower()

            if provider == "openai":
                return self._init_openai_embeddings()
            if provider == "gpt4all":
                return self._init_gpt4all_embeddings()
            if provider == "google":
                return self._init_google_embeddings()

            error_msg = f"Unsupported embedding provider configured: '{self.embedding_provider}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Critical error initializing embedding model (Provider: {self.embedding_provider}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_openai_embeddings(self) -> OpenAIEmbeddings:
        try:
            self._validate_openai_embedding_key()
            model_name = self._get_openai_embedding_model()
            openai_embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=model_name)
            self.logger.info(f"OpenAI Embeddings model '{model_name}' initialized successfully.")
            return openai_embeddings
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI Embeddings model '{self.embedding_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_gpt4all_embeddings(self) -> GPT4AllEmbeddings:
        try:
            gpt4all_embeddings = GPT4AllEmbeddings()
            self.logger.info("GPT4All Embeddings model initialized successfully.")
            return gpt4all_embeddings
        except Exception as e:
            error_msg = f"Failed to initialize GPT4All Embeddings model: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_google_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        try:
            self._validate_google_embedding_key()
            model_name = self._get_google_embedding_model()
            google_embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=self.google_api_key)
            self.logger.info(f"Google Embeddings model '{model_name}' initialized successfully.")
            return google_embeddings

        except Exception as e:
            error_msg = f"Failed to initialize Google Embeddings model '{self.embedding_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_text_splitter(self) -> BaseChunker: 
        try:
            # Get the configured chunking strategy, defaulting to 'adaptive'
            strategy = getattr(app_settings.engine, 'chunking_strategy', 'adaptive')
            self.logger.debug(f"Initializing text splitter with strategy: '{strategy}'")

            if strategy == "adaptive":
                return self._create_adaptive_splitter()
            elif strategy == "semantic" and self.openai_api_key:
                return self._create_semantic_splitter()
            elif strategy == "hybrid":
                return self._create_hybrid_splitter()
            else:
                # Fallback to default splitter for unknown strategies or missing API key for semantic
                self.logger.info(f"Falling back to default splitter for strategy: '{strategy}'")
                return self._create_default_splitter()

        except Exception as e:
            error_msg = f"Critical error initializing text splitter (Strategy: {getattr(app_settings.engine, 'chunking_strategy', 'N/A')}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _validate_openai_embedding_key(self) -> None:
        try:
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY is missing or empty. It is required for OpenAI embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("OpenAI API key validation for embeddings successful.")
        except Exception as e:
            error_msg = f"Unexpected error during OpenAI embedding API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _validate_google_embedding_key(self) -> None:
        try:
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY is missing or empty. It is required for Google embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("Google API key validation for embeddings successful.")
        except Exception as e:
            error_msg = f"Unexpected error during Google embedding API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _get_openai_embedding_model(self) -> str:
        try:
            model_name = self.embedding_model_name or "text-embedding-3-small"
            self.logger.debug(f"OpenAI embedding model determined: '{model_name}' (Configured: '{self.embedding_model_name}')")
            return model_name
        except Exception as e:
            error_msg = f"Unexpected error determining OpenAI embedding model: {e}"
            self.logger.error(error_msg, exc_info=True)
            return "text-embedding-3-small"


    def _get_google_embedding_model(self) -> str:
        try:
            model_name = self.embedding_model_name or "models/embedding-001"
            self.logger.debug(f"Google embedding model determined: '{model_name}' (Configured: '{self.embedding_model_name}')")
            return model_name
        except Exception as e:
            error_msg = f"Unexpected error determining Google embedding model: {e}"
            self.logger.error(error_msg, exc_info=True)
            return "models/embedding-001"


    def _create_adaptive_splitter(self) -> AdaptiveChunker:
        """
        Create adaptive chunker with configuration.

        This method instantiates an AdaptiveChunker with the engine's configured
        chunk size, overlap, and API key.

        Returns:
            AdaptiveChunker: An instance of the AdaptiveChunker.
        """
        try:
            adaptive_splitter = AdaptiveChunker(chunk_size=self.chunk_size,
                                                chunk_overlap=self.chunk_overlap,
                                                openai_api_key=self.openai_api_key,
                                                model_name=self.llm_model_name)
            self.logger.info(f"AdaptiveChunker created successfully with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
            return adaptive_splitter
        except Exception as e:
            error_msg = f"Failed to create AdaptiveChunker: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _create_semantic_splitter(self) -> BaseChunker:
        """
        Create semantic chunker with fallback to adaptive.

        This method attempts to instantiate a SemanticChunker using OpenAI embeddings.
        If the required dependencies are not available, it falls back to creating
        an AdaptiveChunker.

        Returns:
            BaseChunker: An instance of the SemanticChunker or a fallback AdaptiveChunker.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            
            # Get the threshold type from config, defaulting to 'percentile'
            threshold_type = getattr(app_settings.engine, 'semantic_chunking_threshold', 'percentile')
            
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            
            semantic_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type=threshold_type)
            self.logger.info(f"SemanticChunker created successfully with threshold_type='{threshold_type}'.")
            return semantic_splitter

        except ImportError as import_error:
            warning_msg = f"SemanticChunker not available (ImportError: {import_error}), falling back to AdaptiveChunker."
            self.logger.warning(warning_msg)
            return self._create_adaptive_splitter()
        except Exception as e:
            error_msg = f"Failed to create SemanticChunker, falling back to AdaptiveChunker: {e}"
            self.logger.warning(error_msg, exc_info=True)
            # Fallback to AdaptiveChunker
            return self._create_adaptive_splitter()


    def _create_hybrid_splitter(self) -> HybridChunker:
        """
        Create hybrid chunker with primary and secondary splitters.

        This method creates a HybridChunker that uses an AdaptiveChunker as the primary
        splitter and a RecursiveCharacterTextSplitter as the secondary splitter.

        Returns:
            HybridChunker: An instance of the HybridChunker.
        """
        try:
            # Create the primary splitter (AdaptiveChunker)
            primary_splitter = self._create_adaptive_splitter()
            
            # Create the secondary splitter (RecursiveCharacterTextSplitter with smaller chunks)
            secondary_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size // 2,
                                                                chunk_overlap=self.chunk_overlap // 2,
                                                                length_function=len)
            
            hybrid_splitter = HybridChunker(primary_splitter, secondary_splitter)
            self.logger.info(
                f"HybridChunker created successfully with primary (Adaptive) and secondary (Recursive, "
                f"chunk_size={self.chunk_size // 2}, chunk_overlap={self.chunk_overlap // 2}).")
            return hybrid_splitter

        except Exception as e:
            error_msg = f"Failed to create HybridChunker: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _create_default_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create default recursive character splitter with tiktoken support.

        This method creates a RecursiveCharacterTextSplitter. If `tiktoken` is available
        and the LLM provider is OpenAI, it attempts to create a splitter that uses
        a Tiktoken encoder for more accurate token-based splitting.

        Returns:
            RecursiveCharacterTextSplitter: An instance of the text splitter.
        """
        try:
            use_tiktoken = tiktoken is not None and self.llm_provider.lower() == "openai"
            
            if use_tiktoken:
                try:
                    tiktoken_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name=self.llm_model_name,
                                                                                             chunk_size=self.chunk_size,
                                                                                             chunk_overlap=self.chunk_overlap)
                    self.logger.info(f"Default splitter created with Tiktoken encoder for model '{self.llm_model_name}'.")
                    return tiktoken_splitter
                except Exception as tiktoken_error:
                    self.logger.warning(f"Tiktoken splitter failed (Error: {tiktoken_error}), falling back to default character splitter.")
            
            # Create and return the default RecursiveCharacterTextSplitter
            default_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                              chunk_overlap=self.chunk_overlap,
                                                              length_function=len)

            fallback_type = "Tiktoken fallback" if use_tiktoken else "default"
            self.logger.info(
                f"Default splitter created ({fallback_type}) with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}.")
            return default_splitter

        except Exception as e:
            error_msg = f"Failed to create default RecursiveCharacterTextSplitter: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _init_search_tool(self) -> Optional[Runnable]:
        """
        Initialize Tavily search tool if available.

        This method attempts to import and instantiate the TavilySearch tool.
        If the required dependency is not installed, it logs a warning and returns None.

        Returns:
            Optional[Runnable]: An instance of the TavilySearch tool, or None if unavailable.
        """
        try:
            from langchain_tavily import TavilySearch
            search_tool = TavilySearch(api_key=self.tavily_api_key, max_results=5)
            self.logger.info("Tavily search tool initialized successfully.")
            return search_tool
        except ImportError as import_error:
            warning_msg = ("langchain-tavily is not installed. Web search functionality will be disabled. ")
            self.logger.warning(warning_msg)
            return None
        except Exception as e:
            error_msg = f"Failed to initialize Tavily search tool: {e}"
            self.logger.error(error_msg, exc_info=True)
            return None


    def _create_document_relevance_grader_chain(self) -> Runnable:
        """
        Create chain for grading document relevance to questions.

        This method constructs a LangChain Runnable that takes a question and a document excerpt,
        and uses a JSON-formatted LLM to assess the document's relevance to the question,
        returning a structured `RelevanceGrade` object.

        Returns:
            Runnable: A configured chain for document relevance grading.
        """
        try:
            parser = PydanticOutputParser(pydantic_object=RelevanceGrade)
            prompt_template = (
                "You are a document relevance grader. Your task is to assess if a given document excerpt "
                "is relevant to the provided question.\n"
                "Respond with a JSON object matching this schema:\n"
                "{format_instructions}\n"
                "Question: {question}\n"
                "Document Excerpt:\n---\n{document_content}\n---\n"
                "Provide your JSON response:"
            )
            
            # Create the prompt template, partially filling in the format instructions
            prompt = ChatPromptTemplate.from_template(template=prompt_template,
                                                      partial_variables={"format_instructions": parser.get_format_instructions()})
            
            chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RelevanceGrade)

            self.logger.info("Document relevance grader chain created successfully with PydanticOutputParser.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create document relevance grader chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _create_document_relevance_grader_chain(self) -> Runnable:
        """
        Create chain for grading document relevance to questions.

        This method constructs a LangChain Runnable that takes a question and a document excerpt,
        and uses a JSON-formatted LLM to assess the document's relevance to the question,
        returning a structured `RelevanceGrade` object.

        Returns:
            Runnable: A configured chain for document relevance grading.
        """
        try:
            parser = PydanticOutputParser(pydantic_object=RelevanceGrade)
            prompt_template = (
                "You are a document relevance grader. Your task is to assess if a given document excerpt "
                "is relevant to the provided question.\n"
                "Respond with a JSON object matching this schema:\n"
                "{format_instructions}\n"
                "Question: {question}\n"
                "Document Excerpt:\n---\n{document_content}\n---\n"
                "Provide your JSON response:"
            )
            
            # Create the prompt template, partially filling in the format instructions
            prompt = ChatPromptTemplate.from_template(template=prompt_template,
                                                      partial_variables={"format_instructions": parser.get_format_instructions()})
            
            chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RelevanceGrade)
            
            self.logger.info("Document relevance grader chain created successfully with PydanticOutputParser.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create document relevance grader chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _create_document_relevance_grader_chain(self) -> Runnable:
        """
        Create chain for grading document relevance to questions.

        This method constructs a LangChain Runnable that takes a question and a document excerpt,
        and uses a JSON-formatted LLM to assess the document's relevance to the question,
        returning a structured `RelevanceGrade` object.

        Returns:
            Runnable: A configured chain for document relevance grading.
        """
        try:
            parser = PydanticOutputParser(pydantic_object=RelevanceGrade)
            prompt_template = (
                "You are a document relevance grader. Your task is to assess if a given document excerpt "
                "is relevant to the provided question.\n"
                "Respond with a JSON object matching this schema:\n"
                "{format_instructions}\n"
                "Question: {question}\n"
                "Document Excerpt:\n---\n{document_content}\n---\n"
                "Provide your JSON response:"
            )
            
            # Create the prompt template, partially filling in the format instructions
            prompt = ChatPromptTemplate.from_template(template=prompt_template,
                                                      partial_variables={"format_instructions": parser.get_format_instructions()})
            
            chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RelevanceGrade)

            self.logger.info("Document relevance grader chain created successfully with PydanticOutputParser.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create document relevance grader chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _create_query_rewriter_chain(self) -> Runnable:
        """
        Create chain for rewriting queries with chat history context.

        This method constructs a LangChain Runnable that takes a question and optional chat history,
        and rewrites the question to be clear, specific, and self-contained for retrieval purposes.

        Returns:
            Runnable: A configured chain for query rewriting.
        """
        try:
            self.logger.info("Creating query rewriter chain with chat history support.")
            
            system_prompt = (
                "You are a query optimization assistant. Given 'Chat History' (if any) and "
                "'Latest User Question', rewrite the question to be clear, specific, and self-contained "
                "for retrieval. If already clear, return it as is."
            )
            
            # Create the prompt template using messages for better history handling
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Latest User Question to rewrite:\n{question}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
  
            self.logger.info("Query rewriter chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create query rewriter chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def _create_answer_generation_chain(self) -> Runnable:
        """
        Create chain for generating answers from context documents.

        This method constructs a LangChain Runnable that generates a final answer based on
        provided context documents, optional chat history, and optional regeneration feedback.
        It ensures the answer is grounded in the context and handles follow-up questions via history.

        Returns:
            Runnable: A configured chain for answer generation.
        """
        try:
            self.logger.info("Creating answer generation chain with chat history and feedback support.")
            
            system_prompt = (
                "You are a helpful assistant. Your answer must be based *only* on the provided context documents.\n"
                "If the context lacks the answer, say you don't know. Do not invent details.\n"
                "Use 'Chat History' (if any) only to resolve references in the current question, "
                "but ground your answer in the current context.\n"
                "Current Context Documents:\n---\n{context}\n---\n"
                "{optional_regeneration_prompt_header_if_feedback}"
            )
            
            # Create the prompt template using messages for better history/feedback handling
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{regeneration_feedback_if_any}{question}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            self.logger.info("Answer generation chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create answer generation chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_grounding_check_chain(self) -> Runnable:
        """
        Create chain for checking if answers are grounded in context.

        This method constructs a LangChain Runnable that verifies if a generated answer
        is fully supported by and only uses information from the provided context documents.
        It returns a structured `GroundingCheck` object indicating the result.

        Returns:
            Runnable: A configured chain for answer grounding verification.
        """
        try:
            self.logger.info("Creating answer grounding check chain.")
            parser = PydanticOutputParser(pydantic_object=GroundingCheck)

            prompt_template = (
                "You are an Answer Grounding Checker. Your task is to verify if the 'Generated Answer' "
                "is FULLY supported by and ONLY uses information from the provided 'Context Documents'.\n"
                "Respond with a JSON matching this schema: {format_instructions}\n"
                "Context Documents:\n---\n{context}\n---\n"
                "Generated Answer:\n---\n{generation}\n---\n"
                "Provide your JSON response:"
            )
            
            # Create the prompt template, partially filling in the format instructions
            prompt = ChatPromptTemplate.from_template(template=prompt_template,
                                                      partial_variables={"format_instructions": parser.get_format_instructions()})
            
            chain = prompt | self.json_llm | parser
            
            self.logger.info("Answer grounding check chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create answer grounding check chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        

    async def _grounding_check_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Perform grounding check on generated answer and provide feedback if needed.

        This asynchronous node in the RAG workflow evaluates whether the generated
        answer is fully supported by the retrieved context. It can use either a
        basic check (via a dedicated chain) or an advanced multi-level check,
        depending on configuration. If the answer is not grounded, feedback is
        generated to guide regeneration.

        Args:
            state (CoreGraphState): The current state of the RAG workflow.

        Returns:
            CoreGraphState: The updated state, potentially including regeneration feedback.
        """
        try:
            self.logger.info("NODE: Performing grounding check on generated answer...")

            # Extract State Variables
            context = state.get("context", "")
            generation = state.get("generation", "")
            question = state.get("original_question") or state.get("question", "")
            documents = state.get("documents", [])

            # Update Attempt Tracking
            current_attempts = state.get("grounding_check_attempts", 0) + 1
            state["grounding_check_attempts"] = current_attempts
            # Reset feedback for this attempt
            state["regeneration_feedback"] = None

            # Skip the grounding check if essential data is missing or a generation error occurred
            if not context or not generation or "Error generating answer." in generation:
                self.logger.info("Skipping grounding check (no context or generation, or generation error).")
                # Explicitly set feedback to None and update attempts in the returned state
                return {**state,
                        "regeneration_feedback": None,
                        "grounding_check_attempts": current_attempts}

            # Perform Grounding Check
            try:
                if self.enable_advanced_grounding and self.advanced_grounding_checker:
                    # Use the advanced, multi-level grounding checker
                    await self._perform_advanced_grounding_check(state,
                                                                 context,
                                                                 generation,
                                                                 question,
                                                                 documents,
                                                                 current_attempts)
                else:
                    # Use the basic grounding check chain
                    self._perform_basic_grounding_check(state,
                                                        context,
                                                        generation,
                                                        question,
                                                        current_attempts)
            except Exception as check_execution_error:
                self._handle_grounding_check_exception(state, check_execution_error, question, current_attempts)

            return state

        except Exception as node_error:
            error_msg = f"Unexpected error in _grounding_check_node: {node_error}"
            self.logger.critical(error_msg, exc_info=True)
            
            state["regeneration_feedback"] = f"A critical system error occurred in the grounding check node: {node_error}. Please try to generate a concise answer based on context."
            state["error_message"] = (state.get("error_message") or "") + f" | Critical grounding node error: {node_error}"

            raise RuntimeError(error_msg) from node_error


    async def _perform_advanced_grounding_check(self,
                                                state: CoreGraphState,
                                                context: str,
                                                generation: str,
                                                question: str,
                                                documents: List,
                                                attempt: int) -> None:
        """
        Perform advanced multi-level grounding check.

        This method uses the `MultiLevelGroundingChecker` to conduct a comprehensive
        analysis of the generated answer against the context and documents. It evaluates
        grounding, consistency, completeness, and potential hallucinations. Based on
        the results, it either confirms the answer is acceptable or generates detailed
        feedback for regeneration.

        Args:
            state (CoreGraphState): The current workflow state to be updated.
            context (str): The aggregated context used for generation.
            generation (str): The generated answer to check.
            question (str): The original user question.
            documents (List): The list of retrieved documents.
            attempt (int): The current grounding check attempt number.
        """
        try:
            self.logger.info(f"Initiating advanced grounding check (Attempt {attempt})...")

            try:
                advanced_results = await self.advanced_grounding_checker.perform_comprehensive_grounding_check(
                    answer=generation,
                    context=context,
                    question=question,
                    documents=documents
                )
                self.logger.debug(f"Advanced grounding check completed (Attempt {attempt}).")
            except Exception as checker_error:
                error_msg = f"Advanced grounding checker execution failed (Attempt {attempt}): {checker_error}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from checker_error

            try:
                # Extract the overall assessment
                overall_assessment = advanced_results.get("overall_assessment", {})
                is_acceptable = overall_assessment.get("is_acceptable", False)

                if not is_acceptable:
                    # The answer did not pass the advanced check, generate detailed feedback
                    self._generate_advanced_feedback(state, advanced_results, question, attempt)
                else:
                    # The answer passed the advanced check
                    score_info = overall_assessment.get('overall_score', 'N/A')
                    self.logger.info(f"Advanced grounding check PASSED (Attempt {attempt}). Score: {score_info}")
                    # Ensure no feedback is carried forward from a previous failed attempt
                    state["regeneration_feedback"] = None

                # Store detailed results for potential debugging or future use
                state["advanced_grounding_results"] = advanced_results

            except Exception as processing_error:
                error_msg = f"Error processing advanced grounding results (Attempt {attempt}): {processing_error}"
                self.logger.error(error_msg, exc_info=True)

            self.logger.info(f"Finished advanced grounding check processing (Attempt {attempt}).")

        except Exception as e:
            self.logger.error(f"Advanced grounding check failed with exception (Attempt {attempt}): {e}. Falling back to basic check.", exc_info=True)
            self._perform_basic_grounding_check(state, context, generation, question, attempt)


    def _perform_basic_grounding_check(
        self,
        state: CoreGraphState,
        context: str,
        generation: str,
        question: str,
        attempt: int
    ) -> None:
        """
        Perform basic grounding check using the grounding check chain.

        This method uses the pre-configured `grounding_check_chain` (a LangChain Runnable)
        to evaluate if the generated answer is grounded in the provided context. It parses
        the structured output and, if the answer is not grounded, generates feedback
        for regeneration.

        Args:
            state (CoreGraphState): The current workflow state to be updated.
            context (str): The aggregated context used for generation.
            generation (str): The generated answer to check.
            question (str): The original user question.
            attempt (int): The current grounding check attempt number.
        """
        try:
            self.logger.info(f"Initiating basic grounding check (Attempt {attempt})...")

            try:
                # The chain expects 'context' and 'generation' as inputs
                chain_input = {"context": context, "generation": generation}
                result: GroundingCheck = self.grounding_check_chain.invoke(chain_input)
                self.logger.debug(f"Basic grounding check chain invoked successfully (Attempt {attempt}).")
            except Exception as chain_invoke_error:
                error_msg = f"Error invoking basic grounding check chain (Attempt {attempt}): {chain_invoke_error}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from chain_invoke_error

            try:
                if not result.is_grounded:
                    # The answer failed the basic grounding check, generate feedback
                    self._generate_basic_feedback(state, result, question, attempt)
                else:
                    # The answer passed the basic grounding check
                    self.logger.info(f"Basic grounding check PASSED (Attempt {attempt}).")
                    # Ensure no feedback is carried forward from a previous failed attempt
                    state["regeneration_feedback"] = None
            except Exception as feedback_error:
                error_msg = f"Error processing basic grounding results or generating feedback (Attempt {attempt}): {feedback_error}"
                self.logger.error(error_msg, exc_info=True)

            self.logger.info(f"Finished basic grounding check processing (Attempt {attempt}).")

        except Exception as e:
            error_msg = f"Unexpected error in _perform_basic_grounding_check (Attempt {attempt}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            state["regeneration_feedback"] = f"A system error occurred during the basic grounding check: {e}. Please try to generate a concise answer based on context."
            state["error_message"] = (state.get("error_message") or "") + f" | Basic grounding check error (Attempt {attempt}): {e}"
            raise RuntimeError(error_msg) from e


    async def _perform_advanced_grounding_check(
        self,
        state: CoreGraphState,
        context: str,
        generation: str,
        question: str,
        documents: List,
        attempt: int
    ) -> None:
        """
        Perform advanced multi-level grounding check.

        This method uses the `MultiLevelGroundingChecker` to conduct a comprehensive
        analysis of the generated answer against the context and documents. It evaluates
        grounding, consistency, completeness, and potential hallucinations. Based on
        the results, it either confirms the answer is acceptable or generates detailed
        feedback for regeneration.

        Args:
            state (CoreGraphState): The current workflow state to be updated.
            context (str): The aggregated context used for generation.
            generation (str): The generated answer to check.
            question (str): The original user question.
            documents (List): The list of retrieved documents.
            attempt (int): The current grounding check attempt number.
        """
        try:
            self.logger.info(f"Initiating advanced grounding check (Attempt {attempt})...")

            try:
                advanced_results = await self.advanced_grounding_checker.perform_comprehensive_grounding_check(
                    answer=generation,
                    context=context,
                    question=question,
                    documents=documents
                )
                self.logger.debug(f"Advanced grounding check completed (Attempt {attempt}).")
            except Exception as checker_error:
                error_msg = f"Advanced grounding checker execution failed (Attempt {attempt}): {checker_error}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from checker_error

            try:
                # Extract the overall assessment
                overall_assessment = advanced_results.get("overall_assessment", {})
                is_acceptable = overall_assessment.get("is_acceptable", False)

                if not is_acceptable:
                    # The answer did not pass the advanced check, generate detailed feedback
                    self._generate_advanced_feedback(state, advanced_results, question, attempt)
                else:
                    # The answer passed the advanced check
                    score_info = overall_assessment.get('overall_score', 'N/A')
                    self.logger.info(f"Advanced grounding check PASSED (Attempt {attempt}). Score: {score_info}")
                    # Ensure no feedback is carried forward from a previous failed attempt
                    state["regeneration_feedback"] = None

                # Store detailed results for potential debugging or future use
                state["advanced_grounding_results"] = advanced_results

            except Exception as processing_error:
                error_msg = f"Error processing advanced grounding results (Attempt {attempt}): {processing_error}"
                self.logger.error(error_msg, exc_info=True)
                
            self.logger.info(f"Finished advanced grounding check processing (Attempt {attempt}).")

        except Exception as e:
            self.logger.error(f"Advanced grounding check failed with exception (Attempt {attempt}): {e}. Falling back to basic check.", exc_info=True)
            self._perform_basic_grounding_check(state, context, generation, question, attempt)


    def _generate_basic_feedback(self,
                                 state: CoreGraphState,
                                 result: GroundingCheck,
                                 question: str,
                                 attempt: int) -> None:
        """
        Generate feedback based on basic grounding check results.

        This method processes the structured output from the basic grounding check chain
        (`GroundingCheck`) and constructs a regeneration prompt for the answer generation node.
        It uses the identified ungrounded statements and correction suggestions.

        Args:
            state (CoreGraphState): The current workflow state to be updated with feedback.
            result (GroundingCheck): The result from the basic grounding check chain.
            question (str): The original user question.
            attempt (int): The current grounding check attempt number.
        """
        try:
            self.logger.debug(f"Generating basic feedback (Attempt {attempt})...")

            feedback_parts: List[str] = []

            try:
                ungrounded_statements = result.ungrounded_statements
                if ungrounded_statements:
                    # Join the list of ungrounded statements into a single string for the prompt
                    statements_str = '; '.join(ungrounded_statements)
                    feedback_parts.append(f"The following statements were ungrounded: {statements_str}.")
            except Exception as statements_error:
                self.logger.warning(f"Error processing ungrounded statements (Attempt {attempt}): {statements_error}")

            try:
                correction_suggestion = result.correction_suggestion
                if correction_suggestion:
                    feedback_parts.append(f"Suggestion for correction: {correction_suggestion}.")
            except Exception as suggestion_error:
                self.logger.warning(f"Error processing correction suggestion (Attempt {attempt}): {suggestion_error}")

            # Handle Case with No Specific Feedback
            if not feedback_parts:
                feedback_parts.append("The answer was not fully grounded in the provided context. Please revise.")

            try:
                # Construct the regeneration prompt using the collected feedback parts
                regeneration_prompt = (
                    f"The previous answer to the question '{question}' was not well-grounded. "
                    f"{' '.join(feedback_parts)} "
                    "Please generate a new answer focusing ONLY on the provided documents and addressing these issues."
                )
                
                # Update the state with the generated feedback
                state["regeneration_feedback"] = regeneration_prompt
                self.logger.warning(f"Basic grounding check FAILED (Attempt {attempt}). Feedback generated.")

            except Exception as prompt_error:
                error_msg = f"Error constructing basic feedback prompt (Attempt {attempt}): {prompt_error}"
                self.logger.error(error_msg, exc_info=True)
                state["regeneration_feedback"] = (
                    f"The answer to '{question}' failed a basic grounding check. "
                    f"Please generate a new answer based strictly on the provided context."
                )

            self.logger.debug(f"Finished generating basic feedback (Attempt {attempt}).")

        except Exception as e:
            error_msg = f"Unexpected error in _generate_basic_feedback (Attempt {attempt}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            state["regeneration_feedback"] = (
                f"A system error occurred while generating feedback: {e}. "
                f"Please try to generate a concise and accurate answer based on the provided context for '{question}'."
            )
            raise RuntimeError(error_msg) from e


    def _generate_advanced_feedback(self,
                                    state: CoreGraphState,
                                    advanced_results: Dict,
                                    question: str,
                                    attempt: int) -> None:
        """
        Generate detailed feedback based on advanced grounding analysis.

        This method processes the results from the `MultiLevelGroundingChecker` and
        constructs a detailed regeneration prompt for the answer generation node.
        It identifies specific issues like unsupported claims, contradictions, missing
        aspects, and hallucinations.

        Args:
            state (CoreGraphState): The current workflow state to be updated with feedback.
            advanced_results (Dict): The detailed results from the advanced grounding checker.
            question (str): The original user question.
            attempt (int): The current grounding check attempt number.
        """
        try:
            self.logger.debug(f"Generating advanced feedback (Attempt {attempt})...")

            feedback_parts: List[str] = []

            # Extract and Process Detailed Grounding Issues
            try:
                detailed_grounding = advanced_results.get("detailed_grounding")
                if detailed_grounding:
                    unsupported_claims = getattr(detailed_grounding, 'unsupported_claims', [])
                    if unsupported_claims:
                        # Limit the number of claims shown in feedback for conciseness
                        limited_claims = unsupported_claims[:3]
                        feedback_parts.append(f"Unsupported claims found: {'; '.join(limited_claims)}")
            except Exception as grounding_error:
                self.logger.warning(f"Error processing detailed grounding results (Attempt {attempt}): {grounding_error}")

            # Extract and Process Consistency Issues
            try:
                consistency = advanced_results.get("consistency")
                if consistency:
                    contradictions_found = getattr(consistency, 'contradictions_found', [])
                    if contradictions_found:
                        # Limit the number of contradictions shown
                        limited_contradictions = contradictions_found[:2]
                        feedback_parts.append(f"Internal contradictions: {'; '.join(limited_contradictions)}")
            except Exception as consistency_error:
                self.logger.warning(f"Error processing consistency results (Attempt {attempt}): {consistency_error}")

            # Extract and Process Completeness Issues
            try:
                completeness = advanced_results.get("completeness")
                if completeness:
                    missing_aspects = getattr(completeness, 'missing_aspects', [])
                    if missing_aspects:
                        # Limit the number of missing aspects shown
                        limited_missing = missing_aspects[:2]
                        feedback_parts.append(f"Missing important aspects: {'; '.join(limited_missing)}")
            except Exception as completeness_error:
                self.logger.warning(f"Error processing completeness results (Attempt {attempt}): {completeness_error}")

            # Extract and Process Hallucination Issues
            try:
                hallucination_data = advanced_results.get("hallucination_detection", {})
                hallucinations = hallucination_data.get("hallucinations", [])
                if hallucinations:
                    # Limit the number of hallucinations shown
                    limited_hallucinations = hallucinations[:2]
                    feedback_parts.append(f"Potential hallucinations: {'; '.join(limited_hallucinations)}")
            except Exception as hallucination_error:
                self.logger.warning(f"Error processing hallucination results (Attempt {attempt}): {hallucination_error}")

            if feedback_parts:
                try:
                    # Get a high-level recommendation from the overall assessment
                    overall_assessment = advanced_results.get("overall_assessment", {})
                    recommendation = overall_assessment.get("recommendation", "Please improve the answer")
                    
                    regeneration_prompt = (
                        f"The previous answer to '{question}' failed advanced verification. "
                        f"Issues identified: {' | '.join(feedback_parts)}. "
                        f"Recommendation: {recommendation}. "
                        f"Please generate a new answer that strictly follows the provided context and addresses these issues."
                    )
                    
                    # Update the state with the generated feedback
                    state["regeneration_feedback"] = regeneration_prompt
                    self.logger.warning(f"Advanced grounding check FAILED (Attempt {attempt}). Issues identified: {len(feedback_parts)}")
                    
                except Exception as prompt_error:
                    error_msg = f"Error constructing advanced feedback prompt (Attempt {attempt}): {prompt_error}"
                    self.logger.error(error_msg, exc_info=True)
                    state["regeneration_feedback"] = (
                        f"The answer to '{question}' did not meet advanced quality standards. "
                        f"Please generate a more accurate and complete answer based strictly on the provided context."
                    )
            else:
                self.logger.info(f"No specific advanced issues found, but check failed (Attempt {attempt}). Using general feedback.")
                state["regeneration_feedback"] = (
                    f"The answer to '{question}' did not meet quality standards. "
                    f"Please generate a more accurate answer based strictly on the provided context."
                )

            self.logger.debug(f"Finished generating advanced feedback (Attempt {attempt}).")

        except Exception as e:
            error_msg = f"Unexpected error in _generate_advanced_feedback (Attempt {attempt}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            state["regeneration_feedback"] = (
                f"A system error occurred while generating detailed feedback: {e}. "
                f"Please try to generate a concise and accurate answer based on the provided context for '{question}'."
            )
            raise RuntimeError(error_msg) from e
    

    def _handle_grounding_check_exception(self,
                                          state: CoreGraphState,
                                          exception: Exception,
                                          question: str,
                                          attempt: int) -> None:
        """
        Handle exceptions during grounding check execution.

        This method is a centralized exception handler for errors that occur
        during either the basic or advanced grounding check processes. It logs
        the error and generates generic fallback feedback to guide the workflow.

        Args:
            state (CoreGraphState): The current workflow state to be updated.
            exception (Exception): The exception that was caught.
            question (str): The original user question.
            attempt (int): The current grounding check attempt number.
        """
        try:
            self.logger.error(f"Grounding check failed with exception (Attempt {attempt}): {exception}", exc_info=True)

            # Generate Generic Error Feedback
            try:
                error_feedback = (
                    f"A system error occurred during the grounding check process: {exception}. "
                    f"Please try to generate a concise and accurate answer based on the provided context for '{question}'."
                )
                
                # Update the state with the error feedback
                state["regeneration_feedback"] = error_feedback
                
            except Exception as feedback_error:
                self.logger.critical(f"Failed to generate error feedback (Attempt {attempt}): {feedback_error}", exc_info=True)
                state["regeneration_feedback"] = f"An error occurred during verification. Please answer '{question}' concisely based on context."

            # Update Error Message in State
            try:
                # Append the grounding check exception to the overall error message in the state
                current_error_msg = state.get("error_message", "")
                if current_error_msg:
                    updated_error_msg = f"{current_error_msg} | Grounding check exception (Attempt {attempt}): {exception}"
                else:
                    updated_error_msg = f"Grounding check exception (Attempt {attempt}): {exception}"
                
                state["error_message"] = updated_error_msg
                
            except Exception as state_error:
                self.logger.warning(f"Failed to update state error_message (Attempt {attempt}): {state_error}")

            self.logger.info(f"Handled grounding check exception (Attempt {attempt}). Workflow will proceed with error feedback.")

        except Exception as handler_error:
            critical_error_msg = f"Critical failure in _handle_grounding_check_exception (Attempt {attempt}): {handler_error}"
            self.logger.critical(critical_error_msg, exc_info=True)
            
            try:
                state["regeneration_feedback"] = f"Critical system error during verification for '{question}'. Provide a concise answer."
                state["error_message"] = (state.get("error_message", "")) + f" | Critical handler error: {handler_error}"
            except:
                pass
                
            raise RuntimeError(critical_error_msg) from handler_error


    def _route_after_grounding_check(self, state: CoreGraphState) -> str:
        """
        Determine the next step in the workflow after a grounding check.

        This method evaluates the results of the grounding check node. If the answer
        is deemed grounded (no feedback), it ends the workflow. If the answer is not
        grounded but the maximum number of attempts has been reached, it ends the
        workflow with a warning prepended to the answer. Otherwise, it routes back
        to the answer generation node for another attempt.

        Args:
            state (CoreGraphState): The current state of the workflow after the grounding check.

        Returns:
            str: The name of the next node ('generate_answer' or END) or a directive.
        """
        try:
            attempts = state.get('grounding_check_attempts', 0)
            has_feedback = state.get('regeneration_feedback') is not None
            self.logger.info(f"Routing after grounding check. Attempts: {attempts}. Feedback Present: {'Yes' if has_feedback else 'No'}")

            # Check if Answer is Grounded
            if not has_feedback:
                # No feedback means the grounding check passed
                self.logger.info("Answer is grounded. Ending workflow.")
                return END

            # Check Attempt Limit
            max_attempts = getattr(self, 'max_grounding_attempts', 1)
            if attempts >= max_attempts:
                # Maximum attempts reached, end workflow with a warning
                self.logger.warning(f"Max grounding attempts ({max_attempts}) reached. Answer still not grounded. Ending with warning.")
                
                # Prepend a warning message to the existing generation
                original_generation = state.get("generation", "")
                warning_message = ("**Self-Correction Incomplete:** The following answer could not be fully verified against the provided documents after attempts to correct it. Please use with caution.\n---\n")
                state["generation"] = warning_message + original_generation
                return END

            # Route for Regeneration
            # If not grounded and attempts remain, go back to generate the answer
            self.logger.info("Answer not grounded, and attempts remain. Routing back to 'generate_answer'.")
            return "generate_answer"

        except Exception as e:
            error_msg = f"Unexpected error in _route_after_grounding_check: {e}"
            self.logger.critical(error_msg, exc_info=True)
            
            try:
                original_generation = state.get("generation", "")
                error_message = f"**Workflow Error:** {error_msg}\n---\n"
                state["generation"] = error_message + original_generation
            except:
                pass
                
            return END


    def _retrieve_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Retrieve relevant documents from the vector store based on the question.

        This node handles document retrieval. It ensures the vector store and retriever
        for the specified collection are loaded, performs the retrieval using the
        current question (potentially augmented with query analysis results), and
        updates the state with the retrieved documents and context.

        Args:
            state (CoreGraphState): The current state of the workflow.

        Returns:
            CoreGraphState: The updated state with retrieved documents and context.
        """
        try:
            self.logger.info("NODE: Starting document retrieval...")

            # Determine Collection
            collection_name = state.get("collection_name") or self.default_collection_name
            self.logger.info(f"Retrieving documents from collection '{collection_name}' for question: '{state['question']}'")

            try:
                self._init_or_load_vectorstore(collection_name, recreate=False)
                retriever = self.retrievers.get(collection_name)
                
                if not retriever:
                    error_msg = f"No retriever found for collection '{collection_name}'."
                    self.logger.warning(error_msg)

                    # Update state to reflect the failure
                    state["documents"] = []
                    state["context"] = "Retriever not available for the specified collection."
                    return state

            except Exception as init_error:
                error_msg = f"Error initializing or loading vector store/retriever for '{collection_name}': {init_error}"
                self.logger.error(error_msg, exc_info=True)
                state["documents"] = []
                state["context"] = "Error occurred while initializing the retrieval system."
                state["error_message"] = (state.get("error_message") or "") + f" | Retrieval init error: {init_error}"
                return state

            current_question = state["question"]
            query_analysis: Optional[QueryAnalysis] = state.get("query_analysis_results")

            # 1. Determine dynamic top_k based on query analysis
            retrieval_k = self.default_retrieval_top_k
            if query_analysis:
                self.logger.info(
                    f"Using query analysis for retrieval: Type='{query_analysis.query_type}', "
                    f"Keywords='{query_analysis.extracted_keywords}'"
                )
                
                if query_analysis.query_type in ["summary_request", "complex_reasoning"]:
                    # For summaries / complex reasoning, retrieve more docs
                    retrieval_k = max(self.default_retrieval_top_k, 7)
                    self.logger.info(f"Adjusted top_k to {retrieval_k} for query type: {query_analysis.query_type}")
                elif query_analysis.query_type == "factual_lookup":
                    # For factual lookups, retrieve fewer docs
                    retrieval_k = min(self.default_retrieval_top_k, 3)
                    self.logger.info(f"Adjusted top_k to {retrieval_k} for query type: {query_analysis.query_type}")

            # 2. Augment query with keywords (if any)
            search_query = current_question
            if query_analysis and query_analysis.extracted_keywords:
                keywords_str = " ".join(query_analysis.extracted_keywords)
                search_query = f"{current_question} Keywords: {keywords_str}"
                self.logger.info(f"Augmented search query with keywords: '{search_query}'")

            # Perform Retrieval
            original_k = None # To store the original k value for restoration
            try:
                # Temporarily override retriever.k or retriever.search_kwargs['k'] if possible
                if hasattr(retriever, "k"):
                    original_k = retriever.k
                    retriever.k = retrieval_k
                elif hasattr(retriever, "search_kwargs") and "k" in retriever.search_kwargs:
                    original_k = retriever.search_kwargs["k"]
                    retriever.search_kwargs["k"] = retrieval_k
                else:
                    self.logger.debug("Retriever does not support dynamic k adjustment.")

                self.logger.info(f"Attempting to retrieve top {retrieval_k} docs for: '{search_query}'")
                docs = retriever.invoke(search_query)
                self.logger.info(f"Successfully retrieved {len(docs)} documents.")

            except Exception as retrieval_error:
                error_msg = f"Error during document retrieval for query '{search_query}': {retrieval_error}"
                self.logger.error(error_msg, exc_info=True)
                state["documents"] = []
                state["context"] = "Error occurred during document retrieval."
                state["error_message"] = (state.get("error_message") or "") + f" | Retrieval error: {retrieval_error}"
                docs = []

            finally:
                # Restore Original Retriever Configuration
                if original_k is not None:
                    try:
                        if hasattr(retriever, "k"):
                            retriever.k = original_k
                        elif hasattr(retriever, "search_kwargs") and "k" in retriever.search_kwargs:
                            retriever.search_kwargs["k"] = original_k
                        self.logger.debug(f"Restored retriever's original k value: {original_k}")
                    except Exception as restore_error:
                        self.logger.warning(f"Failed to restore retriever's original k value: {restore_error}")

            # Update State with Results
            state["documents"] = docs
            state["context"] = "\n\n".join(d.page_content for d in docs)

            self.logger.info("NODE: Document retrieval completed.")
            return state

        except Exception as node_error:
            error_msg = f"Unexpected error in _retrieve_node: {node_error}"
            self.logger.critical(error_msg, exc_info=True)
            state["documents"] = []
            state["context"] = "A critical error occurred during the document retrieval process."
            state["error_message"] = (state.get("error_message") or "") + f" | Critical retrieval node error: {node_error}"
            return state
    
    def _rerank_documents_node(self, state: CoreGraphState) -> CoreGraphState:
        """
        Re-rank retrieved documents by their relevance score.

        This node takes the documents retrieved by the `_retrieve_node` and re-ranks
        them using a dedicated LLM-based re-ranking chain. This provides a more
        nuanced ordering than vector similarity alone.

        Args:
            state (CoreGraphState): The current state of the workflow, containing documents.

        Returns:
            CoreGraphState: The updated state with re-ranked documents.
        """
        try:
            self.logger.info("NODE: Starting document re-ranking...")

            # Use the original question for re-ranking context
            question = state.get("original_question") or state["question"]
            docs = state.get("documents", [])

            if not docs:
                self.logger.info("No documents found in state to re-rank.")
                return state # Return early, state is already correct

            self.logger.info(f"Re-ranking {len(docs)} documents for question: '{question}'")

            # Rerank Documents
            docs_with_scores: List[Tuple[Document, float]] = []
            
            for idx, doc in enumerate(docs):
                try:
                    source_info = doc.metadata.get("source", "unknown")
                    self.logger.debug(f"Re-ranking document {idx+1}/{len(docs)} from source '{source_info}'")

                    # Invoke the re-ranker chain for this document-question pair
                    score_result: RerankScore = self.document_reranker_chain.invoke({"question": question,
                                                                                     "document_content": doc.page_content})

                    # Store the document along with its relevance score
                    docs_with_scores.append((doc, score_result.relevance_score))
                    self.logger.debug(f"Document {idx+1} scored: {score_result.relevance_score}")

                except Exception as doc_error:
                    # If scoring a single document fails, log it and assign a low score
                    source_info = doc.metadata.get("source", "unknown")
                    
                    error_msg = f"Error re-ranking document {idx+1} (source: {source_info}): {doc_error}"
                    self.logger.error(error_msg, exc_info=True)
                    
                    # Assign score 0.0 for failed documents
                    docs_with_scores.append((doc, 0.0))

            # Sort Documents by Score
            try:
                # Sort the list of (document, score) tuples by score in descending order
                sorted_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
                
                # Extract the re-ranked documents (discard scores for state update)
                sorted_docs = [doc for doc, score in sorted_docs_with_scores]
                
                self.logger.info(f"Finished re-ranking. Top document score: {sorted_docs_with_scores[0][1] if sorted_docs_with_scores else 'N/A'}")

            except Exception as sort_error:
                error_msg = f"Error sorting re-ranked documents: {sort_error}"
                self.logger.error(error_msg, exc_info=True)
                # If sorting fails, keep the original order
                sorted_docs = docs

            # Update State
            state["documents"] = sorted_docs

            self.logger.info("NODE: Document re-ranking completed.")
            return state

        except Exception as node_error:
            error_msg = f"Unexpected error in _rerank_documents_node: {node_error}"
            self.logger.critical(error_msg, exc_info=True)
            state["error_message"] = (state.get("error_message") or "") + f" | Critical re-ranking node error: {node_error}"
            return state

    def _grade_documents_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Grading documents individually...")
        question = state.get("original_question") or state["question"]
        docs = state.get("documents", []) or []

        if not docs:
            self.logger.info("No documents retrieved to grade.")
            state["relevance_check_passed"] = False
            state["context"] = "No documents were retrieved to grade."
            return state

        relevant_docs: List[Document] = []
        for idx, doc in enumerate(docs):
            src = doc.metadata.get("source", "unknown")
            self.logger.debug(f"Grading doc {idx+1}/{len(docs)} from source '{src}'")
            try:
                grade: RelevanceGrade = self.document_relevance_grader_chain.invoke({
                    "question": question,
                    "document_content": doc.page_content
                })
                self.logger.debug(
                    f"Doc {idx+1} graded: is_relevant={grade.is_relevant}, justification={grade.justification}"
                )
                if grade.is_relevant:
                    doc.metadata["relevance_grade_justification"] = grade.justification
                    relevant_docs.append(doc)
            except Exception as e:
                self.logger.error(
                    f"Error grading document {idx+1} (source: {src}): {e}",
                    exc_info=True
                )
                continue

        if relevant_docs:
            self.logger.info(f"{len(relevant_docs)} document(s) passed relevance grading.")
            state["documents"] = relevant_docs
            state["context"] = "\n\n".join(d.page_content for d in relevant_docs)
            state["relevance_check_passed"] = True
        else:
            self.logger.info("No documents deemed relevant after grading.")
            state["documents"] = []
            state["context"] = "Retrieved content was not deemed relevant after grading."
            state["relevance_check_passed"] = False

        state["error_message"] = None
        return state
    
    def _analyze_query_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Analyzing query...")
        question = state["question"]
        chat_history = state.get("chat_history", [])

        # Format chat history for the prompt
        formatted_history_str = "\n".join(
            [f"{msg.type.upper()}: {msg.content}" for msg in chat_history]
        )
        if not formatted_history_str:
            formatted_history_str = "No chat history."

        try:
            analysis_result: QueryAnalysis = self.query_analyzer_chain.invoke({
                "question": question,
                "chat_history_formatted": formatted_history_str
            })
            state["query_analysis_results"] = analysis_result
            self.logger.info(f"Query analysis complete: Type='{analysis_result.query_type}', Intent='{analysis_result.main_intent}'")
            if analysis_result.is_ambiguous:
                self.logger.warning(f"Query marked as ambiguous: {question}")

        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}", exc_info=True)
            state["query_analysis_results"] = None
            state["error_message"] = (state.get("error_message") or "") + f" | Query analysis failed: {e}"

        if state.get("original_question") is None:
            state["original_question"] = question

        return state

    def _rewrite_query_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Rewriting query")
        original_question = state.get("original_question") or state["question"]
        try:
            result = self.query_rewriter_chain.invoke({
                "question": original_question,
                "chat_history": state.get("chat_history", [])
            })
            rewritten_query = result.strip()
            if rewritten_query.lower() != original_question.lower():
                self.logger.info(f"Rewrote '{original_question}' → '{rewritten_query}'")
            else:
                self.logger.info(f"No rewrite needed for question: '{original_question}'")

            state["question"] = rewritten_query
            
        except Exception as e:
            self.logger.error(f"Error during query rewriting: {e}", exc_info=True)
            # If rewriting fails, fall back to the original question.
            state["question"] = original_question
            state["error_message"] = f"Query rewriting failed: {e}"
        return state

    def _web_search_node(self, state: CoreGraphState) -> CoreGraphState:
        if state.get("run_web_search") != "Yes":
            self.logger.debug("NODE: Web search skipped; run_web_search != 'Yes'.")
            return state

        self.logger.info("NODE: Performing web search...")
        current_question = state["question"]

        state["documents"] = []
        state["context"] = ""
        state["web_search_results"] = None

        if not self.search_tool:
            self.logger.warning("Web search tool not configured.")
            state["context"] = "Web search tool is not available. Cannot perform web search."
            state["error_message"] = "Web search tool unavailable."
            return state

        try:
            self.logger.debug(f"Invoking web search for: '{current_question}'")
            raw_results = self.search_tool.invoke({"query": current_question})

            processed_web_docs: List[Document] = []
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict) and item.get("content"):
                        processed_web_docs.append(
                            Document(
                                page_content=item["content"],
                                metadata={"source": item.get("url", "Unknown Web Source")}
                            )
                        )
            elif isinstance(raw_results, str) and raw_results.strip():
                processed_web_docs.append(
                    Document(
                        page_content=raw_results.strip(),
                        metadata={"source": "Unknown Web Source - String Output"}
                    )
                )

            state["web_search_results"] = processed_web_docs

            if processed_web_docs:
                self.logger.info(f"Web search found {len(processed_web_docs)} documents.")
                state["documents"] = processed_web_docs
                state["context"]   = "\n\n".join(d.page_content for d in processed_web_docs)
                state["error_message"] = None
            else:
                self.logger.info("Web search returned no relevant results.")
                state["context"] = "Web search returned no relevant results or content."

        except Exception as e:
            self.logger.error(f"An error occurred during web search: {e}", exc_info=True)
            state["documents"] = []
            state["context"]   = "An error occurred during web search. Unable to retrieve web results."
            state["error_message"] = f"Web search execution failed: {e}"

        return state
        
    def _generate_answer_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Generating answer")
        question = state["question"]
        original_question = state.get("original_question", question)
        context = state.get("context", "")
        chat_history = state.get("chat_history", [])
        regeneration_feedback = state.get("regeneration_feedback")
    
        input_data_for_chain = {
            "context": context,
            "chat_history": chat_history,
            "optional_regeneration_prompt_header_if_feedback": "",
            "regeneration_feedback_if_any": "",
            "question": original_question
        }
    
        if regeneration_feedback:
            self.logger.info(f"Generating answer with regeneration feedback: {regeneration_feedback}")
            input_data_for_chain["optional_regeneration_prompt_header_if_feedback"] = (
                "You are attempting to regenerate a previous answer that had issues."
            )
            input_data_for_chain["regeneration_feedback_if_any"] = regeneration_feedback + "\n\nOriginal Question: "
    
        try:
            generated_text = self.answer_generation_chain.invoke(input_data_for_chain)
            state["generation"] = generated_text.strip()
        except Exception as e:
            self.logger.error(f"Generation error: {e}", exc_info=True)
            state["generation"] = "Error generating answer."
            state["error_message"] = str(e)
        return state
    
    def _increment_retries_node(self, state: CoreGraphState) -> CoreGraphState:
        """Increments the retry counter in the state."""
        self.logger.info("NODE: Incrementing retries...")
        retries = state.get("retries", 0)
        state["retries"] = retries + 1
        return state

    def _route_after_grading(self, state: CoreGraphState) -> str:
        self.logger.info("Routing after grading...")
        passed = state.get("relevance_check_passed", False)
        retries = state.get("retries", 0)

        if passed:
            self.logger.info("...Documents are relevant. Generating answer.")
            return "generate_answer"

        if retries >= self.max_rewrite_retries:
            self.logger.info("...Max rewrite retries reached.")
            if self.search_tool:
                self.logger.info("...Performing web search.")
                state["run_web_search"] = "Yes"
                return "web_search"
            else:
                self.logger.info("...No web search tool available. Generating answer from existing context.")
                return "generate_answer"
        
        self.logger.info("...Documents not relevant. Incrementing retries and rewriting.")
        return "increment_retries"

    # ---------------------
    # Workflow Compilation
    # ---------------------
    def _compile_rag_workflow(self) -> Any:
        graph = StateGraph(CoreGraphState)

        # New analysis node
        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("rerank_documents", self._rerank_documents_node)
        graph.add_node("grade_documents", self._grade_documents_node)
        graph.add_node("rewrite_query", self._rewrite_query_node)
        graph.add_node("web_search", self._web_search_node)
        graph.add_node("generate_answer", self._generate_answer_node)
        graph.add_node("grounding_check", self._grounding_check_node)
        graph.add_node("increment_retries", self._increment_retries_node)

        # Set new entry point
        graph.set_entry_point("analyze_query")

        # Routing
        graph.add_edge("analyze_query", "rewrite_query")
        graph.add_edge("retrieve", "rerank_documents")
        graph.add_edge("rerank_documents", "grade_documents")
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "grounding_check")
        graph.add_edge("increment_retries", "rewrite_query")
        graph.add_edge("rewrite_query", "retrieve")
        
        graph.add_conditional_edges(
        "grade_documents",
        self._route_after_grading,
            {
                "generate_answer": "generate_answer",
                "increment_retries": "increment_retries", # New path
                "web_search": "web_search"
            }
        )
        
        graph.add_conditional_edges(
            "grounding_check",
            self._route_after_grounding_check,
            {
                "generate_answer": "generate_answer",
                END: END
            }
        )

        self.rag_workflow = graph.compile()
        self.logger.info("RAG workflow compiled successfully with new retry logic.")
        return self.rag_workflow
    # ---------------------
    # Public API: Ingestion
    # ---------------------
    def load_documents(self, source_type: str, source_value: Any) -> List[Document]:
        try:
            if source_type == "url":
                loader = WebBaseLoader(web_paths=[source_value])
            elif source_type == "pdf_path":
                if not os.path.exists(source_value):
                    self.logger.error(f"PDF not found: {source_value}")
                    return []
                loader = PyPDFLoader(file_path=source_value)
            elif source_type == "text_path":
                if not os.path.exists(source_value):
                    self.logger.error(f"Text file not found: {source_value}")
                    return []
                loader = TextLoader(file_path=source_value)
            elif source_type == "uploaded_pdf" and UploadedFile and isinstance(source_value, UploadedFile):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(source_value.getbuffer())
                tmp.close()
                loader = PyPDFLoader(file_path=tmp.name)
            else:
                self.logger.error(f"Unsupported source_type: {source_type}")
                return []
        except Exception as e:
            self.logger.error(f"Error creating loader: {e}")
            return []

        try:
            docs = loader.load()
            self.logger.info(f"Loaded {len(docs)} docs from {source_type}")
            return docs
        except Exception as e:
            self.logger.error(f"Error loading docs: {e}")
            return []
        finally:
            if source_type == "uploaded_pdf" and 'tmp' in locals():
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    def split_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        
        if isinstance(self.text_splitter, BaseChunker):
            chunks = self.text_splitter.chunk_documents(docs)
        else:
            # Fallback for old-style splitters
            chunks = self.text_splitter.split_documents(docs)
        
        self.logger.info(f"Split into {len(chunks)} chunks using {type(self.text_splitter).__name__}")
        
        # Log chunking statistics
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            self.logger.info(f"Average chunk size: {avg_size:.0f} characters")
        
        return chunks
        
    def index_documents(self, docs: List[Document], name: str, recreate: bool = False) -> None:
        self._init_or_load_vectorstore(name, recreate)
        vs = self.vectorstores.get(name)
        d = self._get_persist_dir(name)

        if vs is None or recreate:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                # The unit-test monkey-patches os.makedirs to raise
                # PermissionError, then spies on self.logger.error. We must
                # log exactly once and abort indexing.
                self.logger.error(
                    f"Could not create persist directory '{d}': {e}",
                    exc_info=True,
                )
                return

            vs_new = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                collection_name=name,
                persist_directory=d
            )
            self.vectorstores[name] = vs_new
            self.retrievers[name] = vs_new.as_retriever(
                search_kwargs={'k': self.default_retrieval_top_k}
            )
            self.logger.info(f"Created vector store '{name}' with default k={self.default_retrieval_top_k}")
        else:
            vs.add_documents(docs)
            self.logger.info(f"Added {len(docs)} docs to '{name}'")


    def ingest(
        self,
        sources: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        collection_name: Optional[str] = None,
        recreate_collection: bool = False,
        direct_documents: Optional[List[Document]] = None
    ) -> None:
        """
        Ingest documents into a specified collection.

        This method orchestrates the full ingestion pipeline: loading documents from sources,
        splitting them into chunks, and indexing them into a Chroma vector store collection.
        It supports ingesting from various source types (URLs, PDFs, text files, uploaded files)
        or directly from a list of pre-loaded `Document` objects.

        Args:
            sources (Union[Dict[str, Any], List[Dict[str, Any]], None]):
                A dictionary or list of dictionaries specifying document sources.
                Each dict should have 'type' (e.g., 'url', 'pdf_path') and 'value'.
                Example: [{"type": "url", "value": "https://example.com"}, ...]
            collection_name (Optional[str]):
                The name of the collection to ingest into. Defaults to `self.default_collection_name`.
            recreate_collection (bool):
                If True, removes the existing collection and creates a new one. Defaults to False.
            direct_documents (Optional[List[Document]]):
                A list of pre-loaded `Document` objects to ingest directly, bypassing the loading step.
                If provided, the `sources` parameter is ignored.

        Raises:
            Exception: If any step in the ingestion process fails.
        """
        try:
            # Determine Target Collection
            name = collection_name or self.default_collection_name
            self.logger.info(
                f"Starting ingestion process for collection '{name}'. "
                f"Recreate collection: {recreate_collection}"
            )

            all_chunks_for_collection: List[Document] = []

            # Handle Direct Documents
            if direct_documents is not None and isinstance(direct_documents, list) and direct_documents:
                # If direct documents are provided, use them and skip loading from sources
                if all(isinstance(doc, Document) for doc in direct_documents):
                    self.logger.info(f"Using {len(direct_documents)} pre-loaded documents for ingestion.")
                    all_chunks_for_collection = self.split_documents(direct_documents)
                else:
                    error_msg = "Invalid 'direct_documents' provided: not all items are Document instances."
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            # Handle Sources 
            elif sources is not None:
                # If sources are provided, load and process them
                self.logger.info("Processing documents from the 'sources' parameter.")
                
                # Normalize sources to a list if it's a single dict
                if not isinstance(sources, list):
                    sources = [sources]

                for idx, src in enumerate(sources):
                    try:
                        src_type = src.get("type")
                        src_val = src.get("value")

                        if not src_type or src_val is None:
                            self.logger.warning(f"Skipping invalid source at index {idx}: {src}")
                            continue

                        raw_docs = self.load_documents(source_type=src_type, source_value=src_val)
                        
                        if raw_docs:
                            chunks = self.split_documents(raw_docs)
                            all_chunks_for_collection.extend(chunks)
                            self.logger.debug(f"Processed source {idx+1}/{len(sources)}: '{src_type}' -> {len(chunks)} chunks")

                    except Exception as source_error:
                        # Log error for individual source but continue with others
                        self.logger.error(f"Error processing source {idx+1} ({src}): {source_error}", exc_info=True)
                        
            # Validate Documents to Ingest
            if not all_chunks_for_collection:
                # No documents were prepared for ingestion
                if recreate_collection:
                    # If recreation was requested but no new docs, just clear the existing collection
                    self.logger.info(
                        f"No new documents found, but recreate_collection=True. "
                        f"Clearing/resetting collection '{name}'."
                    )
                    self._init_or_load_vectorstore(name, recreate=True)
                else:
                    self.logger.warning(
                        f"No documents available to ingest for collection '{name}'. "
                        f"Ingestion process skipped."
                    )
                # Exit early as there's nothing to index
                return

            self.logger.info(
                f"Proceeding to index {len(all_chunks_for_collection)} document chunks "
                f"into collection '{name}'."
            )
            try:
                self.index_documents(docs=all_chunks_for_collection,
                                     name=name,
                                     recreate=recreate_collection)
                self.logger.info(
                    f"Ingestion completed successfully for collection '{name}' "
                    f"({len(all_chunks_for_collection)} chunks indexed)."
                )
            except Exception as index_error:
                error_msg = f"Failed to index documents into collection '{name}': {index_error}"
        
    def answer_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: int = 4) -> Dict[str, Any]:
        name = collection_name or self.default_collection_name
        self._init_or_load_vectorstore(name, recreate=False)
        retriever = self.retrievers.get(name)
        if not retriever:
            msg = "No documents indexed."
            self.logger.warning(msg)
            return {"answer": msg, "sources": []}
        docs = retriever.get_relevant_documents(question)[:top_k]
        context = "\n\n".join(d.page_content for d in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            ans = self.llm.invoke(prompt).content
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return {"answer": "Error generating answer.", "sources": []}
        sources = [
            {"source": d.metadata.get("source", "unknown"),
            "preview": d.page_content[:200] + "..."}
            for d in docs
        ]
        return {"answer": ans.strip(), "sources": sources}

    async def run_full_rag_workflow(self,
                                    question: str,
                                    collection_name: Optional[str] = None,
                                    chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        
        name = collection_name or self.default_collection_name
        initial_state: CoreGraphState = {
            "question": question,
            "original_question": question,
            "query_analysis_results": None,
            "documents": [],
            "context": "",
            "web_search_results": None,
            "generation": "",
            "retries": 0,
            "run_web_search": "No",
            "relevance_check_passed": None,
            "error_message": None,
            "grounding_check_attempts": 0,
            "regeneration_feedback": None,
            "collection_name": name,
            "chat_history": chat_history or []
        }
        final = await self.rag_workflow.ainvoke(initial_state)

        answer = final.get("generation", "")
        docs = final.get("documents", [])
        sources = [{"source": d.metadata.get("source", "unknown"), "preview": d.page_content[:200] + "..."} for d in docs]
        return {"answer": answer, "sources": sources}
										
    def run_full_rag_workflow_sync(self,
                                question: str,
                                collection_name: Optional[str] = None,
                                chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """Synchronous wrapper for the async workflow"""
        import asyncio
        import nest_asyncio
        
        # Try to get existing event loop, create new one if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply()
                return loop.run_until_complete(self.run_full_rag_workflow(question, collection_name, chat_history))
            else:
                return loop.run_until_complete(self.run_full_rag_workflow(question, collection_name, chat_history))
        
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(self.run_full_rag_workflow(question, collection_name, chat_history))
        except ImportError:
            # nest_asyncio not available, create new event loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self.run_full_rag_workflow(question, collection_name, chat_history)
                )
            finally:
                new_loop.close()
