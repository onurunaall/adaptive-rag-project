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
        # Load from AppSettings if parameters are not explicitly passed
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

        # Persist directory logic
        self.persist_directory_base = persist_directory_base if persist_directory_base is not None else app_settings.engine.persist_directory_base
        base = self.persist_directory_base if self.persist_directory_base is not None else tempfile.gettempdir()
        self.persist_directory_base = os.path.join(base, "core_rag_engine_chroma")
        os.makedirs(self.persist_directory_base, exist_ok=True)

        # API keys
        self.tavily_api_key = tavily_api_key if tavily_api_key is not None else app_settings.api.tavily_api_key
        self.openai_api_key = openai_api_key if openai_api_key is not None else app_settings.api.openai_api_key
        self.google_api_key = google_api_key if google_api_key is not None else app_settings.api.google_api_key

        self.max_rewrite_retries = max_rewrite_retries if max_rewrite_retries is not None else app_settings.engine.max_rewrite_retries
        self.max_grounding_attempts = max_grounding_attempts if max_grounding_attempts is not None else app_settings.engine.max_grounding_attempts
        self.default_retrieval_top_k = default_retrieval_top_k if default_retrieval_top_k is not None else app_settings.engine.default_retrieval_top_k
        
        # Hybrid search settings
        self.enable_hybrid_search = enable_hybrid_search if enable_hybrid_search is not None else getattr(app_settings.engine, 'enable_hybrid_search', False)
        self.hybrid_search_alpha = hybrid_search_alpha if hybrid_search_alpha is not None else getattr(app_settings.engine, 'hybrid_search_alpha', 0.7)
        
        # Logger
        self._setup_logger()

        # Initialize LLMs (normal and JSON-formatted)
        self.llm = self._init_llm(use_json_format=False)
        self.json_llm = self._init_llm(use_json_format=True)

        # Embeddings
        self.embedding_model = self._init_embedding_model()

        # Splitter and search tool
        self.text_splitter = self._init_text_splitter()
                
        self.search_tool = self._init_search_tool()

        # Individual LLM Chains
        self.document_relevance_grader_chain = self._create_document_relevance_grader_chain()
        self.document_reranker_chain = self._create_document_reranker_chain()
        self.query_rewriter_chain = self._create_query_rewriter_chain()
        self.answer_generation_chain = self._create_answer_generation_chain()
        self.grounding_check_chain = self._create_grounding_check_chain()
        self.query_analyzer_chain = self._create_query_analyzer_chain()

        # Compile the RAG workflow graph
        self._compile_rag_workflow()

        # Storage for vector stores and retrievers
        self.vectorstores: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, Any] = {}
                     
        # Advanced grounding settings
        self.enable_advanced_grounding = enable_advanced_grounding if enable_advanced_grounding is not None else getattr(app_settings.engine, 'enable_advanced_grounding', False)
        
        # Initialize advanced grounding checker if enabled
        self.advanced_grounding_checker = None
        if self.enable_advanced_grounding:
            try:
                self.advanced_grounding_checker = MultiLevelGroundingChecker(self.llm)
                self.logger.info("Advanced grounding checker initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced grounding checker: {e}")

        self.document_cache = {}
        self.cache_ttl = 300  # 5 minutes TTL
        self.cache_timestamps = {}
        
        self.logger.info("CoreRAGEngine initialized and workflow compiled.")

    def _get_all_documents_from_collection(self, collection_name: str) -> List[Document]:
        """Retrieve all documents with caching and memory optimization"""
        import time
        
        # Check cache
        cache_key = collection_name
        current_time = time.time()
        
        if cache_key in self.document_cache:
            if current_time - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl:
                self.logger.debug(f"Using cached documents for '{collection_name}'")
                return self.document_cache[cache_key]
        
        try:
            vectorstore = self.vectorstores.get(collection_name)
            if not vectorstore:
                return []
            
            # Use pagination for large collections
            batch_size = 1000
            all_docs = []
            offset = 0
            
            collection = vectorstore._collection
            
            while True:
                results = collection.get(limit=batch_size, offset=offset)
                
                if not results['documents']:
                    break
                    
                for content, metadata in zip(results['documents'], results['metadatas'] or []):
                    doc = Document(page_content=content, metadata=metadata or {})
                    all_docs.append(doc)
                
                if len(results['documents']) < batch_size:
                    break
                    
                offset += batch_size
                
                # Clear memory periodically
                if offset % 5000 == 0:
                    import gc
                    gc.collect()
            
            # Update cache
            self.document_cache[cache_key] = all_docs
            self.cache_timestamps[cache_key] = current_time
            
            # Clean old cache entries
            for key in list(self.cache_timestamps.keys()):
                if current_time - self.cache_timestamps[key] > self.cache_ttl * 2:
                    del self.document_cache[key]
                    del self.cache_timestamps[key]
            
            return all_docs
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents from '{collection_name}': {e}")
            return []

    def clear_document_cache(self, collection_name: Optional[str] = None):
        """Clear document cache to free memory"""
        if collection_name:
            self.document_cache.pop(collection_name, None)
            self.cache_timestamps.pop(collection_name, None)
        else:
            self.document_cache.clear()
            self.cache_timestamps.clear()
        
        import gc
        gc.collect()
        
    def _setup_logger(self) -> None:
        logger = logging.getLogger(self.__class__.__name__)

        # Only add handler if none exist to avoid duplicate logs
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)

        logger.setLevel(logging.INFO)
        self.logger = logger

    def _get_persist_dir(self, collection_name: str) -> str: # Helper method if not already present
        """Returns the specific persistence directory for a given collection name."""
        return os.path.join(self.persist_directory_base, collection_name)

    def _init_or_load_vectorstore(self, collection_name: str, recreate: bool = False) -> None:
        """
        Initializes an in-memory reference to a vector store, loading it from disk
        if it exists and is not being recreated, or preparing for a new one.
        Ensures the retriever is set up with the current engine's default_retrieval_top_k.
        """
        persist_dir = self._get_persist_dir(collection_name)
    
        if collection_name in self.vectorstores and not recreate:
            # Check if the retriever needs to be created or re-configured (e.g., k changed)
            if collection_name not in self.retrievers or self.retrievers[collection_name].search_kwargs.get('k') != self.default_retrieval_top_k:
                if self.enable_hybrid_search:
                    # Create hybrid retriever
                    try:
                        all_docs = self._get_all_documents_from_collection(collection_name)
                        if all_docs:
                            self.retrievers[collection_name] = AdaptiveHybridRetriever(
                                vector_store=self.vectorstores[collection_name],
                                documents=all_docs,
                                k=self.default_retrieval_top_k
                            )
                            self.logger.info(f"Created hybrid retriever for collection '{collection_name}' with k={self.default_retrieval_top_k}")
                        else:
                            # Fallback to standard retriever if no docs are found for BM25
                            self.retrievers[collection_name] = self.vectorstores[collection_name].as_retriever(
                                search_kwargs={'k': self.default_retrieval_top_k}
                            )
                            self.logger.warning(f"Could not create hybrid retriever for '{collection_name}' (no documents found), using standard retriever")
                    except Exception as e:
                        self.logger.warning(f"Failed to create hybrid retriever: {e}, using standard retriever")
                        self.retrievers[collection_name] = self.vectorstores[collection_name].as_retriever(
                            search_kwargs={'k': self.default_retrieval_top_k}
                        )
                else:
                    # Standard retriever
                    self.retrievers[collection_name] = self.vectorstores[collection_name].as_retriever(
                        search_kwargs={'k': self.default_retrieval_top_k}
                    )
                self.logger.info(f"Retriever for collection '{collection_name}' configured with k={self.default_retrieval_top_k}")
            return
    
        if recreate:
            self.logger.info(f"Recreating collection '{collection_name}'. Removing existing if present.")
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            if collection_name in self.vectorstores:
                del self.vectorstores[collection_name]
            if collection_name in self.retrievers:
                del self.retrievers[collection_name]
            return
    
        # If not in memory and not recreating, try to load from disk
        if os.path.exists(persist_dir):
            try:
                self.logger.info(f"Loading existing vector store '{collection_name}' from {persist_dir}")
                self.vectorstores[collection_name] = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=persist_dir
                )
                # After loading from disk, immediately set up the correct retriever
                # This logic is similar to the fix above but for newly loaded collections
                if self.enable_hybrid_search:
                    all_docs = self._get_all_documents_from_collection(collection_name)
                    if all_docs:
                        self.retrievers[collection_name] = AdaptiveHybridRetriever(
                            vector_store=self.vectorstores[collection_name],
                            documents=all_docs,
                            k=self.default_retrieval_top_k
                        )
                        self.logger.info(f"Created hybrid retriever for newly loaded collection '{collection_name}'")
                    else:
                        self.retrievers[collection_name] = self.vectorstores[collection_name].as_retriever(
                            search_kwargs={'k': self.default_retrieval_top_k}
                        )
                else:
                    self.retrievers[collection_name] = self.vectorstores[collection_name].as_retriever(
                        search_kwargs={'k': self.default_retrieval_top_k}
                    )
                self.logger.info(f"Successfully loaded '{collection_name}' with k={self.default_retrieval_top_k}.")
            except Exception as e:
                self.logger.error(f"Error loading vector store '{collection_name}' from {persist_dir}: {e}. A new one may be created if documents are indexed.", exc_info=True)
                if collection_name in self.vectorstores:
                    del self.vectorstores[collection_name]
                if collection_name in self.retrievers:
                    del self.retrievers[collection_name]
        else:
            self.logger.info(f"No persisted vector store found for '{collection_name}' at {persist_dir}. It will be created upon first indexing.")
    
    def _init_llm(self, use_json_format: bool) -> Any:
        provider = self.llm_provider.lower()
        if provider == "openai":
            return self._init_openai_llm(use_json_format)
        if provider == "ollama":
            return self._init_ollama_llm(use_json_format)
        if provider == "google":
            return self._init_google_llm(use_json_format)
        raise ValueError(f"Unsupported LLM provider: {provider}")
    

    def _init_openai_llm(self, use_json: bool) -> ChatOpenAI:
        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY missing")
            raise ValueError("OPENAI_API_KEY is required")
        args: Dict[str, Any] = {
            "model": self.llm_model_name,
            "temperature": self.temperature,
            "openai_api_key": self.openai_api_key,
        }
        if use_json:
            # Request JSON response for structured parsing
            args["model_kwargs"] = {"response_format": {"type": "json_object"}}
            self.logger.info(f"JSON mode enabled for {self.llm_model_name}")
        return ChatOpenAI(**args)

    def _init_ollama_llm(self, use_json: bool) -> ChatOllama:
        fmt = "json" if use_json else None
        # Ollama runs locally, format controls JSON or plain-text output
        return ChatOllama(model=self.llm_model_name, temperature=self.temperature, format=fmt)


    def _init_google_llm(self, use_json: bool) -> ChatGoogleGenerativeAI:
        if not self.google_api_key:
            self.logger.error("GOOGLE_API_KEY missing")
            raise ValueError("GOOGLE_API_KEY is required")
        
        kwargs: Dict[str, Any] = {
            "model": self.llm_model_name,
            "temperature": self.temperature,
            "google_api_key": self.google_api_key,
            "convert_system_message_to_human": True,
        }
        
        if use_json:
            # Request JSON output from Google model
            kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
            self.logger.info(f"JSON mode enabled for Google model {self.llm_model_name}")
        return ChatGoogleGenerativeAI(**kwargs)

    # ---------------------
    # Embedding Initialization
    # ---------------------
    def _init_embedding_model(self) -> Any:
        provider = self.embedding_provider.lower()
        if provider == "openai":
            return self._init_openai_embeddings()
        if provider == "gpt4all":
            return self._init_gpt4all_embeddings()
        if provider == "google":
            return self._init_google_embeddings()
        raise ValueError(f"Unsupported embedding provider: {provider}")

    def _init_openai_embeddings(self) -> OpenAIEmbeddings:
        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY missing for embeddings")
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        model = self.embedding_model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=model)

    def _init_gpt4all_embeddings(self) -> GPT4AllEmbeddings:
        return GPT4AllEmbeddings()

    def _init_google_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if not self.google_api_key:
            self.logger.error("GOOGLE_API_KEY missing for embeddings")
            raise ValueError("GOOGLE_API_KEY is required for embeddings")
        model = self.embedding_model_name or "models/embedding-001"
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=self.google_api_key)

    def _init_text_splitter(self) -> BaseChunker:
        """
        Initialize text splitter based on configuration
        """
        strategy = getattr(app_settings.engine, 'chunking_strategy', 'adaptive')
        
        if strategy == "adaptive":
            return AdaptiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                openai_api_key=self.openai_api_key,
                model_name=self.llm_model_name
            )
        elif strategy == "semantic" and self.openai_api_key:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                return SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type=getattr(app_settings.engine, 'semantic_chunking_threshold', 'percentile')
                )
            except ImportError:
                self.logger.warning("SemanticChunker not available, falling back to adaptive")
                return AdaptiveChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    openai_api_key=self.openai_api_key
                )
        elif strategy == "hybrid":
            primary = AdaptiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                openai_api_key=self.openai_api_key
            )
            secondary = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size // 2,
                chunk_overlap=self.chunk_overlap // 2,
                length_function=len,
            )
            return HybridChunker(primary, secondary)
        else:
            # Default recursive splitter
            use_tok = tiktoken is not None and self.llm_provider.lower() == "openai"
            if use_tok:
                try:
                    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        model_name=self.llm_model_name,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                except Exception:
                    self.logger.warning("Tiktoken splitter failed, falling back to default")
    
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            
    def _init_search_tool(self) -> Optional[Runnable]:
        try:
            from langchain_tavily import TavilySearch
            return TavilySearch(api_key=self.tavily_api_key, max_results=5)
        except ImportError:
            self.logger.warning("langchain-tavily is not installed. Please pip install langchain-tavily")
            return None
    
    def _create_document_relevance_grader_chain(self) -> Runnable:
        parser = PydanticOutputParser(pydantic_object=RelevanceGrade)
        
        prompt_template = (
            "You are a document relevance grader. Your task is to assess if a given document excerpt\n"
            "is relevant to the provided question.\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{format_instructions}\n\n"
            "Question: {question}\n\n"
            "Document Excerpt:\n---\n{document_content}\n---\n\n"
            "Provide your JSON response:"
        )
        
        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RelevanceGrade)
        self.logger.info("Created document_relevance_grader_chain with PydanticOutputParser.")
        return chain
    
    def _create_document_reranker_chain(self) -> Runnable:
        parser = PydanticOutputParser(pydantic_object=RerankScore)

        prompt_template = (
            "You are a document re-ranking expert. Your task is to score a document's relevance "
            "for directly answering the given question.\n"
            "Assign a relevance score from 0.0 (not relevant) to 1.0 (perfectly relevant).\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{format_instructions}\n\n"
            "Question: {question}\n\n"
            "Document Excerpt:\n---\n{document_content}\n---\n\n"
            "Provide your JSON response:"
        )

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RerankScore)

        self.logger.info("Created document_reranker_chain with PydanticOutputParser.")
        return chain
    
    def _create_query_analyzer_chain(self) -> Runnable:
        self.logger.info("Creating query analyzer chain.")
        parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

        prompt_template_str = (
            "You are an expert query understanding system. Analyze the given 'User Question'. "
            "Consider the 'Chat History' (if provided) for context, but primarily focus on the current question. "
            "Your goal is to understand the user's intent, the type of query, extract critical keywords, "
            "and identify any ambiguities.\n\n"
            "Respond with a JSON object matching this schema:\n{format_instructions}\n\n"
            "Chat History (if any):\n{chat_history_formatted}\n\n"
            "User Question:\n{question}\n\n"
            "Provide your JSON response:"
        )

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template_str,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )


        chain = prompt | self.json_llm | parser 
        return chain

    def _create_query_rewriter_chain(self) -> Runnable:
        self.logger.info("Creating query rewriter chain with chat history support.")
        system_prompt = (
            "You are a query optimization assistant. Given 'Chat History' (if any) and "
            "'Latest User Question', rewrite the question to be clear, specific, and self-contained "
            "for retrieval. If already clear, return it as is."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Latest User Question to rewrite:\n{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain

    def _create_answer_generation_chain(self) -> Runnable:
        self.logger.info("Creating answer generation chain with chat history and feedback support.")
        
        system_prompt = (
            "You are a helpful assistant. Your answer must be based *only* on the provided context documents.\n"
            "If the context lacks the answer, say you don't know. Do not invent details.\n"
            "Use 'Chat History' (if any) only to resolve references in the current question, "
            "but ground your answer in the current context.\n\n"
            "Current Context Documents:\n---\n{context}\n---\n\n"
            "{optional_regeneration_prompt_header_if_feedback}\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{regeneration_feedback_if_any}{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()        
        return chain


    def _create_grounding_check_chain(self) -> Runnable:
        self.logger.info("Creating answer grounding check chain.")
        parser = PydanticOutputParser(pydantic_object=GroundingCheck)
        prompt = ChatPromptTemplate.from_template(
            template=(
                "You are an Answer Grounding Checker. Your task is to verify if the 'Generated Answer' "
                "is FULLY supported by and ONLY uses information from the provided 'Context Documents'.\n"
                "Respond with a JSON matching this schema:\n{format_instructions}\n\n"
                "Context Documents:\n---\n{context}\n---\n\n"
                "Generated Answer:\n---\n{generation}\n---\n\n"
                "Provide your JSON response:"
            ),
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.json_llm | parser
        return chain

    async def _grounding_check_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Performing grounding check on generated answer...")
        context = state.get("context", "")
        generation = state.get("generation", "")
        question = state.get("original_question") or state.get("question", "")
        documents = state.get("documents", [])
    
        current_attempts = state.get("grounding_check_attempts", 0) + 1
        state["grounding_check_attempts"] = current_attempts
        state["regeneration_feedback"] = None
    
        if not context or not generation or "Error generating answer." in generation:
            self.logger.info("Skipping grounding check (no context or generation, or generation error).")
            return {**state, "regeneration_feedback": None, "grounding_check_attempts": current_attempts}
    
        try:
            if self.enable_advanced_grounding and self.advanced_grounding_checker:
                # Use advanced multi-level grounding check
                import asyncio
                
                try:
                    # Run advanced grounding check
                    advanced_results = await self.advanced_grounding_checker.perform_comprehensive_grounding_check(answer=generation,
                                                                                                                   context=context,
                                                                                                                   question=question,
                                                                                                                   documents=documents)
                    
                    # Process advanced results
                    overall_assessment = advanced_results.get("overall_assessment", {})
                    is_acceptable = overall_assessment.get("is_acceptable", False)
                    
                    if not is_acceptable:
                        # Generate detailed feedback based on advanced analysis
                        feedback_parts = []
                        
                        detailed_grounding = advanced_results.get("detailed_grounding")
                        if detailed_grounding and hasattr(detailed_grounding, 'unsupported_claims') and detailed_grounding.unsupported_claims:
                            feedback_parts.append(f"Unsupported claims found: {'; '.join(detailed_grounding.unsupported_claims[:3])}")
                        
                        consistency = advanced_results.get("consistency")
                        if consistency and hasattr(consistency, 'contradictions_found') and consistency.contradictions_found:
                            feedback_parts.append(f"Internal contradictions: {'; '.join(consistency.contradictions_found[:2])}")
                        
                        completeness = advanced_results.get("completeness")
                        if completeness and hasattr(completeness, 'missing_aspects') and completeness.missing_aspects:
                            feedback_parts.append(f"Missing important aspects: {'; '.join(completeness.missing_aspects[:2])}")
                        
                        hallucinations = advanced_results.get("hallucination_detection", {}).get("hallucinations", [])
                        if hallucinations:
                            feedback_parts.append(f"Potential hallucinations: {'; '.join(hallucinations[:2])}")
                        
                        if feedback_parts:
                            recommendation = overall_assessment.get("recommendation", "Please improve the answer")
                            regeneration_prompt = (
                                f"The previous answer to '{question}' failed advanced verification. "
                                f"Issues identified: {' | '.join(feedback_parts)}. "
                                f"Recommendation: {recommendation}. "
                                f"Please generate a new answer that strictly follows the provided context and addresses these issues."
                            )
                            state["regeneration_feedback"] = regeneration_prompt
                            self.logger.warning(f"Advanced grounding check FAILED (Attempt {current_attempts}). Issues: {len(feedback_parts)}")
                        else:
                            # No specific issues identified, use general feedback
                            state["regeneration_feedback"] = (
                                f"The answer to '{question}' did not meet quality standards. "
                                f"Please generate a more accurate answer based strictly on the provided context."
                            )
                    else:
                        self.logger.info(f"Advanced grounding check PASSED (Attempt {current_attempts}). Score: {overall_assessment.get('overall_score', 'N/A')}")
                        state["regeneration_feedback"] = None
                    
                    # Store detailed results for potential debugging
                    state["advanced_grounding_results"] = advanced_results
                    
                except Exception as e:
                    self.logger.error(f"Advanced grounding check failed with exception: {e}. Falling back to basic check.")
                    # Fall back to basic grounding check
                    result: GroundingCheck = self.grounding_check_chain.invoke({
                        "context": context,
                        "generation": generation
                    })
                    
                    if not result.is_grounded:
                        feedback_parts = []
                        if result.ungrounded_statements:
                            feedback_parts.append(f"Ungrounded statements: {'; '.join(result.ungrounded_statements)}")
                        if result.correction_suggestion:
                            feedback_parts.append(f"Suggestion: {result.correction_suggestion}")
                        
                        regeneration_prompt = (
                            f"The previous answer to '{question}' was not well-grounded. "
                            f"{' '.join(feedback_parts)} "
                            f"Please generate a new answer focusing ONLY on the provided documents."
                        )
                        state["regeneration_feedback"] = regeneration_prompt
                        self.logger.warning(f"Basic grounding check FAILED (Attempt {current_attempts})")
                    else:
                        state["regeneration_feedback"] = None
                        self.logger.info(f"Basic grounding check PASSED (Attempt {current_attempts})")
            else:
                # Use basic grounding check
                result: GroundingCheck = self.grounding_check_chain.invoke({
                    "context": context,
                    "generation": generation
                })
    
                if not result.is_grounded:
                    feedback_parts = []
                    if result.ungrounded_statements:
                        feedback_parts.append(f"The following statements were ungrounded: {'; '.join(result.ungrounded_statements)}.")
                    if result.correction_suggestion:
                        feedback_parts.append(f"Suggestion for correction: {result.correction_suggestion}.")
    
                    if not feedback_parts:
                        feedback_parts.append("The answer was not fully grounded in the provided context. Please revise.")
    
                    regeneration_prompt = (
                        f"The previous answer to the question '{question}' was not well-grounded. "
                        f"{' '.join(feedback_parts)} "
                        "Please generate a new answer focusing ONLY on the provided documents and addressing these issues."
                    )
                    state["regeneration_feedback"] = regeneration_prompt
                    self.logger.warning(f"Basic grounding check FAILED (Attempt {current_attempts}). Feedback: {regeneration_prompt}")
                else:
                    self.logger.info(f"Basic grounding check PASSED (Attempt {current_attempts}).")
                    state["regeneration_feedback"] = None
    
        except Exception as e:
            self.logger.error(f"Grounding check failed with exception: {e}", exc_info=True)
            state["regeneration_feedback"] = f"A system error occurred during grounding check: {e}. Please try to generate a concise answer based on context."
            state["error_message"] = (state.get("error_message") or "") + f" | Grounding check exception: {e}"
    
        return state
        
    def _route_after_grounding_check(self, state: CoreGraphState) -> str:
        self.logger.info(
            f"Routing after grounding check. Attempts: {state.get('grounding_check_attempts', 0)}. "
            f"Feedback: {'Yes' if state.get('regeneration_feedback') else 'No'}"
        )

        max_attempts = getattr(self, 'max_grounding_attempts', 1)

        if state.get("regeneration_feedback") is None:
            self.logger.info("Answer is grounded. Ending.")
            return END

        if state.get("grounding_check_attempts", 0) >= max_attempts:
            self.logger.warning(f"Max grounding attempts ({max_attempts}) reached. Answer still not grounded. Ending with warning.")
            original_generation = state.get("generation", "")
            warning_message = (
                "**Self-Correction Incomplete:** The following answer could not be fully verified against the provided documents after attempts to correct it. "
                "Please use with caution.\n---\n"
            )
            state["generation"] = warning_message + original_generation
            return END

        self.logger.info("Answer not grounded, and attempts remain. Routing back to generate_answer.")
        return "generate_answer"

    # ---------------------
    # Wrapper Node Methods
    # ---------------------
    def _retrieve_node(self, state: CoreGraphState) -> CoreGraphState:
        collection_name = state.get("collection_name") or self.default_collection_name
        self.logger.info(f"NODE: Retrieving documents from collection '{collection_name}' for question: '{state['question']}'")

        # Ensure vectorstore and retriever exist
        self._init_or_load_vectorstore(collection_name, recreate=False)
        retriever = self.retrievers.get(collection_name)

        if not retriever:
            self.logger.warning(f"No retriever found for collection '{collection_name}'. Returning empty documents.")
            state["documents"] = []
            state["context"] = "Retriever not available for the specified collection."
            return state

        current_question = state["question"]
        query_analysis: Optional[QueryAnalysis] = state.get("query_analysis_results")

        # 1. Determine dynamic top_k
        retrieval_k = self.default_retrieval_top_k
        if query_analysis:
            self.logger.info(f"Using query analysis for retrieval: Type='{query_analysis.query_type}', Keywords='{query_analysis.extracted_keywords}'")
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

        # 3. Temporarily override retriever.k or retriever.search_kwargs['k'] if possible
        original_k = None
        try:
            if hasattr(retriever, "k"):
                original_k = retriever.k
                retriever.k = retrieval_k
            elif hasattr(retriever, "search_kwargs") and "k" in retriever.search_kwargs:
                original_k = retriever.search_kwargs["k"]
                retriever.search_kwargs["k"] = retrieval_k

            self.logger.info(f"Attempting to retrieve top {retrieval_k} docs for: '{search_query}'")
            docs = retriever.invoke(search_query)
            self.logger.info(f"Retrieved {len(docs)} documents.")
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {e}", exc_info=True)
            state["documents"] = []
            state["context"] = "Error occurred during document retrieval."
            state["error_message"] = (state.get("error_message") or "") + f" | Retrieval error: {e}"
            # Restore original k if set
            if original_k is not None:
                if hasattr(retriever, "k"):
                    retriever.k = original_k
                elif hasattr(retriever, "search_kwargs") and "k" in retriever.search_kwargs:
                    retriever.search_kwargs["k"] = original_k
            return state

        # Restore original k if it was changed
        if original_k is not None:
            if hasattr(retriever, "k"):
                retriever.k = original_k
            elif hasattr(retriever, "search_kwargs") and "k" in retriever.search_kwargs:
                retriever.search_kwargs["k"] = original_k

        # 4. Update state
        state["documents"] = docs
        state["context"] = "\n\n".join(d.page_content for d in docs)
        return state
    
    def _rerank_documents_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Re-ranking retrieved documents...")
        question = state.get("original_question") or state["question"]
        docs = state.get("documents", [])

        if not docs:
            self.logger.info("No documents to re-rank.")
            return state

        docs_with_scores = []
        for doc in docs:
            try:
                score_result: RerankScore = self.document_reranker_chain.invoke({
                    "question": question,
                    "document_content": doc.page_content
                })
                docs_with_scores.append((doc, score_result.relevance_score))
                self.logger.debug(f"Document from '{doc.metadata.get('source')}' scored: {score_result.relevance_score}")
            except Exception as e:
                self.logger.error(f"Error re-ranking document: {e}. Assigning score of 0.")
                docs_with_scores.append((doc, 0.0))

        # Sort documents by score in descending order
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        # Update state with the re-ranked documents
        state["documents"] = [doc for doc, score in sorted_docs]
        self.logger.info("Finished re-ranking documents.")
        
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
        try:
            name = collection_name or self.default_collection_name
            self.logger.info(f"Starting ingestion for collection '{name}'. Recreate: {recreate_collection}")

            all_chunks_for_collection: List[Document] = []

            if direct_documents is not None and isinstance(direct_documents, list) and direct_documents:
                if all(isinstance(doc, Document) for doc in direct_documents):
                    self.logger.info(f"Using {len(direct_documents)} preloaded documents for ingestion.")
                    all_chunks_for_collection = self.split_documents(direct_documents)
                else:
                    self.logger.error("'direct_documents' provided but some items are not Document instances.")
            elif sources is not None:
                self.logger.info("Processing documents from `sources` parameter.")
                if not isinstance(sources, list):
                    sources = [sources]
                for src in sources:
                    src_type = src.get("type")
                    src_val  = src.get("value")
                    if not src_type or src_val is None:
                        self.logger.warning(f"Skipping invalid source: {src}")
                        continue
                    raw_docs = self.load_documents(source_type=src_type, source_value=src_val)
                    if raw_docs:
                        chunks = self.split_documents(raw_docs)
                        all_chunks_for_collection.extend(chunks)

            if not all_chunks_for_collection:
                if recreate_collection:
                    self.logger.info(f"No docs but recreate_collection=True. Clearing '{name}'.")
                    self._init_or_load_vectorstore(name, recreate=True)
                else:
                    self.logger.warning(f"No documents to ingest for '{name}'. Skipping.")
                return

            self.index_documents(
                docs=all_chunks_for_collection,
                name=name,
                recreate=recreate_collection
            )
        except Exception as e:
            self.logger.error(f"Ingestion failed for collection '{name}': {e}", exc_info=True)
            raise

    def answer_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
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

    # ---------------------
    # Public API: Full Workflow
    # ---------------------
    def run_full_rag_workflow(
        self,
        question: str,
        collection_name: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
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
        final = self.rag_workflow.invoke(initial_state)

        answer = final.get("generation", "")
        docs   = final.get("documents", [])
        sources = [
            {"source": d.metadata.get("source", "unknown"),
             "preview": d.page_content[:200] + "..."}
            for d in docs
        ]
        return {"answer": answer, "sources": sources}
