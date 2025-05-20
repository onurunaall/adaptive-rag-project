import os
import logging
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Union, TypedDict

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field

# Optional support for Streamlit uploads
try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ImportError:
    UploadedFile = None

load_dotenv()

PROMPT_TEMPLATE = (
    "Answer the question based only on the following context.\n"
    "If the context does not contain the answer, state that you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

try:
    import tiktoken
except ImportError:
    tiktoken = None


class CoreGraphState(TypedDict):
    """
    Placeholder state for future LangGraph-based workflows.
    """
    question: str
    original_question: Optional[str]
    documents: List[Document]
    context: str
    web_search_results: Optional[List[Document]]
    generation: str
    retries: int
    run_web_search: str  # "Yes" or "No"
    error_message: Optional[str]
    collection_name: Optional[str]


class CoreRAGEngine:
    """
    Core RAG engine:
      - Ingest documents from URLs, PDFs, text files, or Streamlit uploads
      - Split and index into Chroma vector stores with persistence
      - Answer queries by retrieving relevant chunks and invoking an LLM
      - Stub for future full LangGraph workflows
    """

    def __init__(
        self,
        llm_provider: str = os.getenv("LLM_PROVIDER", "openai"),
        llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai"),
        embedding_model_name: Optional[str] = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        temperature: float = 0.0,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        default_collection_name: str = "insight_engine_default",
        persist_directory_base: Optional[str] = None,
        tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY"),
        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY"),
    ):
        # Save configuration
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_collection_name = default_collection_name
        self.tavily_api_key = tavily_api_key
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key

        # Determine persistence directory base
        if persist_directory_base:
            self.persist_directory_base = persist_directory_base
        else:
            system_temp = tempfile.gettempdir()
            self.persist_directory_base = os.path.join(system_temp, "core_rag_engine_chroma")
        os.makedirs(self.persist_directory_base, exist_ok=True)

        # Setup logger
        self._setup_logger()

        # Initialize LLMs
        self.llm = self._init_llm(use_json_format=False)
        self.json_llm = self._init_llm(use_json_format=True)

        # Initialize embedding model
        self.embedding_model = self._init_embedding_model()

        # Initialize text splitter
        self.text_splitter = self._init_text_splitter()

        # Initialize optional web search tool
        self.search_tool = self._init_search_tool()

        # Build the document relevance grader chain
        self.document_relevance_grader_chain = self._create_document_relevance_grader_chain()


        # Storage for each collection's vector store and retriever
        self.vectorstores: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, Any] = {}

        # Placeholder for future workflow
        self.rag_workflow: Optional[Any] = None

        self.logger.info(
            f"Engine initialized: LLM={self.llm_provider}/{self.llm_model_name}, "
            f"Embeddings={self.embedding_provider}/{self.embedding_model_name}, "
            f"Chunks={self.chunk_size}/{self.chunk_overlap}, "
            f"Persistence={self.persist_directory_base}"
        )

    def _setup_logger(self) -> None:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self.logger = logger

    # ---------------------
    # LLM Initialization
    # ---------------------
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

        constructor_args: Dict[str, Any] = {
            "model": self.llm_model_name,
            "temperature": self.temperature,
            "openai_api_key": self.openai_api_key,
        }

        if use_json:
            name = self.llm_model_name.lower()
            is_json_model = (
                "gpt-4o" in name
                or "gpt-4" in name
                or name.startswith("gpt-3.5-turbo-")
            )
            constructor_args["model_kwargs"] = {"response_format": {"type": "json_object"}}
            if is_json_model:
                self.logger.info(f"Native JSON mode enabled for {self.llm_model_name}")
            else:
                self.logger.warning(
                    f"JSON mode forced for {self.llm_model_name}; actual support may vary"
                )

        try:
            return ChatOpenAI(**constructor_args)
        except Exception as e:
            self.logger.error(f"OpenAI LLM init failed for model {self.llm_model_name}: {e}")
            raise ValueError(f"Failed to initialize OpenAI LLM {self.llm_model_name}") from e

    def _init_ollama_llm(self, use_json: bool) -> ChatOllama:
        fmt = "json" if use_json else None
        try:
            return ChatOllama(model=self.llm_model_name, temperature=self.temperature, format=fmt)
        except Exception as e:
            self.logger.error(f"Ollama LLM init failed: {e}")
            raise

    def _init_google_llm(self, use_json: bool) -> ChatGoogleGenerativeAI:
        if not self.google_api_key:
            self.logger.error("GOOGLE_API_KEY missing")
            raise ValueError("GOOGLE_API_KEY is required")
        try:
            return ChatGoogleGenerativeAI(
                model=self.llm_model_name,
                temperature=self.temperature,
                google_api_key=self.google_api_key,
                convert_system_message_to_human=True
            )
        except Exception as e:
            self.logger.error(f"Google LLM init failed: {e}")
            raise

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
        try:
            return OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=model)
        except Exception as e:
            self.logger.error(f"OpenAI Embeddings init failed: {e}")
            raise

    def _init_gpt4all_embeddings(self) -> GPT4AllEmbeddings:
        try:
            return GPT4AllEmbeddings()
        except Exception as e:
            self.logger.error(f"GPT4All Embeddings init failed: {e}")
            raise

    def _init_google_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if not self.google_api_key:
            self.logger.error("GOOGLE_API_KEY missing for embeddings")
            raise ValueError("GOOGLE_API_KEY is required for embeddings")
        model = self.embedding_model_name or "models/embedding-001"
        try:
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=self.google_api_key)
        except Exception as e:
            self.logger.error(f"Google Embeddings init failed: {e}")
            raise

    # ---------------------
    # Text Splitter Initialization
    # ---------------------
    def _init_text_splitter(self) -> RecursiveCharacterTextSplitter:
        use_tiktoken = tiktoken is not None and self.llm_provider.lower() == "openai"
        if use_tiktoken:
            try:
                return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name=self.llm_model_name,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            except Exception:
                self.logger.warning("Tiktoken splitter failed, falling back")
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    # ---------------------
    # Search Tool Initialization
    # ---------------------
    def _init_search_tool(self) -> Optional[TavilySearchResults]:
        if not self.tavily_api_key:
            self.logger.warning("TAVILY_API_KEY missing, web search disabled")
            return None
        try:
            return TavilySearchResults(api_key=self.tavily_api_key)
        except Exception as e:
            self.logger.error(f"TavilySearchResults init failed: {e}")
            return None

        def _create_document_relevance_grader_chain(self) -> Any:
        """
        Chain that decides if a document chunk is relevant to the question.
        Inputs:
          - question (str)
          - document_context (str)
        Output: "yes" or "no"
        """
        prompt = ChatPromptTemplate.from_template(
            "Given the question and a document excerpt, answer 'yes' if the excerpt is relevant to answering the question, otherwise answer 'no'.\n\n"
            "Question: {question}\n"
            "Excerpt: {document_context}\n\n"
            "Relevant?"
        )

        chain = LLMChain(
            llm=self.json_llm,
            prompt=prompt,
            output_parser=StrOutputParser()
        )

        return chain

    # ---------------------
    # Persistence Helpers
    # ---------------------
    def _get_persist_dir(self, collection_name: str) -> str:
        return os.path.join(self.persist_directory_base, collection_name)

    def _init_or_load_vectorstore(self, collection_name: str, recreate: bool) -> None:
        persist_dir = self._get_persist_dir(collection_name)

        if recreate and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        if os.path.exists(persist_dir) and collection_name in self.vectorstores:
            return

        if os.path.exists(persist_dir):
            vs = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=persist_dir
            )
            self.vectorstores[collection_name] = vs
            self.retrievers[collection_name] = vs.as_retriever()
        else:
            self.vectorstores[collection_name] = None
            self.retrievers[collection_name] = None

    # ---------------------
    # Document I/O
    # ---------------------
    def load_documents(self, source_type: str, source_value: Any) -> List[Document]:
        """
        Load documents from:
          - "url": source_value is URL str
          - "pdf_path": source_value is file path str
          - "text_path": source_value is file path str
          - "uploaded_pdf": source_value is Streamlit UploadedFile
        """
        # Instantiate loader
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
            self.logger.error(f"Error creating loader for {source_type}: {e}")
            return []

        # Load docs
        try:
            docs = loader.load()
            self.logger.info(f"Loaded {len(docs)} docs from {source_type}")
            return docs
        except Exception as e:
            self.logger.error(f"Error loading docs from {source_type}: {e}")
            return []
        finally:
            if source_type == "uploaded_pdf" and 'tmp' in locals():
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split loaded documents into smaller chunks.
        """
        if not documents:
            return []
        chunks = self.text_splitter.split_documents(documents)
        self.logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    # ---------------------
    # Indexing
    # ---------------------
    def index_documents(self, documents: List[Document], collection_name: str, recreate: bool = False) -> None:
        """
        Index document chunks into Chroma. Recreate store if requested.
        """
        self._init_or_load_vectorstore(collection_name, recreate)
        vs = self.vectorstores[collection_name]
        persist_dir = self._get_persist_dir(collection_name)

        if vs is None or recreate:
            vs_new = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection_name=collection_name,
                persist_directory=persist_dir
            )
            self.vectorstores[collection_name] = vs_new
            self.retrievers[collection_name] = vs_new.as_retriever()
            self.logger.info(f"Created new vector store '{collection_name}'")
        else:
            vs.add_documents(documents=documents)
            self.logger.info(f"Added {len(documents)} docs to '{collection_name}'")

    # ---------------------
    # Public API: ingest
    # ---------------------
    def ingest(
        self,
        sources: Union[Dict[str, Any], List[Dict[str, Any]]],
        collection_name: Optional[str] = None,
        recreate_collection: bool = False
    ) -> None:
        """
        Ingest one or more sources into the named collection.
        Each source is a dict: {"type": ..., "value": ...}.
        """
        name = collection_name or self.default_collection_name
        if not isinstance(sources, list):
            sources = [sources]

        all_chunks: List[Document] = []
        for src in sources:
            src_type = src.get("type")
            src_val = src.get("value")
            raw = self.load_documents(src_type, src_val)
            chunks = self.split_documents(raw)
            all_chunks.extend(chunks)

        if all_chunks:
            self.index_documents(all_chunks, name, recreate_collection)
        elif recreate_collection:
            # Clear existing store if requested
            self._init_or_load_vectorstore(name, recreate=True)

    # ---------------------
    # Public API: answer_query
    # ---------------------
    def answer_query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieve context and generate an answer.
        Returns {"answer": str, "sources": List[{"source":..., "preview":...}]}.
        """
        name = collection_name or self.default_collection_name
        self._init_or_load_vectorstore(name, recreate=False)
        retriever = self.retrievers.get(name)
        if not retriever:
            msg = "No documents indexed in this collection."
            self.logger.warning(msg)
            return {"answer": msg, "sources": []}

        docs = retriever.get_relevant_documents(question)[:top_k]
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            answer = self.llm.predict(prompt)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return {"answer": "Error generating answer.", "sources": []}

        sources_info = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:200] + "..."
            sources_info.append({"source": src, "preview": preview})

        return {"answer": answer.strip(), "sources": sources_info}

    # ---------------------
    # Public API: full workflow stub
    # ---------------------
    def run_full_rag_workflow(
        self,
        question: str,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Placeholder for future LangGraph-based RAG workflow.
        Currently delegates to basic answer_query.
        """
        self.logger.info("Running full RAG workflow (stub)")
        return self.answer_query(question, collection_name)
