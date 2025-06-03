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
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field

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
    ungrounded_statements: Optional[List[str]] = Field(
        default=None,
        description="List any specific statements in the answer that are NOT supported by the context."
    )
    correction_suggestion: Optional[str] = Field(
        default=None,
        description="If not grounded, suggest how to rephrase the answer to be grounded, or state no grounded answer is possible."
    )

class CoreGraphState(TypedDict):
    question: str
    original_question: Optional[str]
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

class RelevanceGrade(BaseModel):
    """
    Structured output for document relevance grading.
    """
    is_relevant: bool = Field(
        description="Is the document excerpt relevant to the question? True or False."
    )
    justification: Optional[str] = Field(
        default=None,
        description="Brief justification for the relevance decision (1-2 sentences)."
    )

class CoreRAGEngine:
    """
    Core RAG engine: ingestion, indexing, and adaptive querying via LangGraph.
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
        max_rewrite_retries: int = 1,
        max_grounding_attempts: int = 1,
    ):
        # Configuration
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
        self.max_rewrite_retries = max_rewrite_retries
        self.max_grounding_attempts = max_grounding_attempts

        # Persistence directory
        base = persist_directory_base or tempfile.gettempdir()
        self.persist_directory_base = os.path.join(base, "core_rag_engine_chroma")
        os.makedirs(self.persist_directory_base, exist_ok=True)

        # Logger
        self._setup_logger()

        # LLMs
        self.llm = self._init_llm(use_json_format=False)
        self.json_llm = self._init_llm(use_json_format=True)

        # Embeddings
        self.embedding_model = self._init_embedding_model()

        # Splitter and search tool
        self.text_splitter = self._init_text_splitter()
        self.search_tool = self._init_search_tool()

        # Chains
        self.document_relevance_grader_chain = self._create_document_relevance_grader_chain()
        self.query_rewriter_chain         = self._create_query_rewriter_chain()
        self.answer_generation_chain      = self._create_answer_generation_chain()
        self.grounding_check_chain        = self._create_grounding_check_chain()

        # Compile workflow
        self._compile_rag_workflow()

        # Storage for indexes
        self.vectorstores: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, Any] = {}

        self.logger.info("CoreRAGEngine initialized and workflow compiled.")

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
        p = self.llm_provider.lower()
        if p == "openai":
            return self._init_openai_llm(use_json_format)
        if p == "ollama":
            return self._init_ollama_llm(use_json_format)
        if p == "google":
            return self._init_google_llm(use_json_format)
        raise ValueError(f"Unsupported LLM provider: {p}")

    def _init_openai_llm(self, use_json: bool) -> ChatOpenAI:
        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY missing")
            raise ValueError("OPENAI_API_KEY is required")
        args: Dict[str, Any] = {
            "model": self.llm_model_name,
            "temperature": self.temperature,
            "openai_api_key": self.openai_api_key
        }
        if use_json:
            args["model_kwargs"] = {"response_format": {"type": "json_object"}}
            self.logger.info(f"JSON mode enabled for {self.llm_model_name}")
        return ChatOpenAI(**args)

    def _init_ollama_llm(self, use_json: bool) -> ChatOllama:
        fmt = "json" if use_json else None
        return ChatOllama(model=self.llm_model_name, temperature=self.temperature, format=fmt)

    def _init_google_llm(self, use_json: bool) -> ChatGoogleGenerativeAI:
        if not self.google_api_key:
            self.logger.error("GOOGLE_API_KEY missing")
            raise ValueError("GOOGLE_API_KEY is required")
        kwargs: Dict[str, Any] = {
            "model": self.llm_model_name,
            "temperature": self.temperature,
            "google_api_key": self.google_api_key,
            "convert_system_message_to_human": True
        }
        if use_json:
            kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
            self.logger.info(f"JSON mode enabled for Google model {self.llm_model_name}")
        return ChatGoogleGenerativeAI(**kwargs)

    # ---------------------
    # Embedding Initialization
    # ---------------------
    def _init_embedding_model(self) -> Any:
        p = self.embedding_provider.lower()
        if p == "openai":
            return self._init_openai_embeddings()
        if p == "gpt4all":
            return self._init_gpt4all_embeddings()
        if p == "google":
            return self._init_google_embeddings()
        raise ValueError(f"Unsupported embedding provider: {p}")

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

    # ---------------------
    # Text Splitter
    # ---------------------
    def _init_text_splitter(self) -> RecursiveCharacterTextSplitter:
        use_tok = tiktoken is not None and self.llm_provider.lower() == "openai"
        if use_tok:
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
    # Search Tool
    # ---------------------
    def _init_search_tool(self) -> Optional[TavilySearchResults]:
        if not self.tavily_api_key:
            self.logger.warning("TAVILY_API_KEY missing, web search disabled")
            return None
        return TavilySearchResults(api_key=self.tavily_api_key)

    # ---------------------
    # Chain Factories
    # ---------------------
    def _create_document_relevance_grader_chain(self) -> LLMChain:
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
        chain = LLMChain(
            llm=self.json_llm,
            prompt=prompt,
            output_parser=parser,
            verbose=False
        )
        self.logger.info("Created document_relevance_grader_chain with PydanticOutputParser.")
        return chain

    def _create_query_rewriter_chain(self) -> LLMChain:
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
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            verbose=False
        )

    def _create_answer_generation_chain(self) -> LLMChain:
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
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            verbose=False
        )

    def _create_grounding_check_chain(self) -> LLMChain:
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
        return LLMChain(
            llm=self.json_llm,
            prompt=prompt,
            output_parser=parser,
            verbose=False
        )

    def _grounding_check_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Performing grounding check on generated answer...")
        context    = state.get("context", "")
        generation = state.get("generation", "")
        question   = state.get("original_question") or state.get("question", "")

        current_attempts = state.get("grounding_check_attempts", 0) + 1
        state["grounding_check_attempts"] = current_attempts
        state["regeneration_feedback"] = None

        if not context or not generation or "Error generating answer." in generation:
            self.logger.info("Skipping grounding check (no context or generation, or generation error).")
            return {**state, "regeneration_feedback": None, "grounding_check_attempts": current_attempts}

        try:
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
                self.logger.warning(f"Grounding check FAILED (Attempt {current_attempts}). Feedback: {regeneration_prompt}")
            else:
                self.logger.info(f"Grounding check PASSED (Attempt {current_attempts}).")
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
                "⚠️ **Self-Correction Incomplete:** The following answer could not be fully verified against the provided documents after attempts to correct it. "
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
        name = state.get("collection_name") or self.default_collection_name
        self._init_or_load_vectorstore(name, recreate=False)
        retriever = self.retrievers.get(name)
        if not retriever:
            state["documents"] = []
            state["context"] = ""
            return state
        docs = retriever.get_relevant_documents(state["question"])
        state["documents"] = docs
        state["context"] = "\n\n".join(d.page_content for d in docs)
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

    def _rewrite_query_node(self, state: CoreGraphState) -> CoreGraphState:
        self.logger.info("NODE: Rewriting query")
        original_question = state.get("original_question") or state["question"]
        try:
            result = self.query_rewriter_chain.invoke({
                "question": original_question,
                "chat_history": state.get("chat_history", [])
            })
            rewritten_query = result.get("text", "").strip()
            if rewritten_query and rewritten_query.lower() != original_question.lower():
                self.logger.info(f"Rewrote '{original_question}' → '{rewritten_query}'")
                state["question"] = rewritten_query
            else:
                self.logger.info(f"No rewrite needed. Keeping '{original_question}'")
                state["question"] = original_question
        except Exception as e:
            self.logger.error(f"Error during query rewriting: {e}", exc_info=True)
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
            generated_text = self.answer_generation_chain.run(input_data_for_chain)
            state["generation"] = generated_text.strip()
        except Exception as e:
            self.logger.error(f"Generation error: {e}", exc_info=True)
            state["generation"] = "Error generating answer."
            state["error_message"] = str(e)
        return state

    def _route_after_grading(self, state: CoreGraphState) -> str:
        passed = state.get("relevance_check_passed", False)
        retries = state.get("retries", 0)
        self.logger.info(f"Routing after grading: passed={passed}, retries={retries}")
        if passed:
            return "generate_answer"
        if retries < self.max_rewrite_retries:
            state["retries"] = retries + 1
            return "rewrite_query"
        if self.search_tool:
            state["run_web_search"] = "Yes"
            return "web_search"
        state["context"] = state.get("context", "") or "No relevant information found."
        return "generate_answer"

    # ---------------------
    # Workflow Compilation
    # ---------------------
    def _compile_rag_workflow(self) -> Any:
        graph = StateGraph(CoreGraphState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("grade_documents", self._grade_documents_node)
        graph.add_node("rewrite_query", self._rewrite_query_node)
        graph.add_node("web_search", self._web_search_node)
        graph.add_node("generate_answer", self._generate_answer_node)
        graph.add_node("grounding_check", self._grounding_check_node)
        graph.set_entry_point("retrieve")

        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "web_search": "web_search"
            }
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "grounding_check")
        graph.add_conditional_edges(
            "grounding_check",
            self._route_after_grounding_check,
            {
                "generate_answer": "generate_answer",
                END: END
            }
        )

        self.rag_workflow = graph.compile()
        self.logger.info("RAG workflow compiled successfully.")

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
        chunks = self.text_splitter.split_documents(docs)
        self.logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    def index_documents(self, docs: List[Document], name: str, recreate: bool = False) -> None:
        self._init_or_load_vectorstore(name, recreate)
        vs = self.vectorstores.get(name)
        d = self._get_persist_dir(name)
        if vs is None or recreate:
            vs_new = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                collection_name=name,
                persist_directory=d
            )
            self.vectorstores[name] = vs_new
            self.retrievers[name] = vs_new.as_retriever()
            self.logger.info(f"Created vector store '{name}'")
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
        name = collection_name or self.default_collection_name
        self.logger.info(f"Starting ingestion for collection '{name}'. Recreate: {recreate_collection}")

        all_chunks_for_collection: List[Document] = []

        if direct_documents is not None and isinstance(direct_documents, list) and direct_documents:
            if all(isinstance(doc, Document) for doc in direct_documents):
                self.logger.info(f"Using {len(direct_documents)} preloaded documents for ingestion.")
                all_chunks_for_collection = self.split_documents(direct_documents)
            else:
                self.logger.error("`direct_documents` provided but some items are not Document instances.")
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

    # ---------------------
    # Public API: Legacy Query
    # ---------------------
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
            ans = self.llm.predict(prompt)
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
