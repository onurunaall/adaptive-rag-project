"""
Factory for creating LangChain chains used in the RAG workflow.
"""
import logging
from typing import Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from src.rag.models import GroundingCheck, RelevanceGrade, RerankScore, QueryAnalysis


class ChainFactory:
    """Factory class for creating various LangChain chains used in RAG workflows."""

    def __init__(
        self,
        llm: Any,
        json_llm: Any,
        tavily_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the chain factory.

        Args:
            llm: The primary LLM instance for generation
            json_llm: The JSON-formatted LLM instance for structured outputs
            tavily_api_key: API key for Tavily search (optional)
            logger: Logger instance for logging
        """
        self.llm = llm
        self.json_llm = json_llm
        self.tavily_api_key = tavily_api_key
        self.logger = logger or logging.getLogger(__name__)

    def create_query_analyzer_chain(self) -> Runnable:
        """
        Create chain for analyzing query characteristics.

        This method constructs a LangChain Runnable that analyzes a user's query to
        determine its type, intent, key terms, and whether clarification is needed.
        Returns a structured `QueryAnalysis` object.

        Returns:
            Runnable: A configured chain for query analysis.
        """
        try:
            self.logger.info("Creating query analyzer chain.")
            parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

            prompt_template = (
                "You are a query analysis expert. Your task is to analyze the user's query "
                "and chat history to understand intent, extract key information, and classify the query type.\n\n"
                "Chat History:\n---\n{chat_history_formatted}\n---\n\n"
                "Current User Question: {question}\n\n"
                "Analyze this query and respond with a JSON object matching this schema:\n"
                "{format_instructions}\n\n"
                "Query Type Guidelines:\n"
                "- 'factual_lookup': Questions seeking specific facts or definitions\n"
                "- 'comparison': Questions comparing multiple items/concepts\n"
                "- 'summary_request': Requests for summaries or overviews\n"
                "- 'complex_reasoning': Questions requiring multi-step reasoning\n"
                "- 'ambiguous': Unclear or vague questions needing clarification\n"
                "- 'keyword_search_sufficient': Simple keyword-based lookups\n"
                "- 'greeting': Casual greetings or chitchat\n"
                "- 'not_a_question': Statements or commands\n\n"
                "Provide your JSON response:"
            )

            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(
                template=prompt_template, partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt | self.json_llm | parser

            self.logger.info("Query analyzer chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create query analyzer chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def create_document_reranker_chain(self) -> Runnable:
        """
        Create chain for re-ranking documents by relevance score.
        """
        try:
            self.logger.info("Creating document re-ranker chain.")
            parser = PydanticOutputParser(pydantic_object=RerankScore)

            prompt_template = (
                "You are a document relevance scorer. Rate how relevant this document is "
                "to answering the given question.\n\n"
                "Question: {question}\n\n"
                "Document Content:\n---\n{document_content}\n---\n\n"
                "Respond with a JSON object matching this schema:\n"
                "{format_instructions}\n\n"
                "Provide your JSON response:"
            )

            prompt = ChatPromptTemplate.from_template(
                template=prompt_template, partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt | self.json_llm | parser

            self.logger.info("Document re-ranker chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create document re-ranker chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def create_document_relevance_grader_chain(self) -> Runnable:
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
            prompt = ChatPromptTemplate.from_template(
                template=prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RelevanceGrade)

            self.logger.info("Document relevance grader chain created successfully with PydanticOutputParser.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create document relevance grader chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def create_query_rewriter_chain(self) -> Runnable:
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
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "Latest User Question to rewrite:\n{question}"),
                ]
            )

            chain = prompt | self.llm | StrOutputParser()

            self.logger.info("Query rewriter chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create query rewriter chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def create_answer_generation_chain(self) -> Runnable:
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
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{regeneration_feedback_if_any}{question}"),
                ]
            )

            chain = prompt | self.llm | StrOutputParser()

            self.logger.info("Answer generation chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create answer generation chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def create_grounding_check_chain(self) -> Runnable:
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
            prompt = ChatPromptTemplate.from_template(
                template=prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | self.json_llm | parser

            self.logger.info("Answer grounding check chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create answer grounding check chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def create_search_tool(self) -> Optional[Runnable]:
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
        except ImportError:
            warning_msg = "langchain-tavily is not installed. Web search functionality will be disabled. "
            self.logger.warning(warning_msg)
            return None
        except Exception as e:
            error_msg = f"Failed to initialize Tavily search tool: {e}"
            self.logger.error(error_msg, exc_info=True)
            return None
