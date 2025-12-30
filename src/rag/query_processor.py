"""
Query Processor Module

Handles query analysis and rewriting operations.
Extracted from CoreRAGEngine to improve modularity and maintainability.
"""

import logging
from typing import List, Optional, Any

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import BaseMessage

from src.rag.models import QueryAnalysis


class QueryProcessor:
    """
    Manages query analysis and rewriting operations.

    Responsibilities:
    - Analyze queries to determine type, intent, and characteristics
    - Rewrite queries to be clear, specific, and self-contained
    - Format chat history for query context
    """

    def __init__(
        self,
        llm: Any,
        json_llm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize QueryProcessor.

        Args:
            llm: Language model for text generation (query rewriting)
            json_llm: Language model for JSON-structured output (query analysis)
            logger: Optional logger instance
        """
        self.llm = llm
        self.json_llm = json_llm
        self.logger = logger or logging.getLogger(__name__)

        self.query_analyzer_chain = self._create_query_analyzer_chain()
        self.query_rewriter_chain = self._create_query_rewriter_chain()

    def _create_query_analyzer_chain(self) -> Runnable:
        """
        Create chain for analyzing query characteristics.

        Returns:
            Runnable chain for query analysis
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

    def _create_query_rewriter_chain(self) -> Runnable:
        """
        Create chain for rewriting queries with chat history context.

        Returns:
            Runnable chain for query rewriting
        """
        try:
            self.logger.info("Creating query rewriter chain with chat history support.")

            system_prompt = (
                "You are a query optimization assistant. Given 'Chat History' (if any) and "
                "'Latest User Question', rewrite the question to be clear, specific, and self-contained "
                "for retrieval. If already clear, return it as is."
            )

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

    def format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        """
        Format chat history for prompt inclusion.

        Args:
            chat_history: List of chat messages

        Returns:
            Formatted string representation of chat history
        """
        if not chat_history:
            return "No chat history."

        formatted_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in chat_history])
        return formatted_history

    def analyze_query(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> Optional[QueryAnalysis]:
        """
        Analyze a query to determine its characteristics.

        Args:
            question: The user's question
            chat_history: Optional chat history for context

        Returns:
            QueryAnalysis object or None if analysis fails
        """
        chat_history = chat_history or []
        formatted_history = self.format_chat_history(chat_history)

        try:
            analysis_result: QueryAnalysis = self.query_analyzer_chain.invoke(
                {"question": question, "chat_history_formatted": formatted_history}
            )
            self.logger.info(
                f"Query analysis complete: Type='{analysis_result.query_type}', Intent='{analysis_result.main_intent}'"
            )
            if analysis_result.is_ambiguous:
                self.logger.warning(f"Query marked as ambiguous: {question}")

            return analysis_result

        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}", exc_info=True)
            return None

    def rewrite_query(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Rewrite a query to be clear, specific, and self-contained.

        Args:
            question: The original question
            chat_history: Optional chat history for context

        Returns:
            Rewritten query string (or original if rewriting fails)
        """
        chat_history = chat_history or []

        try:
            result = self.query_rewriter_chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )
            rewritten_query = result.strip()

            if rewritten_query.lower() != question.lower():
                self.logger.info(f"Rewrote '{question}' → '{rewritten_query}'")
            else:
                self.logger.info(f"No rewrite needed for question: '{question}'")

            return rewritten_query

        except Exception as e:
            self.logger.error(f"Error during query rewriting: {e}", exc_info=True)
            return question

    def should_use_web_search(self, query_analysis: Optional[QueryAnalysis]) -> bool:
        """
        Determine if web search should be used based on query analysis.

        Args:
            query_analysis: Query analysis results

        Returns:
            True if web search is recommended, False otherwise
        """
        if not query_analysis:
            return False

        web_search_types = {
            "factual_lookup",
            "comparison",
            "summary_request",
            "complex_reasoning",
        }

        return query_analysis.query_type in web_search_types
