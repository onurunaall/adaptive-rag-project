"""
Document Grader Module

Handles document relevance grading and reranking operations.
Extracted from CoreRAGEngine to improve modularity and maintainability.
"""

import logging
from typing import List, Tuple, Optional, Any

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

from src.rag.models import RelevanceGrade, RerankScore


class DocumentGrader:
    """
    Manages document relevance grading and reranking operations.

    Responsibilities:
    - Grade documents for relevance to a question
    - Rerank documents by relevance scores
    - Filter out irrelevant documents
    """

    def __init__(
        self,
        json_llm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize DocumentGrader.

        Args:
            json_llm: Language model for JSON-structured output
            logger: Optional logger instance
        """
        self.json_llm = json_llm
        self.logger = logger or logging.getLogger(__name__)

        self.document_relevance_grader_chain = self._create_document_relevance_grader_chain()
        self.document_reranker_chain = self._create_document_reranker_chain()

    def _create_document_relevance_grader_chain(self) -> Runnable:
        """
        Create chain for grading document relevance to questions.

        Returns:
            Runnable chain for document relevance grading
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

            prompt = ChatPromptTemplate.from_template(
                template=prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | self.json_llm | PydanticOutputParser(pydantic_object=RelevanceGrade)

            self.logger.info("Document relevance grader chain created successfully.")
            return chain

        except Exception as e:
            error_msg = f"Failed to create document relevance grader chain: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_document_reranker_chain(self) -> Runnable:
        """
        Create chain for re-ranking documents by relevance score.

        Returns:
            Runnable chain for document reranking
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

    def grade_documents(self, documents: List[Document], question: str) -> List[Document]:
        """
        Grade documents for relevance and filter out irrelevant ones.

        Args:
            documents: List of documents to grade
            question: The question to grade documents against

        Returns:
            List of relevant documents with justifications in metadata
        """
        if not documents:
            self.logger.info("No documents to grade.")
            return []

        self.logger.info(f"Grading {len(documents)} documents for relevance...")
        relevant_docs: List[Document] = []

        for idx, doc in enumerate(documents):
            src = doc.metadata.get("source", "unknown")
            self.logger.debug(f"Grading doc {idx+1}/{len(documents)} from source '{src}'")
            try:
                grade: RelevanceGrade = self.document_relevance_grader_chain.invoke(
                    {"question": question, "document_content": doc.page_content}
                )
                self.logger.debug(f"Doc {idx+1} graded: is_relevant={grade.is_relevant}, justification={grade.justification}")

                if grade.is_relevant:
                    doc.metadata["relevance_grade_justification"] = grade.justification
                    relevant_docs.append(doc)

            except Exception as e:
                self.logger.error(
                    f"Error grading document {idx+1} (source: {src}): {e}",
                    exc_info=True,
                )
                continue

        self.logger.info(f"{len(relevant_docs)}/{len(documents)} documents passed relevance grading.")
        return relevant_docs

    def rerank_documents(self, documents: List[Document], question: str) -> List[Document]:
        """
        Rerank documents by relevance scores.

        Args:
            documents: List of documents to rerank
            question: The question to rerank documents against

        Returns:
            List of documents sorted by relevance score (highest first)
        """
        if not documents:
            self.logger.info("No documents to rerank.")
            return []

        self.logger.info(f"Re-ranking {len(documents)} documents for question: '{question}'")

        docs_with_scores: List[Tuple[Document, float]] = []

        for idx, doc in enumerate(documents):
            try:
                source_info = doc.metadata.get("source", "unknown")
                self.logger.debug(f"Re-ranking document {idx+1}/{len(documents)} from source '{source_info}'")

                score_result: RerankScore = self.document_reranker_chain.invoke(
                    {"question": question, "document_content": doc.page_content}
                )

                docs_with_scores.append((doc, score_result.relevance_score))
                self.logger.debug(f"Document {idx+1} scored: {score_result.relevance_score}")

            except Exception as doc_error:
                source_info = doc.metadata.get("source", "unknown")
                error_msg = f"Error re-ranking document {idx+1} (source: {source_info}): {doc_error}"
                self.logger.error(error_msg, exc_info=True)
                docs_with_scores.append((doc, 0.0))

        try:
            sorted_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
            sorted_docs = [doc for doc, score in sorted_docs_with_scores]

            if sorted_docs_with_scores:
                top_score = sorted_docs_with_scores[0][1]
                self.logger.info(f"Finished re-ranking. Top document score: {top_score}")
            else:
                self.logger.info("No documents to rank.")

            return sorted_docs

        except Exception as sort_error:
            error_msg = f"Error sorting re-ranked documents: {sort_error}"
            self.logger.error(error_msg, exc_info=True)
            return documents

    def calculate_relevance_score(self, document: Document, question: str) -> float:
        """
        Calculate relevance score for a single document.

        Args:
            document: Document to score
            question: Question to score against

        Returns:
            Relevance score (0.0 to 1.0)
        """
        try:
            score_result: RerankScore = self.document_reranker_chain.invoke(
                {"question": question, "document_content": document.page_content}
            )
            return score_result.relevance_score

        except Exception as e:
            self.logger.error(f"Error calculating relevance score: {e}", exc_info=True)
            return 0.0

    def is_relevant(self, document: Document, question: str) -> bool:
        """
        Check if a document is relevant to a question.

        Args:
            document: Document to check
            question: Question to check against

        Returns:
            True if relevant, False otherwise
        """
        try:
            grade: RelevanceGrade = self.document_relevance_grader_chain.invoke(
                {"question": question, "document_content": document.page_content}
            )
            return grade.is_relevant

        except Exception as e:
            self.logger.error(f"Error checking document relevance: {e}", exc_info=True)
            return False
