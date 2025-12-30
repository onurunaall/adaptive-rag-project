"""
Answer Generator Module

Handles answer generation from context and grounding validation.
Extracted from CoreRAGEngine to improve modularity and maintainability.
"""

import logging
from typing import List, Optional, Any, Dict

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import BaseMessage

from src.rag.models import GroundingCheck


class AnswerGenerator:
    """
    Manages answer generation and grounding validation operations.

    Responsibilities:
    - Generate answers from context documents
    - Validate answer grounding (basic check)
    - Generate feedback for answer regeneration
    - Support chat history and regeneration feedback
    """

    def __init__(
        self,
        llm: Any,
        json_llm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize AnswerGenerator.

        Args:
            llm: Language model for text generation
            json_llm: Language model for JSON-structured output
            logger: Optional logger instance
        """
        self.llm = llm
        self.json_llm = json_llm
        self.logger = logger or logging.getLogger(__name__)

        self.answer_generation_chain = self._create_answer_generation_chain()
        self.grounding_check_chain = self._create_grounding_check_chain()

    def _create_answer_generation_chain(self) -> Runnable:
        """
        Create chain for generating answers from context documents.

        Returns:
            Runnable chain for answer generation
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

    def _create_grounding_check_chain(self) -> Runnable:
        """
        Create chain for checking if answers are grounded in context.

        Returns:
            Runnable chain for grounding validation
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

    def generate_answer(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[BaseMessage]] = None,
        regeneration_feedback: Optional[str] = None,
    ) -> str:
        """
        Generate an answer from context documents.

        Args:
            question: The user's question
            context: Context documents as string
            chat_history: Optional chat history for context
            regeneration_feedback: Optional feedback for regeneration

        Returns:
            Generated answer string
        """
        chat_history = chat_history or []

        input_data = {
            "context": context,
            "chat_history": chat_history,
            "optional_regeneration_prompt_header_if_feedback": "",
            "regeneration_feedback_if_any": "",
            "question": question,
        }

        if regeneration_feedback:
            self.logger.info("Generating answer with regeneration feedback")
            input_data["optional_regeneration_prompt_header_if_feedback"] = (
                "You are attempting to regenerate a previous answer that had issues."
            )
            input_data["regeneration_feedback_if_any"] = regeneration_feedback + "\n\nOriginal Question: "

        try:
            generated_text = self.answer_generation_chain.invoke(input_data)
            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"Error generating answer: {e}", exc_info=True)
            return "Error generating answer."

    def check_grounding(self, context: str, generation: str) -> Optional[GroundingCheck]:
        """
        Check if generated answer is grounded in context.

        Args:
            context: Context documents as string
            generation: Generated answer to validate

        Returns:
            GroundingCheck object or None if check fails
        """
        try:
            chain_input = {"context": context, "generation": generation}
            result: GroundingCheck = self.grounding_check_chain.invoke(chain_input)
            return result

        except Exception as e:
            self.logger.error(f"Error checking grounding: {e}", exc_info=True)
            return None

    def generate_basic_feedback(
        self,
        grounding_result: GroundingCheck,
        question: str,
    ) -> str:
        """
        Generate feedback for answer regeneration based on grounding check.

        Args:
            grounding_result: Result from grounding check
            question: Original question

        Returns:
            Regeneration feedback string
        """
        try:
            feedback_parts: List[str] = []

            try:
                ungrounded_statements = grounding_result.ungrounded_statements
                if ungrounded_statements:
                    statements_str = "; ".join(ungrounded_statements)
                    feedback_parts.append(f"The following statements were ungrounded: {statements_str}.")
            except Exception as statements_error:
                self.logger.warning(f"Error processing ungrounded statements: {statements_error}")

            try:
                correction_suggestion = grounding_result.correction_suggestion
                if correction_suggestion:
                    feedback_parts.append(f"Suggestion for correction: {correction_suggestion}.")
            except Exception as suggestion_error:
                self.logger.warning(f"Error processing correction suggestion: {suggestion_error}")

            if not feedback_parts:
                feedback_parts.append("The answer was not fully grounded in the provided context. Please revise.")

            regeneration_prompt = (
                f"The previous answer to the question '{question}' was not well-grounded. "
                f"{' '.join(feedback_parts)} "
                "Please generate a new answer focusing ONLY on the provided documents and addressing these issues."
            )

            return regeneration_prompt

        except Exception as e:
            self.logger.error(f"Error generating feedback: {e}", exc_info=True)
            return (
                f"The answer to '{question}' failed grounding check. "
                f"Please generate a new answer based strictly on the provided context."
            )

    def generate_advanced_feedback(
        self,
        advanced_results: Dict,
        question: str,
    ) -> str:
        """
        Generate detailed feedback from advanced grounding analysis.

        Args:
            advanced_results: Results from advanced grounding checker
            question: Original question

        Returns:
            Detailed regeneration feedback string
        """
        try:
            feedback_parts: List[str] = []

            try:
                detailed_grounding = advanced_results.get("detailed_grounding")
                if detailed_grounding:
                    unsupported_claims = getattr(detailed_grounding, "unsupported_claims", [])
                    if unsupported_claims:
                        limited_claims = unsupported_claims[:3]
                        feedback_parts.append(f"Unsupported claims found: {'; '.join(limited_claims)}")
            except Exception as grounding_error:
                self.logger.warning(f"Error processing detailed grounding results: {grounding_error}")

            try:
                consistency = advanced_results.get("consistency")
                if consistency:
                    contradictions_found = getattr(consistency, "contradictions_found", [])
                    if contradictions_found:
                        limited_contradictions = contradictions_found[:2]
                        feedback_parts.append(f"Internal contradictions: {'; '.join(limited_contradictions)}")
            except Exception as consistency_error:
                self.logger.warning(f"Error processing consistency results: {consistency_error}")

            try:
                completeness = advanced_results.get("completeness")
                if completeness:
                    missing_aspects = getattr(completeness, "missing_aspects", [])
                    if missing_aspects:
                        limited_missing = missing_aspects[:2]
                        feedback_parts.append(f"Missing important aspects: {'; '.join(limited_missing)}")
            except Exception as completeness_error:
                self.logger.warning(f"Error processing completeness results: {completeness_error}")

            try:
                hallucination_data = advanced_results.get("hallucination_detection", {})
                hallucinations = hallucination_data.get("hallucinations", [])
                if hallucinations:
                    limited_hallucinations = hallucinations[:2]
                    feedback_parts.append(f"Potential hallucinations: {'; '.join(limited_hallucinations)}")
            except Exception as hallucination_error:
                self.logger.warning(f"Error processing hallucination results: {hallucination_error}")

            if feedback_parts:
                overall_assessment = advanced_results.get("overall_assessment", {})
                recommendation = overall_assessment.get("recommendation", "Please improve the answer")

                regeneration_prompt = (
                    f"The previous answer to '{question}' failed advanced verification. "
                    f"Issues identified: {' | '.join(feedback_parts)}. "
                    f"Recommendation: {recommendation}. "
                    f"Please generate a new answer that strictly follows the provided context and addresses these issues."
                )
            else:
                regeneration_prompt = (
                    f"The answer to '{question}' did not meet quality standards. "
                    f"Please generate a more accurate answer based strictly on the provided context."
                )

            return regeneration_prompt

        except Exception as e:
            self.logger.error(f"Error generating advanced feedback: {e}", exc_info=True)
            return (
                f"The answer to '{question}' did not meet advanced quality standards. "
                f"Please generate a more accurate and complete answer based strictly on the provided context."
            )

    def is_grounded(self, context: str, generation: str) -> bool:
        """
        Check if an answer is grounded in context (boolean result).

        Args:
            context: Context documents as string
            generation: Generated answer to validate

        Returns:
            True if grounded, False otherwise
        """
        result = self.check_grounding(context, generation)
        if result is None:
            return False
        return result.is_grounded

    def format_context(self, documents: List[Any]) -> str:
        """
        Format documents into context string for answer generation.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        try:
            context_parts = []
            for idx, doc in enumerate(documents):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                context_parts.append(f"[Document {idx+1} from {source}]\n{content}")

            return "\n\n".join(context_parts)

        except Exception as e:
            self.logger.error(f"Error formatting context: {e}", exc_info=True)
            return ""
