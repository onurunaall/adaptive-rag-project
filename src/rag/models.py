"""
Pydantic models and TypedDict definitions for the RAG engine.
"""
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GroundingCheck(BaseModel):
    """
    Structured output for answer grounding and hallucination check.
    """

    is_grounded: bool = Field(description="Is the generated answer fully supported by the provided context? True or False.")

    ungrounded_statements: Optional[List[str]] = Field(
        default=None,
        description="List any specific statements in the answer that are NOT supported by the context.",
    )

    correction_suggestion: Optional[str] = Field(
        default=None,
        description="If not grounded, suggest how to rephrase the answer to be grounded, or state no grounded answer is possible.",
    )


class RelevanceGrade(BaseModel):
    """
    Structured output for document relevance grading.
    """

    is_relevant: bool = Field(description="Is the document excerpt relevant to the question? True or False.")

    justification: Optional[str] = Field(
        default=None,
        description="Brief justification for the relevance decision (1-2 sentences).",
    )


class RerankScore(BaseModel):
    """
    Pydantic model for contextual re-ranking scores.
    """

    relevance_score: float = Field(
        description="A score from 0.0 to 1.0 indicating the document's direct relevance to answering the question."
    )
    justification: str = Field(description="A brief justification for the assigned score.")


class QueryAnalysis(BaseModel):
    """
    Structured output for sophisticated query analysis.
    """

    query_type: str = Field(
        description="Classify the query's primary type (e.g., 'factual_lookup', 'comparison', 'summary_request', 'complex_reasoning', 'ambiguous', 'keyword_search_sufficient', 'greeting', 'not_a_question')."
    )

    main_intent: str = Field(
        description="A concise sentence describing the primary user intent or what the user wants to achieve."
    )

    extracted_keywords: List[str] = Field(
        default_factory=list,
        description="A list of key nouns, verbs, and named entities from the query that are critical for effective retrieval.",
    )

    is_ambiguous: bool = Field(
        default=False,
        description="True if the query is vague, unclear, or open to multiple interpretations and might benefit from clarification before proceeding.",
    )


class CoreGraphState(TypedDict):
    """State dictionary for the RAG workflow graph."""

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
    context_was_truncated: Optional[bool]
