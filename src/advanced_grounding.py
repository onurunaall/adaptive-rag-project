from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
import logging


@dataclass
class FactualClaim:
    text: str
    context_support: Optional[str] = None
    confidence: float = 0.0
    is_supported: bool = False
    evidence_sources: List[str] = None


class AdvancedGroundingCheck(BaseModel):
    """Enhanced grounding check with detailed analysis"""

    is_fully_grounded: bool = Field(description="Is the entire answer fully supported by the context?")
    overall_confidence: float = Field(description="Overall confidence score from 0.0 to 1.0")

    factual_claims: List[str] = Field(description="List of specific factual claims made in the answer")

    supported_claims: List[str] = Field(
        default_factory=list,
        description="Claims that are well-supported by the context",
    )

    unsupported_claims: List[str] = Field(default_factory=list, description="Claims that lack support in the context")

    partially_supported_claims: List[str] = Field(
        default_factory=list, description="Claims that have some but incomplete support"
    )

    hallucination_indicators: List[str] = Field(
        default_factory=list,
        description="Specific indicators of potential hallucination",
    )

    citation_accuracy: float = Field(description="Score from 0.0 to 1.0 indicating accuracy of any citations or references")

    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific suggestions for improving the answer's grounding",
    )

    context_coverage: float = Field(description="Score indicating how well the answer covers the available context (0.0-1.0)")


class ConsistencyCheck(BaseModel):
    """Check for internal consistency within the answer"""

    is_internally_consistent: bool = Field(description="Is the answer internally consistent?")
    contradictions_found: List[str] = Field(default_factory=list, description="Any contradictions found within the answer")
    confidence_score: float = Field(description="Confidence in consistency assessment")


class CompletenessCheck(BaseModel):
    """Check if the answer adequately addresses the original query"""

    addresses_main_question: bool = Field(description="Does the answer address the main question?")
    addresses_sub_questions: List[str] = Field(default_factory=list, description="Sub-questions or aspects that are addressed")
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Important aspects of the question that are not addressed",
    )
    completeness_score: float = Field(description="Overall completeness score (0.0-1.0)")


class MultiLevelGroundingChecker:
    """
    Advanced grounding checker with multiple levels of verification
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize specialized chains
        self.detailed_grounding_chain = self._create_detailed_grounding_chain()
        self.consistency_chain = self._create_consistency_chain()
        self.completeness_chain = self._create_completeness_chain()
        self.hallucination_detector_chain = self._create_hallucination_detector_chain()

    def _create_detailed_grounding_chain(self):
        """Create chain for detailed grounding analysis"""
        parser = PydanticOutputParser(pydantic_object=AdvancedGroundingCheck)

        prompt_template = (
            "You are an expert fact-checker analyzing whether an AI-generated answer is properly "
            "grounded in the provided context. Perform a comprehensive analysis.\n\n"
            "Context Documents:\n---\n{context}\n---\n\n"
            "Generated Answer:\n---\n{answer}\n---\n\n"
            "Original Question:\n---\n{question}\n---\n\n"
            "Analyze the answer systematically:\n"
            "1. Identify all factual claims in the answer\n"
            "2. For each claim, check if it's supported by the context\n"
            "3. Look for potential hallucinations (information not in context)\n"
            "4. Assess citation accuracy if any references are made\n"
            "5. Evaluate how well the answer uses the available context\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{format_instructions}\n\n"
            "Provide your detailed JSON analysis:"
        )

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser

    def _create_consistency_chain(self):
        """Create chain for consistency checking"""
        parser = PydanticOutputParser(pydantic_object=ConsistencyCheck)

        prompt_template = (
            "You are analyzing an AI-generated answer for internal consistency. "
            "Look for any contradictions, conflicting statements, or logical inconsistencies within the answer itself.\n\n"
            "Answer to Analyze:\n---\n{answer}\n---\n\n"
            "Check for:\n"
            "- Contradictory statements\n"
            "- Conflicting facts or figures\n"
            "- Logical inconsistencies\n"
            "- Timeline contradictions\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{format_instructions}\n\n"
            "Provide your JSON analysis:"
        )

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser

    def _create_completeness_chain(self):
        """Create chain for completeness checking"""
        parser = PydanticOutputParser(pydantic_object=CompletenessCheck)

        prompt_template = (
            "You are evaluating whether an AI-generated answer adequately addresses the original question. "
            "Assess completeness and identify any missing important aspects.\n\n"
            "Original Question:\n---\n{question}\n---\n\n"
            "Generated Answer:\n---\n{answer}\n---\n\n"
            "Available Context:\n---\n{context}\n---\n\n"
            "Evaluate:\n"
            "1. Does the answer directly address the main question?\n"
            "2. What aspects of the question are well-addressed?\n"
            "3. What important aspects are missing that could be answered from the context?\n"
            "4. Is the answer appropriately comprehensive given the available information?\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{format_instructions}\n\n"
            "Provide your JSON analysis:"
        )

        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser

    def _create_hallucination_detector_chain(self):
        """Create specialized chain for hallucination detection"""
        prompt_template = (
            "You are a specialized hallucination detector. Your job is to identify any information "
            "in the answer that is NOT present in or supported by the provided context.\n\n"
            "Context:\n---\n{context}\n---\n\n"
            "Answer:\n---\n{answer}\n---\n\n"
            "Identify any hallucinated content (information not in context). List each hallucination "
            "as a separate line starting with 'HALLUCINATION:'. If no hallucinations found, respond with 'NONE'.\n\n"
            "Response:"
        )

        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm

    async def perform_comprehensive_grounding_check(
        self, answer: str, context: str, question: str, documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-level grounding check
        """
        results = {}

        try:
            # 1. Detailed grounding analysis
            detailed_result = await self.detailed_grounding_chain.ainvoke(
                {"context": context, "answer": answer, "question": question}
            )
            results["detailed_grounding"] = detailed_result

            # 2. Consistency check
            consistency_result = await self.consistency_chain.ainvoke({"answer": answer})
            results["consistency"] = consistency_result

            # 3. Completeness check
            completeness_result = await self.completeness_chain.ainvoke(
                {"question": question, "answer": answer, "context": context}
            )
            results["completeness"] = completeness_result

            # 4. Hallucination detection
            hallucination_result = await self.hallucination_detector_chain.ainvoke({"context": context, "answer": answer})
            results["hallucination_detection"] = self._parse_hallucination_result(hallucination_result)

            # 5. Calculate overall assessment
            overall_assessment = self._calculate_overall_assessment(results)
            results["overall_assessment"] = overall_assessment

        except Exception as e:
            self.logger.error(f"Error in comprehensive grounding check: {e}")
            results["error"] = str(e)

        return results

    def _parse_hallucination_result(self, result: str) -> Dict[str, Any]:
        """Parse hallucination detection result"""
        if "NONE" in result.upper():
            return {
                "hallucinations_found": False,
                "hallucinations": [],
                "confidence": 0.9,
            }

        hallucinations = []
        for line in result.split("\n"):
            if line.startswith("HALLUCINATION:"):
                hallucinations.append(line.replace("HALLUCINATION:", "").strip())

        return {
            "hallucinations_found": len(hallucinations) > 0,
            "hallucinations": hallucinations,
            "confidence": 0.8 if hallucinations else 0.9,
        }

    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall grounding assessment"""
        try:
            detailed = results.get("detailed_grounding")
            consistency = results.get("consistency")
            completeness = results.get("completeness")
            hallucination = results.get("hallucination_detection", {})

            # Calculate weighted score
            scores = []

            if detailed and hasattr(detailed, "overall_confidence"):
                scores.append(detailed.overall_confidence * 0.4)  # 40% weight

            if consistency and hasattr(consistency, "confidence_score"):
                consistency_score = consistency.confidence_score if consistency.is_internally_consistent else 0.0
                scores.append(consistency_score * 0.2)  # 20% weight

            if completeness and hasattr(completeness, "completeness_score"):
                scores.append(completeness.completeness_score * 0.3)  # 30% weight

            if hallucination:
                hallucination_score = 0.0 if hallucination.get("hallucinations_found") else 1.0
                scores.append(hallucination_score * 0.1)  # 10% weight

            overall_score = sum(scores) / len(scores) if scores else 0.0

            # Determine if answer should be accepted
            is_acceptable = (
                overall_score >= 0.7
                and (not detailed or detailed.is_fully_grounded)
                and (not consistency or consistency.is_internally_consistent)
                and (not hallucination or not hallucination.get("hallucinations_found"))
            )

            return {
                "overall_score": overall_score,
                "is_acceptable": is_acceptable,
                "recommendation": self._get_recommendation(overall_score, is_acceptable),
                "key_issues": self._identify_key_issues(results),
            }

        except Exception as e:
            self.logger.error(f"Error calculating overall assessment: {e}")
            return {
                "overall_score": 0.0,
                "is_acceptable": False,
                "recommendation": "Manual review required due to assessment error",
                "error": str(e),
            }

    def _get_recommendation(self, score: float, is_acceptable: bool) -> str:
        """Get recommendation based on assessment"""
        if is_acceptable and score >= 0.9:
            return "High quality answer - acceptable as-is"
        elif is_acceptable and score >= 0.7:
            return "Acceptable answer with minor improvements possible"
        elif score >= 0.5:
            return "Answer needs significant improvement before use"
        else:
            return "Answer not suitable for use - major grounding issues"

    def _identify_key_issues(self, results: Dict[str, Any]) -> List[str]:
        """Identify the most critical issues"""
        issues = []

        detailed = results.get("detailed_grounding")
        if detailed and hasattr(detailed, "unsupported_claims") and detailed.unsupported_claims:
            issues.append(f"Unsupported claims: {len(detailed.unsupported_claims)}")

        consistency = results.get("consistency")
        if consistency and hasattr(consistency, "contradictions_found") and consistency.contradictions_found:
            issues.append(f"Internal contradictions: {len(consistency.contradictions_found)}")

        completeness = results.get("completeness")
        if completeness and hasattr(completeness, "missing_aspects") and completeness.missing_aspects:
            issues.append(f"Missing aspects: {len(completeness.missing_aspects)}")

        hallucination = results.get("hallucination_detection", {})
        if hallucination.get("hallucinations_found"):
            issues.append(f"Hallucinations detected: {len(hallucination.get('hallucinations', []))}")

        return issues
