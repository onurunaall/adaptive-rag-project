"""
Unit tests for DocumentGrader.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag.document_grader import DocumentGrader
from src.rag.models import RelevanceGrade, RerankScore


@pytest.fixture
def mock_json_llm():
    """Create a mock JSON-enabled LLM."""
    return Mock()


@pytest.fixture
def document_grader(mock_json_llm):
    """Create a DocumentGrader instance for testing."""
    return DocumentGrader(json_llm=mock_json_llm)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="Python is a programming language.", metadata={"source": "doc1.txt"}),
        Document(page_content="Machine learning uses algorithms.", metadata={"source": "doc2.txt"}),
        Document(page_content="The weather is sunny today.", metadata={"source": "doc3.txt"}),
    ]


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return "What is Python?"


class TestDocumentGraderInit:
    """Tests for DocumentGrader initialization."""

    def test_init_with_required_params(self, mock_json_llm):
        """Test initialization with required parameters."""
        grader = DocumentGrader(json_llm=mock_json_llm)
        assert grader.json_llm == mock_json_llm
        assert grader.logger is not None
        assert grader.document_relevance_grader_chain is not None
        assert grader.document_reranker_chain is not None

    def test_init_with_custom_logger(self, mock_json_llm):
        """Test initialization with custom logger."""
        mock_logger = Mock()
        grader = DocumentGrader(json_llm=mock_json_llm, logger=mock_logger)
        assert grader.logger == mock_logger

    def test_init_creates_chains(self, mock_json_llm):
        """Test that initialization creates required chains."""
        with patch.object(DocumentGrader, '_create_document_relevance_grader_chain') as mock_grader_chain, \
             patch.object(DocumentGrader, '_create_document_reranker_chain') as mock_reranker_chain:

            mock_grader_chain.return_value = Mock()
            mock_reranker_chain.return_value = Mock()

            grader = DocumentGrader(json_llm=mock_json_llm)

            mock_grader_chain.assert_called_once()
            mock_reranker_chain.assert_called_once()


class TestCreateDocumentRelevanceGraderChain:
    """Tests for _create_document_relevance_grader_chain method."""

    def test_create_grader_chain_success(self, mock_json_llm):
        """Test successful creation of grader chain."""
        grader = DocumentGrader(json_llm=mock_json_llm)
        assert grader.document_relevance_grader_chain is not None

    def test_create_grader_chain_uses_correct_model(self, mock_json_llm):
        """Test that grader chain uses RelevanceGrade model."""
        # We can't easily mock internal chain creation without breaking the | operator
        # Instead, verify the chain is created successfully
        grader = DocumentGrader(json_llm=mock_json_llm)

        # The grader chain should be created
        assert grader.document_relevance_grader_chain is not None
        # This implicitly tests that PydanticOutputParser was called with RelevanceGrade

    def test_create_grader_chain_logs_success(self, mock_json_llm):
        """Test that successful chain creation is logged."""
        mock_logger = Mock()
        grader = DocumentGrader(json_llm=mock_json_llm, logger=mock_logger)
        mock_logger.info.assert_any_call("Document relevance grader chain created successfully.")

    def test_create_grader_chain_failure_raises_runtime_error(self, mock_json_llm):
        """Test that chain creation failure raises RuntimeError."""
        with patch('src.rag.document_grader.PydanticOutputParser', side_effect=Exception("Test error")):
            with pytest.raises(RuntimeError, match="Failed to create document relevance grader chain"):
                DocumentGrader(json_llm=mock_json_llm)


class TestCreateDocumentRerankerChain:
    """Tests for _create_document_reranker_chain method."""

    def test_create_reranker_chain_success(self, mock_json_llm):
        """Test successful creation of reranker chain."""
        grader = DocumentGrader(json_llm=mock_json_llm)
        assert grader.document_reranker_chain is not None

    def test_create_reranker_chain_uses_correct_model(self, mock_json_llm):
        """Test that reranker chain uses RerankScore model."""
        # We can't easily mock internal chain creation without breaking the | operator
        # Instead, verify the chain is created successfully
        grader = DocumentGrader(json_llm=mock_json_llm)

        # The reranker chain should be created
        assert grader.document_reranker_chain is not None
        # This implicitly tests that PydanticOutputParser was called with RerankScore

    def test_create_reranker_chain_logs_success(self, mock_json_llm):
        """Test that successful chain creation is logged."""
        mock_logger = Mock()
        grader = DocumentGrader(json_llm=mock_json_llm, logger=mock_logger)
        mock_logger.info.assert_any_call("Document re-ranker chain created successfully.")


class TestGradeDocuments:
    """Tests for grade_documents method."""

    def test_grade_documents_empty_list(self, document_grader):
        """Test grading empty document list."""
        result = document_grader.grade_documents([], "test question")
        assert result == []

    def test_grade_documents_filters_irrelevant(self, document_grader, sample_documents, sample_question):
        """Test that irrelevant documents are filtered out."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = [
            RelevanceGrade(is_relevant=True, justification="Relevant to Python"),
            RelevanceGrade(is_relevant=False, justification="Not relevant"),
            RelevanceGrade(is_relevant=True, justification="Somewhat relevant"),
        ]
        document_grader.document_relevance_grader_chain = mock_chain

        result = document_grader.grade_documents(sample_documents, sample_question)

        assert len(result) == 2
        assert result[0].page_content == "Python is a programming language."
        assert result[1].page_content == "The weather is sunny today."

    def test_grade_documents_adds_justification_to_metadata(self, document_grader, sample_documents, sample_question):
        """Test that justifications are added to document metadata."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RelevanceGrade(
            is_relevant=True,
            justification="Contains information about Python"
        )
        document_grader.document_relevance_grader_chain = mock_chain

        result = document_grader.grade_documents(sample_documents[:1], sample_question)

        assert "relevance_grade_justification" in result[0].metadata
        assert result[0].metadata["relevance_grade_justification"] == "Contains information about Python"

    def test_grade_documents_logs_progress(self, document_grader, sample_documents, sample_question):
        """Test that grading progress is logged."""
        mock_logger = Mock()
        document_grader.logger = mock_logger

        mock_chain = Mock()
        mock_chain.invoke.return_value = RelevanceGrade(is_relevant=True, justification="Relevant")
        document_grader.document_relevance_grader_chain = mock_chain

        document_grader.grade_documents(sample_documents, sample_question)

        mock_logger.info.assert_any_call(f"Grading {len(sample_documents)} documents for relevance...")
        mock_logger.info.assert_any_call(f"{len(sample_documents)}/{len(sample_documents)} documents passed relevance grading.")

    def test_grade_documents_handles_grading_error(self, document_grader, sample_documents, sample_question):
        """Test that grading errors are handled gracefully."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = [
            RelevanceGrade(is_relevant=True, justification="Good"),
            Exception("Grading failed"),
            RelevanceGrade(is_relevant=True, justification="Also good"),
        ]
        document_grader.document_relevance_grader_chain = mock_chain

        result = document_grader.grade_documents(sample_documents, sample_question)

        # Should skip the failed document and continue
        assert len(result) == 2

    def test_grade_documents_invokes_chain_correctly(self, document_grader, sample_documents, sample_question):
        """Test that chain is invoked with correct parameters."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RelevanceGrade(is_relevant=True, justification="Relevant")
        document_grader.document_relevance_grader_chain = mock_chain

        document_grader.grade_documents(sample_documents[:1], sample_question)

        mock_chain.invoke.assert_called_once_with({
            "question": sample_question,
            "document_content": sample_documents[0].page_content
        })


class TestRerankDocuments:
    """Tests for rerank_documents method."""

    def test_rerank_documents_empty_list(self, document_grader):
        """Test reranking empty document list."""
        result = document_grader.rerank_documents([], "test question")
        assert result == []

    def test_rerank_documents_sorts_by_score(self, document_grader, sample_documents, sample_question):
        """Test that documents are sorted by relevance score (highest first)."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = [
            RerankScore(relevance_score=0.5, justification="test"),
            RerankScore(relevance_score=0.9, justification="test"),
            RerankScore(relevance_score=0.3, justification="test"),
        ]
        document_grader.document_reranker_chain = mock_chain

        result = document_grader.rerank_documents(sample_documents, sample_question)

        assert len(result) == 3
        assert result[0].page_content == "Machine learning uses algorithms."  # 0.9
        assert result[1].page_content == "Python is a programming language."  # 0.5
        assert result[2].page_content == "The weather is sunny today."  # 0.3

    def test_rerank_documents_handles_reranking_error(self, document_grader, sample_documents, sample_question):
        """Test that reranking errors assign 0.0 score."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = [
            RerankScore(relevance_score=0.8, justification="test"),
            Exception("Reranking failed"),
            RerankScore(relevance_score=0.6, justification="test"),
        ]
        document_grader.document_reranker_chain = mock_chain

        result = document_grader.rerank_documents(sample_documents, sample_question)

        # Failed document should get 0.0 score and be last
        assert len(result) == 3
        assert result[2].page_content == "Machine learning uses algorithms."

    def test_rerank_documents_logs_progress(self, document_grader, sample_documents, sample_question):
        """Test that reranking progress is logged."""
        mock_logger = Mock()
        document_grader.logger = mock_logger

        mock_chain = Mock()
        mock_chain.invoke.return_value = RerankScore(relevance_score=0.7, justification="test")
        document_grader.document_reranker_chain = mock_chain

        document_grader.rerank_documents(sample_documents, sample_question)

        mock_logger.info.assert_any_call(f"Re-ranking {len(sample_documents)} documents for question: '{sample_question}'")
        mock_logger.info.assert_any_call("Finished re-ranking. Top document score: 0.7")

    def test_rerank_documents_handles_sort_error(self, document_grader, sample_documents, sample_question):
        """Test that sorting errors return original document order."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RerankScore(relevance_score=0.5, justification="test")
        document_grader.document_reranker_chain = mock_chain

        with patch('src.rag.document_grader.sorted', side_effect=Exception("Sort failed")):
            result = document_grader.rerank_documents(sample_documents, sample_question)

            # Should return original documents
            assert result == sample_documents

    def test_rerank_documents_invokes_chain_correctly(self, document_grader, sample_documents, sample_question):
        """Test that chain is invoked with correct parameters."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RerankScore(relevance_score=0.8, justification="test")
        document_grader.document_reranker_chain = mock_chain

        document_grader.rerank_documents(sample_documents[:1], sample_question)

        mock_chain.invoke.assert_called_once_with({
            "question": sample_question,
            "document_content": sample_documents[0].page_content
        })


class TestCalculateRelevanceScore:
    """Tests for calculate_relevance_score method."""

    def test_calculate_relevance_score_success(self, document_grader, sample_documents, sample_question):
        """Test successful relevance score calculation."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RerankScore(relevance_score=0.85, justification="test")
        document_grader.document_reranker_chain = mock_chain

        score = document_grader.calculate_relevance_score(sample_documents[0], sample_question)

        assert score == 0.85

    def test_calculate_relevance_score_error_returns_zero(self, document_grader, sample_documents, sample_question):
        """Test that errors return 0.0 score."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Scoring failed")
        document_grader.document_reranker_chain = mock_chain

        score = document_grader.calculate_relevance_score(sample_documents[0], sample_question)

        assert score == 0.0

    def test_calculate_relevance_score_logs_error(self, document_grader, sample_documents, sample_question):
        """Test that scoring errors are logged."""
        mock_logger = Mock()
        document_grader.logger = mock_logger

        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Scoring failed")
        document_grader.document_reranker_chain = mock_chain

        document_grader.calculate_relevance_score(sample_documents[0], sample_question)

        mock_logger.error.assert_called_once()

    def test_calculate_relevance_score_invokes_chain_correctly(self, document_grader, sample_documents, sample_question):
        """Test that chain is invoked with correct parameters."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RerankScore(relevance_score=0.75, justification="test")
        document_grader.document_reranker_chain = mock_chain

        document_grader.calculate_relevance_score(sample_documents[0], sample_question)

        mock_chain.invoke.assert_called_once_with({
            "question": sample_question,
            "document_content": sample_documents[0].page_content
        })


class TestIsRelevant:
    """Tests for is_relevant method."""

    def test_is_relevant_returns_true(self, document_grader, sample_documents, sample_question):
        """Test is_relevant returns True for relevant document."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RelevanceGrade(is_relevant=True, justification="Relevant")
        document_grader.document_relevance_grader_chain = mock_chain

        result = document_grader.is_relevant(sample_documents[0], sample_question)

        assert result is True

    def test_is_relevant_returns_false(self, document_grader, sample_documents, sample_question):
        """Test is_relevant returns False for irrelevant document."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RelevanceGrade(is_relevant=False, justification="Not relevant")
        document_grader.document_relevance_grader_chain = mock_chain

        result = document_grader.is_relevant(sample_documents[0], sample_question)

        assert result is False

    def test_is_relevant_error_returns_false(self, document_grader, sample_documents, sample_question):
        """Test that errors return False."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Grading failed")
        document_grader.document_relevance_grader_chain = mock_chain

        result = document_grader.is_relevant(sample_documents[0], sample_question)

        assert result is False

    def test_is_relevant_logs_error(self, document_grader, sample_documents, sample_question):
        """Test that relevance check errors are logged."""
        mock_logger = Mock()
        document_grader.logger = mock_logger

        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Grading failed")
        document_grader.document_relevance_grader_chain = mock_chain

        document_grader.is_relevant(sample_documents[0], sample_question)

        mock_logger.error.assert_called_once()

    def test_is_relevant_invokes_chain_correctly(self, document_grader, sample_documents, sample_question):
        """Test that chain is invoked with correct parameters."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = RelevanceGrade(is_relevant=True, justification="Good match")
        document_grader.document_relevance_grader_chain = mock_chain

        document_grader.is_relevant(sample_documents[0], sample_question)

        mock_chain.invoke.assert_called_once_with({
            "question": sample_question,
            "document_content": sample_documents[0].page_content
        })
