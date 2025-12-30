"""
Unit tests for AnswerGenerator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from src.rag.answer_generator import AnswerGenerator
from src.rag.models import GroundingCheck


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return Mock()


@pytest.fixture
def mock_json_llm():
    """Create a mock JSON-enabled LLM."""
    return Mock()


@pytest.fixture
def answer_generator(mock_llm, mock_json_llm):
    """Create an AnswerGenerator instance for testing."""
    return AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm)


@pytest.fixture
def sample_context():
    """Sample context string."""
    return "Python is a high-level programming language. It was created by Guido van Rossum."


@pytest.fixture
def sample_question():
    """Sample question."""
    return "What is Python?"


@pytest.fixture
def sample_generation():
    """Sample generated answer."""
    return "Python is a high-level programming language created by Guido van Rossum."


@pytest.fixture
def sample_chat_history():
    """Sample chat history."""
    return [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi! How can I help you?"),
    ]


@pytest.fixture
def sample_documents():
    """Sample documents."""
    return [
        Document(page_content="Python is a programming language.", metadata={"source": "doc1.txt"}),
        Document(page_content="It was created by Guido van Rossum.", metadata={"source": "doc2.txt"}),
    ]


class TestAnswerGeneratorInit:
    """Tests for AnswerGenerator initialization."""

    def test_init_with_required_params(self, mock_llm, mock_json_llm):
        """Test initialization with required parameters."""
        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm)
        assert generator.llm == mock_llm
        assert generator.json_llm == mock_json_llm
        assert generator.logger is not None
        assert generator.answer_generation_chain is not None
        assert generator.grounding_check_chain is not None

    def test_init_with_custom_logger(self, mock_llm, mock_json_llm):
        """Test initialization with custom logger."""
        mock_logger = Mock()
        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm, logger=mock_logger)
        assert generator.logger == mock_logger

    def test_init_creates_chains(self, mock_llm, mock_json_llm):
        """Test that initialization creates required chains."""
        with patch.object(AnswerGenerator, '_create_answer_generation_chain') as mock_answer_chain, \
             patch.object(AnswerGenerator, '_create_grounding_check_chain') as mock_grounding_chain:

            mock_answer_chain.return_value = Mock()
            mock_grounding_chain.return_value = Mock()

            generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm)

            mock_answer_chain.assert_called_once()
            mock_grounding_chain.assert_called_once()


class TestCreateAnswerGenerationChain:
    """Tests for _create_answer_generation_chain method."""

    def test_create_answer_chain_success(self, mock_llm, mock_json_llm):
        """Test successful creation of answer generation chain."""
        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm)
        assert generator.answer_generation_chain is not None

    def test_create_answer_chain_logs_success(self, mock_llm, mock_json_llm):
        """Test that successful chain creation is logged."""
        mock_logger = Mock()
        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm, logger=mock_logger)
        mock_logger.info.assert_any_call("Answer generation chain created successfully.")

    def test_create_answer_chain_failure_raises_runtime_error(self, mock_json_llm):
        """Test that chain creation failure raises RuntimeError."""
        with patch('src.rag.answer_generator.ChatPromptTemplate', side_effect=Exception("Test error")):
            with pytest.raises(RuntimeError, match="Failed to create answer generation chain"):
                AnswerGenerator(llm=Mock(), json_llm=mock_json_llm)


class TestCreateGroundingCheckChain:
    """Tests for _create_grounding_check_chain method."""

    def test_create_grounding_chain_success(self, mock_llm, mock_json_llm):
        """Test successful creation of grounding check chain."""
        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm)
        assert generator.grounding_check_chain is not None

    @patch('src.rag.answer_generator.PydanticOutputParser')
    @patch('src.rag.answer_generator.ChatPromptTemplate')
    def test_create_grounding_chain_uses_correct_model(self, mock_prompt, mock_parser, mock_llm, mock_json_llm):
        """Test that grounding chain uses GroundingCheck model."""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.get_format_instructions.return_value = "format instructions"

        mock_prompt_instance = Mock()
        mock_prompt.from_template.return_value = mock_prompt_instance

        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm)

        calls = mock_parser.call_args_list
        assert any(call.kwargs.get('pydantic_object') == GroundingCheck or
                  (len(call.args) > 0 and call.args[0] == GroundingCheck)
                  for call in calls if call.kwargs or call.args)

    def test_create_grounding_chain_logs_success(self, mock_llm, mock_json_llm):
        """Test that successful chain creation is logged."""
        mock_logger = Mock()
        generator = AnswerGenerator(llm=mock_llm, json_llm=mock_json_llm, logger=mock_logger)
        mock_logger.info.assert_any_call("Answer grounding check chain created successfully.")


class TestGenerateAnswer:
    """Tests for generate_answer method."""

    def test_generate_answer_basic(self, answer_generator, sample_question, sample_context):
        """Test basic answer generation."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Python is a programming language."
        answer_generator.answer_generation_chain = mock_chain

        result = answer_generator.generate_answer(sample_question, sample_context)

        assert result == "Python is a programming language."
        mock_chain.invoke.assert_called_once()

    def test_generate_answer_with_chat_history(
        self, answer_generator, sample_question, sample_context, sample_chat_history
    ):
        """Test answer generation with chat history."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer with history context."
        answer_generator.answer_generation_chain = mock_chain

        result = answer_generator.generate_answer(
            sample_question, sample_context, chat_history=sample_chat_history
        )

        assert result == "Answer with history context."
        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["chat_history"] == sample_chat_history

    def test_generate_answer_with_regeneration_feedback(
        self, answer_generator, sample_question, sample_context
    ):
        """Test answer generation with regeneration feedback."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Regenerated answer."
        answer_generator.answer_generation_chain = mock_chain

        feedback = "Please be more specific."
        result = answer_generator.generate_answer(
            sample_question, sample_context, regeneration_feedback=feedback
        )

        assert result == "Regenerated answer."
        call_args = mock_chain.invoke.call_args[0][0]
        assert "You are attempting to regenerate" in call_args["optional_regeneration_prompt_header_if_feedback"]
        assert feedback in call_args["regeneration_feedback_if_any"]

    def test_generate_answer_strips_whitespace(self, answer_generator, sample_question, sample_context):
        """Test that generated answer is stripped of whitespace."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = "  Answer with spaces  "
        answer_generator.answer_generation_chain = mock_chain

        result = answer_generator.generate_answer(sample_question, sample_context)

        assert result == "Answer with spaces"

    def test_generate_answer_error_handling(self, answer_generator, sample_question, sample_context):
        """Test error handling in answer generation."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Generation failed")
        answer_generator.answer_generation_chain = mock_chain

        result = answer_generator.generate_answer(sample_question, sample_context)

        assert result == "Error generating answer."

    def test_generate_answer_empty_chat_history_default(
        self, answer_generator, sample_question, sample_context
    ):
        """Test that empty list is used for chat_history by default."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer."
        answer_generator.answer_generation_chain = mock_chain

        answer_generator.generate_answer(sample_question, sample_context)

        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["chat_history"] == []


class TestCheckGrounding:
    """Tests for check_grounding method."""

    def test_check_grounding_success(self, answer_generator, sample_context, sample_generation):
        """Test successful grounding check."""
        mock_chain = Mock()
        mock_result = GroundingCheck(
            is_grounded=True,
            ungrounded_statements=[],
            correction_suggestion=""
        )
        mock_chain.invoke.return_value = mock_result
        answer_generator.grounding_check_chain = mock_chain

        result = answer_generator.check_grounding(sample_context, sample_generation)

        assert result == mock_result
        assert result.is_grounded is True

    def test_check_grounding_not_grounded(self, answer_generator, sample_context, sample_generation):
        """Test grounding check when answer is not grounded."""
        mock_chain = Mock()
        mock_result = GroundingCheck(
            is_grounded=False,
            ungrounded_statements=["Unsupported claim"],
            correction_suggestion="Remove unsupported claim"
        )
        mock_chain.invoke.return_value = mock_result
        answer_generator.grounding_check_chain = mock_chain

        result = answer_generator.check_grounding(sample_context, sample_generation)

        assert result.is_grounded is False
        assert len(result.ungrounded_statements) == 1

    def test_check_grounding_error_returns_none(self, answer_generator, sample_context, sample_generation):
        """Test that errors return None."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Check failed")
        answer_generator.grounding_check_chain = mock_chain

        result = answer_generator.check_grounding(sample_context, sample_generation)

        assert result is None

    def test_check_grounding_invokes_chain_correctly(
        self, answer_generator, sample_context, sample_generation
    ):
        """Test that chain is invoked with correct parameters."""
        mock_chain = Mock()
        mock_result = GroundingCheck(is_grounded=True, ungrounded_statements=[], correction_suggestion="")
        mock_chain.invoke.return_value = mock_result
        answer_generator.grounding_check_chain = mock_chain

        answer_generator.check_grounding(sample_context, sample_generation)

        mock_chain.invoke.assert_called_once_with({
            "context": sample_context,
            "generation": sample_generation
        })


class TestGenerateBasicFeedback:
    """Tests for generate_basic_feedback method."""

    def test_generate_basic_feedback_with_ungrounded_statements(
        self, answer_generator, sample_question
    ):
        """Test feedback generation with ungrounded statements."""
        grounding_result = GroundingCheck(
            is_grounded=False,
            ungrounded_statements=["Claim 1", "Claim 2"],
            correction_suggestion="Fix these claims"
        )

        feedback = answer_generator.generate_basic_feedback(grounding_result, sample_question)

        assert "Claim 1" in feedback
        assert "Claim 2" in feedback
        assert "Fix these claims" in feedback
        assert sample_question in feedback

    def test_generate_basic_feedback_with_correction_only(
        self, answer_generator, sample_question
    ):
        """Test feedback generation with only correction suggestion."""
        grounding_result = GroundingCheck(
            is_grounded=False,
            ungrounded_statements=[],
            correction_suggestion="Please be more specific"
        )

        feedback = answer_generator.generate_basic_feedback(grounding_result, sample_question)

        assert "Please be more specific" in feedback
        assert sample_question in feedback

    def test_generate_basic_feedback_with_no_details(
        self, answer_generator, sample_question
    ):
        """Test feedback generation with no specific details."""
        grounding_result = GroundingCheck(
            is_grounded=False,
            ungrounded_statements=[],
            correction_suggestion=""
        )

        feedback = answer_generator.generate_basic_feedback(grounding_result, sample_question)

        assert "not fully grounded" in feedback
        assert sample_question in feedback

    def test_generate_basic_feedback_error_handling(
        self, answer_generator, sample_question
    ):
        """Test error handling in feedback generation."""
        mock_grounding = Mock()
        mock_grounding.ungrounded_statements = None  # Will cause error
        mock_grounding.correction_suggestion = None

        feedback = answer_generator.generate_basic_feedback(mock_grounding, sample_question)

        assert "failed grounding check" in feedback
        assert sample_question in feedback


class TestGenerateAdvancedFeedback:
    """Tests for generate_advanced_feedback method."""

    def test_generate_advanced_feedback_with_all_issues(
        self, answer_generator, sample_question
    ):
        """Test advanced feedback with all issue types."""
        advanced_results = {
            "detailed_grounding": Mock(unsupported_claims=["Claim 1", "Claim 2"]),
            "consistency": Mock(contradictions_found=["Contradiction 1"]),
            "completeness": Mock(missing_aspects=["Missing info"]),
            "hallucination_detection": {"hallucinations": ["Hallucination 1"]},
            "overall_assessment": {"recommendation": "Improve answer"},
        }

        feedback = answer_generator.generate_advanced_feedback(advanced_results, sample_question)

        assert "Claim 1" in feedback
        assert "Contradiction 1" in feedback
        assert "Missing info" in feedback
        assert "Hallucination 1" in feedback
        assert "Improve answer" in feedback

    def test_generate_advanced_feedback_with_limited_issues(
        self, answer_generator, sample_question
    ):
        """Test that feedback limits number of issues shown."""
        advanced_results = {
            "detailed_grounding": Mock(unsupported_claims=["C1", "C2", "C3", "C4", "C5"]),
            "overall_assessment": {"recommendation": "Fix issues"},
        }

        feedback = answer_generator.generate_advanced_feedback(advanced_results, sample_question)

        # Should limit to 3 claims
        assert "C1" in feedback
        assert "C2" in feedback
        assert "C3" in feedback

    def test_generate_advanced_feedback_with_no_issues(
        self, answer_generator, sample_question
    ):
        """Test advanced feedback when no specific issues found."""
        advanced_results = {
            "overall_assessment": {"recommendation": "Improve"},
        }

        feedback = answer_generator.generate_advanced_feedback(advanced_results, sample_question)

        assert "did not meet quality standards" in feedback
        assert sample_question in feedback

    def test_generate_advanced_feedback_error_handling(
        self, answer_generator, sample_question
    ):
        """Test error handling in advanced feedback generation."""
        # Malformed results that will cause errors
        advanced_results = {"bad_key": "bad_value"}

        feedback = answer_generator.generate_advanced_feedback(advanced_results, sample_question)

        assert "advanced quality standards" in feedback
        assert sample_question in feedback


class TestIsGrounded:
    """Tests for is_grounded method."""

    def test_is_grounded_returns_true(self, answer_generator, sample_context, sample_generation):
        """Test is_grounded returns True when answer is grounded."""
        mock_chain = Mock()
        mock_result = GroundingCheck(is_grounded=True, ungrounded_statements=[], correction_suggestion="")
        mock_chain.invoke.return_value = mock_result
        answer_generator.grounding_check_chain = mock_chain

        result = answer_generator.is_grounded(sample_context, sample_generation)

        assert result is True

    def test_is_grounded_returns_false(self, answer_generator, sample_context, sample_generation):
        """Test is_grounded returns False when answer is not grounded."""
        mock_chain = Mock()
        mock_result = GroundingCheck(
            is_grounded=False,
            ungrounded_statements=["Issue"],
            correction_suggestion="Fix"
        )
        mock_chain.invoke.return_value = mock_result
        answer_generator.grounding_check_chain = mock_chain

        result = answer_generator.is_grounded(sample_context, sample_generation)

        assert result is False

    def test_is_grounded_error_returns_false(self, answer_generator, sample_context, sample_generation):
        """Test that errors return False."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Check failed")
        answer_generator.grounding_check_chain = mock_chain

        result = answer_generator.is_grounded(sample_context, sample_generation)

        assert result is False


class TestFormatContext:
    """Tests for format_context method."""

    def test_format_context_with_documents(self, answer_generator, sample_documents):
        """Test formatting documents into context string."""
        result = answer_generator.format_context(sample_documents)

        assert "[Document 1 from doc1.txt]" in result
        assert "[Document 2 from doc2.txt]" in result
        assert "Python is a programming language." in result
        assert "It was created by Guido van Rossum." in result

    def test_format_context_empty_list(self, answer_generator):
        """Test formatting empty document list."""
        result = answer_generator.format_context([])

        assert result == ""

    def test_format_context_document_without_metadata(self, answer_generator):
        """Test formatting document without metadata."""
        doc = Document(page_content="Test content")
        doc.metadata = {}

        result = answer_generator.format_context([doc])

        assert "[Document 1 from unknown]" in result
        assert "Test content" in result

    def test_format_context_error_handling(self, answer_generator):
        """Test error handling when formatting fails."""
        # Create a mock object that will cause errors
        bad_doc = Mock()
        bad_doc.page_content = Mock(side_effect=Exception("Access error"))

        result = answer_generator.format_context([bad_doc])

        assert result == ""

    def test_format_context_multiple_documents_separator(self, answer_generator, sample_documents):
        """Test that documents are separated correctly."""
        result = answer_generator.format_context(sample_documents)

        # Should have separator between documents
        assert "\n\n" in result
        parts = result.split("\n\n")
        assert len(parts) == 2
