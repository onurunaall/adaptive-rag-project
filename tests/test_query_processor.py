"""
Unit tests for QueryProcessor.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.rag.query_processor import QueryProcessor
from src.rag.models import QueryAnalysis


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return Mock()


@pytest.fixture
def mock_json_llm():
    """Create a mock JSON LLM."""
    return Mock()


@pytest.fixture
def query_processor(mock_llm, mock_json_llm):
    """Create a QueryProcessor instance for testing."""
    return QueryProcessor(llm=mock_llm, json_llm=mock_json_llm)


@pytest.fixture
def sample_query_analysis():
    """Create a sample QueryAnalysis object."""
    return QueryAnalysis(
        query_type="factual_lookup",
        main_intent="Find information about Python",
        key_terms=["Python", "programming"],
        is_ambiguous=False,
        needs_clarification=False,
    )


@pytest.fixture
def sample_chat_history():
    """Create sample chat history."""
    return [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
    ]


class TestQueryProcessorInit:
    """Tests for QueryProcessor initialization."""

    def test_init_with_required_params(self, mock_llm, mock_json_llm):
        """Test initialization with required parameters."""
        processor = QueryProcessor(llm=mock_llm, json_llm=mock_json_llm)
        assert processor.llm == mock_llm
        assert processor.json_llm == mock_json_llm
        assert processor.query_analyzer_chain is not None
        assert processor.query_rewriter_chain is not None

    def test_init_creates_chains(self, mock_llm, mock_json_llm):
        """Test that initialization creates analyzer and rewriter chains."""
        with patch.object(QueryProcessor, "_create_query_analyzer_chain") as mock_analyzer:
            with patch.object(QueryProcessor, "_create_query_rewriter_chain") as mock_rewriter:
                mock_analyzer.return_value = Mock()
                mock_rewriter.return_value = Mock()

                processor = QueryProcessor(llm=mock_llm, json_llm=mock_json_llm)

                mock_analyzer.assert_called_once()
                mock_rewriter.assert_called_once()


class TestFormatChatHistory:
    """Tests for format_chat_history method."""

    def test_format_empty_chat_history(self, query_processor):
        """Test formatting empty chat history."""
        result = query_processor.format_chat_history([])
        assert result == "No chat history."

    def test_format_chat_history_with_messages(self, query_processor, sample_chat_history):
        """Test formatting chat history with messages."""
        result = query_processor.format_chat_history(sample_chat_history)
        assert "HUMAN: What is Python?" in result
        assert "AI: Python is a programming language." in result

    def test_format_chat_history_preserves_order(self, query_processor):
        """Test that chat history order is preserved."""
        history = [
            HumanMessage(content="First message"),
            AIMessage(content="Second message"),
            HumanMessage(content="Third message"),
        ]
        result = query_processor.format_chat_history(history)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "First message" in lines[0]
        assert "Second message" in lines[1]
        assert "Third message" in lines[2]


class TestAnalyzeQuery:
    """Tests for analyze_query method."""

    def test_analyze_query_success(self, query_processor, sample_query_analysis):
        """Test successful query analysis."""
        query_processor.query_analyzer_chain = Mock()
        query_processor.query_analyzer_chain.invoke.return_value = sample_query_analysis

        result = query_processor.analyze_query("What is Python?")

        assert result == sample_query_analysis
        assert result.query_type == "factual_lookup"
        assert result.main_intent == "Find information about Python"

    def test_analyze_query_with_chat_history(self, query_processor, sample_query_analysis, sample_chat_history):
        """Test query analysis with chat history."""
        query_processor.query_analyzer_chain = Mock()
        query_processor.query_analyzer_chain.invoke.return_value = sample_query_analysis

        result = query_processor.analyze_query("What is it used for?", chat_history=sample_chat_history)

        assert result is not None
        query_processor.query_analyzer_chain.invoke.assert_called_once()
        call_args = query_processor.query_analyzer_chain.invoke.call_args[0][0]
        assert "question" in call_args
        assert "chat_history_formatted" in call_args

    def test_analyze_query_handles_error(self, query_processor):
        """Test that analyze_query returns None on error."""
        query_processor.query_analyzer_chain = Mock()
        query_processor.query_analyzer_chain.invoke.side_effect = Exception("Analysis failed")

        result = query_processor.analyze_query("Test question")

        assert result is None

    def test_analyze_query_logs_ambiguous_queries(self, query_processor):
        """Test that ambiguous queries are logged with warning."""
        ambiguous_analysis = QueryAnalysis(
            query_type="ambiguous",
            main_intent="Unclear",
            key_terms=[],
            is_ambiguous=True,
            needs_clarification=True,
        )

        query_processor.query_analyzer_chain = Mock()
        query_processor.query_analyzer_chain.invoke.return_value = ambiguous_analysis

        with patch.object(query_processor.logger, "warning") as mock_warning:
            result = query_processor.analyze_query("ambiguous question")

            assert result.is_ambiguous is True
            mock_warning.assert_called_once()


class TestRewriteQuery:
    """Tests for rewrite_query method."""

    def test_rewrite_query_success(self, query_processor):
        """Test successful query rewriting."""
        query_processor.query_rewriter_chain = Mock()
        query_processor.query_rewriter_chain.invoke.return_value = "What is Python programming language?"

        result = query_processor.rewrite_query("What is it?")

        assert result == "What is Python programming language?"

    def test_rewrite_query_with_chat_history(self, query_processor, sample_chat_history):
        """Test query rewriting with chat history."""
        query_processor.query_rewriter_chain = Mock()
        query_processor.query_rewriter_chain.invoke.return_value = "What is Python used for?"

        result = query_processor.rewrite_query("What is it used for?", chat_history=sample_chat_history)

        assert result == "What is Python used for?"
        query_processor.query_rewriter_chain.invoke.assert_called_once()
        call_args = query_processor.query_rewriter_chain.invoke.call_args[0][0]
        assert "question" in call_args
        assert "chat_history" in call_args

    def test_rewrite_query_no_change_needed(self, query_processor):
        """Test when query doesn't need rewriting."""
        original_query = "What is Python?"
        query_processor.query_rewriter_chain = Mock()
        query_processor.query_rewriter_chain.invoke.return_value = original_query

        with patch.object(query_processor.logger, "info") as mock_info:
            result = query_processor.rewrite_query(original_query)

            assert result == original_query
            assert any("No rewrite needed" in str(call) for call in mock_info.call_args_list)

    def test_rewrite_query_handles_error(self, query_processor):
        """Test that rewrite_query returns original on error."""
        original_query = "Test question"
        query_processor.query_rewriter_chain = Mock()
        query_processor.query_rewriter_chain.invoke.side_effect = Exception("Rewrite failed")

        result = query_processor.rewrite_query(original_query)

        assert result == original_query

    def test_rewrite_query_strips_whitespace(self, query_processor):
        """Test that rewritten query is stripped of whitespace."""
        query_processor.query_rewriter_chain = Mock()
        query_processor.query_rewriter_chain.invoke.return_value = "  Rewritten query  "

        result = query_processor.rewrite_query("Original")

        assert result == "Rewritten query"

    def test_rewrite_query_logs_changes(self, query_processor):
        """Test that query changes are logged."""
        query_processor.query_rewriter_chain = Mock()
        query_processor.query_rewriter_chain.invoke.return_value = "Rewritten query"

        with patch.object(query_processor.logger, "info") as mock_info:
            result = query_processor.rewrite_query("Original query")

            assert any("Rewrote" in str(call) for call in mock_info.call_args_list)


class TestShouldUseWebSearch:
    """Tests for should_use_web_search method."""

    def test_should_use_web_search_for_factual_lookup(self, query_processor):
        """Test that factual lookup queries should use web search."""
        analysis = QueryAnalysis(
            query_type="factual_lookup",
            main_intent="Find facts",
            key_terms=["test"],
            is_ambiguous=False,
            needs_clarification=False,
        )

        result = query_processor.should_use_web_search(analysis)

        assert result is True

    def test_should_use_web_search_for_comparison(self, query_processor):
        """Test that comparison queries should use web search."""
        analysis = QueryAnalysis(
            query_type="comparison",
            main_intent="Compare items",
            key_terms=["test"],
            is_ambiguous=False,
            needs_clarification=False,
        )

        result = query_processor.should_use_web_search(analysis)

        assert result is True

    def test_should_use_web_search_for_summary(self, query_processor):
        """Test that summary requests should use web search."""
        analysis = QueryAnalysis(
            query_type="summary_request",
            main_intent="Summarize",
            key_terms=["test"],
            is_ambiguous=False,
            needs_clarification=False,
        )

        result = query_processor.should_use_web_search(analysis)

        assert result is True

    def test_should_use_web_search_for_complex_reasoning(self, query_processor):
        """Test that complex reasoning queries should use web search."""
        analysis = QueryAnalysis(
            query_type="complex_reasoning",
            main_intent="Reason through",
            key_terms=["test"],
            is_ambiguous=False,
            needs_clarification=False,
        )

        result = query_processor.should_use_web_search(analysis)

        assert result is True

    def test_should_not_use_web_search_for_greeting(self, query_processor):
        """Test that greetings should not use web search."""
        analysis = QueryAnalysis(
            query_type="greeting",
            main_intent="Greet",
            key_terms=[],
            is_ambiguous=False,
            needs_clarification=False,
        )

        result = query_processor.should_use_web_search(analysis)

        assert result is False

    def test_should_not_use_web_search_for_keyword_search(self, query_processor):
        """Test that keyword searches should not use web search."""
        analysis = QueryAnalysis(
            query_type="keyword_search_sufficient",
            main_intent="Simple search",
            key_terms=["test"],
            is_ambiguous=False,
            needs_clarification=False,
        )

        result = query_processor.should_use_web_search(analysis)

        assert result is False

    def test_should_not_use_web_search_when_analysis_none(self, query_processor):
        """Test that None analysis returns False."""
        result = query_processor.should_use_web_search(None)

        assert result is False


class TestCreateChains:
    """Tests for chain creation methods."""

    def test_create_query_analyzer_chain(self, mock_llm, mock_json_llm):
        """Test creating query analyzer chain."""
        processor = QueryProcessor(llm=mock_llm, json_llm=mock_json_llm)
        assert processor.query_analyzer_chain is not None

    def test_create_query_rewriter_chain(self, mock_llm, mock_json_llm):
        """Test creating query rewriter chain."""
        processor = QueryProcessor(llm=mock_llm, json_llm=mock_json_llm)
        assert processor.query_rewriter_chain is not None

    def test_create_analyzer_chain_error_handling(self, mock_llm, mock_json_llm):
        """Test that chain creation errors are handled."""
        with patch("src.rag.query_processor.PydanticOutputParser") as mock_parser:
            mock_parser.side_effect = Exception("Parser creation failed")

            with pytest.raises(RuntimeError, match="Failed to create query analyzer chain"):
                QueryProcessor(llm=mock_llm, json_llm=mock_json_llm)


class TestIntegration:
    """Integration tests for QueryProcessor."""

    def test_full_query_analysis_and_rewrite_workflow(self, query_processor, sample_query_analysis):
        """Test complete workflow of analyzing and rewriting a query."""
        query_processor.query_analyzer_chain = Mock()
        query_processor.query_rewriter_chain = Mock()

        query_processor.query_analyzer_chain.invoke.return_value = sample_query_analysis
        query_processor.query_rewriter_chain.invoke.return_value = "What is the Python programming language?"

        original_query = "What is it?"
        chat_history = [HumanMessage(content="Tell me about Python")]

        analysis = query_processor.analyze_query(original_query, chat_history)
        rewritten = query_processor.rewrite_query(original_query, chat_history)
        should_search = query_processor.should_use_web_search(analysis)

        assert analysis is not None
        assert analysis.query_type == "factual_lookup"
        assert rewritten == "What is the Python programming language?"
        assert should_search is True
