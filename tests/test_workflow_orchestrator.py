"""
Unit tests for WorkflowOrchestrator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from langgraph.graph import END

from src.rag.workflow_orchestrator import WorkflowOrchestrator
from src.rag.models import CoreGraphState


@pytest.fixture
def mock_node_functions():
    """Create mock node functions."""
    return {
        "analyze_query": Mock(return_value={"question": "test"}),
        "rewrite_query": Mock(return_value={"question": "rewritten"}),
        "retrieve": Mock(return_value={"documents": []}),
        "rerank_documents": Mock(return_value={"documents": []}),
        "grade_documents": Mock(return_value={"relevance_check_passed": True}),
        "generate_answer": Mock(return_value={"generation": "answer"}),
        "grounding_check": Mock(return_value={"generation": "answer"}),
        "increment_retries": Mock(return_value={"retries": 1}),
        "web_search": Mock(return_value={"documents": []}),
    }


@pytest.fixture
def mock_routing_functions():
    """Create mock routing functions."""
    return {
        "route_after_grading": Mock(return_value="generate_answer"),
        "route_after_grounding_check": Mock(return_value=END),
    }


@pytest.fixture
def workflow_orchestrator(mock_node_functions, mock_routing_functions):
    """Create a WorkflowOrchestrator instance for testing."""
    return WorkflowOrchestrator(
        node_functions=mock_node_functions,
        routing_functions=mock_routing_functions,
    )


@pytest.fixture
def sample_initial_state():
    """Create sample initial state."""
    return CoreGraphState(
        question="What is Python?",
        original_question="What is Python?",
        documents=[],
        context="",
        generation="",
        chat_history=[],
    )


class TestWorkflowOrchestratorInit:
    """Tests for WorkflowOrchestrator initialization."""

    def test_init_with_required_params(self, mock_node_functions):
        """Test initialization with required parameters."""
        orchestrator = WorkflowOrchestrator(node_functions=mock_node_functions)
        assert orchestrator.node_functions == mock_node_functions
        assert orchestrator.routing_functions == {}
        assert orchestrator.logger is not None
        assert orchestrator.rag_workflow is None
        assert orchestrator.graph is None

    def test_init_with_all_params(self, mock_node_functions, mock_routing_functions):
        """Test initialization with all parameters."""
        mock_logger = Mock()
        orchestrator = WorkflowOrchestrator(
            node_functions=mock_node_functions,
            routing_functions=mock_routing_functions,
            logger=mock_logger,
        )
        assert orchestrator.node_functions == mock_node_functions
        assert orchestrator.routing_functions == mock_routing_functions
        assert orchestrator.logger == mock_logger


class TestCompileWorkflow:
    """Tests for compile_workflow method."""

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_workflow_basic(self, mock_state_graph, workflow_orchestrator):
        """Test basic workflow compilation."""
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        edges = [("analyze_query", "rewrite_query")]

        result = workflow_orchestrator.compile_workflow(
            entry_point="analyze_query",
            edges=edges,
        )

        assert result == mock_compiled
        assert workflow_orchestrator.rag_workflow == mock_compiled
        mock_graph_instance.set_entry_point.assert_called_once_with("analyze_query")

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_workflow_adds_nodes(self, mock_state_graph, workflow_orchestrator):
        """Test that all nodes are added during compilation."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        # Should add all nodes
        assert mock_graph_instance.add_node.call_count == len(workflow_orchestrator.node_functions)

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_workflow_adds_edges(self, mock_state_graph, workflow_orchestrator):
        """Test that edges are added correctly."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        edges = [
            ("analyze_query", "rewrite_query"),
            ("rewrite_query", "retrieve"),
        ]

        workflow_orchestrator.compile_workflow(entry_point="analyze_query", edges=edges)

        assert mock_graph_instance.add_edge.call_count == len(edges)

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_workflow_adds_conditional_edges(
        self, mock_state_graph, workflow_orchestrator
    ):
        """Test that conditional edges are added correctly."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        conditional_edges = [
            ("grade_documents", "route_after_grading", {"generate_answer": "generate_answer"}),
        ]

        workflow_orchestrator.compile_workflow(
            entry_point="analyze_query",
            conditional_edges=conditional_edges,
        )

        mock_graph_instance.add_conditional_edges.assert_called_once()

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_workflow_missing_routing_function(
        self, mock_state_graph, workflow_orchestrator
    ):
        """Test handling of missing routing function."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        conditional_edges = [
            ("grade_documents", "nonexistent_route", {"generate_answer": "generate_answer"}),
        ]

        # Should not raise exception, just log warning
        workflow_orchestrator.compile_workflow(
            entry_point="analyze_query",
            conditional_edges=conditional_edges,
        )

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_workflow_failure_raises_runtime_error(
        self, mock_state_graph, workflow_orchestrator
    ):
        """Test that compilation failure raises RuntimeError."""
        mock_state_graph.side_effect = Exception("Compilation failed")

        with pytest.raises(RuntimeError, match="Failed to compile RAG workflow"):
            workflow_orchestrator.compile_workflow(entry_point="analyze_query")


class TestCompileDefaultRAGWorkflow:
    """Tests for compile_default_rag_workflow method."""

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_compile_default_workflow(self, mock_state_graph, workflow_orchestrator):
        """Test compiling default RAG workflow."""
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        result = workflow_orchestrator.compile_default_rag_workflow()

        assert result == mock_compiled
        assert workflow_orchestrator.rag_workflow == mock_compiled

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_default_workflow_has_correct_structure(
        self, mock_state_graph, workflow_orchestrator
    ):
        """Test that default workflow has expected structure."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_default_rag_workflow()

        # Check entry point
        mock_graph_instance.set_entry_point.assert_called_with("analyze_query")

        # Check that edges and conditional edges were added
        assert mock_graph_instance.add_edge.called
        assert mock_graph_instance.add_conditional_edges.called


class TestAddNode:
    """Tests for add_node method."""

    def test_add_node(self, workflow_orchestrator):
        """Test adding a node."""
        new_node_func = Mock()

        workflow_orchestrator.add_node("new_node", new_node_func)

        assert "new_node" in workflow_orchestrator.node_functions
        assert workflow_orchestrator.node_functions["new_node"] == new_node_func

    def test_add_node_overwrites_existing(self, workflow_orchestrator):
        """Test that adding a node with existing name overwrites it."""
        original_func = workflow_orchestrator.node_functions["analyze_query"]
        new_func = Mock()

        workflow_orchestrator.add_node("analyze_query", new_func)

        assert workflow_orchestrator.node_functions["analyze_query"] == new_func
        assert workflow_orchestrator.node_functions["analyze_query"] != original_func


class TestAddRoutingFunction:
    """Tests for add_routing_function method."""

    def test_add_routing_function(self, workflow_orchestrator):
        """Test adding a routing function."""
        new_routing_func = Mock()

        workflow_orchestrator.add_routing_function("new_route", new_routing_func)

        assert "new_route" in workflow_orchestrator.routing_functions
        assert workflow_orchestrator.routing_functions["new_route"] == new_routing_func

    def test_add_routing_function_overwrites_existing(self, workflow_orchestrator):
        """Test that adding routing function with existing name overwrites it."""
        original_func = workflow_orchestrator.routing_functions["route_after_grading"]
        new_func = Mock()

        workflow_orchestrator.add_routing_function("route_after_grading", new_func)

        assert workflow_orchestrator.routing_functions["route_after_grading"] == new_func


class TestGetWorkflow:
    """Tests for get_workflow method."""

    def test_get_workflow_before_compilation(self, workflow_orchestrator):
        """Test getting workflow before compilation."""
        result = workflow_orchestrator.get_workflow()

        assert result is None

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_get_workflow_after_compilation(self, mock_state_graph, workflow_orchestrator):
        """Test getting workflow after compilation."""
        mock_compiled = Mock()
        mock_graph_instance = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        result = workflow_orchestrator.get_workflow()

        assert result == mock_compiled


class TestInvokeWorkflow:
    """Tests for invoke_workflow method."""

    def test_invoke_workflow_not_compiled_raises_error(
        self, workflow_orchestrator, sample_initial_state
    ):
        """Test that invoking uncompiled workflow raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Workflow not compiled"):
            workflow_orchestrator.invoke_workflow(sample_initial_state)

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_invoke_workflow_success(
        self, mock_state_graph, workflow_orchestrator, sample_initial_state
    ):
        """Test successful workflow invocation."""
        mock_final_state = CoreGraphState(
            question="test",
            original_question="test",
            documents=[],
            context="",
            generation="final answer",
            chat_history=[],
        )

        mock_compiled = Mock()
        mock_compiled.invoke.return_value = mock_final_state
        mock_graph_instance = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        result = workflow_orchestrator.invoke_workflow(sample_initial_state)

        assert result == mock_final_state
        mock_compiled.invoke.assert_called_once_with(sample_initial_state)

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_invoke_workflow_error_raises_runtime_error(
        self, mock_state_graph, workflow_orchestrator, sample_initial_state
    ):
        """Test that workflow execution error raises RuntimeError."""
        mock_compiled = Mock()
        mock_compiled.invoke.side_effect = Exception("Execution failed")
        mock_graph_instance = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        with pytest.raises(RuntimeError, match="Error during workflow execution"):
            workflow_orchestrator.invoke_workflow(sample_initial_state)


class TestAinvokeWorkflow:
    """Tests for ainvoke_workflow method."""

    @pytest.mark.asyncio
    async def test_ainvoke_workflow_not_compiled_raises_error(
        self, workflow_orchestrator, sample_initial_state
    ):
        """Test that async invoking uncompiled workflow raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Workflow not compiled"):
            await workflow_orchestrator.ainvoke_workflow(sample_initial_state)

    @pytest.mark.asyncio
    @patch('src.rag.workflow_orchestrator.StateGraph')
    async def test_ainvoke_workflow_success(
        self, mock_state_graph, workflow_orchestrator, sample_initial_state
    ):
        """Test successful async workflow invocation."""
        mock_final_state = CoreGraphState(
            question="test",
            original_question="test",
            documents=[],
            context="",
            generation="final answer",
            chat_history=[],
        )

        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = mock_final_state
        mock_graph_instance = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        result = await workflow_orchestrator.ainvoke_workflow(sample_initial_state)

        assert result == mock_final_state


class TestStreamWorkflow:
    """Tests for stream_workflow method."""

    def test_stream_workflow_not_compiled_raises_error(
        self, workflow_orchestrator, sample_initial_state
    ):
        """Test that streaming uncompiled workflow raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Workflow not compiled"):
            list(workflow_orchestrator.stream_workflow(sample_initial_state))

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_stream_workflow_success(
        self, mock_state_graph, workflow_orchestrator, sample_initial_state
    ):
        """Test successful workflow streaming."""
        mock_updates = [{"step": 1}, {"step": 2}, {"step": 3}]

        mock_compiled = Mock()
        mock_compiled.stream.return_value = iter(mock_updates)
        mock_graph_instance = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        result = list(workflow_orchestrator.stream_workflow(sample_initial_state))

        assert result == mock_updates


class TestGetGraphVisualization:
    """Tests for get_graph_visualization method."""

    def test_get_graph_visualization_before_compilation(self, workflow_orchestrator):
        """Test getting visualization before compilation."""
        result = workflow_orchestrator.get_graph_visualization()

        assert result is None

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_get_graph_visualization_after_compilation(
        self, mock_state_graph, workflow_orchestrator
    ):
        """Test getting visualization after compilation."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        result = workflow_orchestrator.get_graph_visualization()

        assert result is not None
        assert "Workflow Nodes:" in result


class TestValidateWorkflow:
    """Tests for validate_workflow method."""

    def test_validate_workflow_no_nodes(self):
        """Test validation fails when no nodes defined."""
        orchestrator = WorkflowOrchestrator(node_functions={})

        result = orchestrator.validate_workflow()

        assert result is False

    def test_validate_workflow_not_compiled(self, workflow_orchestrator):
        """Test validation fails when workflow not compiled."""
        result = workflow_orchestrator.validate_workflow()

        assert result is False

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_validate_workflow_success(self, mock_state_graph, workflow_orchestrator):
        """Test successful workflow validation."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        result = workflow_orchestrator.validate_workflow()

        assert result is True


class TestResetWorkflow:
    """Tests for reset_workflow method."""

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_reset_workflow(self, mock_state_graph, workflow_orchestrator):
        """Test resetting workflow."""
        mock_graph_instance = Mock()
        mock_state_graph.return_value = mock_graph_instance

        workflow_orchestrator.compile_workflow(entry_point="analyze_query")

        assert workflow_orchestrator.rag_workflow is not None
        assert workflow_orchestrator.graph is not None

        workflow_orchestrator.reset_workflow()

        assert workflow_orchestrator.rag_workflow is None
        assert workflow_orchestrator.graph is None


class TestIntegration:
    """Integration tests for WorkflowOrchestrator."""

    @patch('src.rag.workflow_orchestrator.StateGraph')
    def test_full_workflow_lifecycle(
        self, mock_state_graph, mock_node_functions, mock_routing_functions, sample_initial_state
    ):
        """Test full lifecycle: create, compile, execute, reset."""
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(
            node_functions=mock_node_functions,
            routing_functions=mock_routing_functions,
        )

        # Compile workflow
        mock_compiled = Mock()
        mock_compiled.invoke.return_value = sample_initial_state
        mock_graph_instance = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        orchestrator.compile_default_rag_workflow()

        # Validate
        assert orchestrator.validate_workflow() is True

        # Execute
        result = orchestrator.invoke_workflow(sample_initial_state)
        assert result == sample_initial_state

        # Reset
        orchestrator.reset_workflow()
        assert orchestrator.rag_workflow is None
