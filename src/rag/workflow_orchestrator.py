"""
Workflow Orchestrator Module

Handles LangGraph workflow orchestration for RAG pipeline.
Extracted from CoreRAGEngine to improve modularity and maintainability.

This module coordinates all the RAG components through a state graph workflow.
"""

import logging
from typing import Any, Optional, Dict, List, Callable
from langgraph.graph import StateGraph, END

from src.rag.models import CoreGraphState


class WorkflowOrchestrator:
    """
    Manages LangGraph workflow orchestration for RAG pipeline.

    Responsibilities:
    - Define workflow graph structure (nodes and edges)
    - Compile LangGraph workflow
    - Coordinate routing logic between nodes
    - Execute workflow with state management
    """

    def __init__(
        self,
        node_functions: Dict[str, Callable],
        routing_functions: Optional[Dict[str, Callable]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize WorkflowOrchestrator.

        Args:
            node_functions: Dictionary mapping node names to node functions
            routing_functions: Dictionary mapping routing decision points to routing functions
            logger: Optional logger instance
        """
        self.node_functions = node_functions
        self.routing_functions = routing_functions or {}
        self.logger = logger or logging.getLogger(__name__)

        self.rag_workflow = None
        self.graph = None

    def compile_workflow(
        self,
        entry_point: str = "analyze_query",
        edges: Optional[List[tuple]] = None,
        conditional_edges: Optional[List[tuple]] = None,
    ) -> Any:
        """
        Compile the RAG workflow graph.

        Args:
            entry_point: Name of the entry node
            edges: List of (from_node, to_node) tuples for simple edges
            conditional_edges: List of (from_node, routing_function_name, mapping) tuples

        Returns:
            Compiled workflow graph
        """
        try:
            self.logger.info("Compiling RAG workflow...")

            graph = StateGraph(CoreGraphState)
            self.graph = graph

            # Add all nodes
            for node_name, node_function in self.node_functions.items():
                self.logger.debug(f"Adding node: {node_name}")
                graph.add_node(node_name, node_function)

            # Set entry point
            graph.set_entry_point(entry_point)
            self.logger.debug(f"Set entry point: {entry_point}")

            # Add simple edges
            if edges:
                for from_node, to_node in edges:
                    self.logger.debug(f"Adding edge: {from_node} -> {to_node}")
                    graph.add_edge(from_node, to_node)

            # Add conditional edges
            if conditional_edges:
                for from_node, routing_func_name, mapping in conditional_edges:
                    routing_func = self.routing_functions.get(routing_func_name)
                    if routing_func:
                        self.logger.debug(f"Adding conditional edge from: {from_node}")
                        graph.add_conditional_edges(from_node, routing_func, mapping)
                    else:
                        self.logger.warning(
                            f"Routing function '{routing_func_name}' not found for node '{from_node}'"
                        )

            # Compile the graph
            self.rag_workflow = graph.compile()
            self.logger.info("RAG workflow compiled successfully.")

            return self.rag_workflow

        except Exception as e:
            error_msg = f"Failed to compile RAG workflow: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def compile_default_rag_workflow(self) -> Any:
        """
        Compile the default RAG workflow with standard structure.

        This creates a workflow with the following flow:
        analyze_query -> rewrite_query -> retrieve -> rerank -> grade ->
        (conditional routing) -> generate_answer -> grounding_check -> END

        Returns:
            Compiled workflow graph
        """
        # Define simple edges
        edges = [
            ("analyze_query", "rewrite_query"),
            ("retrieve", "rerank_documents"),
            ("rerank_documents", "grade_documents"),
            ("web_search", "generate_answer"),
            ("generate_answer", "grounding_check"),
            ("increment_retries", "rewrite_query"),
            ("rewrite_query", "retrieve"),
        ]

        # Define conditional edges
        conditional_edges = [
            (
                "grade_documents",
                "route_after_grading",
                {
                    "generate_answer": "generate_answer",
                    "increment_retries": "increment_retries",
                    "web_search": "web_search",
                },
            ),
            (
                "grounding_check",
                "route_after_grounding_check",
                {
                    "generate_answer": "generate_answer",
                    END: END,
                },
            ),
        ]

        return self.compile_workflow(
            entry_point="analyze_query",
            edges=edges,
            conditional_edges=conditional_edges,
        )

    def add_node(self, node_name: str, node_function: Callable) -> None:
        """
        Add a node to the workflow.

        Args:
            node_name: Name of the node
            node_function: Function to execute for this node
        """
        self.node_functions[node_name] = node_function
        self.logger.debug(f"Added node: {node_name}")

    def add_routing_function(self, name: str, routing_function: Callable) -> None:
        """
        Add a routing function for conditional edges.

        Args:
            name: Name of the routing function
            routing_function: Function that determines next node
        """
        self.routing_functions[name] = routing_function
        self.logger.debug(f"Added routing function: {name}")

    def get_workflow(self) -> Optional[Any]:
        """
        Get the compiled workflow.

        Returns:
            Compiled workflow or None if not yet compiled
        """
        return self.rag_workflow

    def invoke_workflow(self, initial_state: CoreGraphState) -> CoreGraphState:
        """
        Invoke the workflow with initial state.

        Args:
            initial_state: Initial state for the workflow

        Returns:
            Final state after workflow execution
        """
        if not self.rag_workflow:
            raise RuntimeError("Workflow not compiled. Call compile_workflow() first.")

        try:
            self.logger.info("Invoking RAG workflow...")
            final_state = self.rag_workflow.invoke(initial_state)
            self.logger.info("RAG workflow completed successfully.")
            return final_state

        except Exception as e:
            error_msg = f"Error during workflow execution: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def ainvoke_workflow(self, initial_state: CoreGraphState) -> CoreGraphState:
        """
        Asynchronously invoke the workflow with initial state.

        Args:
            initial_state: Initial state for the workflow

        Returns:
            Final state after workflow execution
        """
        if not self.rag_workflow:
            raise RuntimeError("Workflow not compiled. Call compile_workflow() first.")

        try:
            self.logger.info("Asynchronously invoking RAG workflow...")
            final_state = await self.rag_workflow.ainvoke(initial_state)
            self.logger.info("RAG workflow completed successfully.")
            return final_state

        except Exception as e:
            error_msg = f"Error during async workflow execution: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def stream_workflow(self, initial_state: CoreGraphState):
        """
        Stream workflow execution.

        Args:
            initial_state: Initial state for the workflow

        Yields:
            State updates as workflow progresses
        """
        if not self.rag_workflow:
            raise RuntimeError("Workflow not compiled. Call compile_workflow() first.")

        try:
            self.logger.info("Streaming RAG workflow...")
            for state_update in self.rag_workflow.stream(initial_state):
                yield state_update

            self.logger.info("RAG workflow streaming completed.")

        except Exception as e:
            error_msg = f"Error during workflow streaming: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def astream_workflow(self, initial_state: CoreGraphState):
        """
        Asynchronously stream workflow execution.

        Args:
            initial_state: Initial state for the workflow

        Yields:
            State updates as workflow progresses
        """
        if not self.rag_workflow:
            raise RuntimeError("Workflow not compiled. Call compile_workflow() first.")

        try:
            self.logger.info("Asynchronously streaming RAG workflow...")
            async for state_update in self.rag_workflow.astream(initial_state):
                yield state_update

            self.logger.info("RAG workflow async streaming completed.")

        except Exception as e:
            error_msg = f"Error during async workflow streaming: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_graph_visualization(self) -> Optional[str]:
        """
        Get a text representation of the workflow graph.

        Returns:
            String representation of the graph structure or None
        """
        if not self.graph:
            return None

        try:
            nodes = list(self.node_functions.keys())
            return f"Workflow Nodes: {', '.join(nodes)}"
        except Exception as e:
            self.logger.error(f"Error getting graph visualization: {e}", exc_info=True)
            return None

    def validate_workflow(self) -> bool:
        """
        Validate that the workflow has been properly configured.

        Returns:
            True if workflow is valid, False otherwise
        """
        try:
            if not self.node_functions:
                self.logger.error("No node functions defined")
                return False

            if not self.rag_workflow:
                self.logger.error("Workflow not compiled")
                return False

            self.logger.info("Workflow validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error during workflow validation: {e}", exc_info=True)
            return False

    def reset_workflow(self) -> None:
        """
        Reset the compiled workflow (useful for reconfiguration).
        """
        self.rag_workflow = None
        self.graph = None
        self.logger.info("Workflow reset")
