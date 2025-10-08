import pytest
import os
import shutil
from unittest.mock import Mock

from langchain_core.agents import AgentFinish
from langgraph.graph import StateGraph, END

from src.core_rag_engine import CoreRAGEngine
from src.loop import AgentLoopState, AgentLoopWorkflow, Plan, PlanStep


@pytest.fixture(scope="module")
def core_engine_for_agent():
    test_dir = "agent_test_chroma_db_e2e"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    engine = CoreRAGEngine(
        persist_directory_base=test_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY", "dummy_key"),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "dummy_key"),
    )
    yield engine
    shutil.rmtree(test_dir)


def test_agent_plan_and_execute_workflow(core_engine_for_agent, mocker):
    """
    Tests that the agent can follow a mocked plan, execute each tool step,
    and produce a final summary.
    """
    mock_stock_tool = Mock()
    mock_stock_tool.invoke.return_value = "Successfully fetched news for GOOG."
    mock_stock_tool.name = "FetchStockNews"

    # 2. Define the plan we want the agent to execute
    mock_plan = Plan(
        steps=[
            PlanStep(
                tool="FetchStockNews",
                tool_input={"tickers_input": "GOOG", "max_articles_per_ticker": 1},
                reasoning="Fetch news for testing.",
            )
        ]
    )

    # 3. Create the agent instance, but we will mock its compiled graph
    agent = AgentLoopWorkflow(
        openai_api_key=os.getenv("OPENAI_API_KEY", "dummy_key"),
        core_rag_engine_instance=core_engine_for_agent,
    )
    # Replace the agent's tools for the execute_step to find
    agent.tools = [mock_stock_tool]

    # 4. Create a simple graph for the test, mocking the plan_step
    test_graph = StateGraph(AgentLoopState)
    # Mock the plan_step to return our predefined plan
    test_graph.add_node("plan_step", lambda state: {"plan": mock_plan, "current_step_index": 0})
    # Use the real execute and summarize steps
    test_graph.add_node("execute_step", agent.execute_step)
    test_graph.add_node("summarize_step", agent.summarize_step)

    test_graph.set_entry_point("plan_step")
    test_graph.add_edge("plan_step", "execute_step")
    test_graph.add_conditional_edges(
        "execute_step",
        agent.should_continue,
        {"continue": "execute_step", "end": "summarize_step"},
    )
    test_graph.add_edge("summarize_step", END)

    # 5. Patch the agent's compiled_graph with our test_graph
    agent.compiled_graph = test_graph.compile()

    # 6. Run the workflow
    final_state = agent.run_workflow(goal="Test goal")

    # 7. Assert the results
    assert final_state.get("error") is None
    assert len(final_state["past_steps"]) == 1
    assert "Successfully fetched news" in final_state["final_summary"]
    assert isinstance(final_state.get("agent_outcome"), AgentFinish)

    # Verify the mock tool was called correctly
    mock_stock_tool.invoke.assert_called_once_with({"tickers_input": "GOOG", "max_articles_per_ticker": 1})
