import pytest
import os
import shutil
from unittest.mock import Mock

from src.core_rag_engine import CoreRAGEngine
from src.loop import AgentLoopWorkflow, Plan, PlanStep
from langchain_core.agents import AgentFinish

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
    # 1. Define the plan we want the agent to execute
    mock_plan = Plan(steps=[
        PlanStep(
            tool="FetchStockNews",
            tool_input={"tickers_input": "GOOG", "max_articles_per_ticker": 1},
            reasoning="Fetch news for testing."
        )
    ])

    # 2. Mock the tools to avoid real API calls
    # Mock the stock tool to return a predictable result
    mock_stock_tool = Mock()
    mock_stock_tool.run.return_value = "Successfully fetched news for GOOG."
    
    # 3. Create the agent instance
    key = os.getenv("OPENAI_API_KEY", "dummy_key")
    agent = AgentLoopWorkflow(
        openai_api_key=key,
        core_rag_engine_instance=core_engine_for_agent
    )

    # 4. Replace the agent's real tools with our mocks
    # This is a robust way to control tool behavior in tests
    agent.tools = [mock_stock_tool]
    # Match the name used in the plan
    mock_stock_tool.name = "FetchStockNews"

    # 5. Mock the 'plan_step' to inject our plan into the workflow
    mocker.patch.object(
        agent, 
        'plan_step', 
        return_value={"plan": mock_plan, "current_step_index": 0}
    )

    # 6. Run the workflow
    final_state = agent.run_workflow(goal="Test goal")

    # 7. Assert the results
    assert final_state.get("error") is None
    assert len(final_state["past_steps"]) == 1
    assert "Successfully fetched news" in final_state["final_summary"]
    assert isinstance(final_state.get("agent_outcome"), AgentFinish)
    
    # Verify the mock tool was called correctly
    mock_stock_tool.run.assert_called_once_with(
        {"tickers_input": "GOOG", "max_articles_per_ticker": 1}
    )

