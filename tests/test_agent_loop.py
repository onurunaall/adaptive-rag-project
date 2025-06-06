import pytest
import os
import shutil
from unittest.mock import Mock

from src.core_rag_engine import CoreRAGEngine
from src.loop import AgentLoopWorkflow, Plan, PlanStep
from langchain_core.agents import AgentAction, AgentFinish

@pytest.fixture(scope="module")
def core_engine_for_agent():
    """
    Set up a CoreRAGEngine instance with a temporary persistence directory.
    Clean up after tests complete.
    """
    test_dir = "agent_test_chroma_db_e2e"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    engine = CoreRAGEngine(
        persist_directory_base=test_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY_TEST", os.getenv("OPENAI_API_KEY")),
        tavily_api_key=os.getenv("TAVILY_API_KEY_TEST", os.getenv("TAVILY_API_KEY")),
    )
    yield engine

    shutil.rmtree(test_dir)

def test_agent_plan_and_execute_workflow(core_engine_for_agent, mocker):
    """
    Tests that the agent can follow a mocked plan, execute each tool step,
    reflect in the scratchpad, and produce a final summary (AgentFinish).
    """

    mock_plan = Plan(steps=[
        PlanStep(tool="FetchStockNews", tool_input={}, reasoning="..."),
        PlanStep(tool="InsightEngineIngest", tool_input={}, reasoning="...")])
    
    mock_planner_chain = Mock()
    mock_planner_chain.invoke.return_value = mock_plan
    mocker.patch("src.loop.AgentLoopWorkflow._create_planner_chain", return_value=mock_planner_chain)

    key = os.getenv("OPENAI_API_KEY_TEST", os.getenv("OPENAI_API_KEY"))
    if not key:
        pytest.skip("OpenAI API key not set.")

    agent = AgentLoopWorkflow(
        openai_api_key=key,
        core_rag_engine_instance=core_engine_for_agent,
        enable_tavily_search=False
    )

    final_state = agent.run_workflow(goal="Test goal")

    assert final_state.get("error") is None
    assert len(final_state["past_steps"]) == len(mock_plan.steps)
    assert isinstance(final_state.get("agent_outcome"), AgentFinish)

    