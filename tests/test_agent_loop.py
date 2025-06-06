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
    # Define a mocked plan with three sequential steps
    mock_plan = Plan(steps=[
        PlanStep(
            tool="FetchStockNews",
            tool_input={"tickers_input": "NVDA", "max_articles_per_ticker": 1},
            reasoning="First, I need to fetch the latest news for the requested ticker."
        ),
        PlanStep(
            tool="InsightEngineIngest",
            tool_input={"collection_name": "agent_test_nvda_news", "recreate_collection": True},
            reasoning="Next, I need to ingest the fetched news into a new collection to make it queryable."
        ),
        PlanStep(
            tool="InsightEngineRAGWorkflow",
            tool_input={"collection_name": "agent_test_nvda_news", "question": "What was the key announcement in the news?"},
            reasoning="Finally, I will query the ingested news to answer the user's core question."
        )
    ])

    # Patch the planner chain so it returns our mock_plan
    mock_planner_chain = Mock()
    mock_planner_chain.invoke.return_value = mock_plan
    mocker.patch(
        "src.loop.AgentLoopWorkflow._create_planner_chain",
        return_value=mock_planner_chain
    )

    # Spy on execute_step to verify each plan step is executed
    spy_tool_executor = mocker.spy(agent.tool_executor, "invoke")

    # Ensure an OpenAI API key is available; otherwise skip this test
    key = os.getenv("OPENAI_API_KEY_TEST", os.getenv("OPENAI_API_KEY"))
    if not key:
        pytest.skip("OpenAI API key for agent LLM not set.")

    # Instantiate the agent workflow with the mocked CoreRAGEngine
    agent = AgentLoopWorkflow(
        openai_api_key=key,
        core_rag_engine_instance=core_engine_for_agent,
        enable_tavily_search=False
    )

    # Run the workflow with a sample goal
    goal = "Fetch news for NVDA, ingest it, and find out the key announcement."
    final_state = agent.run_workflow(goal=goal)

    # === Assertions ===

    # 1. No error should be present in the final state
    assert final_state.get("error") is None, (
        f"Workflow failed with error: {final_state.get('error')}"
    )

    # 2. Each of the three plan steps should have been executed
    assert spy_tool_executor.call_count == len(mock_plan.steps)

    # 3. The agent outcome must be an AgentFinish instance
    agent_outcome = final_state.get("agent_outcome")
    assert isinstance(agent_outcome, AgentFinish)

    # 4. The final summary (output) must be a non-empty string
    final_output = agent_outcome.return_values.get("output", "")
    assert isinstance(final_output, str)
    assert len(final_output) > 0

    # 5. The scratchpad should contain one reflection per plan step
    assert "scratchpad" in final_state
    assert len(final_state["scratchpad"]) == len(mock_plan.steps)