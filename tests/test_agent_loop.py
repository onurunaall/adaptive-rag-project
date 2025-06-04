import pytest
import os
import shutil
from src.core_rag_engine import CoreRAGEngine
from src.loop import AgentLoopWorkflow, AgentLoopState
from langchain.schema import Document
from langchain.agents import AgentFinish

@pytest.fixture(scope="module")
def core_engine_for_agent():
    test_dir = "agent_test_chroma_db"
    os.makedirs(test_dir, exist_ok=True)
    engine = CoreRAGEngine(
        persist_directory_base=test_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY_TEST", os.getenv("OPENAI_API_KEY")),
        tavily_api_key=os.getenv("TAVILY_API_KEY_TEST", os.getenv("TAVILY_API_KEY")),
    )
    yield engine
    shutil.rmtree(test_dir)

def test_agent_fetch_ingest_query(core_engine_for_agent, mocker):
    mock_doc = Document(page_content="NVIDIA announced a new GPU today.", metadata={"source": "mock_news_nvda"})
    mocker.patch('src.loop.fetch_stock_news_documents', return_value=[mock_doc])

    spy_ingest = mocker.spy(core_engine_for_agent, 'ingest')
    spy_rag = mocker.spy(core_engine_for_agent, 'run_full_rag_workflow')

    key = os.getenv("OPENAI_API_KEY_TEST", os.getenv("OPENAI_API_KEY"))

    if not key:
        pytest.skip("OpenAI API key for agent LLM not set.")

    agent = AgentLoopWorkflow(
        openai_api_key=key,
        core_rag_engine_instance=core_engine_for_agent,
        enable_tavily_search=False
    )

    goal = (
        "Fetch 1 news article for NVDA, "
        "ingest it into collection 'agent_nvda_news' (recreate it), "
        "then tell me what NVIDIA announced."
    )

    state = agent.run_workflow(goal=goal)
    assert isinstance(state.get("agent_outcome"), AgentFinish)

    out = state["agent_outcome"].return_values.get("output", "").lower()
    assert "new gpu" in out

    tool_calls = [action.tool for action, obs in state.get("intermediate_steps", [])]
    assert "FetchStockNews" in tool_calls
    assert "InsightEngineIngest" in tool_calls
    assert "InsightEngineRAGWorkflow" in tool_calls

    spy_ingest.assert_called_once()
    args_ingest, kwargs_ingest = spy_ingest.call_args
    assert kwargs_ingest.get("collection_name") == "agent_nvda_news"
    assert any("NVIDIA announced a new GPU today" in doc.page_content for doc in kwargs_ingest.get("direct_documents", []))
    
    spy_rag.assert_called_once()
    args_rag, kwargs_rag = spy_rag.call_args
    assert kwargs_rag.get("collection_name") == "agent_nvda_news"
    assert "what nvidia announced" in kwargs_rag.get("question", "").lower()
