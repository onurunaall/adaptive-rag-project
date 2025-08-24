import pytest
import os
import shutil
import json

from unittest.mock import Mock, patch
from langgraph.graph import END
from langchain.schema import Document

from src.core_rag_engine import CoreRAGEngine, GroundingCheck, RerankScore, QueryAnalysis, RelevanceGrade

@pytest.fixture
def mock_embedding():
    """Fixture using LangChain's FakeEmbeddings for testing."""
    from langchain_core.embeddings.fake import FakeEmbeddings
    return FakeEmbeddings(size=1536)
    
@pytest.fixture(scope="module")
def populated_rag_engine(rag_engine):
    """
    Fixture that ingests two documents into a collection named 'rag_test_data'.
    """
    from langchain_core.embeddings.fake import FakeEmbeddings
    # Replace embedding model with FakeEmbeddings
    rag_engine.embedding_model = FakeEmbeddings(size=1536)
    
    cname = "rag_test_data"
    docs = [
        Document(page_content="Paris is the capital of France. It is known for the Eiffel Tower.", metadata={"source": "france_doc"}),
        Document(page_content="The OpenAI GPT-4 model is a large language model.", metadata={"source": "openai_doc"})
    ]
    
    rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)
    return rag_engine, cname

@pytest.fixture(scope="module")
def rag_engine():
    """
    Initialize a CoreRAGEngine with a temporary persistence directory.
    Cleans up the directory after tests complete.
    """
    test_dir = "test_chroma_db"
    os.makedirs(test_dir, exist_ok=True)

    engine = CoreRAGEngine(
        llm_provider="openai",
        embedding_provider="openai",
        persist_directory_base=test_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY_TEST", os.getenv("OPENAI_API_KEY")),
        tavily_api_key=os.getenv("TAVILY_API_KEY_TEST", os.getenv("TAVILY_API_KEY")),
    )

    yield engine

    # Remove the test directory after all tests in this module run
    shutil.rmtree(test_dir)
    
def test_ingest_direct_documents(rag_engine):
    """Tests that ingesting documents creates a queryable vector store."""
    from langchain_core.embeddings.fake import FakeEmbeddings
    rag_engine.embedding_model = FakeEmbeddings(size=1536)
    
    cname = "test_ingest_direct"
    docs = [Document(page_content="LangGraph is a library.")]
    rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)
    
    path = os.path.join(rag_engine.persist_directory_base, cname, "chroma.sqlite3")
    assert os.path.exists(path)
    
    # Create complete state for _retrieve_node
    state = {
        "question": "What is LangGraph?",
        "original_question": "What is LangGraph?",
        "query_analysis_results": None,
        "documents": [],
        "context": "",
        "web_search_results": None,
        "generation": "",
        "retries": 0,
        "run_web_search": "No",
        "relevance_check_passed": None,
        "error_message": None,
        "grounding_check_attempts": 0,
        "regeneration_feedback": None,
        "collection_name": cname,
        "chat_history": []
    }
    res = rag_engine._retrieve_node(state)
    assert any("LangGraph" in doc.page_content for doc in res.get("documents", []))


def test_rag_direct_answer(populated_rag_engine, mocker):
    engine, collection = populated_rag_engine

    mock_analyzer = Mock()
    mock_analyzer.invoke.return_value = QueryAnalysis(
        query_type="factual_lookup", main_intent="testing", extracted_keywords=[], is_ambiguous=False
    )
    mocker.patch.object(engine, 'query_analyzer_chain', mock_analyzer)

    mock_rewriter = Mock()
    # Note: the output of the rewriter is the full object, not just the text
    mock_rewriter.invoke.return_value = "What is the capital city of France?"
    mocker.patch.object(engine, 'query_rewriter_chain', mock_rewriter)

    mock_answer_gen = Mock()
    mock_answer_gen.invoke.return_value = "The answer is Paris."
    mocker.patch.object(engine, 'answer_generation_chain', mock_answer_gen)

    # This mock is for a node, not a chain, so it's fine as is
    mocker.patch.object(engine, '_grounding_check_node', return_value={"regeneration_feedback": None})

    res = engine.run_full_rag_workflow("What is the capital of France?", collection_name=collection)
    assert "Paris" in res["answer"]

def test_rag_web_search_fallback(rag_engine, mocker):
    """
    Tests that the graph correctly follows the web_search path when
    the router decides to.
    """
    engine = rag_engine
    engine.tavily_api_key = "fake_key"

    # Mock the preceding nodes to prevent real API calls
    mocker.patch.object(engine, '_retrieve_node', return_value={"documents": [], "context": ""})
    
    mock_query_analysis = QueryAnalysis(query_type="test", main_intent="test", extracted_keywords=[], is_ambiguous=False)
    mocker.patch.object(engine, 'query_analyzer_chain', return_value=mock_query_analysis)
    
    mocker.patch.object(engine, 'query_rewriter_chain', return_value="test question")
    
    mocker.patch.object(engine, 'document_reranker_chain')
    mocker.patch.object(engine, 'document_relevance_grader_chain')

    # Mock the router to force the graph down the "web_search" branch
    mocker.patch.object(engine, '_route_after_grading', return_value="web_search")

    # Mock the components that are called after the routing decision
    mock_answer_gen = Mock()
    mock_answer_gen.invoke.return_value = "Web result: AlphaFold3 is an AI model."
    mocker.patch.object(engine, 'answer_generation_chain', mock_answer_gen)

    mock_search_tool = Mock()
    mock_search_tool.invoke.return_value = """{
        "query": "What is AlphaFold 3?",
        "follow_up_questions": null,
        "answer": null,
        "images": [],
        "results": [
            {
                "title": "AlphaFold 3 - AI Model",
                "url": "http://fake.url",
                "content": "AlphaFold3 is an AI model developed by Google DeepMind.",
                "score": 0.95
            }
        ]
    }"""
    mocker.patch.object(engine, 'search_tool', mock_search_tool)

    # Mock the final step to prevent any other issues
    mocker.patch.object(engine, '_grounding_check_node', return_value={"regeneration_feedback": None})
    
    res = engine.run_full_rag_workflow("What is AlphaFold 3?")
    
    assert "AlphaFold3" in res["answer"]
    mock_search_tool.invoke.assert_called_once()

def test_grounding_check_node_on_failure(rag_engine, mocker):
    mock_failure_output = GroundingCheck(
        is_grounded=False,
        ungrounded_statements=["The sky is green."],
        correction_suggestion="The answer should stick to the context."
    )
    
    mock_chain = Mock()
    mock_chain.invoke.return_value = mock_failure_output
    mocker.patch.object(rag_engine, 'grounding_check_chain', mock_chain)
    
    initial_state = {
        "question": "What color is the sky?",
        "original_question": "What color is the sky?",
        "query_analysis_results": None,
        "documents": [],
        "context": "The context says the sky is blue.",
        "web_search_results": None,
        "generation": "The sky is green.",
        "retries": 0,
        "run_web_search": "No",
        "relevance_check_passed": None,
        "error_message": None,
        "grounding_check_attempts": 0,
        "regeneration_feedback": None,
        "collection_name": None,
        "chat_history": []
    }
    result_state = rag_engine._grounding_check_node(initial_state)

    assert result_state["regeneration_feedback"] is not None
    assert "The following statements were ungrounded" in result_state["regeneration_feedback"]
    assert result_state["grounding_check_attempts"] == 1


def test_grounding_check_node_on_success(rag_engine, mocker):
    mock_success_output = GroundingCheck(is_grounded=True)
    
    mock_chain = Mock()
    mock_chain.invoke.return_value = mock_success_output
    mocker.patch.object(rag_engine, 'grounding_check_chain', mock_chain)

    initial_state = {
        "question": "What color is the sky?",
        "original_question": "What color is the sky?",
        "query_analysis_results": None,
        "documents": [],
        "context": "The sky is blue.",
        "web_search_results": None,
        "generation": "The sky is blue.",
        "retries": 0,
        "run_web_search": "No",
        "relevance_check_passed": None,
        "error_message": None,
        "grounding_check_attempts": 0,
        "regeneration_feedback": None,
        "collection_name": None,
        "chat_history": []
    }
    result_state = rag_engine._grounding_check_node(initial_state)

    assert result_state["regeneration_feedback"] is None
    assert result_state["grounding_check_attempts"] == 1


def test_route_after_grounding_check(rag_engine):
    """
    Tests the routing logic after grounding check:
      - If no feedback: END
      - If feedback and attempts remain: 'generate_answer'
      - If feedback and max attempts reached: END
    """
    # Case 1: Grounding passed (no feedback) -> should return END
    state_success = {"regeneration_feedback": None}
    assert rag_engine._route_after_grounding_check(state_success) == END

    # Case 2: Grounding failed, attempts remain -> should return 'generate_answer'
    state_retry = {
        "regeneration_feedback": "Please fix.",
        "grounding_check_attempts": 1
    }
    rag_engine.max_grounding_attempts = 2
    assert rag_engine._route_after_grounding_check(state_retry) == "generate_answer"

    # Case 3: Grounding failed, max attempts reached -> should return END
    state_max_attempts = {
        "regeneration_feedback": "Please fix.",
        "grounding_check_attempts": 2
    }
    rag_engine.max_grounding_attempts = 2
    assert rag_engine._route_after_grounding_check(state_max_attempts) == END

def test_grade_documents_node_handles_parsing_error(rag_engine, mocker):
    mock_chain = Mock()
    mock_chain.invoke.side_effect = Exception("LLM or parsing failed")
    mocker.patch.object(rag_engine, 'document_relevance_grader_chain', mock_chain)

    state = {
        "question": "AI?",
        "documents": [Document(page_content="Some content.")],
    }
    result = rag_engine._grade_documents_node(state)

    assert result["relevance_check_passed"] is False
    assert len(result["documents"]) == 0
    
def test_rerank_documents_node_sorts_correctly(rag_engine, mocker):
    docs = [Document(page_content="Low"), Document(page_content="High")]
    mock_scores = [
        RerankScore(relevance_score=0.1, justification=""),
        RerankScore(relevance_score=0.9, justification="")
    ]
    
    mock_chain = Mock()
    mock_chain.invoke.side_effect = mock_scores
    mocker.patch.object(rag_engine, 'document_reranker_chain', mock_chain)

    result_state = rag_engine._rerank_documents_node({"documents": docs, "question": "test"})
    assert result_state["documents"][0].page_content == "High"


def test_document_relevance_grader_chain_parsing(rag_engine, mocker):
    correct_response = RelevanceGrade(is_relevant=True, justification="Matches user question")
    
    mock_chain = Mock()
    mock_chain.invoke.return_value = correct_response
    mocker.patch.object(rag_engine, 'document_relevance_grader_chain', mock_chain)

    state = {
        "question": "What is AI?",
        "documents": [Document(page_content="AI is the simulation of human intelligence.", metadata={})],
        "collection_name": "test"
    }
    result = rag_engine._grade_documents_node(state)

    assert result["relevance_check_passed"] is True
    
def test_document_relevance_grader_chain_bad_json(rag_engine, mocker):
    mock_chain = Mock()
    mock_chain.invoke.side_effect = json.JSONDecodeError("Expecting value", "NOT JSON", 0)
    mocker.patch.object(rag_engine, 'document_relevance_grader_chain', mock_chain)

    state = {
        "question": "AI?",
        "documents": [Document(page_content="No relevance here.", metadata={})],
        "collection_name": "test"
    }
    result = rag_engine._grade_documents_node(state)

    assert result["relevance_check_passed"] is False


def test_query_rewriter_chain_parsing(rag_engine, mocker):
    mock_chain = Mock()
    mock_chain.invoke.return_value = "Rewritten: What is LangGraph?"
    mocker.patch.object(rag_engine, 'query_rewriter_chain', mock_chain)

    state = {"question": "What about LG?", "original_question": "What about LG?"}
    result = rag_engine._rewrite_query_node(state)
    
    # In the source, the rewriter output is now just the string, not a dict
    assert result["question"] == "Rewritten: What is LangGraph?"

def test_ingest_handles_oserror(monkeypatch, rag_engine, mocker, mock_embedding):
    """Simulates a PermissionError and ensures the error is logged."""
    spy_logger = mocker.spy(rag_engine.logger, 'error')
    monkeypatch.setattr(os, "makedirs", Mock(side_effect=PermissionError("No permission")))
    
    rag_engine.ingest(
        direct_documents=[Document(page_content="fail")],
        collection_name="err_test",
        recreate_collection=True
    )
    spy_logger.assert_called_once()
