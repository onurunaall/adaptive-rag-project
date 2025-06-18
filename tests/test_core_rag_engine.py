import pytest
import os
import shutil
import json

from unittest.mock import Mock, patch
from langgraph.graph import END
from langchain.schema import Document

from src.core_rag_engine import CoreRAGEngine, GroundingCheck, RerankScore, QueryAnalysis, RelevanceGrade


@pytest.fixture
def mock_embedding(mocker):
    """Fixture to mock the OpenAI embedding function."""
    return mocker.patch('langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents', return_value=[[0.1] * 1536])

@pytest.fixture(scope="module")
def populated_rag_engine(rag_engine): # Add 'mocker' if not present, though patch is used here
    """
    Fixture that ingests two documents into a collection named 'rag_test_data'.
    """
    cname = "rag_test_data"
    docs = [Document(page_content="Paris is the capital of France. It is known for the Eiffel Tower.", metadata={"source": "france_doc"}),
            Document(page_content="The OpenAI GPT-4 model is a large language model.", metadata={"source": "openai_doc"})]
    
    with patch('langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents') as mock_embed:
        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
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

def test_ingest_direct_documents(rag_engine, mock_embedding):
    """Tests that ingesting documents creates a queryable vector store."""
    cname = "test_ingest_direct"
    docs = [Document(page_content="LangGraph is a library.")]
    # The mock_embedding fixture is now used automatically
    rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)
    
    path = os.path.join(rag_engine.persist_directory_base, cname, "chroma.sqlite3")
    assert os.path.exists(path)
    
    res = rag_engine._retrieve_node({"question": "What is LangGraph?", "collection_name": cname})
    assert any("LangGraph" in doc.page_content for doc in res.get("documents", []))


def test_rag_direct_answer(populated_rag_engine, mocker):
    """Tests the full RAG workflow, mocking all sub-chains."""
    engine, collection = populated_rag_engine
    
    mock_analysis = QueryAnalysis(
        query_type="factual_lookup",
        main_intent="testing",
        extracted_keywords=[],
        is_ambiguous=False
    )
    
    mocker.patch.object(engine.query_analyzer_chain, '__call__', return_value=mock_analysis)
    
    mocker.patch.object(engine.query_rewriter_chain, '__call__', return_value={'text': 'What is the capital of France?'})
    mocker.patch.object(engine.answer_generation_chain, '__call__', return_value={'text': 'The answer is Paris.'})
    mocker.patch.object(engine, '_grounding_check_node', return_value={"regeneration_feedback": None})
    
    res = engine.run_full_rag_workflow("What is the capital of France?", collection_name=collection)
    assert "Paris" in res["answer"]

def test_rag_web_search_fallback(rag_engine, mocker):
    """Tests that the engine falls back to web search."""
    engine = rag_engine
    engine.tavily_api_key = "fake_key"

    mocker.patch.object(engine, '_retrieve_node', return_value={"documents": []})
    mocker.patch.object(engine, '_grade_documents_node', return_value={"relevance_check_passed": False, "documents": []})

    mocker.patch.object(engine.query_rewriter_chain, '__call__', return_value={'text': 'What is AlphaFold 3?'})
    mocker.patch.object(engine.answer_generation_chain, '__call__', return_value={'text': 'Web result: AlphaFold3 is an AI model.'})

    mocker.patch.object(engine.search_tool, 'run', return_value=[{"content": "AlphaFold3 is an AI model."}])

    res = engine.run_full_rag_workflow("What is AlphaFold 3?")
    assert "AlphaFold3" in res["answer"]

def test_grounding_check_node_on_failure(rag_engine, mocker):
    """
    Tests that if the grounding check fails, the node correctly populates
    the 'regeneration_feedback' field and increments attempts.
    """
    mock_failure_output = GroundingCheck(
        is_grounded=False,
        ungrounded_statements=["The sky is green."],
        correction_suggestion="The answer should stick to the context."
    )
    mock_chain = Mock()
    mock_chain.__call__.return_value = mock_failure_output

    mocker.patch.object(rag_engine, 'grounding_check_chain', mock_chain)

    initial_state = {
        "question": "What color is the sky?",
        "context": "The context says the sky is blue.",
        "generation": "The sky is green.",
        "grounding_check_attempts": 0,
    }

    result_state = rag_engine._grounding_check_node(initial_state)

    assert result_state["regeneration_feedback"] is not None
    assert "The following statements were ungrounded" in result_state["regeneration_feedback"]
    assert "Suggestion for correction" in result_state["regeneration_feedback"]
    assert result_state["grounding_check_attempts"] == 1


def test_grounding_check_node_on_success(rag_engine, mocker):
    """
    Tests that if the grounding check passes, 'regeneration_feedback'
    remains None and attempts increment.
    """
    mock_success_output = GroundingCheck(is_grounded=True)
    mock_chain = Mock()
    mock_chain.__call__.return_value = mock_success_output
    mocker.patch.object(rag_engine, 'grounding_check_chain', mock_chain)
    
    initial_state = {
        "context": "The sky is blue.",
        "generation": "The sky is blue.",
        "grounding_check_attempts": 0,
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
    """
    Tests that _grade_documents_node continues gracefully if the grader chain
    fails with a parsing or execution error.
    """
    mocker.patch.object(
        rag_engine.document_relevance_grader_chain, 
        '__call__', 
        side_effect=Exception("LLM or parsing failed")
    )
    
    state = {
        "question": "AI?",
        "documents": [Document(page_content="Some content.")],
    }

    result = rag_engine._grade_documents_node(state)

    assert result["relevance_check_passed"] is False
    assert len(result["documents"]) == 0

def test_rerank_documents_node_sorts_correctly(rag_engine, mocker):
    """Tests that the rerank node sorts documents by score."""
    docs = [Document(page_content="Low"),
            Document(page_content="High")]
    
    mock_scores = [RerankScore(relevance_score=0.1, justification=""),
                   RerankScore(relevance_score=0.9, justification="")]
    
    mock_chain = Mock()
    mock_chain.__call__.side_effect = mock_scores
    mocker.patch.object(rag_engine, 'document_reranker_chain', mock_chain)
    
    result_state = rag_engine._rerank_documents_node({"documents": docs, "question": "test"})
    assert result_state["documents"][0].page_content == "High"


def test_document_relevance_grader_chain_parsing(monkeypatch, rag_engine):
    """
    Simulate correct JSON output from relevance grader chain and ensure
    _grade_documents_node sets 'relevance_check_passed' in the returned state.
    """
    correct_response = RelevanceGrade(is_relevant=True, justification="Matches user question")
    fake_chain = Mock()

    fake_chain.__call__.return_value = correct_response

    monkeypatch.setattr(
        rag_engine,
        "document_relevance_grader_chain",
        fake_chain
    )

    state = {
        "question": "What is AI?",
        "documents": [Document(page_content="AI is the simulation of human intelligence.", metadata={})],
        "collection_name": "test"
    }
    result = rag_engine._grade_documents_node(state)

    assert "relevance_check_passed" in result
    assert result["relevance_check_passed"] is True
    
def test_document_relevance_grader_chain_bad_json(rag_engine, mocker):
    """
    Simulate malformed output (non-JSON) from relevance grader chain and
    verify _grade_documents_node handles it without crashing.
    """
    mocker.patch.object(
        rag_engine.document_relevance_grader_chain,
        "__call__",
        side_effect=json.JSONDecodeError("Expecting value", "NOT JSON", 0)
    )

    state = {
        "question": "AI?",
        "documents": [Document(page_content="No relevance here.", metadata={})],
        "collection_name": "test"
    }
    result = rag_engine._grade_documents_node(state)

    # The node should handle the error and mark relevance as False.
    assert result["relevance_check_passed"] is False


def test_query_rewriter_chain_parsing(rag_engine, mocker):
    """Simulates LLM output for query rewriting."""
    mock_chain = Mock()
    mock_chain.__call__.return_value = {"text": "Rewritten: What is LangGraph?"}
    mocker.patch.object(rag_engine, 'query_rewriter_chain', mock_chain)

    state = {"question": "What about LG?", "original_question": "What about LG?"}
    result = rag_engine._rewrite_query_node(state)
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