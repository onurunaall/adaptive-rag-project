import pytest
import os
import shutil
import json

from unittest.mock import Mock, patch
from langgraph.graph import END
from langchain.schema import Document

from src.core_rag_engine import CoreRAGEngine, GroundingCheck, RerankScore

@pytest.fixture(scope="module")
def populated_rag_engine(rag_engine): # Add 'mocker' if not present, though patch is used here
    """
    Fixture that ingests two documents into a collection named 'rag_test_data'.
    """
    cname = "rag_test_data"
    docs = [
        Document(page_content="Paris is the capital of France. It is known for the Eiffel Tower.", metadata={"source": "france_doc"}),
        Document(page_content="The OpenAI GPT-4 model is a large language model.", metadata={"source": "openai_doc"})
    ]
    # Use patch to mock the embedding call during ingestion
    with patch('langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents') as mock_embed:
        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536] # Return fake vectors
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
    """
    Test that ingesting a list of Document instances indexes them into Chroma,
    and that retrieving documents returns expected content.
    """
    cname = "test_ingest_direct"
    docs = [
        Document(page_content="All about AI agents.", metadata={"source": "doc1"}),
        Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.", metadata={"source": "doc2"})
    ]

    # Ingest directly provided documents, recreate collection
    rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)

    # Verify the Chroma SQLite file exists in the persistence directory
    path = os.path.join(rag_engine.persist_directory_base, cname, "chroma.sqlite3")
    assert os.path.exists(path)

    # Retrieve documents for a query and verify expected content
    state = {"question": "What is LangGraph?", "collection_name": cname}
    res = rag_engine._retrieve_node(state)
    docs_retrieved = res.get("documents", [])

    with patch('langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents') as mock_embed:
        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
        rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)

    assert len(docs_retrieved) > 0
    assert any("LangGraph" in doc.page_content for doc in docs_retrieved)

def test_rag_direct_answer(populated_rag_engine):
    """
    Test that running the full RAG workflow on a simple factual question
    returns an answer containing 'Paris' and includes source metadata.
    """
    engine, collection = populated_rag_engine
    res = engine.run_full_rag_workflow("What is the capital of France?", collection_name=collection)

    answer = res["answer"]
    sources = res["sources"]

    assert "Paris" in answer
    assert any("france_doc" in src["source"] for src in sources)

def test_rag_web_search_fallback(populated_rag_engine, mocker):
    """
    Test that when document grading fails (no relevant docs), the engine
    falls back to web search. Requires a valid Tavily API key.
    """
    engine, collection = populated_rag_engine

    if not engine.tavily_api_key:
        pytest.skip("Tavily API key not set.")

    # Patch _grade_documents_node to simulate no relevant documents
    def mock_grade(state):
        return {
            "documents": [],
            "context": "",
            "relevance_check_passed": False
        }

    mocker.patch.object(
        engine, "_grade_documents_node",
        side_effect=mock_grade
    )

    # Spy on the search_tool.invoke method to ensure it's called
    spy_tavily = mocker.spy(engine.search_tool, "invoke")

    question = "What is the latest news on AlphaFold 3?"
    res = engine.run_full_rag_workflow(
        question,
        collection_name=collection
    )

    spy_tavily.assert_called_once()
    assert res["answer"]
    assert any("http" in src.get("source", "") for src in res["sources"])

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
    mock_chain.invoke.return_value = mock_failure_output

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
    mock_chain.invoke.return_value = mock_success_output
    mocker.patch.object(rag_engine, 'grounding_check_chain', mock_chain)
    # 2. Define initial state where generation matches context
    initial_state = {
        "context": "The sky is blue.",
        "generation": "The sky is blue.",
        "grounding_check_attempts": 0,
    }

    # 3. Execute the grounding check node
    result_state = rag_engine._grounding_check_node(initial_state)

    # 4. Verify no feedback and that attempt count incremented
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


def test_rerank_documents_node_sorts_correctly(rag_engine, mocker):
    """
    Test that _rerank_documents_node sorts documents based on descending
    relevance scores returned by a mocked reranker chain.
    """
    docs = [
        Document(page_content="Low relevance doc.", metadata={"source": "c"}),
        Document(page_content="High relevance doc.", metadata={"source": "a"}),
        Document(page_content="Medium relevance doc.", metadata={"source": "b"})
    ]

    # Mock scores: intentionally out of input order
    mock_scores = [
        RerankScore(relevance_score=0.1, justification="Low"),
        RerankScore(relevance_score=0.9, justification="High"),
        RerankScore(relevance_score=0.5, justification="Medium"),
    ]

    # Patch the reranker chain's invoke method to return mock scores in sequence
    mocker.patch.object(rag_engine.document_reranker_chain,"invoke",side_effect=mock_scores)
    initial_state = {"question": "A test question","original_question": "A test question","documents": docs}
    result_state = rag_engine._rerank_documents_node(initial_state)

    sorted_docs = result_state["documents"]

    assert len(sorted_docs) == 3
    assert sorted_docs[0].metadata["source"] == "a"  # highest score 0.9
    assert sorted_docs[1].metadata["source"] == "b"  # next score 0.5
    assert sorted_docs[2].metadata["source"] == "c"  # lowest score 0.1


def test_document_relevance_grader_chain_parsing(monkeypatch, rag_engine):
    """
    Simulate correct JSON output from relevance grader chain and ensure
    _grade_documents_node sets 'relevance_check_passed' in the returned state.
    """
    correct_response = '{"relevant": true, "reason": "Matches user question"}'
    fake_chain = Mock()
    fake_chain.run.return_value = correct_response

    monkeypatch.setattr(
        rag_engine,
        "_create_document_relevance_grader_chain",
        lambda: fake_chain
    )

    state = {
        "question": "What is AI?",
        "documents": [Document(page_content="AI is the simulation of human intelligence.", metadata={})],
        "collection_name": "test"
    }
    result = rag_engine._grade_documents_node(state)

    assert "relevance_check_passed" in result

def test_document_relevance_grader_chain_bad_json(monkeypatch, rag_engine):
    """
    Simulate malformed output (non-JSON) from relevance grader chain and
    verify _grade_documents_node handles it without crashing.
    """
    fake_chain = Mock()
    fake_chain.run.return_value = "NOT JSON"

    monkeypatch.setattr(
        rag_engine,
        "_create_document_relevance_grader_chain",
        lambda: fake_chain
    )

    state = {
        "question": "AI?",
        "documents": [Document(page_content="No relevance here.", metadata={})],
            "collection_name": "test"
    }
    result = rag_engine._grade_documents_node(state)

    assert "relevance_check_passed" in result


def test_query_rewriter_chain_parsing(monkeypatch, rag_engine):
    """
    Simulate LLM output for query rewriting and ensure
    _rewrite_query_node sets 'rewritten_question' in the state.
    """
    fake_chain = Mock()
    fake_chain.run.return_value = "Rewritten: What is LangGraph?"

    monkeypatch.setattr(
        rag_engine,
        "_create_query_rewriter_chain",
        lambda: fake_chain
    )

    state = {"question": "What about LG?", "history": []}
    result = rag_engine._rewrite_query_node(state)

    assert "rewritten_question" in result


def test_ingest_handles_oserror(monkeypatch, rag_engine):
    """
    Simulate a PermissionError when creating directories during ingest.
    Verify ingest propagates that error.
    """
    monkeypatch.setattr(
        "os.makedirs",
        Mock(side_effect=PermissionError("No permission"))
    )

    with pytest.raises(PermissionError):
        rag_engine.ingest(
            direct_documents=[Document(page_content="fail", metadata={})],
            collection_name="err_test",
            recreate_collection=True
        )

