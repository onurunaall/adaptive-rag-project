mport pytest
import os
import shutil
import json
from unittest.mock import Mock
from src.core_rag_engine import CoreRAGEngine, CoreGraphState
from langchain.schema import Document

@pytest.fixture(scope="module")
def rag_engine():
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
    shutil.rmtree(test_dir)

def test_ingest_direct_documents(rag_engine):
    cname = "test_ingest_direct"
    docs = [
        Document(page_content="All about AI agents.", metadata={"source": "doc1"}),
        Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.", metadata={"source": "doc2"})
    ]
    rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)
    path = os.path.join(rag_engine.persist_directory_base, cname, "chroma.sqlite3")
    assert os.path.exists(path)
    res = rag_engine._retrieve_node({"question": "What is LangGraph?", "collection_name": cname})
    assert len(res.get("documents", [])) > 0
    assert any("LangGraph" in doc.page_content for doc in res["documents"])

@pytest.fixture(scope="module")
def populated_rag_engine(rag_engine):
    cname = "rag_test_data"
    docs = [
        Document(page_content="Paris is the capital of France. It is known for the Eiffel Tower.", metadata={"source": "france_doc"}),
        Document(page_content="The OpenAI GPT-4 model is a large language model.", metadata={"source": "openai_doc"})
    ]
    rag_engine.ingest(direct_documents=docs, collection_name=cname, recreate_collection=True)
    return rag_engine, cname

def test_rag_direct_answer(populated_rag_engine):
    engine, collection = populated_rag_engine
    res = engine.run_full_rag_workflow("What is the capital of France?", collection_name=collection)
    assert "Paris" in res["answer"]
    assert any("france_doc" in src["source"] for src in res["sources"])

def test_rag_web_search_fallback(populated_rag_engine, mocker):
    engine, collection = populated_rag_engine
    if not engine.tavily_api_key:
        pytest.skip("Tavily API key not set.")
    def mock_grade(_): return {"documents": [], "context": "", "relevance_check_passed": False}
    mocker.patch.object(engine, '_grade_documents_node', side_effect=mock_grade)
    spy_tavily = mocker.spy(engine.search_tool, 'invoke')
    question = "What is the latest news on AlphaFold 3?"
    res = engine.run_full_rag_workflow(question, collection_name=collection)
    spy_tavily.assert_called_once()
    assert res["answer"]
    assert any("http" in src.get("source", "") for src in res["sources"])

def test_rag_grounding_check_and_self_correction(populated_rag_engine, mocker):
    engine, collection = populated_rag_engine
    q = "Does the Eiffel Tower grant wishes according to the document about France?"
    spy_ground = mocker.spy(engine, '_grounding_check_node')
    spy_generate = mocker.spy(engine, '_generate_answer_node')
    engine.max_grounding_attempts = 2
    res = engine.run_full_rag_workflow(q, collection_name=collection)
    assert spy_ground.call_count > 0
    answer = res["answer"].lower()
    assert "does not say" in answer or "no" in answer or "⚠️" in answer
    engine.max_grounding_attempts = 1

def test_document_relevance_grader_chain_parsing(monkeypatch, rag_engine):
    # Simulate LLM chain output: correct JSON
    correct_response = '{"relevant": true, "reason": "Matches user question"}'
    fake_chain = Mock()
    fake_chain.run.return_value = correct_response
    monkeypatch.setattr(rag_engine, "_create_document_relevance_grader_chain", lambda: fake_chain)

    result = rag_engine._grade_documents_node({
        "question": "What is AI?",
        "documents": [Document(page_content="AI is the simulation of human intelligence.", metadata={})],
        "collection_name": "test"
    })
    assert "relevance_check_passed" in result

def test_document_relevance_grader_chain_bad_json(monkeypatch, rag_engine):
    # Simulate LLM chain output: bad JSON
    fake_chain = Mock()
    fake_chain.run.return_value = 'NOT JSON'
    monkeypatch.setattr(rag_engine, "_create_document_relevance_grader_chain", lambda: fake_chain)
    # Should not crash, should mark relevance as False or similar fallback
    result = rag_engine._grade_documents_node({
        "question": "AI?",
        "documents": [Document(page_content="No relevance here.", metadata={})],
        "collection_name": "test"
    })
    assert "relevance_check_passed" in result

def test_query_rewriter_chain_parsing(monkeypatch, rag_engine):
    # Simulate LLM chain output for query rewriting
    fake_chain = Mock()
    fake_chain.run.return_value = "Rewritten: What is LangGraph?"
    monkeypatch.setattr(rag_engine, "_create_query_rewriter_chain", lambda: fake_chain)
    state = {"question": "What about LG?", "history": []}
    result = rag_engine._rewrite_query_node(state)
    assert "rewritten_question" in result

def test_ingest_handles_oserror(monkeypatch, rag_engine):
    # Simulate disk failure on ingest (e.g., permission denied)
    monkeypatch.setattr("os.makedirs", Mock(side_effect=PermissionError("No permission")))
    with pytest.raises(PermissionError):
        rag_engine.ingest(direct_documents=[Document(page_content="fail", metadata={})], collection_name="err_test", recreate_collection=True)

def test_run_full_rag_workflow_handles_llm_error(monkeypatch, rag_engine):
    # Simulate LLM failure in answer generation
    monkeypatch.setattr(rag_engine, "_generate_answer_node", Mock(side_effect=Exception("LLM failure")))
    with pytest.raises(Exception):
        rag_engine.run_full_rag_workflow("Some Q?", collection_name="failcase")
