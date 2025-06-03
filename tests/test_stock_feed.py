import pytest
from src.stock import fetch_stock_news_documents
from langchain.schema import Document

def test_fetch_stock_news_single_ticker():
    tickers = "AAPL"
    max_articles = 1
    docs = fetch_stock_news_documents(tickers, max_articles)
    assert isinstance(docs, list)
    if docs:
        assert len(docs) <= max_articles
        for doc in docs:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata
            assert "title" in doc.metadata
            assert "tickers" in doc.metadata
            assert tickers in doc.metadata["tickers"]

def test_fetch_stock_news_multiple_tickers():
    tickers = ["MSFT", "GOOGL"]
    max_articles = 1
    docs = fetch_stock_news_documents(tickers, max_articles)
    assert isinstance(docs, list)
    if docs:
        for doc in docs:
            assert isinstance(doc, Document)
            assert "tickers" in doc.metadata

def test_fetch_stock_news_empty_input():
    assert fetch_stock_news_documents("", 0) == []
    assert fetch_stock_news_documents([], 0) == []

def test_fetch_stock_news_mocked(mocker):
    mock_article = {"title": "Test", "link": "http://test.com", "published": "Now", "summary": "Test summary."}
    mock_tool = mocker.Mock()
    mock_tool.run.return_value = [mock_article]
    mocker.patch('src.stock.YahooFinanceNewsTool', return_value=mock_tool)
    docs = fetch_stock_news_documents("FAKE", 1)
    assert len(docs) == 1
    assert docs[0].page_content == "Test summary."
    mock_tool.run.assert_called_once_with("FAKE")

def test_fetch_stock_news_documents_handles_api_exception(mocker):
    # Simulate the YahooFinanceNewsTool raising an error
    mock_tool = mocker.Mock()
    mock_tool.run.side_effect = Exception("API down")
    mocker.patch('src.stock.YahooFinanceNewsTool', return_value=mock_tool)
    try:
        docs = fetch_stock_news_documents("AAPL", 1)
        assert docs == [] or docs is None  # Your impl: should handle error gracefully, not crash
    except Exception:
        pytest.fail("Should not raise, should handle error internally.")

def test_fetch_stock_news_documents_handles_malformed_article(mocker):
    # Simulate malformed tool response
    mock_tool = mocker.Mock()
    mock_tool.run.return_value = [{"wrong_key": "No summary"}]
    mocker.patch('src.stock.YahooFinanceNewsTool', return_value=mock_tool)
    docs = fetch_stock_news_documents("AAPL", 1)
    assert isinstance(docs, list)  # Shouldn't crash on bad dict
