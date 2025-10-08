from src.stock import fetch_stock_news_documents
from langchain_core.documents import Document


def test_fetch_stock_news_single_ticker(mocker):
    # Mock YahooFinanceNewsTool to avoid live API calls
    mock_article_data = [
        {
            "title": "Apple Reports Strong Q4 Results",
            "link": "https://finance.yahoo.com/news/apple-results",
            "published": "2024-01-15",
            "summary": "Apple Inc. reported better than expected quarterly results.",
        }
    ]
    mock_tool = mocker.Mock()
    mock_tool.run.return_value = mock_article_data
    mocker.patch("src.stock.YahooFinanceNewsTool", return_value=mock_tool)

    tickers = "AAPL"
    max_articles = 1
    docs = fetch_stock_news_documents(tickers, max_articles)

    assert isinstance(docs, list)
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert "source" in doc.metadata
    assert "title" in doc.metadata
    assert "tickers" in doc.metadata
    assert tickers in doc.metadata["tickers"]
    assert doc.page_content == "Apple Inc. reported better than expected quarterly results."
    mock_tool.run.assert_called_once_with("AAPL")


def test_fetch_stock_news_multiple_tickers(mocker):
    # Mock multiple articles for multiple tickers
    mock_articles = [
        {
            "title": "Microsoft Cloud Growth",
            "link": "https://finance.yahoo.com/news/msft-cloud",
            "published": "2024-01-15",
            "summary": "Microsoft sees strong cloud growth.",
        },
        {
            "title": "Google AI Advances",
            "link": "https://finance.yahoo.com/news/googl-ai",
            "published": "2024-01-15",
            "summary": "Google announces new AI capabilities.",
        },
    ]
    mock_tool = mocker.Mock()
    mock_tool.run.return_value = mock_articles
    mocker.patch("src.stock.YahooFinanceNewsTool", return_value=mock_tool)

    tickers = ["MSFT", "GOOGL"]
    max_articles = 1
    docs = fetch_stock_news_documents(tickers, max_articles)

    assert isinstance(docs, list)
    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, Document)
        assert "tickers" in doc.metadata
        assert "MSFT,GOOGL" in doc.metadata["tickers"]
    mock_tool.run.assert_called_once_with("MSFT,GOOGL")


def test_fetch_stock_news_empty_input():
    assert fetch_stock_news_documents("", 0) == []
    assert fetch_stock_news_documents([], 0) == []


def test_fetch_stock_news_mocked(mocker):
    mock_article = {
        "title": "Test",
        "link": "http://test.com",
        "published": "Now",
        "summary": "Test summary.",
    }
    mock_tool = mocker.Mock()
    mock_tool.run.return_value = [mock_article]
    mocker.patch("src.stock.YahooFinanceNewsTool", return_value=mock_tool)
    docs = fetch_stock_news_documents("FAKE", 1)
    assert len(docs) == 1
    assert docs[0].page_content == "Test summary."
    mock_tool.run.assert_called_once_with("FAKE")


def test_fetch_stock_news_documents_handles_api_exception(mocker):
    # Simulate the YahooFinanceNewsTool raising an error
    mock_tool = mocker.Mock()
    mock_tool.run.side_effect = Exception("API down")
    mocker.patch("src.stock.YahooFinanceNewsTool", return_value=mock_tool)

    docs = fetch_stock_news_documents("AAPL", 1)
    # Function should return empty list on API failure, not raise exception
    assert docs == []
    mock_tool.run.assert_called_once_with("AAPL")


def test_fetch_stock_news_documents_handles_malformed_article(mocker):
    # Simulate malformed tool response
    mock_tool = mocker.Mock()
    mock_tool.run.return_value = [{"wrong_key": "No summary"}]
    mocker.patch("src.stock.YahooFinanceNewsTool", return_value=mock_tool)
    docs = fetch_stock_news_documents("AAPL", 1)
    assert isinstance(docs, list)  # Shouldn't crash on bad dict
