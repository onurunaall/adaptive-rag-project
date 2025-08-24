import pytest
from src.scraper import scrape_urls_as_documents
from langchain.schema import Document

def test_scrape_single_url(mocker):
    # Mock WebBaseLoader to avoid network calls
    mock_doc = Document(
        page_content="This is sample content about AI agents from the web.",
        metadata={"source": "https://example.com/ai-agents"}
    )
    mock_loader = mocker.Mock()
    mock_loader.load.return_value = [mock_doc]
    mocker.patch('src.scraper.WebBaseLoader', return_value=mock_loader)
    
    test_urls = ["https://example.com/ai-agents"]
    docs = scrape_urls_as_documents(test_urls, user_goal_for_scraping="test_goal")
    
    assert isinstance(docs, list)
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.page_content == "This is sample content about AI agents from the web."
    assert "source" in doc.metadata
    assert doc.metadata["source"] == "https://example.com/ai-agents"
    assert doc.metadata.get("user_goal_for_scraping") == "test_goal"
    
    # Verify WebBaseLoader was called correctly
    mock_loader.load.assert_called_once()

def test_scrape_empty_url_list():
    assert scrape_urls_as_documents([]) == []

def test_scrape_with_invalid_url(mocker):
    mock_loader = mocker.Mock()
    mock_loader.load.return_value = []
    mocker.patch('src.scraper.WebBaseLoader', return_value=mock_loader)
    docs = scrape_urls_as_documents(["http://nonexistenturl123abc.com"], "test_invalid")
    assert docs == []

def test_scrape_urls_as_documents_handles_loader_exception(mocker):
    # Simulate loader failure
    mock_loader = mocker.Mock()
    mock_loader.load.side_effect = Exception("Loader failed")
    mocker.patch('src.scraper.WebBaseLoader', return_value=mock_loader)
    docs = scrape_urls_as_documents(["http://fail.com"], user_goal_for_scraping="test")
    assert docs == []
