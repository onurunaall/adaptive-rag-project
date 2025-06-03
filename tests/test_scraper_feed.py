import pytest
from src.scraper import scrape_urls_as_documents
from langchain.schema import Document

@pytest.fixture
def sample_urls():
    return ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

def test_scrape_single_url(sample_urls):
    docs = scrape_urls_as_documents(sample_urls[:1], user_goal_for_scraping="test_goal")
    assert isinstance(docs, list)
    assert len(docs) >= 1
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert "source" in doc.metadata
        assert doc.metadata["source"] == sample_urls[0]
        assert doc.metadata.get("user_goal_for_scraping") == "test_goal"

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
