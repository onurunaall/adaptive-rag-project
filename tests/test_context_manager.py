from langchain_core.documents import Document
from src.context_manager import ContextManager


def test_context_manager_initialization():
    """Test basic initialization."""
    cm = ContextManager(model_name="gpt-4", max_context_tokens=4000)
    assert cm.max_context_tokens == 4000


def test_token_counting():
    """Test token counting with and without tiktoken."""
    cm = ContextManager(model_name="gpt-4")

    text = "This is a test sentence."
    tokens = cm.count_tokens(text)

    # Should be roughly 5-7 tokens
    assert 3 < tokens < 10


def test_truncate_smart_strategy():
    """Test smart truncation keeps top documents."""
    cm = ContextManager(model_name="gpt-4", max_context_tokens=100)

    docs = [
        Document(page_content="A" * 200),  # Too large
        Document(page_content="B" * 200),  # Too large
        Document(page_content="C" * 200),  # Too large
    ]

    truncated, was_truncated = cm.truncate_documents(docs, question="", strategy="smart")

    assert was_truncated
    assert len(truncated) >= 1  # Should keep at least first doc (truncated)
    assert "A" in truncated[0].page_content


def test_truncate_balanced_strategy():
    """Test balanced truncation gives equal space."""
    cm = ContextManager(model_name="gpt-4", max_context_tokens=120)

    docs = [
        Document(page_content="A" * 200),
        Document(page_content="B" * 200),
        Document(page_content="C" * 200),
    ]

    truncated, was_truncated = cm.truncate_documents(docs, question="", strategy="balanced")

    assert was_truncated
    assert len(truncated) == 3  # Should keep all documents but truncated

    # Each should be roughly equal length (accounting for truncation marker)
    truncation_marker = "\n[...truncated]"
    lengths = [len(doc.page_content.replace(truncation_marker, "")) for doc in truncated]
    # Lengths should be similar (within 20% variance)
    avg_length = sum(lengths) / len(lengths)
    for length in lengths:
        assert abs(length - avg_length) / avg_length < 0.2


def test_no_truncation_needed():
    """Test documents that fit don't get truncated."""
    cm = ContextManager(model_name="gpt-4", max_context_tokens=10000)

    docs = [
        Document(page_content="Short doc 1"),
        Document(page_content="Short doc 2"),
    ]

    truncated, was_truncated = cm.truncate_documents(docs, question="test", strategy="smart")

    assert not was_truncated
    assert len(truncated) == 2
    assert truncated[0].page_content == "Short doc 1"


def test_context_summary():
    """Test context usage summary."""
    cm = ContextManager(model_name="gpt-4", max_context_tokens=1000)

    docs = [
        Document(page_content="A" * 100),
        Document(page_content="B" * 100),
    ]

    summary = cm.get_context_summary(docs)

    assert summary["document_count"] == 2
    assert summary["total_tokens"] > 0
    assert 0 <= summary["token_utilization"] <= 1
