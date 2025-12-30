"""
Unit tests for DocumentManager.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag.document_manager import DocumentManager
from src.chunking import AdaptiveChunker, HybridChunker, BaseChunker


@pytest.fixture
def document_manager():
    """Create a DocumentManager instance for testing."""
    return DocumentManager(
        chunk_size=1000,
        chunk_overlap=200,
        openai_api_key="test-key",
        llm_provider="openai",
        llm_model_name="gpt-3.5-turbo",
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="This is the first test document.", metadata={"source": "test1"}),
        Document(page_content="This is the second test document.", metadata={"source": "test2"}),
    ]


class TestDocumentManagerInit:
    """Tests for DocumentManager initialization."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        manager = DocumentManager()
        assert manager.chunk_size == 1000
        assert manager.chunk_overlap == 200
        assert manager.llm_provider == "openai"
        assert manager.text_splitter is not None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        manager = DocumentManager(
            chunk_size=500,
            chunk_overlap=100,
            openai_api_key="custom-key",
            llm_provider="google",
            llm_model_name="custom-model",
        )
        assert manager.chunk_size == 500
        assert manager.chunk_overlap == 100
        assert manager.openai_api_key == "custom-key"
        assert manager.llm_provider == "google"
        assert manager.llm_model_name == "custom-model"


class TestLoadDocuments:
    """Tests for load_documents method."""

    def test_load_from_url(self, document_manager):
        """Test loading documents from URL."""
        with patch("src.rag.document_manager.WebBaseLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Document(page_content="Web content", metadata={"source": "http://example.com"})
            ]
            mock_loader.return_value = mock_instance

            docs = document_manager.load_documents("url", "http://example.com")

            assert len(docs) == 1
            assert docs[0].page_content == "Web content"
            mock_loader.assert_called_once_with(web_paths=["http://example.com"])

    def test_load_from_pdf_path(self, document_manager):
        """Test loading documents from PDF file path."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(b"PDF content")

        try:
            with patch("src.rag.document_manager.PyPDFLoader") as mock_loader:
                mock_instance = Mock()
                mock_instance.load.return_value = [Document(page_content="PDF content", metadata={"source": tmp_path})]
                mock_loader.return_value = mock_instance

                docs = document_manager.load_documents("pdf_path", tmp_path)

                assert len(docs) == 1
                mock_loader.assert_called_once_with(file_path=tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_from_text_path(self, document_manager):
        """Test loading documents from text file path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write("Text content")

        try:
            with patch("src.rag.document_manager.TextLoader") as mock_loader:
                mock_instance = Mock()
                mock_instance.load.return_value = [Document(page_content="Text content", metadata={"source": tmp_path})]
                mock_loader.return_value = mock_instance

                docs = document_manager.load_documents("text_path", tmp_path)

                assert len(docs) == 1
                mock_loader.assert_called_once_with(file_path=tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_pdf(self, document_manager):
        """Test loading from non-existent PDF path returns empty list."""
        docs = document_manager.load_documents("pdf_path", "/nonexistent/file.pdf")
        assert docs == []

    def test_load_nonexistent_text(self, document_manager):
        """Test loading from non-existent text path returns empty list."""
        docs = document_manager.load_documents("text_path", "/nonexistent/file.txt")
        assert docs == []

    def test_load_unsupported_source_type(self, document_manager):
        """Test loading with unsupported source type returns empty list."""
        docs = document_manager.load_documents("unsupported_type", "some_value")
        assert docs == []

    def test_load_documents_error_handling(self, document_manager):
        """Test error handling when document loading fails."""
        with patch("src.rag.document_manager.WebBaseLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.side_effect = Exception("Load failed")
            mock_loader.return_value = mock_instance

            docs = document_manager.load_documents("url", "http://example.com")

            assert docs == []


class TestSplitDocuments:
    """Tests for split_documents method."""

    def test_split_empty_document_list(self, document_manager):
        """Test splitting empty document list returns empty list."""
        chunks = document_manager.split_documents([])
        assert chunks == []

    def test_split_with_basechunker(self, document_manager, sample_documents):
        """Test splitting documents with BaseChunker interface."""
        mock_chunker = Mock(spec=BaseChunker)
        mock_chunker.chunk_documents.return_value = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
        ]
        document_manager.text_splitter = mock_chunker

        chunks = document_manager.split_documents(sample_documents)

        assert len(chunks) == 2
        mock_chunker.chunk_documents.assert_called_once_with(sample_documents)

    def test_split_with_legacy_splitter(self, document_manager, sample_documents):
        """Test splitting documents with legacy splitter interface."""
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
        ]
        document_manager.text_splitter = mock_splitter

        chunks = document_manager.split_documents(sample_documents)

        assert len(chunks) == 2
        mock_splitter.split_documents.assert_called_once_with(sample_documents)


class TestTextSplitterCreation:
    """Tests for text splitter creation methods."""

    def test_create_adaptive_splitter(self, document_manager):
        """Test creating adaptive splitter."""
        splitter = document_manager._create_adaptive_splitter()
        assert isinstance(splitter, AdaptiveChunker)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200

    def test_create_hybrid_splitter(self, document_manager):
        """Test creating hybrid splitter."""
        splitter = document_manager._create_hybrid_splitter()
        assert isinstance(splitter, HybridChunker)

    def test_create_default_splitter(self, document_manager):
        """Test creating default recursive character text splitter."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = document_manager._create_default_splitter()
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_default_splitter_with_tiktoken(self, document_manager):
        """Test creating default splitter with tiktoken support."""
        with patch("src.rag.document_manager.tiktoken", None):
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = document_manager._create_default_splitter()
            assert isinstance(splitter, RecursiveCharacterTextSplitter)

    @patch("src.rag.document_manager.app_settings")
    def test_init_text_splitter_adaptive_strategy(self, mock_settings, document_manager):
        """Test initializing text splitter with adaptive strategy."""
        mock_settings.engine.chunking_strategy = "adaptive"
        splitter = document_manager._init_text_splitter()
        assert isinstance(splitter, AdaptiveChunker)

    @patch("src.rag.document_manager.app_settings")
    def test_init_text_splitter_hybrid_strategy(self, mock_settings, document_manager):
        """Test initializing text splitter with hybrid strategy."""
        mock_settings.engine.chunking_strategy = "hybrid"
        splitter = document_manager._init_text_splitter()
        assert isinstance(splitter, HybridChunker)

    @patch("src.rag.document_manager.app_settings")
    def test_init_text_splitter_unknown_strategy(self, mock_settings, document_manager):
        """Test initializing text splitter with unknown strategy falls back to default."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        mock_settings.engine.chunking_strategy = "unknown_strategy"
        splitter = document_manager._init_text_splitter()
        assert isinstance(splitter, (RecursiveCharacterTextSplitter, AdaptiveChunker))


class TestSemanticSplitter:
    """Tests for semantic splitter creation."""

    def test_create_semantic_splitter_success(self, document_manager):
        """Test creating semantic splitter successfully."""
        with patch("src.rag.document_manager.app_settings") as mock_settings:
            mock_settings.engine.semantic_chunking_threshold = "percentile"

            with patch("src.rag.document_manager.OpenAIEmbeddings") as mock_embeddings:
                with patch("langchain_experimental.text_splitter.SemanticChunker") as mock_semantic:
                    mock_semantic_instance = Mock()
                    mock_semantic.return_value = mock_semantic_instance

                    splitter = document_manager._create_semantic_splitter()

                    assert splitter == mock_semantic_instance

    def test_create_semantic_splitter_import_error_fallback(self, document_manager):
        """Test semantic splitter falls back to adaptive on import error."""
        with patch("src.rag.document_manager.OpenAIEmbeddings"):
            with patch(
                "langchain_experimental.text_splitter.SemanticChunker", side_effect=ImportError("Module not found")
            ):
                splitter = document_manager._create_semantic_splitter()
                assert isinstance(splitter, AdaptiveChunker)

    def test_create_semantic_splitter_exception_fallback(self, document_manager):
        """Test semantic splitter falls back to adaptive on exception."""
        with patch("src.rag.document_manager.OpenAIEmbeddings", side_effect=Exception("API error")):
            splitter = document_manager._create_semantic_splitter()
            assert isinstance(splitter, AdaptiveChunker)


class TestIntegration:
    """Integration tests for DocumentManager."""

    def test_full_document_loading_and_splitting_workflow(self):
        """Test complete workflow: load and split documents."""
        manager = DocumentManager(chunk_size=50, chunk_overlap=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write("This is a long document that should be split into multiple chunks. " * 5)

        try:
            docs = manager.load_documents("text_path", tmp_path)
            assert len(docs) > 0

            chunks = manager.split_documents(docs)
            assert len(chunks) >= 1
        finally:
            os.unlink(tmp_path)
