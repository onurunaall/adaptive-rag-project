"""
Unit tests for VectorStoreManager.
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from langchain_core.documents import Document

from src.rag.vector_store_manager import VectorStoreManager


@pytest.fixture
def temp_persist_dir():
    """Create a temporary persistence directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    return Mock()


@pytest.fixture
def vector_store_manager(mock_embedding_model, temp_persist_dir):
    """Create a VectorStoreManager instance for testing."""
    return VectorStoreManager(
        embedding_model=mock_embedding_model,
        persist_directory_base=temp_persist_dir,
        default_retrieval_top_k=5,
        enable_hybrid_search=False,
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="Test document 1", metadata={"source": "test1"}),
        Document(page_content="Test document 2", metadata={"source": "test2"}),
    ]


class TestVectorStoreManagerInit:
    """Tests for VectorStoreManager initialization."""

    def test_init_with_required_params(self, mock_embedding_model, temp_persist_dir):
        """Test initialization with required parameters."""
        manager = VectorStoreManager(
            embedding_model=mock_embedding_model,
            persist_directory_base=temp_persist_dir,
        )
        assert manager.embedding_model == mock_embedding_model
        assert manager.persist_directory_base == temp_persist_dir
        assert manager.default_retrieval_top_k == 4
        assert manager.enable_hybrid_search is False
        assert isinstance(manager.vectorstores, dict)
        assert isinstance(manager.retrievers, dict)

    def test_init_with_all_params(self, mock_embedding_model, temp_persist_dir):
        """Test initialization with all parameters."""
        mock_callback = Mock()
        manager = VectorStoreManager(
            embedding_model=mock_embedding_model,
            persist_directory_base=temp_persist_dir,
            default_retrieval_top_k=10,
            enable_hybrid_search=True,
            get_all_documents_callback=mock_callback,
            stream_documents_callback=mock_callback,
            invalidate_cache_callback=mock_callback,
        )
        assert manager.default_retrieval_top_k == 10
        assert manager.enable_hybrid_search is True
        assert manager.get_all_documents_callback == mock_callback

    def test_init_creates_persist_directory(self, mock_embedding_model):
        """Test that initialization creates persistence directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_dir = os.path.join(temp_dir, "test_persist")
            manager = VectorStoreManager(
                embedding_model=mock_embedding_model,
                persist_directory_base=persist_dir,
            )
            assert os.path.exists(persist_dir)


class TestGetPersistDir:
    """Tests for get_persist_dir method."""

    def test_get_persist_dir_valid_collection(self, vector_store_manager):
        """Test getting persist directory for valid collection name."""
        collection_name = "test_collection"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        expected_path = os.path.join(vector_store_manager.persist_directory_base, collection_name)
        assert persist_dir == expected_path

    def test_get_persist_dir_empty_collection_raises_error(self, vector_store_manager):
        """Test that empty collection name raises ValueError."""
        with pytest.raises(ValueError, match="collection_name cannot be None or empty"):
            vector_store_manager.get_persist_dir("")

    def test_get_persist_dir_non_string_raises_error(self, vector_store_manager):
        """Test that non-string collection name raises TypeError."""
        with pytest.raises(TypeError, match="collection_name must be a string"):
            vector_store_manager.get_persist_dir(123)


class TestInitOrLoadVectorstore:
    """Tests for init_or_load_vectorstore method."""

    def test_init_new_collection_not_on_disk(self, vector_store_manager):
        """Test initializing a new collection that doesn't exist on disk."""
        collection_name = "new_collection"
        vector_store_manager.init_or_load_vectorstore(collection_name)
        assert collection_name not in vector_store_manager.vectorstores

    @patch("src.rag.vector_store_manager.Chroma")
    def test_load_existing_collection_from_disk(self, mock_chroma, vector_store_manager, temp_persist_dir):
        """Test loading an existing collection from disk."""
        collection_name = "existing_collection"
        persist_dir = os.path.join(temp_persist_dir, collection_name)
        os.makedirs(persist_dir, exist_ok=True)

        mock_vs_instance = Mock()
        mock_vs_instance.as_retriever.return_value = Mock()
        mock_chroma.return_value = mock_vs_instance

        vector_store_manager.init_or_load_vectorstore(collection_name)

        mock_chroma.assert_called_once()
        assert collection_name in vector_store_manager.vectorstores

    def test_init_with_recreate_true(self, vector_store_manager):
        """Test initializing with recreate=True removes existing data."""
        collection_name = "recreate_collection"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        os.makedirs(persist_dir, exist_ok=True)

        test_file = os.path.join(persist_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test data")

        assert os.path.exists(test_file)

        vector_store_manager.init_or_load_vectorstore(collection_name, recreate=True)

        assert not os.path.exists(persist_dir)

    def test_init_existing_in_memory_collection(self, vector_store_manager):
        """Test initializing a collection that's already in memory."""
        collection_name = "memory_collection"
        mock_vs = Mock()
        mock_vs.as_retriever.return_value = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs

        vector_store_manager.init_or_load_vectorstore(collection_name)

        assert collection_name in vector_store_manager.vectorstores


class TestSetupRetrieverForCollection:
    """Tests for setup_retriever_for_collection method."""

    def test_setup_standard_retriever(self, vector_store_manager):
        """Test setting up standard retriever."""
        collection_name = "test_collection"
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever.return_value = mock_retriever
        vector_store_manager.vectorstores[collection_name] = mock_vs

        vector_store_manager.setup_retriever_for_collection(collection_name)

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert vector_store_manager.retrievers[collection_name] == mock_retriever

    def test_setup_retriever_vectorstore_not_found(self, vector_store_manager):
        """Test that setup fails when vectorstore doesn't exist."""
        with pytest.raises(ValueError, match="Vectorstore for 'nonexistent' not found"):
            vector_store_manager.setup_retriever_for_collection("nonexistent")

    @patch("src.rag.vector_store_manager.AdaptiveHybridRetriever")
    def test_setup_hybrid_retriever_small_collection(self, mock_hybrid, vector_store_manager):
        """Test setting up hybrid retriever for small collection."""
        vector_store_manager.enable_hybrid_search = True
        collection_name = "small_collection"

        mock_vs = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_vs._collection = mock_collection
        vector_store_manager.vectorstores[collection_name] = mock_vs

        sample_docs = [Document(page_content="test")]
        vector_store_manager.get_all_documents_callback = Mock(return_value=sample_docs)

        mock_hybrid_instance = Mock()
        mock_hybrid.return_value = mock_hybrid_instance

        vector_store_manager.setup_retriever_for_collection(collection_name)

        mock_hybrid.assert_called_once()
        assert vector_store_manager.retrievers[collection_name] == mock_hybrid_instance

    def test_setup_hybrid_retriever_large_collection(self, vector_store_manager):
        """Test setting up hybrid retriever for large collection with streaming."""
        vector_store_manager.enable_hybrid_search = True
        collection_name = "large_collection"

        mock_vs = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 15000
        mock_vs._collection = mock_collection
        vector_store_manager.vectorstores[collection_name] = mock_vs

        def mock_stream(coll_name, batch_size):
            yield [Document(page_content=f"doc {i}") for i in range(100)]

        vector_store_manager.stream_documents_callback = mock_stream

        with patch("src.rag.vector_store_manager.AdaptiveHybridRetriever") as mock_hybrid:
            mock_hybrid_instance = Mock()
            mock_hybrid.return_value = mock_hybrid_instance

            vector_store_manager.setup_retriever_for_collection(collection_name)

            assert vector_store_manager.retrievers[collection_name] == mock_hybrid_instance

    def test_setup_hybrid_retriever_fallback_to_standard(self, vector_store_manager):
        """Test fallback to standard retriever when hybrid fails."""
        vector_store_manager.enable_hybrid_search = True
        collection_name = "fallback_collection"

        mock_vs = Mock()
        mock_collection = Mock()
        mock_collection.count.side_effect = Exception("Count failed")
        mock_vs._collection = mock_collection
        mock_vs.as_retriever.return_value = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs

        vector_store_manager.setup_retriever_for_collection(collection_name)

        assert collection_name in vector_store_manager.retrievers


class TestIndexDocuments:
    """Tests for index_documents method."""

    @patch("src.rag.vector_store_manager.Chroma")
    def test_index_documents_new_collection(self, mock_chroma, vector_store_manager, sample_documents):
        """Test indexing documents into a new collection."""
        collection_name = "new_index_collection"

        mock_vs_instance = Mock()
        mock_vs_instance.as_retriever.return_value = Mock()
        mock_chroma.from_documents.return_value = mock_vs_instance

        vector_store_manager.index_documents(sample_documents, collection_name)

        mock_chroma.from_documents.assert_called_once()
        assert collection_name in vector_store_manager.vectorstores
        assert collection_name in vector_store_manager.retrievers

    @patch("src.rag.vector_store_manager.Chroma")
    def test_index_documents_existing_collection(self, mock_chroma, vector_store_manager, sample_documents):
        """Test adding documents to existing collection."""
        collection_name = "existing_index_collection"

        mock_vs = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs

        vector_store_manager.index_documents(sample_documents, collection_name)

        mock_vs.add_documents.assert_called_once_with(sample_documents)

    @patch("src.rag.vector_store_manager.Chroma")
    def test_index_documents_with_recreate(self, mock_chroma, vector_store_manager, sample_documents):
        """Test indexing documents with recreate=True."""
        collection_name = "recreate_index_collection"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        os.makedirs(persist_dir, exist_ok=True)

        mock_vs_instance = Mock()
        mock_vs_instance.as_retriever.return_value = Mock()
        mock_chroma.from_documents.return_value = mock_vs_instance

        vector_store_manager.index_documents(sample_documents, collection_name, recreate=True)

        mock_chroma.from_documents.assert_called_once()

    def test_index_documents_calls_cache_callback(self, vector_store_manager, sample_documents):
        """Test that cache invalidation callback is called."""
        collection_name = "cache_test_collection"
        mock_cache_callback = Mock()
        vector_store_manager.invalidate_cache_callback = mock_cache_callback

        mock_vs = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs

        vector_store_manager.index_documents(sample_documents, collection_name)

        mock_cache_callback.assert_called_with(collection_name)


class TestDeleteCollection:
    """Tests for delete_collection method."""

    def test_delete_existing_collection(self, vector_store_manager):
        """Test deleting an existing collection."""
        collection_name = "delete_test_collection"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        os.makedirs(persist_dir, exist_ok=True)

        mock_vs = Mock()
        mock_retriever = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs
        vector_store_manager.retrievers[collection_name] = mock_retriever

        vector_store_manager.delete_collection(collection_name)

        assert not os.path.exists(persist_dir)
        assert collection_name not in vector_store_manager.vectorstores
        assert collection_name not in vector_store_manager.retrievers


class TestListCollections:
    """Tests for list_collections method."""

    def test_list_collections_empty(self, vector_store_manager):
        """Test listing collections when none exist."""
        collections = vector_store_manager.list_collections()
        assert collections == []

    def test_list_collections_with_multiple(self, vector_store_manager):
        """Test listing multiple collections."""
        collection_names = ["collection1", "collection2", "collection3"]

        for name in collection_names:
            persist_dir = vector_store_manager.get_persist_dir(name)
            os.makedirs(persist_dir, exist_ok=True)

        collections = vector_store_manager.list_collections()

        assert len(collections) == 3
        assert set(collections) == set(collection_names)
        assert collections == sorted(collections)

    def test_list_collections_with_files(self, vector_store_manager):
        """Test that files in persist directory are not listed as collections."""
        collection_name = "valid_collection"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        os.makedirs(persist_dir, exist_ok=True)

        test_file = os.path.join(vector_store_manager.persist_directory_base, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")

        collections = vector_store_manager.list_collections()

        assert collections == [collection_name]


class TestGetVectorstoreAndRetriever:
    """Tests for get_vectorstore and get_retriever methods."""

    def test_get_vectorstore_existing(self, vector_store_manager):
        """Test getting an existing vectorstore."""
        collection_name = "test_vs"
        mock_vs = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs

        result = vector_store_manager.get_vectorstore(collection_name)

        assert result == mock_vs

    def test_get_vectorstore_nonexistent(self, vector_store_manager):
        """Test getting a non-existent vectorstore returns None."""
        result = vector_store_manager.get_vectorstore("nonexistent")
        assert result is None

    def test_get_retriever_existing(self, vector_store_manager):
        """Test getting an existing retriever."""
        collection_name = "test_retriever"
        mock_retriever = Mock()
        vector_store_manager.retrievers[collection_name] = mock_retriever

        result = vector_store_manager.get_retriever(collection_name)

        assert result == mock_retriever

    def test_get_retriever_nonexistent(self, vector_store_manager):
        """Test getting a non-existent retriever returns None."""
        result = vector_store_manager.get_retriever("nonexistent")
        assert result is None


class TestHandleRecreateCollection:
    """Tests for _handle_recreate_collection method."""

    def test_handle_recreate_removes_persist_dir(self, vector_store_manager):
        """Test that recreate removes persistence directory."""
        collection_name = "recreate_test"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        os.makedirs(persist_dir, exist_ok=True)

        test_file = os.path.join(persist_dir, "data.txt")
        with open(test_file, "w") as f:
            f.write("test data")

        vector_store_manager._handle_recreate_collection(collection_name, persist_dir)

        assert not os.path.exists(persist_dir)

    def test_handle_recreate_removes_memory_references(self, vector_store_manager):
        """Test that recreate removes in-memory references."""
        collection_name = "recreate_memory_test"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)

        mock_vs = Mock()
        mock_retriever = Mock()
        vector_store_manager.vectorstores[collection_name] = mock_vs
        vector_store_manager.retrievers[collection_name] = mock_retriever

        vector_store_manager._handle_recreate_collection(collection_name, persist_dir)

        assert collection_name not in vector_store_manager.vectorstores
        assert collection_name not in vector_store_manager.retrievers

    def test_handle_recreate_calls_cache_callback(self, vector_store_manager):
        """Test that recreate calls cache invalidation callback."""
        collection_name = "cache_recreate_test"
        persist_dir = vector_store_manager.get_persist_dir(collection_name)
        mock_callback = Mock()
        vector_store_manager.invalidate_cache_callback = mock_callback

        vector_store_manager._handle_recreate_collection(collection_name, persist_dir)

        mock_callback.assert_called_with(collection_name)
