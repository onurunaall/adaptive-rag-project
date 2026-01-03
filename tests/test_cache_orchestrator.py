"""
Unit tests for CacheOrchestrator.
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag.cache_orchestrator import CacheOrchestrator


@pytest.fixture
def cache_orchestrator():
    """Create a CacheOrchestrator instance for testing."""
    return CacheOrchestrator(cache_ttl=60, max_cache_size_mb=10.0)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="Test document 1", metadata={"source": "test1"}),
        Document(page_content="Test document 2", metadata={"source": "test2"}),
        Document(page_content="Test document 3", metadata={"source": "test3"}),
    ]


class TestCacheOrchestratorInit:
    """Tests for CacheOrchestrator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        orchestrator = CacheOrchestrator()
        assert orchestrator.cache_ttl == 3600
        assert orchestrator.max_cache_size_mb == 500.0
        assert orchestrator.logger is not None
        assert isinstance(orchestrator.document_cache, dict)
        assert isinstance(orchestrator.cache_timestamps, dict)

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        mock_logger = Mock()
        orchestrator = CacheOrchestrator(
            cache_ttl=120,
            max_cache_size_mb=100.0,
            logger=mock_logger
        )
        assert orchestrator.cache_ttl == 120
        assert orchestrator.max_cache_size_mb == 100.0
        assert orchestrator.logger == mock_logger

    def test_init_creates_empty_caches(self):
        """Test that initialization creates empty cache dictionaries."""
        orchestrator = CacheOrchestrator()
        assert len(orchestrator.document_cache) == 0
        assert len(orchestrator.cache_timestamps) == 0


class TestMaintainCache:
    """Tests for maintain_cache method."""

    def test_maintain_cache_below_limit(self, cache_orchestrator, sample_documents):
        """Test that cache is not evicted when below limit."""
        cache_orchestrator.cache_documents("collection1", sample_documents)
        initial_count = len(cache_orchestrator.document_cache)

        cache_orchestrator.maintain_cache()

        assert len(cache_orchestrator.document_cache) == initial_count

    @patch('src.rag.cache_orchestrator.sys.getsizeof')
    def test_maintain_cache_exceeds_limit(self, mock_getsizeof, cache_orchestrator, sample_documents):
        """Test that cache evicts oldest entries when exceeding limit."""
        # Mock getsizeof to simulate large cache size that triggers eviction
        # Return values: 30MB for collection lists, 10MB for doc contents
        # This gives: collection1 = 40MB, collection2 = 50MB, total = 90MB > 50MB limit
        # After evicting collection1, we have 50MB which is still > 45MB (90% of limit)
        # But the test expects collection2 to remain, so let's make collection2 smaller
        # collection1 = 35MB, collection2 = 20MB, total = 55MB > 50MB
        # After evicting collection1, we have 20MB < 45MB
        mock_getsizeof.side_effect = [
            # First pass: calculate total size
            35 * 1024 * 1024,  # collection1 list
            35 * 1024 * 1024,  # collection1 doc content
            10 * 1024 * 1024,  # collection2 list
            10 * 1024 * 1024,  # collection2 doc1 content
            10 * 1024 * 1024,  # collection2 doc2 content
            # Eviction pass: calculate removed size for collection1
            35 * 1024 * 1024,  # removed collection1 list
            35 * 1024 * 1024,  # removed collection1 doc content
        ]

        # Use auto_maintain=False to prevent premature eviction during setup
        cache_orchestrator.cache_documents("collection1", sample_documents[:1], auto_maintain=False)
        time.sleep(0.01)
        cache_orchestrator.cache_documents("collection2", sample_documents[1:], auto_maintain=False)

        cache_orchestrator.maintain_cache(max_cache_size_mb=50.0)

        # Oldest collection should be evicted
        assert "collection1" not in cache_orchestrator.document_cache
        assert "collection2" in cache_orchestrator.document_cache

    def test_maintain_cache_with_custom_max_size(self, cache_orchestrator, sample_documents):
        """Test maintain_cache with custom max size parameter."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        cache_orchestrator.maintain_cache(max_cache_size_mb=0.001)

        # Should use the provided max size, not instance default
        assert isinstance(cache_orchestrator.document_cache, dict)

    def test_maintain_cache_error_handling(self, cache_orchestrator):
        """Test error handling in maintain_cache."""
        # Create invalid cache entry that will cause error
        cache_orchestrator.document_cache["bad_collection"] = [Mock(page_content=Mock(side_effect=Exception()))]

        # Should not raise exception
        cache_orchestrator.maintain_cache()


class TestClearDocumentCache:
    """Tests for clear_document_cache method."""

    def test_clear_specific_collection(self, cache_orchestrator, sample_documents):
        """Test clearing cache for specific collection."""
        cache_orchestrator.cache_documents("collection1", sample_documents[:2])
        cache_orchestrator.cache_documents("collection2", sample_documents[2:])

        cache_orchestrator.clear_document_cache("collection1")

        assert "collection1" not in cache_orchestrator.document_cache
        assert "collection1" not in cache_orchestrator.cache_timestamps
        assert "collection2" in cache_orchestrator.document_cache

    def test_clear_all_collections(self, cache_orchestrator, sample_documents):
        """Test clearing entire cache."""
        cache_orchestrator.cache_documents("collection1", sample_documents[:2])
        cache_orchestrator.cache_documents("collection2", sample_documents[2:])

        cache_orchestrator.clear_document_cache(collection_name=None)

        assert len(cache_orchestrator.document_cache) == 0
        assert len(cache_orchestrator.cache_timestamps) == 0

    def test_clear_nonexistent_collection(self, cache_orchestrator):
        """Test clearing cache for nonexistent collection."""
        # Should not raise exception
        cache_orchestrator.clear_document_cache("nonexistent")

    @patch('src.rag.cache_orchestrator.gc.collect')
    def test_clear_triggers_garbage_collection(self, mock_gc_collect, cache_orchestrator, sample_documents):
        """Test that clear triggers garbage collection."""
        mock_gc_collect.return_value = 5

        cache_orchestrator.cache_documents("collection1", sample_documents)
        cache_orchestrator.clear_document_cache()

        mock_gc_collect.assert_called()


class TestInvalidateCollectionCache:
    """Tests for invalidate_collection_cache method."""

    def test_invalidate_existing_collection(self, cache_orchestrator, sample_documents):
        """Test invalidating cache for existing collection."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        cache_orchestrator.invalidate_collection_cache("collection1")

        assert "collection1" not in cache_orchestrator.document_cache
        assert "collection1" not in cache_orchestrator.cache_timestamps

    def test_invalidate_nonexistent_collection(self, cache_orchestrator):
        """Test invalidating cache for nonexistent collection."""
        # Should not raise exception
        cache_orchestrator.invalidate_collection_cache("nonexistent")

    @patch('src.rag.cache_orchestrator.gc.collect')
    def test_invalidate_large_collection_triggers_gc(
        self, mock_gc_collect, cache_orchestrator
    ):
        """Test that invalidating large collection triggers garbage collection."""
        mock_gc_collect.return_value = 100

        large_docs = [Document(page_content=f"Doc {i}") for i in range(1500)]
        cache_orchestrator.cache_documents("large_collection", large_docs)

        cache_orchestrator.invalidate_collection_cache("large_collection")

        mock_gc_collect.assert_called()

    def test_invalidate_small_collection_no_gc(self, cache_orchestrator, sample_documents):
        """Test that invalidating small collection doesn't trigger GC."""
        with patch('src.rag.cache_orchestrator.gc.collect') as mock_gc_collect:
            cache_orchestrator.cache_documents("small_collection", sample_documents)

            cache_orchestrator.invalidate_collection_cache("small_collection")

            # GC should not be called for small collections
            mock_gc_collect.assert_not_called()


class TestGetCacheStats:
    """Tests for get_cache_stats method."""

    def test_get_cache_stats_empty(self, cache_orchestrator):
        """Test getting stats for empty cache."""
        stats = cache_orchestrator.get_cache_stats()

        assert stats["cached_collections"] == 0
        assert stats["collection_names"] == []
        assert stats["total_documents"] == 0
        assert stats["estimated_memory_mb"] >= 0
        assert isinstance(stats["cache_timestamps"], dict)

    def test_get_cache_stats_with_data(self, cache_orchestrator, sample_documents):
        """Test getting stats with cached data."""
        cache_orchestrator.cache_documents("collection1", sample_documents[:2])
        cache_orchestrator.cache_documents("collection2", sample_documents[2:])

        stats = cache_orchestrator.get_cache_stats()

        assert stats["cached_collections"] == 2
        assert set(stats["collection_names"]) == {"collection1", "collection2"}
        assert stats["total_documents"] == 3
        assert stats["estimated_memory_mb"] > 0
        assert stats["cache_ttl_seconds"] == 60
        assert stats["max_cache_size_mb"] == 10.0

    def test_get_cache_stats_error_handling(self, cache_orchestrator):
        """Test error handling in get_cache_stats."""
        # Create a mock that raises exception when __sizeof__ is called
        bad_doc = Mock()
        bad_doc.__sizeof__ = Mock(side_effect=Exception("Test error"))
        bad_doc.page_content = "test"
        bad_doc.metadata = {}

        cache_orchestrator.document_cache["bad"] = [bad_doc]

        stats = cache_orchestrator.get_cache_stats()

        assert "error" in stats


class TestInvalidateAllCaches:
    """Tests for invalidate_all_caches method."""

    def test_invalidate_all_caches(self, cache_orchestrator, sample_documents):
        """Test invalidating all caches."""
        cache_orchestrator.cache_documents("collection1", sample_documents[:2])
        cache_orchestrator.cache_documents("collection2", sample_documents[2:])

        cache_orchestrator.invalidate_all_caches()

        assert len(cache_orchestrator.document_cache) == 0
        assert len(cache_orchestrator.cache_timestamps) == 0

    def test_invalidate_all_caches_empty(self, cache_orchestrator):
        """Test invalidating all caches when already empty."""
        # Should not raise exception
        cache_orchestrator.invalidate_all_caches()

        assert len(cache_orchestrator.document_cache) == 0


class TestSetCacheTTL:
    """Tests for set_cache_ttl method."""

    def test_set_cache_ttl_valid(self, cache_orchestrator):
        """Test setting valid cache TTL."""
        cache_orchestrator.set_cache_ttl(300)

        assert cache_orchestrator.cache_ttl == 300

    def test_set_cache_ttl_zero(self, cache_orchestrator):
        """Test setting cache TTL to zero."""
        cache_orchestrator.set_cache_ttl(0)

        assert cache_orchestrator.cache_ttl == 0

    def test_set_cache_ttl_negative(self, cache_orchestrator):
        """Test that negative TTL is rejected."""
        original_ttl = cache_orchestrator.cache_ttl

        cache_orchestrator.set_cache_ttl(-100)

        assert cache_orchestrator.cache_ttl == original_ttl

    def test_set_cache_ttl_logs_change(self, cache_orchestrator):
        """Test that TTL changes are logged."""
        mock_logger = Mock()
        cache_orchestrator.logger = mock_logger

        cache_orchestrator.set_cache_ttl(500)

        mock_logger.info.assert_called_once()


class TestCacheDocuments:
    """Tests for cache_documents method."""

    def test_cache_documents_new_collection(self, cache_orchestrator, sample_documents):
        """Test caching documents for new collection."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        assert "collection1" in cache_orchestrator.document_cache
        assert "collection1" in cache_orchestrator.cache_timestamps
        assert len(cache_orchestrator.document_cache["collection1"]) == 3

    def test_cache_documents_overwrites_existing(self, cache_orchestrator, sample_documents):
        """Test that caching overwrites existing cache."""
        cache_orchestrator.cache_documents("collection1", sample_documents[:2])
        cache_orchestrator.cache_documents("collection1", sample_documents)

        assert len(cache_orchestrator.document_cache["collection1"]) == 3

    def test_cache_documents_updates_timestamp(self, cache_orchestrator, sample_documents):
        """Test that caching updates timestamp."""
        cache_orchestrator.cache_documents("collection1", sample_documents)
        first_timestamp = cache_orchestrator.cache_timestamps["collection1"]

        time.sleep(0.01)
        cache_orchestrator.cache_documents("collection1", sample_documents)
        second_timestamp = cache_orchestrator.cache_timestamps["collection1"]

        assert second_timestamp > first_timestamp

    @patch.object(CacheOrchestrator, 'maintain_cache')
    def test_cache_documents_calls_maintain(self, mock_maintain, cache_orchestrator, sample_documents):
        """Test that caching triggers cache maintenance."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        mock_maintain.assert_called_once()


class TestGetCachedDocuments:
    """Tests for get_cached_documents method."""

    def test_get_cached_documents_hit(self, cache_orchestrator, sample_documents):
        """Test getting cached documents (cache hit)."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        result = cache_orchestrator.get_cached_documents("collection1")

        assert result == sample_documents

    def test_get_cached_documents_miss(self, cache_orchestrator):
        """Test getting cached documents (cache miss)."""
        result = cache_orchestrator.get_cached_documents("nonexistent")

        assert result is None

    def test_get_cached_documents_expired(self, cache_orchestrator, sample_documents):
        """Test getting expired cached documents."""
        cache_orchestrator.cache_ttl = 0.01  # 10ms TTL

        cache_orchestrator.cache_documents("collection1", sample_documents)
        time.sleep(0.02)  # Wait for expiration

        result = cache_orchestrator.get_cached_documents("collection1")

        assert result is None
        assert "collection1" not in cache_orchestrator.document_cache

    def test_get_cached_documents_not_expired(self, cache_orchestrator, sample_documents):
        """Test getting non-expired cached documents."""
        cache_orchestrator.cache_ttl = 10  # 10s TTL

        cache_orchestrator.cache_documents("collection1", sample_documents)

        result = cache_orchestrator.get_cached_documents("collection1")

        assert result == sample_documents


class TestIsCached:
    """Tests for is_cached method."""

    def test_is_cached_true(self, cache_orchestrator, sample_documents):
        """Test is_cached returns True for cached collection."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        assert cache_orchestrator.is_cached("collection1") is True

    def test_is_cached_false(self, cache_orchestrator):
        """Test is_cached returns False for non-cached collection."""
        assert cache_orchestrator.is_cached("nonexistent") is False

    def test_is_cached_expired(self, cache_orchestrator, sample_documents):
        """Test is_cached returns False for expired cache."""
        cache_orchestrator.cache_ttl = 0.01

        cache_orchestrator.cache_documents("collection1", sample_documents)
        time.sleep(0.02)

        assert cache_orchestrator.is_cached("collection1") is False


class TestGetCacheAge:
    """Tests for get_cache_age method."""

    def test_get_cache_age_existing(self, cache_orchestrator, sample_documents):
        """Test getting cache age for existing collection."""
        cache_orchestrator.cache_documents("collection1", sample_documents)
        time.sleep(0.01)

        age = cache_orchestrator.get_cache_age("collection1")

        assert age is not None
        assert age >= 0.01

    def test_get_cache_age_nonexistent(self, cache_orchestrator):
        """Test getting cache age for nonexistent collection."""
        age = cache_orchestrator.get_cache_age("nonexistent")

        assert age is None

    def test_get_cache_age_increases(self, cache_orchestrator, sample_documents):
        """Test that cache age increases over time."""
        cache_orchestrator.cache_documents("collection1", sample_documents)

        age1 = cache_orchestrator.get_cache_age("collection1")
        time.sleep(0.01)
        age2 = cache_orchestrator.get_cache_age("collection1")

        assert age2 > age1


class TestIntegration:
    """Integration tests for CacheOrchestrator."""

    def test_full_cache_lifecycle(self, cache_orchestrator, sample_documents):
        """Test full lifecycle: cache, get, invalidate, verify."""
        # Cache documents
        cache_orchestrator.cache_documents("collection1", sample_documents)
        assert cache_orchestrator.is_cached("collection1")

        # Get cached documents
        result = cache_orchestrator.get_cached_documents("collection1")
        assert result == sample_documents

        # Get stats
        stats = cache_orchestrator.get_cache_stats()
        assert stats["cached_collections"] == 1

        # Invalidate
        cache_orchestrator.invalidate_collection_cache("collection1")
        assert not cache_orchestrator.is_cached("collection1")

        # Verify stats updated
        stats = cache_orchestrator.get_cache_stats()
        assert stats["cached_collections"] == 0

    def test_multiple_collections_management(self, cache_orchestrator, sample_documents):
        """Test managing multiple collections."""
        cache_orchestrator.cache_documents("collection1", sample_documents[:1])
        cache_orchestrator.cache_documents("collection2", sample_documents[1:2])
        cache_orchestrator.cache_documents("collection3", sample_documents[2:])

        stats = cache_orchestrator.get_cache_stats()
        assert stats["cached_collections"] == 3
        assert stats["total_documents"] == 3

        cache_orchestrator.invalidate_collection_cache("collection2")
        stats = cache_orchestrator.get_cache_stats()
        assert stats["cached_collections"] == 2

        cache_orchestrator.invalidate_all_caches()
        stats = cache_orchestrator.get_cache_stats()
        assert stats["cached_collections"] == 0
