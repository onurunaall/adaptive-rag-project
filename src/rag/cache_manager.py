"""
Cache Manager for RAG document caching.

Handles in-memory document caching with size limits, TTL, and LRU eviction.
"""
import sys
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from langchain_core.documents import Document


class CacheManager:
    """
    Manages document caching with size limits and TTL.

    Features:
    - In-memory document cache with TTL (time-to-live)
    - Size-based eviction (LRU) when cache exceeds max_size_mb
    - Per-collection caching
    - Cache statistics and monitoring
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        max_cache_size_mb: float = 500.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cache manager.

        Args:
            cache_ttl: Time-to-live for cache entries in seconds (default: 300 = 5 min)
            max_cache_size_mb: Maximum cache size in megabytes (default: 500 MB)
            logger: Optional logger instance
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size_mb = max_cache_size_mb
        self.logger = logger or logging.getLogger(__name__)

        # Cache structure: {collection_name: {"timestamp": datetime, "documents": List[Document]}}
        self.document_cache: Dict[str, Dict] = {}

    def get(self, collection_name: str) -> Optional[List[Document]]:
        """
        Get documents from cache if available and not expired.

        Args:
            collection_name: Name of the collection

        Returns:
            List of documents if cache hit and not expired, None otherwise
        """
        if collection_name not in self.document_cache:
            self.logger.debug(f"Cache miss for collection '{collection_name}' (not in cache)")
            return None

        cache_entry = self.document_cache[collection_name]
        cached_time = cache_entry["timestamp"]
        age_seconds = (datetime.now() - cached_time).total_seconds()

        if age_seconds > self.cache_ttl:
            self.logger.debug(f"Cache expired for collection '{collection_name}' (age: {age_seconds:.1f}s > TTL: {self.cache_ttl}s)")
            # Remove expired entry
            del self.document_cache[collection_name]
            return None

        self.logger.debug(f"Cache hit for collection '{collection_name}' (age: {age_seconds:.1f}s, {len(cache_entry['documents'])} docs)")
        return cache_entry["documents"]

    def set(self, collection_name: str, documents: List[Document]) -> None:
        """
        Store documents in cache.

        Args:
            collection_name: Name of the collection
            documents: List of documents to cache
        """
        self.document_cache[collection_name] = {
            "timestamp": datetime.now(),
            "documents": documents
        }
        self.logger.debug(f"Cached {len(documents)} documents for collection '{collection_name}'")

        # Check and maintain cache size
        self._maintain_cache_size()

    def invalidate(self, collection_name: str) -> bool:
        """
        Remove a collection from cache.

        Args:
            collection_name: Name of the collection to invalidate

        Returns:
            True if collection was in cache and removed, False otherwise
        """
        if collection_name in self.document_cache:
            del self.document_cache[collection_name]
            self.logger.info(f"Invalidated cache for collection '{collection_name}'")
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = len(self.document_cache)
        self.document_cache.clear()
        self.logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats including size, entry count, and age info
        """
        if not self.document_cache:
            return {
                "entry_count": 0,
                "total_documents": 0,
                "estimated_size_mb": 0.0,
                "oldest_entry_age_seconds": None,
                "newest_entry_age_seconds": None
            }

        total_docs = sum(len(entry["documents"]) for entry in self.document_cache.values())
        current_size_mb = self._estimate_cache_size_mb()

        # Calculate age stats
        now = datetime.now()
        ages = [(now - entry["timestamp"]).total_seconds() for entry in self.document_cache.values()]

        return {
            "entry_count": len(self.document_cache),
            "total_documents": total_docs,
            "estimated_size_mb": round(current_size_mb, 2),
            "max_size_mb": self.max_cache_size_mb,
            "cache_ttl_seconds": self.cache_ttl,
            "oldest_entry_age_seconds": round(max(ages), 1) if ages else None,
            "newest_entry_age_seconds": round(min(ages), 1) if ages else None,
            "collections": list(self.document_cache.keys())
        }

    def _estimate_cache_size_mb(self) -> float:
        """
        Estimate total cache size in megabytes.

        Returns:
            Estimated size in MB
        """
        total_size = 0
        for entry in self.document_cache.values():
            for doc in entry["documents"]:
                # Estimate: page_content + metadata
                total_size += sys.getsizeof(doc.page_content)
                total_size += sys.getsizeof(doc.metadata)

        return total_size / (1024 * 1024)  # Convert bytes to MB

    def _maintain_cache_size(self) -> None:
        """
        Maintain cache size by evicting oldest entries if size exceeds limit.

        Uses LRU (Least Recently Used) eviction strategy based on cache timestamp.
        """
        current_size_mb = self._estimate_cache_size_mb()

        if current_size_mb <= self.max_cache_size_mb * 0.9:  # 90% threshold
            return

        self.logger.warning(
            f"Cache size ({current_size_mb:.2f} MB) approaching limit ({self.max_cache_size_mb} MB). "
            f"Evicting oldest entries..."
        )

        # Sort collections by timestamp (oldest first)
        sorted_collections = sorted(
            self.document_cache.items(),
            key=lambda x: x[1]["timestamp"]
        )

        # Evict oldest entries until we're under 80% of max size
        target_size_mb = self.max_cache_size_mb * 0.8
        evicted_count = 0

        for collection_name, _ in sorted_collections:
            if current_size_mb <= target_size_mb:
                break

            # Calculate size before deletion
            entry = self.document_cache[collection_name]
            entry_size_mb = sum(
                sys.getsizeof(doc.page_content) + sys.getsizeof(doc.metadata)
                for doc in entry["documents"]
            ) / (1024 * 1024)

            # Delete entry
            del self.document_cache[collection_name]
            current_size_mb -= entry_size_mb
            evicted_count += 1

            self.logger.debug(
                f"Evicted collection '{collection_name}' ({entry_size_mb:.2f} MB) from cache"
            )

        if evicted_count > 0:
            self.logger.info(
                f"Cache maintenance complete: Evicted {evicted_count} collections, "
                f"new size: {current_size_mb:.2f} MB"
            )

    def cleanup_expired(self) -> int:
        """
        Remove all expired cache entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_collections = [
            collection_name
            for collection_name, entry in self.document_cache.items()
            if (now - entry["timestamp"]).total_seconds() > self.cache_ttl
        ]

        for collection_name in expired_collections:
            del self.document_cache[collection_name]

        if expired_collections:
            self.logger.info(f"Cleaned up {len(expired_collections)} expired cache entries")

        return len(expired_collections)
