"""
Cache Orchestrator Module

Handles cache management coordination across document cache and semantic cache.
Extracted from CoreRAGEngine to improve modularity and maintainability.
"""

import gc
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class CacheOrchestrator:
    """
    Manages cache coordination across different caching systems.

    Responsibilities:
    - Maintain document cache with size limits
    - Invalidate collection-specific caches
    - Provide cache statistics
    - Coordinate with semantic cache (CacheManager)
    - Manage cache TTL and eviction policies
    """

    def __init__(
        self,
        cache_ttl: int = 3600,
        max_cache_size_mb: float = 500.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize CacheOrchestrator.

        Args:
            cache_ttl: Time-to-live for cache entries in seconds (default: 3600)
            max_cache_size_mb: Maximum cache size in megabytes (default: 500)
            logger: Optional logger instance
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size_mb = max_cache_size_mb
        self.logger = logger or logging.getLogger(__name__)

        # Document cache stores List[Document] keyed by collection_name
        self.document_cache: Dict[str, List[Any]] = {}
        # Timestamps track when each collection was last cached
        self.cache_timestamps: Dict[str, float] = {}

    def maintain_cache(self, max_cache_size_mb: Optional[float] = None) -> None:
        """
        Maintain cache by removing old entries if memory usage is too high.

        Args:
            max_cache_size_mb: Maximum cache size in megabytes (uses instance default if None)
        """
        max_size = max_cache_size_mb or self.max_cache_size_mb

        try:
            total_size_bytes = 0
            for docs in self.document_cache.values():
                total_size_bytes += sys.getsizeof(docs)
                for doc in docs:
                    total_size_bytes += sys.getsizeof(doc.page_content)

            current_size_mb = total_size_bytes / (1024 * 1024)

            if current_size_mb > max_size:
                self.logger.warning(
                    f"Cache size ({current_size_mb:.1f} MB) exceeds limit ({max_size} MB). "
                    f"Removing oldest entries..."
                )

                sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])

                for collection_name, timestamp in sorted_items:
                    if current_size_mb <= max_size * 0.9:
                        break

                    removed_docs = self.document_cache.pop(collection_name, None)
                    self.cache_timestamps.pop(collection_name, None)

                    if removed_docs:
                        # Calculate removed size consistently with initial calculation
                        removed_size_bytes = sys.getsizeof(removed_docs)
                        for doc in removed_docs:
                            removed_size_bytes += sys.getsizeof(doc.page_content)
                        removed_size = removed_size_bytes / (1024 * 1024)
                        current_size_mb -= removed_size
                        self.logger.info(
                            f"Evicted '{collection_name}' from cache ({len(removed_docs)} docs, "
                            f"~{removed_size:.1f} MB freed)"
                        )

                collected = gc.collect()
                self.logger.debug(f"Garbage collected {collected} objects after cache eviction")

        except Exception as e:
            self.logger.warning(f"Error during cache maintenance: {e}", exc_info=True)

    def clear_document_cache(self, collection_name: Optional[str] = None) -> None:
        """
        Clear document cache to free memory.

        Args:
            collection_name: Collection name to clear cache for, or None to clear all
        """
        try:
            if collection_name:
                removed_docs = self.document_cache.pop(collection_name, None)
                removed_timestamp = self.cache_timestamps.pop(collection_name, None)

                if removed_docs is not None or removed_timestamp is not None:
                    self.logger.debug(f"Cleared cache for collection '{collection_name}'.")
                else:
                    self.logger.debug(f"No cache found for collection '{collection_name}' to clear.")
            else:
                cache_size_before = len(self.document_cache)
                self.document_cache.clear()
                self.cache_timestamps.clear()
                self.logger.info(f"Cleared entire document cache ({cache_size_before} entries).")

        except Exception as e:
            error_msg = f"Error occurred while clearing document cache: {e}"
            self.logger.error(error_msg, exc_info=True)

        finally:
            try:
                collected = gc.collect()
                if collected > 0:
                    self.logger.debug(f"Garbage collector freed {collected} objects after cache clear.")
            except Exception as gc_error:
                self.logger.warning(f"Error during garbage collection after cache clear: {gc_error}")

    def invalidate_collection_cache(self, collection_name: str) -> None:
        """
        Invalidate document cache for a specific collection.

        Should be called when collection is modified (documents added/removed/recreated).

        Args:
            collection_name: Name of collection whose cache to invalidate
        """
        try:
            cache_key = collection_name

            if cache_key in self.document_cache:
                removed_docs = self.document_cache.pop(cache_key, None)
                removed_timestamp = self.cache_timestamps.pop(cache_key, None)

                doc_count = len(removed_docs) if removed_docs else 0
                self.logger.debug(
                    f"Invalidated cache for collection '{collection_name}' "
                    f"({doc_count} documents removed from cache)"
                )
            else:
                self.logger.debug(
                    f"No cache entry found for collection '{collection_name}' "
                    f"(nothing to invalidate)"
                )

            if removed_docs and len(removed_docs) > 1000:
                try:
                    collected = gc.collect()
                    self.logger.debug(f"Garbage collected {collected} objects after cache invalidation")
                except Exception as gc_error:
                    self.logger.debug(f"GC after invalidation failed: {gc_error}")

        except Exception as e:
            error_msg = f"Error invalidating cache for '{collection_name}': {e}"
            self.logger.warning(error_msg, exc_info=True)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document cache.

        Returns:
            Dictionary with cache statistics
        """
        try:
            total_docs = sum(len(docs) for docs in self.document_cache.values())

            total_memory_bytes = 0

            for docs in self.document_cache.values():
                total_memory_bytes += sys.getsizeof(docs)
                for doc in docs:
                    total_memory_bytes += sys.getsizeof(doc)
                    total_memory_bytes += sys.getsizeof(doc.page_content)
                    total_memory_bytes += sys.getsizeof(doc.metadata)

            return {
                "cached_collections": len(self.document_cache),
                "collection_names": list(self.document_cache.keys()),
                "total_documents": total_docs,
                "estimated_memory_mb": round(total_memory_bytes / (1024 * 1024), 2),
                "cache_timestamps": {name: timestamp for name, timestamp in self.cache_timestamps.items()},
                "cache_ttl_seconds": self.cache_ttl,
                "max_cache_size_mb": self.max_cache_size_mb,
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}", exc_info=True)
            return {"error": str(e)}

    def invalidate_all_caches(self) -> None:
        """
        Invalidate all document caches across all collections.

        Useful for freeing memory, forcing fresh reads, or troubleshooting.
        """
        try:
            cache_count = len(self.document_cache)
            doc_count = sum(len(docs) for docs in self.document_cache.values())

            self.clear_document_cache(collection_name=None)

            self.logger.info(
                f"Invalidated all caches: {cache_count} collections, "
                f"{doc_count} documents removed"
            )
        except Exception as e:
            error_msg = f"Error invalidating all caches: {e}"
            self.logger.error(error_msg, exc_info=True)

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """
        Update the Time-To-Live for cached documents.

        Args:
            ttl_seconds: New TTL in seconds (e.g., 300 = 5 minutes)
        """
        if ttl_seconds < 0:
            self.logger.warning(f"Invalid TTL {ttl_seconds}, must be >= 0")
            return

        old_ttl = self.cache_ttl
        self.cache_ttl = ttl_seconds

        self.logger.info(f"Cache TTL updated: {old_ttl}s → {ttl_seconds}s")

    def cache_documents(self, collection_name: str, documents: List[Any]) -> None:
        """
        Cache documents for a collection.

        Args:
            collection_name: Name of collection
            documents: List of documents to cache
        """
        try:
            self.document_cache[collection_name] = documents
            self.cache_timestamps[collection_name] = datetime.now().timestamp()

            self.logger.debug(
                f"Cached {len(documents)} documents for collection '{collection_name}'"
            )

            self.maintain_cache()

        except Exception as e:
            self.logger.error(f"Error caching documents for '{collection_name}': {e}", exc_info=True)

    def get_cached_documents(self, collection_name: str) -> Optional[List[Any]]:
        """
        Get cached documents for a collection.

        Args:
            collection_name: Name of collection

        Returns:
            List of cached documents or None if not cached or expired
        """
        try:
            if collection_name not in self.document_cache:
                return None

            timestamp = self.cache_timestamps.get(collection_name)
            if timestamp is None:
                return None

            current_time = datetime.now().timestamp()
            age_seconds = current_time - timestamp

            if age_seconds > self.cache_ttl:
                self.logger.debug(
                    f"Cache expired for collection '{collection_name}' "
                    f"(age: {age_seconds:.1f}s, TTL: {self.cache_ttl}s)"
                )
                self.invalidate_collection_cache(collection_name)
                return None

            documents = self.document_cache.get(collection_name)
            if documents:
                self.logger.debug(
                    f"Cache hit for collection '{collection_name}' "
                    f"({len(documents)} documents)"
                )

            return documents

        except Exception as e:
            self.logger.error(f"Error getting cached documents for '{collection_name}': {e}", exc_info=True)
            return None

    def is_cached(self, collection_name: str) -> bool:
        """
        Check if a collection's documents are cached and valid.

        Args:
            collection_name: Name of collection

        Returns:
            True if cached and not expired, False otherwise
        """
        return self.get_cached_documents(collection_name) is not None

    def get_cache_age(self, collection_name: str) -> Optional[float]:
        """
        Get the age of cached data for a collection in seconds.

        Args:
            collection_name: Name of collection

        Returns:
            Age in seconds or None if not cached
        """
        try:
            timestamp = self.cache_timestamps.get(collection_name)
            if timestamp is None:
                return None

            current_time = datetime.now().timestamp()
            return current_time - timestamp

        except Exception as e:
            self.logger.error(f"Error getting cache age for '{collection_name}': {e}", exc_info=True)
            return None
