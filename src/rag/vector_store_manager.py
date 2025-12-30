"""
Vector Store Manager Module

Handles vector store initialization, persistence, and retrieval operations.
Extracted from CoreRAGEngine to improve modularity and maintainability.
"""

import os
import logging
import shutil
from typing import Dict, List, Any, Optional, Callable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.hybrid_search import AdaptiveHybridRetriever


class VectorStoreManager:
    """
    Manages vector store operations including initialization, persistence, and retrieval.

    Responsibilities:
    - Initialize and load vector stores from disk
    - Manage collection lifecycle (create, recreate, delete)
    - Setup retrievers (standard and hybrid search)
    - Index documents into collections
    - Handle persistence directories
    """

    def __init__(
        self,
        embedding_model: Any,
        persist_directory_base: str,
        default_retrieval_top_k: int = 4,
        enable_hybrid_search: bool = False,
        logger: Optional[logging.Logger] = None,
        get_all_documents_callback: Optional[Callable[[str, bool], List[Document]]] = None,
        stream_documents_callback: Optional[Callable[[str, int], Any]] = None,
        invalidate_cache_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize VectorStoreManager.

        Args:
            embedding_model: Embedding model instance for vector operations
            persist_directory_base: Base directory for persisting collections
            default_retrieval_top_k: Default number of documents to retrieve
            enable_hybrid_search: Whether to use hybrid search (vector + BM25/TF-IDF)
            logger: Optional logger instance
            get_all_documents_callback: Callback to get all documents from a collection
            stream_documents_callback: Callback to stream documents from a collection
            invalidate_cache_callback: Callback to invalidate collection cache
        """
        self.embedding_model = embedding_model
        self.persist_directory_base = persist_directory_base
        self.default_retrieval_top_k = default_retrieval_top_k
        self.enable_hybrid_search = enable_hybrid_search
        self.logger = logger or logging.getLogger(__name__)

        self.get_all_documents_callback = get_all_documents_callback
        self.stream_documents_callback = stream_documents_callback
        self.invalidate_cache_callback = invalidate_cache_callback

        self.vectorstores: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, Any] = {}

        os.makedirs(self.persist_directory_base, exist_ok=True)

    def get_persist_dir(self, collection_name: str) -> str:
        """
        Get the persistence directory path for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Full path to the collection's persistence directory

        Raises:
            ValueError: If collection_name is empty
            TypeError: If collection_name is not a string
        """
        if not isinstance(collection_name, str):
            error_msg = f"collection_name must be a string, got {type(collection_name).__name__}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)

        if not collection_name:
            error_msg = "collection_name cannot be None or empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            persist_dir = os.path.join(self.persist_directory_base, collection_name)
            return persist_dir
        except Exception as e:
            error_msg = f"Failed to construct persist directory path for collection '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def init_or_load_vectorstore(self, collection_name: str, recreate: bool = False) -> None:
        """
        Initialize or load a vector store for a collection.

        Args:
            collection_name: Name of the collection
            recreate: If True, remove existing data and create fresh collection
        """
        try:
            persist_dir = self.get_persist_dir(collection_name)

            if recreate:
                self._handle_recreate_collection(collection_name, persist_dir)
                return

            if collection_name in self.vectorstores:
                self._handle_existing_collection_in_memory(collection_name)
                return

            self._handle_load_from_disk(collection_name, persist_dir)

        except Exception as e:
            error_msg = f"Critical error in init_or_load_vectorstore for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _handle_existing_collection_in_memory(self, collection_name: str) -> None:
        """
        Handle collection that already exists in memory.

        Args:
            collection_name: Name of the collection
        """
        try:
            current_k = None
            retriever = self.retrievers.get(collection_name)

            if retriever:
                try:
                    if hasattr(retriever, "k"):
                        current_k = retriever.k
                    elif hasattr(retriever, "search_kwargs"):
                        current_k = retriever.search_kwargs.get("k")
                except Exception as attr_error:
                    self.logger.warning(
                        f"Could not determine current 'k' for retriever of collection '{collection_name}': {attr_error}"
                    )

            needs_retriever_setup = collection_name not in self.retrievers or current_k != self.default_retrieval_top_k

            if needs_retriever_setup:
                try:
                    self.setup_retriever_for_collection(collection_name)
                    self.logger.info(
                        f"Retriever for collection '{collection_name}' configured with k={self.default_retrieval_top_k}"
                    )
                except Exception as setup_error:
                    error_msg = (
                        f"Failed to set up retriever for existing in-memory collection '{collection_name}': {setup_error}"
                    )
                    self.logger.error(error_msg, exc_info=True)

        except Exception as e:
            error_msg = f"Unexpected error in _handle_existing_collection_in_memory for '{collection_name}': {e}"
            self.logger.error(error_msg, exc_info=True)

    def _handle_load_from_disk(self, collection_name: str, persist_dir: str) -> None:
        """
        Load a collection from disk if it exists.

        Args:
            collection_name: Name of the collection
            persist_dir: Path to the collection's persistence directory
        """
        try:
            if os.path.exists(persist_dir):
                try:
                    self.logger.info(f"Attempting to load existing vector store '{collection_name}' from {persist_dir}")

                    loaded_vectorstore = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory=persist_dir,
                    )

                    self.vectorstores[collection_name] = loaded_vectorstore
                    self.setup_retriever_for_collection(collection_name)

                    self.logger.info(
                        f"Successfully loaded vector store '{collection_name}' from disk "
                        f"and configured retriever with k={self.default_retrieval_top_k}."
                    )

                except Exception as load_error:
                    error_msg = (
                        f"Error loading vector store '{collection_name}' from {persist_dir}: {load_error}. "
                        f"A new one may be created if documents are indexed."
                    )
                    self.logger.error(error_msg, exc_info=True)
                    self.vectorstores.pop(collection_name, None)
                    self.retrievers.pop(collection_name, None)

            else:
                self.logger.info(
                    f"No persisted vector store found for '{collection_name}' at {persist_dir}. "
                    f"It will be created upon first indexing."
                )

        except Exception as e:
            error_msg = f"Unexpected error in _handle_load_from_disk for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def setup_retriever_for_collection(self, collection_name: str) -> None:
        """
        Set up retriever for a collection with optional hybrid search.

        Args:
            collection_name: Name of the collection
        """
        try:
            vectorstore = self.vectorstores.get(collection_name)
            if not vectorstore:
                error_msg = f"Cannot set up retriever: Vectorstore for '{collection_name}' not found."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            if self.enable_hybrid_search:
                self.logger.debug(f"Setting up hybrid retriever for '{collection_name}'...")
                try:
                    collection = vectorstore._collection
                    try:
                        doc_count = collection.count()
                    except Exception:
                        doc_count = None

                    if doc_count and doc_count > 10000:
                        max_hybrid_docs = 10000
                        self.logger.warning(
                            f"Large collection ({doc_count} docs) - limiting hybrid search to {max_hybrid_docs} docs "
                            f"to prevent memory issues."
                        )

                        if self.stream_documents_callback:
                            all_docs = []
                            for batch in self.stream_documents_callback(collection_name, 1000):
                                all_docs.extend(batch)
                                if len(all_docs) >= max_hybrid_docs:
                                    self.logger.info(
                                        f"Loaded {len(all_docs)} documents for hybrid search (limit: {max_hybrid_docs})"
                                    )
                                    break
                        else:
                            all_docs = []
                            self.logger.warning(
                                f"No stream callback available for large collection '{collection_name}', "
                                f"falling back to standard retriever"
                            )
                    else:
                        if self.get_all_documents_callback:
                            all_docs = self.get_all_documents_callback(collection_name, True)
                        else:
                            all_docs = []
                            self.logger.warning(
                                f"No document callback available for collection '{collection_name}', "
                                f"falling back to standard retriever"
                            )

                    if all_docs:
                        hybrid_retriever = AdaptiveHybridRetriever(
                            vector_store=vectorstore,
                            documents=all_docs,
                            k=self.default_retrieval_top_k,
                        )
                        self.retrievers[collection_name] = hybrid_retriever
                        self.logger.info(
                            f"Hybrid retriever created for '{collection_name}' with "
                            f"{len(all_docs)} docs, k={self.default_retrieval_top_k}"
                        )
                    else:
                        standard_retriever = vectorstore.as_retriever(search_kwargs={"k": self.default_retrieval_top_k})
                        self.retrievers[collection_name] = standard_retriever
                        self.logger.warning(
                            f"No documents for hybrid retriever '{collection_name}', using standard retriever"
                        )

                except Exception as hybrid_error:
                    self.logger.warning(
                        f"Failed to create hybrid retriever for '{collection_name}': {hybrid_error}. "
                        f"Falling back to standard retriever",
                        exc_info=True,
                    )
                    standard_retriever = vectorstore.as_retriever(search_kwargs={"k": self.default_retrieval_top_k})
                    self.retrievers[collection_name] = standard_retriever

            else:
                self.logger.debug(f"Setting up standard retriever for '{collection_name}'...")
                standard_retriever = vectorstore.as_retriever(search_kwargs={"k": self.default_retrieval_top_k})
                self.retrievers[collection_name] = standard_retriever
                self.logger.debug(f"Standard retriever set up for '{collection_name}'.")

        except Exception as e:
            error_msg = f"Unexpected error in setup_retriever_for_collection for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def index_documents(self, docs: List[Document], name: str, recreate: bool = False) -> None:
        """
        Index documents into a collection.

        Args:
            docs: List of documents to index
            name: Collection name
            recreate: If True, recreate the collection
        """
        self.init_or_load_vectorstore(name, recreate)
        vs = self.vectorstores.get(name)
        d = self.get_persist_dir(name)

        if vs is None or recreate:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Could not create persist directory '{d}': {e}", exc_info=True)
                return

            vs_new = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                collection_name=name,
                persist_directory=d,
            )
            self.vectorstores[name] = vs_new
            self.retrievers[name] = vs_new.as_retriever(search_kwargs={"k": self.default_retrieval_top_k})
            self.logger.info(f"Created vector store '{name}' with default k={self.default_retrieval_top_k}")

            if self.invalidate_cache_callback:
                self.invalidate_cache_callback(name)

        else:
            vs.add_documents(docs)
            self.logger.info(f"Added {len(docs)} docs to '{name}'")

            if self.invalidate_cache_callback:
                self.invalidate_cache_callback(name)

    def _handle_recreate_collection(self, collection_name: str, persist_dir: str) -> None:
        """
        Handle recreation of a collection by removing existing data.

        Args:
            collection_name: Name of the collection
            persist_dir: Path to the collection's persistence directory
        """
        try:
            self.logger.info(f"Recreating collection '{collection_name}'. Removing existing data if present.")

            if os.path.exists(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                    self.logger.debug(f"Removed persisted data directory: {persist_dir}")
                except OSError as os_error:
                    self.logger.error(f"OS error removing persisted directory '{persist_dir}': {os_error}")
                    raise RuntimeError(f"Failed to remove persisted directory '{persist_dir}'") from os_error
                except Exception as remove_error:
                    self.logger.critical(
                        f"Unexpected error removing persisted directory '{persist_dir}': {remove_error}",
                        exc_info=True,
                    )
                    raise RuntimeError(f"Failed to remove persisted directory '{persist_dir}'") from remove_error

            if collection_name in self.vectorstores:
                removed_vs = self.vectorstores.pop(collection_name, None)
                if removed_vs:
                    self.logger.debug(f"Removed in-memory vectorstore reference for '{collection_name}'.")

            if collection_name in self.retrievers:
                removed_retriever = self.retrievers.pop(collection_name, None)
                if removed_retriever:
                    self.logger.debug(f"Removed in-memory retriever reference for '{collection_name}'.")

            if self.invalidate_cache_callback:
                self.invalidate_cache_callback(collection_name)

            self.logger.info(f"Completed recreation preparation for collection '{collection_name}'.")

        except Exception as e:
            error_msg = f"Unexpected error in _handle_recreate_collection for '{collection_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and its persisted data.

        Args:
            collection_name: Name of the collection to delete
        """
        persist_dir = self.get_persist_dir(collection_name)
        self._handle_recreate_collection(collection_name, persist_dir)
        self.logger.info(f"Deleted collection '{collection_name}'")

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        try:
            if not os.path.exists(self.persist_directory_base):
                return []

            collections = [
                name
                for name in os.listdir(self.persist_directory_base)
                if os.path.isdir(os.path.join(self.persist_directory_base, name))
            ]
            return sorted(collections)
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}", exc_info=True)
            return []

    def get_vectorstore(self, collection_name: str) -> Optional[Chroma]:
        """
        Get the vector store for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Chroma vectorstore instance or None if not found
        """
        return self.vectorstores.get(collection_name)

    def get_retriever(self, collection_name: str) -> Optional[Any]:
        """
        Get the retriever for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Retriever instance or None if not found
        """
        return self.retrievers.get(collection_name)
