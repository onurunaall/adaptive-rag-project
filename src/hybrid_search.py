from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_core.retrievers import BaseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines semantic (vector) search with keyword (BM25) search
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        documents: List[Document],
        alpha: float = 0.7,  # Weight for semantic search (1-alpha for keyword search)
        k: int = 10
    ):
        self.vector_store = vector_store
        self.documents = documents
        self.alpha = alpha  # Semantic search weight
        self.k = k
        
        # Build BM25 index
        self.corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Build TF-IDF index for additional keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        
        # Document mapping
        self.doc_id_to_doc = {i: doc for i, doc in enumerate(documents)}
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid search
        """
        # 1. Semantic search
        semantic_results = self._semantic_search(query, k=self.k * 2)
        
        # 2. Keyword search (BM25)
        bm25_results = self._bm25_search(query, k=self.k * 2)
        
        # 3. TF-IDF search
        tfidf_results = self._tfidf_search(query, k=self.k * 2)
        
        # 4. Combine and rank results
        combined_results = self._combine_results(
            query, semantic_results, bm25_results, tfidf_results
        )
        
        return combined_results[:self.k]
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform semantic search using vector store"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc, 1.0 - score) for doc, score in results]  # Convert distance to similarity
        except Exception as e:
            # Fallback if similarity_search_with_score not available
            docs = self.vector_store.similarity_search(query, k=k)
            return [(doc, 0.5) for doc in docs]  # Default similarity score
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc = self.documents[idx]
                results.append((doc, scores[idx]))
        
        return results
    
    def _tfidf_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform TF-IDF search"""
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    doc = self.documents[idx]
                    results.append((doc, similarities[idx]))
            
            return results
        except Exception:
            return []  # Return empty list if TF-IDF search fails
    
    def _combine_results(
        self,
        query: str,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        tfidf_results: List[Tuple[Document, float]]
    ) -> List[Document]:
        """
        Combine results from different search methods using weighted scoring
        """
        # Normalize scores for each method
        semantic_scores = self._normalize_scores([score for _, score in semantic_results])
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        tfidf_scores = self._normalize_scores([score for _, score in tfidf_results])
        
        # Create document score mapping
        doc_scores: Dict[str, float] = {}
        
        # Add semantic search scores
        for (doc, _), norm_score in zip(semantic_results, semantic_scores):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + self.alpha * norm_score
        
        # Add BM25 scores
        beta = (1 - self.alpha) * 0.6  # 60% of remaining weight for BM25
        for (doc, _), norm_score in zip(bm25_results, bm25_scores):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + beta * norm_score
        
        # Add TF-IDF scores
        gamma = (1 - self.alpha) * 0.4  # 40% of remaining weight for TF-IDF
        for (doc, _), norm_score in zip(tfidf_results, tfidf_scores):
            doc_key = self._get_doc_key(doc)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + gamma * norm_score
        
        # Create document mapping
        doc_mapping: Dict[str, Document] = {}
        for doc, _ in semantic_results + bm25_results + tfidf_results:
            doc_key = self._get_doc_key(doc)
            doc_mapping[doc_key] = doc
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return ranked documents
        result = []
        for doc_key, score in sorted_docs:
            if doc_key in doc_mapping:
                doc = doc_mapping[doc_key]
                # Add combined score to metadata
                doc.metadata = doc.metadata or {}
                doc.metadata['hybrid_score'] = score
                result.append(doc)
        
        return result
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _get_doc_key(self, doc: Document) -> str:
        """Generate unique key for document"""
        # Use hash of content + source for uniqueness
        content_hash = hash(doc.page_content[:100])  # First 100 chars
        source = doc.metadata.get('source', 'unknown')
        return f"{source}_{content_hash}"


class AdaptiveHybridRetriever(HybridRetriever):
    """
    Hybrid retriever that adapts search strategy based on query characteristics
    """
    
    def __init__(self, vector_store: VectorStore, documents: List[Document], k: int = 10):
        super().__init__(vector_store, documents, alpha=0.7, k=k)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Adapt search strategy based on query characteristics
        """
        # Analyze query to determine optimal search strategy
        query_analysis = self._analyze_query(query)
        
        # Adjust alpha based on query type
        if query_analysis['is_factual']:
            self.alpha = 0.8  # More weight on semantic search for factual queries
        elif query_analysis['is_keyword_heavy']:
            self.alpha = 0.4  # More weight on keyword search
        elif query_analysis['is_conceptual']:
            self.alpha = 0.9  # Heavy semantic search for conceptual queries
        else:
            self.alpha = 0.7  # Balanced approach
        
        return super()._get_relevant_documents(query)
    
    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """
        Analyze query characteristics to determine search strategy
        """
        query_lower = query.lower()
        
        # Factual query indicators
        factual_indicators = ['what is', 'who is', 'when did', 'where is', 'how many']
        is_factual = any(indicator in query_lower for indicator in factual_indicators)
        
        # Keyword-heavy query indicators (many specific terms)
        words = query.split()
        is_keyword_heavy = len(words) > 8 or any(word.isupper() for word in words)
        
        # Conceptual query indicators
        conceptual_indicators = ['explain', 'describe', 'analyze', 'compare', 'discuss']
        is_conceptual = any(indicator in query_lower for indicator in conceptual_indicators)
        
        return {
            'is_factual': is_factual,
            'is_keyword_heavy': is_keyword_heavy,
            'is_conceptual': is_conceptual
        }