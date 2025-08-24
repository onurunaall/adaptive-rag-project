from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

try:
    import tiktoken
except ImportError:
    tiktoken = None


class BaseChunker(ABC):
    """Abstract base class for document chunkers"""
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        pass


class AdaptiveChunker(BaseChunker):
    """
    Adaptive chunker that selects the best chunking strategy based on document type
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Initialize different chunkers
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Token-based chunker for OpenAI models
        self.token_chunker = None
        if tiktoken is not None:
            try:
                self.token_chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name=model_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception:
                pass
        
        # Semantic chunker (requires OpenAI API key)
        self.semantic_chunker = None
        if openai_api_key:
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                self.semantic_chunker = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile"
                )
            except Exception:
                pass
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using adaptive strategy based on document characteristics
        """
        chunked_docs = []
        
        for doc in documents:
            doc_type = self._detect_document_type(doc)
            chunker = self._select_chunker(doc_type, doc)
            
            # Add document type to metadata for tracking
            doc.metadata = doc.metadata or {}
            doc.metadata["detected_type"] = doc_type
            doc.metadata["chunker_used"] = chunker.__class__.__name__
            
            chunks = chunker.split_documents([doc])
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def _detect_document_type(self, doc: Document) -> str:
        """
        Detect document type based on content and metadata
        """
        content = doc.page_content.lower()
        source = doc.metadata.get("source", "").lower()
        
        # Code detection
        if self._is_code_document(content, source):
            return "code"
        
        # Academic paper detection
        if self._is_academic_document(content, source):
            return "academic"
        
        # Financial document detection
        if self._is_financial_document(content, source):
            return "financial"
        
        # News article detection
        if self._is_news_document(content, source):
            return "news"
        
        # Technical documentation
        if self._is_technical_document(content, source):
            return "technical"
        
        return "general"
    
    def _is_code_document(self, content: str, source: str) -> bool:
        """Detect if document contains code"""
        code_indicators = [
            "def ", "class ", "function", "import ", "from ",
            "<?php", "<html", "<script", "SELECT ", "CREATE TABLE",
            "git ", "npm ", "pip install"
        ]
        file_extensions = [".py", ".js", ".html", ".css", ".sql", ".php", ".java", ".cpp"]
        
        has_code_content = sum(1 for indicator in code_indicators if indicator in content) >= 3
        has_code_extension = any(ext in source for ext in file_extensions)
        
        return has_code_content or has_code_extension
    
    def _is_academic_document(self, content: str, source: str) -> bool:
        """Detect academic papers"""
        academic_indicators = [
            "abstract", "introduction", "methodology", "results", "conclusion",
            "references", "bibliography", "doi:", "arxiv", "journal",
            "university", "research", "study shows", "according to"
        ]
        
        indicator_count = sum(1 for indicator in academic_indicators if indicator in content)
        return indicator_count >= 4
    
    def _is_financial_document(self, content: str, source: str) -> bool:
        """Detect financial documents"""
        financial_indicators = [
            "revenue", "profit", "loss", "earnings", "financial",
            "balance sheet", "income statement", "cash flow",
            "quarterly", "annual report", "sec filing", "10-k", "10-q"
        ]
        
        indicator_count = sum(1 for indicator in financial_indicators if indicator in content)
        return indicator_count >= 3
    
    def _is_news_document(self, content: str, source: str) -> bool:
        """Detect news articles"""
        news_indicators = [
            "breaking:", "updated:", "published:", "reporter", "news",
            "according to sources", "spokesperson said", "press release"
        ]
        news_domains = ["reuters", "bloomberg", "cnn", "bbc", "wsj", "ft.com"]
        
        has_news_content = any(indicator in content for indicator in news_indicators)
        has_news_source = any(domain in source for domain in news_domains)
        
        return has_news_content or has_news_source
    
    def _is_technical_document(self, content: str, source: str) -> bool:
        """Detect technical documentation"""
        tech_indicators = [
            "api", "configuration", "installation", "documentation",
            "parameter", "example:", "usage:", "note:", "warning:",
            "requirements", "dependencies"
        ]
        
        indicator_count = sum(1 for indicator in tech_indicators if indicator in content)
        return indicator_count >= 3
    
    def _select_chunker(self, doc_type: str, doc: Document) -> TextSplitter:
        """
        Select the most appropriate chunker based on document type
        """
        if doc_type == "code":
            return self._get_code_chunker()
        elif doc_type == "academic":
            return self._get_semantic_chunker() or self._get_paragraph_aware_chunker()
        elif doc_type == "financial":
            return self._get_structured_chunker()
        elif doc_type == "news":
            return self._get_paragraph_aware_chunker()
        elif doc_type == "technical":
            return self._get_structured_chunker()
        else:
            return self.token_chunker or self.recursive_chunker
    
    def _get_code_chunker(self) -> TextSplitter:
        """Get chunker optimized for code"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=50,  # Less overlap for code
            separators=["\n\nclass ", "\n\ndef ", "\n\nfunction ", "\n\n", "\n", " ", ""],
            length_function=len,
        )
    
    def _get_semantic_chunker(self) -> Optional[TextSplitter]:
        """Get semantic chunker if available"""
        return self.semantic_chunker
    
    def _get_paragraph_aware_chunker(self) -> TextSplitter:
        """Get chunker that preserves paragraph boundaries"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", ".\n", ". ", "\n", " ", ""],
            length_function=len,
        )
    
    def _get_structured_chunker(self) -> TextSplitter:
        """Get chunker for structured documents"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # Larger chunks for structured content
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )


class HybridChunker(BaseChunker):
    """
    Hybrid chunker that combines multiple chunking strategies
    """
    
    def __init__(self, primary_chunker: BaseChunker, secondary_chunker: BaseChunker):
        self.primary_chunker = primary_chunker
        self.secondary_chunker = secondary_chunker
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Use primary chunker, then refine with secondary chunker if needed
        """
        primary_chunks = self.primary_chunker.chunk_documents(documents)
        
        # Refine chunks that are too large or too small
        refined_chunks = []
        for chunk in primary_chunks:
            if len(chunk.page_content) > 1000:  # Too large, split further
                sub_chunks = self.secondary_chunker.chunk_documents([chunk])
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks