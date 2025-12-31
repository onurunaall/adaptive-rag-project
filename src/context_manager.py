from typing import List, Tuple, Optional
from langchain_core.documents import Document
import logging

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class ContextManager:
    """
    Manages context truncation to fit within LLM token limits.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_context_tokens: Optional[int] = None,
        reserved_tokens: int = 1000,  # Reserve for question + system prompt + response
    ):
        """
        Initialize context manager.

        Args:
            model_name: LLM model name for tokenization
            max_context_tokens: Maximum tokens for context (auto-detected if None)
            reserved_tokens: Tokens to reserve for non-context parts
        """
        self.model_name = model_name
        self.reserved_tokens = reserved_tokens
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(model_name)
                self.logger.info(f"Initialized tiktoken encoder for {model_name}")
            except KeyError:
                try:
                    self.logger.warning(f"Model {model_name} not found in tiktoken, using cl100k_base")
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    self.logger.warning(f"Failed to load tiktoken encoding: {e}. Using character-based estimation")
                    self.encoder = None
            except Exception as e:
                # Handle network errors or other exceptions during tiktoken initialization
                self.logger.warning(f"Failed to initialize tiktoken encoder: {e}. Using character-based estimation")
                self.encoder = None
        else:
            self.encoder = None
            self.logger.warning("tiktoken not available, using character-based estimation")

        # Set max context tokens based on model
        if max_context_tokens:
            self.max_context_tokens = max_context_tokens
        else:
            self.max_context_tokens = self._get_model_context_limit() - reserved_tokens

        self.logger.info(f"Context manager initialized: max_context_tokens={self.max_context_tokens}")

    def _get_model_context_limit(self) -> int:
        """Get the total context window size for the model."""
        model_limits = {
            # OpenAI models
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            # Google models
            "gemini-pro": 32768,
            "gemini-1.5-pro": 1048576,
            "gemini-1.5-flash": 1048576,
            # Ollama/local models (conservative estimates)
            "llama3": 8192,
            "llama3.1": 131072,
            "mistral": 8192,
        }

        # Check for exact match or partial match
        for model_key, limit in model_limits.items():
            if model_key in self.model_name.lower():
                self.logger.debug(f"Detected context limit {limit} for model {self.model_name}")
                return limit

        # Default conservative estimate
        self.logger.warning(f"Unknown model {self.model_name}, using default 8192 tokens")
        return 8192

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                self.logger.error(f"Error encoding text: {e}")
                # Fallback to character estimation
                return len(text) // 4
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4

    def truncate_documents(
        self, documents: List[Document], question: str = "", strategy: str = "smart"
    ) -> Tuple[List[Document], bool]:
        """
        Truncate documents to fit within token limits.

        Args:
            documents: List of documents to truncate
            question: User question (to account for its tokens)
            strategy: Truncation strategy:
                - "smart": Prioritize document ranking, truncate from bottom
                - "balanced": Give each document equal space
                - "first": Keep first documents until limit reached
                - "chunk": Split documents into chunks if too large

        Returns:
            Tuple of (truncated_documents, was_truncated)
        """
        if not documents:
            return [], False

        # Calculate available tokens
        question_tokens = self.count_tokens(question)
        available_tokens = self.max_context_tokens - question_tokens

        if available_tokens <= 0:
            self.logger.error(f"Question uses {question_tokens} tokens, exceeds limit!")
            return [], True

        self.logger.debug(f"Available tokens for context: {available_tokens}")

        # Apply truncation strategy
        if strategy == "smart":
            return self._truncate_smart(documents, available_tokens)
        elif strategy == "balanced":
            return self._truncate_balanced(documents, available_tokens)
        elif strategy == "first":
            return self._truncate_first(documents, available_tokens)
        elif strategy == "chunk":
            return self._truncate_chunk(documents, available_tokens)
        else:
            self.logger.warning(f"Unknown strategy {strategy}, using smart")
            return self._truncate_smart(documents, available_tokens)

    def _truncate_smart(self, documents: List[Document], available_tokens: int) -> Tuple[List[Document], bool]:
        """
        Smart truncation: prioritize top documents, truncate from bottom.
        Assumes documents are already ranked by relevance.
        """
        truncated_docs = []
        total_tokens = 0
        was_truncated = False

        for doc in documents:
            doc_tokens = self.count_tokens(doc.page_content)

            if total_tokens + doc_tokens <= available_tokens:
                # Document fits completely
                truncated_docs.append(doc)
                total_tokens += doc_tokens
            else:
                # Check if we can fit a truncated version
                remaining_tokens = available_tokens - total_tokens

                if remaining_tokens > 100:  # Only include if we can fit meaningful content
                    truncated_content = self._truncate_text(doc.page_content, remaining_tokens)

                    truncated_doc = Document(
                        page_content=truncated_content + "\n[...truncated]",
                        metadata=doc.metadata,
                    )
                    truncated_docs.append(truncated_doc)

                was_truncated = True
                break

        if len(truncated_docs) < len(documents):
            was_truncated = True
            self.logger.warning(f"Truncated {len(documents) - len(truncated_docs)} documents " f"(kept {len(truncated_docs)})")

        return truncated_docs, was_truncated

    def _truncate_balanced(self, documents: List[Document], available_tokens: int) -> Tuple[List[Document], bool]:
        """
        Balanced truncation: give each document equal token allocation.
        """
        if not documents:
            return [], False

        tokens_per_doc = available_tokens // len(documents)
        truncated_docs = []
        was_truncated = False

        for doc in documents:
            doc_tokens = self.count_tokens(doc.page_content)

            if doc_tokens <= tokens_per_doc:
                # Document fits
                truncated_docs.append(doc)
            else:
                # Truncate document
                truncated_content = self._truncate_text(doc.page_content, tokens_per_doc)
                truncated_doc = Document(
                    page_content=truncated_content + "\n[...truncated]",
                    metadata=doc.metadata,
                )
                truncated_docs.append(truncated_doc)
                was_truncated = True

        return truncated_docs, was_truncated

    def _truncate_first(self, documents: List[Document], available_tokens: int) -> Tuple[List[Document], bool]:
        """
        First-priority truncation: keep documents in order until limit.
        """
        truncated_docs = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = self.count_tokens(doc.page_content)

            if total_tokens + doc_tokens <= available_tokens:
                truncated_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        was_truncated = len(truncated_docs) < len(documents)
        return truncated_docs, was_truncated

    def _truncate_chunk(self, documents: List[Document], available_tokens: int) -> Tuple[List[Document], bool]:
        """
        Chunk truncation: split large documents into smaller chunks.
        """
        truncated_docs = []
        total_tokens = 0
        was_truncated = False

        for doc in documents:
            doc_tokens = self.count_tokens(doc.page_content)

            if doc_tokens <= available_tokens - total_tokens:
                # Document fits completely
                truncated_docs.append(doc)
                total_tokens += doc_tokens
            else:
                # Split document into chunks
                remaining_tokens = available_tokens - total_tokens

                if remaining_tokens > 100:
                    # Create chunk from remaining space
                    chunk_content = self._truncate_text(doc.page_content, remaining_tokens)

                    chunk_doc = Document(
                        page_content=chunk_content + "\n[...document continues]",
                        metadata={**doc.metadata, "is_chunk": True, "chunk_index": 0},
                    )
                    truncated_docs.append(chunk_doc)

                was_truncated = True
                break

        return truncated_docs, was_truncated

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to specified token count.
        Tries to truncate at sentence boundaries when possible.
        """
        if self.encoder:
            # Tokenize and truncate
            tokens = self.encoder.encode(text)

            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoder.decode(truncated_tokens)

            # Try to end at a sentence boundary
            for delimiter in [". ", ".\n", "! ", "?\n", "? ", "\n\n"]:
                last_delimiter = truncated_text.rfind(delimiter)
                if last_delimiter > len(truncated_text) * 0.8:  # Only if we keep 80%+
                    return truncated_text[: last_delimiter + len(delimiter)]

            return truncated_text
        else:
            # Character-based estimation (4 chars ≈ 1 token)
            max_chars = max_tokens * 4

            if len(text) <= max_chars:
                return text

            truncated = text[:max_chars]

            # Try to end at sentence boundary
            for delimiter in [". ", ".\n", "! ", "?\n", "? ", "\n\n"]:
                last_delimiter = truncated.rfind(delimiter)
                if last_delimiter > len(truncated) * 0.8:
                    return truncated[: last_delimiter + len(delimiter)]

            return truncated

    def get_context_summary(self, documents: List[Document]) -> dict:
        """
        Get summary statistics about context usage.

        Returns:
            Dictionary with token counts and limits
        """
        total_tokens = sum(self.count_tokens(doc.page_content) for doc in documents)

        return {
            "document_count": len(documents),
            "total_tokens": total_tokens,
            "max_context_tokens": self.max_context_tokens,
            "token_utilization": (total_tokens / self.max_context_tokens if self.max_context_tokens > 0 else 0),
            "would_exceed_limit": total_tokens > self.max_context_tokens,
            "tokens_over_limit": max(0, total_tokens - self.max_context_tokens),
        }
