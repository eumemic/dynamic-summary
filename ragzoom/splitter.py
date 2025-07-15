"""Text splitting functionality for RagZoom."""

from typing import List, Optional

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragzoom.config import RagZoomConfig


class TextSplitter:
    """Boundary-aware text splitter for creating leaf chunks."""

    def __init__(self, config: RagZoomConfig):
        """Initialize the splitter with configuration."""
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

        # Use token counts directly since our length_function returns tokens
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.leaf_tokens,
            chunk_overlap=config.leaf_overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=self._token_length,
            is_separator_regex=False,
        )

    def _token_length(self, text: str) -> int:
        """Calculate token length of text."""
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[str]:
        """Split text into leaf chunks."""
        return self.splitter.split_text(text)

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into chunks with metadata preserved."""
        all_chunks = []

        for doc in documents:
            chunks = self.split_text(doc["text"])

            # Calculate character positions for each chunk
            current_pos = 0
            for i, chunk in enumerate(chunks):
                chunk_start = doc["text"].find(chunk, current_pos)
                chunk_end = chunk_start + len(chunk)

                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": i,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "source_doc_id": doc.get("id", "unknown"),
                    }
                })

                # For character position, estimate overlap in characters (rough: 1 token ≈ 4 chars)
                overlap_chars = self.config.leaf_overlap_tokens * 4
                current_pos = chunk_start + len(chunk) - overlap_chars

        return all_chunks

    def get_adjacent_context(
        self, chunks: List[str], chunk_index: int
    ) -> tuple[Optional[str], Optional[str]]:
        """Get adjacent context for a chunk (for summarization)."""
        prev_context = None
        next_context = None

        if chunk_index > 0 and chunks[chunk_index - 1]:
            prev_text = chunks[chunk_index - 1]
            prev_tokens = self.tokenizer.encode(prev_text)
            if len(prev_tokens) > self.config.adjacent_context_tokens:
                # Take last N tokens
                context_tokens = prev_tokens[-self.config.adjacent_context_tokens:]
                prev_context = self.tokenizer.decode(context_tokens)
            else:
                prev_context = prev_text

        if chunk_index < len(chunks) - 1 and chunks[chunk_index + 1]:
            next_text = chunks[chunk_index + 1]
            next_tokens = self.tokenizer.encode(next_text)
            if len(next_tokens) > self.config.adjacent_context_tokens:
                # Take first N tokens
                context_tokens = next_tokens[:self.config.adjacent_context_tokens]
                next_context = self.tokenizer.decode(context_tokens)
            else:
                next_context = next_text

        return prev_context, next_context
