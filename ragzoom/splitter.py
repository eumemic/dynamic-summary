"""Text splitting functionality for RagZoom."""

import logging

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragzoom.config import IndexConfig

logger = logging.getLogger(__name__)


class TextSplitter:
    """Boundary-aware text splitter for creating leaf chunks."""

    def __init__(self, config: IndexConfig):
        """Initialize the splitter with configuration."""
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

        # Use token counts directly since our length_function returns tokens
        # Set overlap to 0 since RagZoom requires non-overlapping sequential chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.target_chunk_tokens,
            chunk_overlap=0,  # No overlap - RagZoom needs sequential chunks
            separators=[
                "\n\n",  # Paragraph breaks
                r"(?<=[.!?])\s+",  # After sentence-ending punctuation
                "\n",  # Line breaks
                " ",  # Spaces
                "",  # Any character
            ],
            length_function=self._token_length,
            is_separator_regex=True,  # Enable regex for better sentence detection
            keep_separator="end",  # Keep separator at end of chunk to avoid gaps
        )

    def _token_length(self, text: str) -> int:
        """Calculate token length of text."""
        return len(self.tokenizer.encode(text))

    def _preprocess_text(self, text: str) -> str:
        """Remove formatting line breaks while preserving semantic breaks.

        Replaces single newlines that don't follow punctuation with spaces.
        Preserves newlines after sentence-ending punctuation and paragraph breaks.
        """
        # First, protect paragraph breaks by replacing \n\n with a placeholder
        paragraph_marker = "\x00PARAGRAPH\x00"
        text = text.replace("\n\n", paragraph_marker)

        # Process each newline
        import re

        # Pattern: any character followed by newline
        pattern = r"(.)\n"

        def replace_newline(match: re.Match[str]) -> str:
            char_before = match.group(1)
            # Only keep newlines after sentence-ending punctuation
            if char_before in ".!?":
                # Keep the newline after sentence boundaries
                return match.group(0)
            else:
                # Replace newline with space for everything else
                return char_before + " "

        text = re.sub(pattern, replace_newline, text)

        # Restore paragraph breaks
        text = text.replace(paragraph_marker, "\n\n")

        return text

    def _reconstruct_chunks_with_whitespace(
        self, original_text: str, raw_chunks: list[str]
    ) -> list[str]:
        """Fill ALL gaps between chunks by appending to the previous chunk.

        This ensures complete coverage with no gaps. Any content between chunks
        (whitespace or otherwise) is appended to the previous chunk.
        """
        if not raw_chunks:
            return []

        if len(raw_chunks) == 1:
            return raw_chunks

        # Find positions of each chunk in the original text
        chunk_positions = []
        search_start = 0

        for chunk in raw_chunks:
            pos = original_text.find(chunk, search_start)
            if pos != -1:
                chunk_positions.append((pos, pos + len(chunk), chunk))
                search_start = pos + len(chunk)
            else:
                # Chunk not found exactly - shouldn't happen with our splitter
                logger.warning(f"Chunk not found in original text: {chunk[:50]}...")
                return raw_chunks  # Return as-is

        # Fill ALL gaps by appending to previous chunk
        reconstructed_chunks = []

        for i, (start_pos, end_pos, chunk) in enumerate(chunk_positions):
            if i == 0:
                # First chunk - check if there's content before it
                if start_pos > 0:
                    # Include any content from the beginning
                    reconstructed_chunks.append(original_text[0:end_pos])
                else:
                    reconstructed_chunks.append(chunk)
            else:
                # Check for gap before this chunk
                prev_end = chunk_positions[i - 1][1]

                if start_pos > prev_end:
                    # There's a gap - append it to previous chunk
                    gap = original_text[prev_end:start_pos]
                    reconstructed_chunks[-1] += gap

                reconstructed_chunks.append(chunk)

        # Check if there's content after the last chunk
        if chunk_positions:
            last_end = chunk_positions[-1][1]
            if last_end < len(original_text):
                # Append remaining content to last chunk
                reconstructed_chunks[-1] += original_text[last_end:]

        return reconstructed_chunks

    def split_text(self, text: str) -> list[str]:
        """Split text into leaf chunks with gap reconstruction."""
        # Get initial chunks from LangChain splitter
        raw_chunks = self.splitter.split_text(text)

        # Reconstruct chunks with ALL gaps filled
        return self._reconstruct_chunks_with_whitespace(text, raw_chunks)

    def get_adjacent_context(
        self, chunks: list[str], chunk_index: int
    ) -> tuple[str | None, str | None]:
        """Get adjacent context for a chunk (for summarization)."""
        prev_context = None
        next_context = None

        if chunk_index > 0 and chunks[chunk_index - 1]:
            prev_text = chunks[chunk_index - 1]
            prev_tokens = self.tokenizer.encode(prev_text)
            if len(prev_tokens) > self.config.preceding_context_tokens:
                # Take last N tokens
                context_tokens = prev_tokens[-self.config.preceding_context_tokens :]
                prev_context = self.tokenizer.decode(context_tokens)
            else:
                prev_context = prev_text

        if chunk_index < len(chunks) - 1 and chunks[chunk_index + 1]:
            next_text = chunks[chunk_index + 1]
            next_tokens = self.tokenizer.encode(next_text)
            if len(next_tokens) > self.config.preceding_context_tokens:
                # Take first N tokens
                context_tokens = next_tokens[: self.config.preceding_context_tokens]
                next_context = self.tokenizer.decode(context_tokens)
            else:
                next_context = next_text

        return prev_context, next_context
