"""Text chunking module for splitting documents into overlapping chunks."""
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.text_splitter import TokenTextSplitter
import numpy as np
import tiktoken

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    start_idx: int
    end_idx: int
    token_count: int
    source_file: str
    source_page: int

class TextChunker:
    """Split text into fixed-size overlapping chunks with metadata using tokens."""
    
    def __init__(
        self,
        tokens_per_chunk: int = 800,
        overlap_tokens: int = 400,
        encoding_name: str = "cl100k_base"  # OpenAI's default encoding
    ):
        """Initialize chunker with token-based parameters.
        
        Args:
            tokens_per_chunk: Target size of each chunk in tokens
            overlap_tokens: Number of overlapping tokens between chunks
            encoding_name: Name of the tokenizer encoding to use
        """
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Initialize Langchain token splitter
        self.text_splitter = TokenTextSplitter(
            chunk_size=tokens_per_chunk,
            chunk_overlap=overlap_tokens,
            encoding_name=encoding_name
        )
    
    def create_chunks(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata using token-based splitting.
        
        Args:
            text: Text content to chunk
            metadata: Source file metadata
            
        Returns:
            List of dictionaries containing:
            - content: Chunk text content
            - metadata: Chunk metadata
        """
        # Get full text tokens for position tracking
        full_text_tokens = self.tokenizer.encode(text)
        
        # Split text using Langchain
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )
        
        # Process chunks and add metadata
        processed_chunks = []
        current_pos = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            chunk_tokens = self.tokenizer.encode(chunk_text)
            token_count = len(chunk_tokens)
            
            # Find token positions in the full text
            if i == 0:
                start_idx = 0
            else:
                # Find the start of this chunk in the full text
                chunk_start_tokens = chunk_tokens[:min(50, token_count)]  # Use first 50 tokens to find position
                for j in range(current_pos, len(full_text_tokens) - len(chunk_start_tokens) + 1):
                    if full_text_tokens[j:j + len(chunk_start_tokens)] == chunk_start_tokens:
                        start_idx = j
                        break
                else:
                    start_idx = current_pos  # Fallback if exact match not found
            
            end_idx = start_idx + token_count
            current_pos = end_idx - self.overlap_tokens  # Update for next chunk
            
            # Create chunk metadata
            chunk_id = f"{metadata['file_hash']}_chunk_{i}"
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=end_idx,
                token_count=token_count,
                source_file=metadata['file_name'],
                source_page=self._estimate_page_number(start_idx, len(full_text_tokens), metadata['num_pages'])
            )
            
            # Only add chunk if it has content
            if token_count > 0:
                processed_chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk": chunk_metadata.__dict__
                    }
                })
        
        return processed_chunks
    
    def _estimate_page_number(self, start_idx: int, total_tokens: int, total_pages: int) -> int:
        """Estimate page number based on token position."""
        if total_pages <= 1:
            return 1
        # Estimate page number based on token position
        tokens_per_page = total_tokens / total_pages
        estimated_page = int(np.ceil(start_idx / tokens_per_page)) + 1
        return min(estimated_page, total_pages)
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the chunks."""
        num_chunks = len(chunks)
        total_tokens = sum(chunk["metadata"]["chunk"]["token_count"] for chunk in chunks)
        
        chunk_sizes = [chunk["metadata"]["chunk"]["token_count"] for chunk in chunks]
        avg_chunk_size = np.mean(chunk_sizes) if chunk_sizes else 0
        std_chunk_size = np.std(chunk_sizes) if chunk_sizes else 0
        
        return {
            "num_chunks": num_chunks,
            "total_tokens": total_tokens,
            "avg_chunk_size": avg_chunk_size,
            "std_chunk_size": std_chunk_size,
            "chunk_size_range": (min(chunk_sizes, default=0), max(chunk_sizes, default=0)),
            "target_chunk_size": self.tokens_per_chunk,
            "target_overlap": self.overlap_tokens,
            "encoding": self.encoding_name
        } 