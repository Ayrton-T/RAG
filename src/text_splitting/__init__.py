"""Text chunking module for RAG applications.

This module provides text chunking functionality specifically for RAG (Retrieval Augmented Generation)
applications. It is distinct from Python's stdlib 'chunk' module which handles IFF chunks.
"""

from .text_chunker import TextChunker, ChunkMetadata

__all__ = ['TextChunker', 'ChunkMetadata'] 