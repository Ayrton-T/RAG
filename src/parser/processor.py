"""File processor that combines parsing and chunking functionality."""
from typing import List, Dict, Any, Union
from pathlib import Path
import json
from datetime import datetime
from .file_parser import FileParser
from ..text_splitting.text_chunker import TextChunker
from ..db.file_db import FileDB

class FileProcessor:
    """Process files through parsing and chunking stages."""
    
    def __init__(
        self,
        tokens_per_chunk: int = 800,
        overlap_tokens: int = 400,
        upload_dir: str = "uploads",
        file_db: FileDB = None
    ):
        """Initialize processor with chunking parameters.
        
        Args:
            tokens_per_chunk: Target size of each chunk in tokens
            overlap_tokens: Number of overlapping tokens between chunks
            upload_dir: Directory to store uploaded files
            file_db: FileDB instance for storing processed files
        """
        self.parser = FileParser()
        self.chunker = TextChunker(tokens_per_chunk, overlap_tokens)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.file_db = file_db or FileDB(upload_dir=upload_dir)
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a file and extract its content with metadata."""
        file_path = Path(file_path)
        
        # Get file info
        file_info = self._get_file_info(file_path)
        file_type = file_info["file_type"]
        
        # Get appropriate parser
        parser = self._get_parser(file_type)
        if not parser:
            raise ValueError(f"No parser available for file type: {file_type}")
        
        # Parse file
        parsed_data = parser.parse_file(file_path)
        
        # Get file hash
        file_hash = self._get_file_hash(file_path)
        
        # Create metadata
        metadata = {
            "file_name": file_path.name,
            "file_type": file_type,
            "file_hash": file_hash,  # Required for chunk IDs
            "file_size": file_info["file_size"],
            "created_at": file_info["created_at"],
            "modified_at": file_info["modified_at"],
            "word_count": parsed_data.get("word_count", 0),
            "num_pages": parsed_data.get("num_pages", 1)
        }
        
        # Chunk content
        chunks = []
        if "content" in parsed_data:
            chunks = self.chunker.create_chunks(
                text=parsed_data["content"],
                metadata=metadata
            )
        
        return {
            "chunks": chunks,
            "metadata": metadata
        }
    
    def save_processed_file(
        self,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save processed file data to FileDB.
        
        Args:
            processed_data: Output from process_file
            
        Returns:
            File metadata
        """
        # Save to FileDB
        if self.file_db:
            self.file_db.add_file_chunks(
                chunks=processed_data["chunks"],
                file_metadata=processed_data["metadata"]
            )
        
        return {
            "metadata": processed_data["metadata"]
        }
    
    def _generate_description(
        self,
        file_metadata: Dict[str, Any],
        chunk_stats: Dict[str, Any]
    ) -> str:
        """Generate a human-readable description of the processed file."""
        file_type_map = {
            'text/plain': 'Text',
            'application/pdf': 'PDF',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word'
        }
        
        file_type = file_type_map.get(file_metadata['file_type'], 'Unknown')
        size_mb = file_metadata['file_size'] / (1024 * 1024)
        
        description = (
            f"{file_type} document '{file_metadata['file_name']}' "
            f"({size_mb:.1f}MB) containing {file_metadata['word_count']} words "
            f"across {file_metadata['num_pages']} page(s). "
            f"Split into {chunk_stats['num_chunks']} chunks "
            f"averaging {chunk_stats['avg_chunk_size']:.0f} words each."
        )
        
        return description

    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file information."""
        # Map extensions to MIME types
        mime_map = {
            '.txt': 'text/plain',
            '.md': 'text/plain',
            '.markdown': 'text/plain',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pdf': 'application/pdf'
        }
        
        file_type = mime_map.get(file_path.suffix.lower(), f"text/{file_path.suffix.lower().strip('.')}")
        
        stats = file_path.stat()
        return {
            "file_type": file_type,
            "file_size": stats.st_size,
            "created_at": stats.st_ctime,
            "modified_at": stats.st_mtime
        }

    def _get_parser(self, file_type: str) -> FileParser | None:
        """Get the appropriate parser for a given file type."""
        # This is a placeholder implementation. You might want to implement
        # a more robust parser selection logic based on the file type.
        return self.parser

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for a given file."""
        # This is a placeholder implementation. You might want to implement
        # a more robust file hashing logic based on the file content.
        return str(file_path.stat().st_size)

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for a given file."""
        # This is a placeholder implementation. You might want to implement
        # a more robust file hashing logic based on the file content.
        return str(file_path.stat().st_size) 