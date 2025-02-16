"""File parser module for handling different file types."""
from typing import Dict, Any, List
import os
from pathlib import Path
import PyPDF2
import docx
import magic
import hashlib

class FileParser:
    """Parser for different file types."""
    
    def __init__(self):
        self.supported_types = {
            'text/plain': self._parse_text,
            'application/pdf': self._parse_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._parse_docx
        }
    
    def parse_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Parse a file and return its content with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing:
            - content: Extracted text content
            - metadata: File metadata (type, size, etc.)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file type using python-magic
        file_type = magic.from_file(str(file_path), mime=True)
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Get file stats
        file_stats = file_path.stat()
        
        # Basic metadata
        metadata = {
            "file_name": file_path.name,
            "file_type": file_type,
            "file_size": file_stats.st_size,
            "created_at": file_stats.st_ctime,
            "modified_at": file_stats.st_mtime,
            "file_hash": file_hash,
            "num_pages": None,
            "word_count": None,
            "description": None  # Will be generated later
        }
        
        # Parse content based on file type
        if file_type in self.supported_types:
            content, extra_metadata = self.supported_types[file_type](file_path)
            metadata.update(extra_metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _parse_text(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count words
        word_count = len(content.split())
        
        return content, {
            "word_count": word_count,
            "num_pages": 1  # Text files are considered one page
        }
    
    def _parse_pdf(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse PDF file."""
        content = []
        word_count = 0
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                content.append(page_text)
                word_count += len(page_text.split())
        
        return '\n'.join(content), {
            "word_count": word_count,
            "num_pages": num_pages
        }
    
    def _parse_docx(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse DOCX file."""
        doc = docx.Document(file_path)
        content = []
        word_count = 0
        
        for paragraph in doc.paragraphs:
            text = paragraph.text
            content.append(text)
            word_count += len(text.split())
        
        return '\n'.join(content), {
            "word_count": word_count,
            "num_pages": len(doc.sections)  # Approximate page count
        } 