from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from qdrant_client.http import models
from .hybrid_db import HybridDatabase
from .config import SearchConfig

class FileDB(HybridDatabase):
    """Specialized database for storing and searching uploaded file chunks with metadata filtering."""
    
    def __init__(
        self,
        collection_name: str = "uploaded-files",
        config: Optional[SearchConfig] = None,
        upload_dir: Optional[str] = "uploads"
    ):
        """Initialize database for uploaded files.
        
        Args:
            collection_name: Name of the Qdrant collection
            config: Search configuration
            upload_dir: Directory for storing uploaded files
        """
        super().__init__(collection_name, config)
        self.upload_dir = Path(upload_dir) if upload_dir else None
        if self.upload_dir:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def add_file_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_metadata: Dict[str, Any]
    ) -> None:
        """Add chunks from a processed file.
        
        Args:
            chunks: List of chunks with content and metadata
            file_metadata: Common metadata for all chunks from this file
        """
        texts = []
        metadatas = []
        
        for chunk in chunks:
            texts.append(chunk["content"])
            # Combine file metadata with chunk metadata
            metadata = {
                "type": "file_chunk",
                "file_name": file_metadata["file_name"],
                "file_type": file_metadata["file_type"],
                "chunk": chunk["metadata"]["chunk"],
                "original_text": chunk["content"],
                # Add additional metadata fields for filtering
                "created_at": file_metadata.get("created_at"),
                "modified_at": file_metadata.get("modified_at"),
                "word_count": file_metadata.get("word_count"),
                "num_pages": file_metadata.get("num_pages"),
                "source_page": chunk["metadata"]["chunk"].get("source_page", 1),
                "token_count": chunk["metadata"]["chunk"].get("token_count", 0)
            }
            metadatas.append(metadata)
        
        self.add_documents(texts, metadatas)
    
    def search_with_filters(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        config: Optional[SearchConfig] = None
    ) -> Dict[str, Any]:
        """Search through documents with metadata filters.
        
        Args:
            query: Search query
            filters: Dictionary of metadata filters:
                - file_type: List[str] - Filter by file types
                - file_name: List[str] - Filter by file names
            config: Optional search configuration override
            
        Returns:
            Dictionary containing search results
        """
        config = config or self.config
        
        try:
            # Generate query embeddings
            dense_query = next(self.dense_model.embed([query]))
            sparse_query = next(self.bm25_model.embed([query]))
            
            # Build filter conditions
            filter_conditions = []
            if filters:
                for key, value in filters.items():
                    if not value:
                        continue
                        
                    if key in ["file_type", "file_name"]:
                        # Match conditions for lists
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
            
            # Create combined filter
            search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            # Get double the results for initial recall
            initial_limit = config.final_top_k * 2
            
            # 1. Vector search with filters
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dense_query.tolist(),
                limit=initial_limit,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Handle empty results
            if not results:
                return {
                    "vector_only": [],
                    "vector_sparse": [],
                    "full_hybrid": []
                }
            
            # Store vector-only results
            vector_results = []
            for result in results[:config.final_top_k]:
                content = result.payload.get("original_text") or result.payload.get("text", "")
                # Filter out sparse indices and values from metadata
                metadata = {k: v for k, v in result.payload.items() 
                          if k not in ["text", "sparse_indices", "sparse_values"]}
                vector_results.append({
                    "content": content,
                    "metadata": metadata
                })
            
            # 2. Vector + Sparse search
            query_indices = sparse_query.indices.tolist()
            query_values = sparse_query.values.tolist()
            
            # Calculate sparse scores for all results
            for result in results:
                if "sparse_indices" in result.payload and "sparse_values" in result.payload:
                    doc_indices = result.payload["sparse_indices"]
                    doc_values = result.payload["sparse_values"]
                    
                    # Calculate dot product for matching indices
                    sparse_score = 0.0
                    i, j = 0, 0
                    while i < len(query_indices) and j < len(doc_indices):
                        if query_indices[i] == doc_indices[j]:
                            sparse_score += query_values[i] * doc_values[j]
                            i += 1
                            j += 1
                        elif query_indices[i] < doc_indices[j]:
                            i += 1
                        else:
                            j += 1
                    
                    # Combine scores (weighted average)
                    result.score = (
                        config.dense_weight * result.score +
                        config.bm25_weight * sparse_score
                    )
            
            # Sort by combined score
            results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # Store vector + sparse results
            vector_sparse_results = []
            for result in results[:config.final_top_k]:
                content = result.payload.get("original_text") or result.payload.get("text", "")
                # Filter out sparse indices and values from metadata
                metadata = {k: v for k, v in result.payload.items() 
                          if k not in ["text", "sparse_indices", "sparse_values"]}
                vector_sparse_results.append({
                    "content": content,
                    "metadata": metadata
                })
            
            # 3. Full hybrid with reranking
            pairs = [[query, result.payload.get("original_text") or result.payload.get("text", "")] 
                    for result in results[:initial_limit]]
            
            # Handle empty pairs
            if not pairs:
                return {
                    "vector_only": vector_results,
                    "vector_sparse": vector_sparse_results,
                    "full_hybrid": []
                }
            
            rerank_scores = self.reranker.predict(pairs)
            
            # Create reranked results
            reranked = list(zip(results[:initial_limit], rerank_scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k reranked results
            full_hybrid_results = []
            for result, _ in reranked[:config.final_top_k]:
                content = result.payload.get("original_text") or result.payload.get("text", "")
                # Filter out sparse indices and values from metadata
                metadata = {k: v for k, v in result.payload.items() 
                          if k not in ["text", "sparse_indices", "sparse_values"]}
                full_hybrid_results.append({
                    "content": content,
                    "metadata": metadata
                })
            
            return {
                "vector_only": vector_results,
                "vector_sparse": vector_sparse_results,
                "full_hybrid": full_hybrid_results
            }
            
        except Exception as e:
            return {
                "vector_only": [],
                "vector_sparse": [],
                "full_hybrid": [],
                "error": f"Search failed: {str(e)}"
            }
    