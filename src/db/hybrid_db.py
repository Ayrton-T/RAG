from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import CrossEncoder
from fastembed import TextEmbedding, SparseTextEmbedding
import requests
from .config import SearchConfig

class HybridDatabase:
    def __init__(
        self,
        collection_name: str,
        config: Optional[SearchConfig] = None,
    ):
        """Initialize the hybrid database with in-memory Qdrant.
        
        Args:
            collection_name: Name of the Qdrant collection
            config: Search configuration
        
        Raises:
            ConnectionError: If cannot connect to local Qdrant
        """
        self.config = config or SearchConfig()
        self.collection_name = collection_name
        
        # Initialize in-memory Qdrant client
        self.client = QdrantClient(":memory:")
        
        # Initialize embedding models
        self.dense_model = TextEmbedding(self.config.dense_model)
        self.bm25_model = SparseTextEmbedding(self.config.bm25_model)
        
        # Initialize reranker
        self.reranker = CrossEncoder(self.config.rerank_model)
        
        # Create collection with appropriate configuration
        self._create_collection()
        
    def _create_collection(self):
        """Create Qdrant collection with appropriate vector configurations."""
        try:
            # Get sample embeddings to determine vector sizes
            sample_text = "Sample text for initialization"
            dense_emb = next(self.dense_model.embed([sample_text]))
            
            # Create collection
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(dense_emb),
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={"bm25": {}},
                hnsw_config=models.HnswConfigDiff(
                    m=16,  # Number of edges per node in the index graph
                    ef_construct=100,  # Number of neighbours to consider during the index building
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")
            
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "vector_size": collection_info.config.params.vectors.get(
                    self.config.vector_names["dense"]
                ).size,
                "has_sparse": bool(collection_info.config.params.sparse_vectors)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get collection info: {str(e)}")

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents with multiple embedding types.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        # Generate embeddings
        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.bm25_model.embed(texts))
        
        # Prepare points
        points = []
        for i, (text, dense_emb) in enumerate(zip(texts, dense_embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata["text"] = text
            metadata["doc_id"] = metadata.get("doc_id", str(i))
            metadata["original_index"] = metadata.get("original_index", i)
            
            # Create base point data
            point_data = {
                "id": i,
                "vector": dense_emb.tolist(),
                "payload": metadata
            }
            
            # Add sparse vectors
            sparse_dict = sparse_embeddings[i]
            metadata["sparse_indices"] = sparse_dict.indices.tolist()
            metadata["sparse_values"] = sparse_dict.values.tolist()
            
            points.append(models.PointStruct(**point_data))
        
        # Upload all points in batch
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def _compute_rank_fusion_scores(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        config: SearchConfig
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Compute rank fusion scores using weighted reciprocal rank fusion."""
        # Extract IDs and create unique set
        dense_ids = [(hit.payload["doc_id"], hit.payload["original_index"]) for hit in dense_results]
        bm25_ids = [(hit.payload["doc_id"], hit.payload["original_index"]) for hit in bm25_results]
        all_ids = list(set(dense_ids + bm25_ids))
        
        # Calculate initial weighted scores (vector search is always used)
        id_to_score = {}
        for chunk_id in all_ids:
            # Initialize with vector search score
            score = 0
            is_from_dense = chunk_id in dense_ids
            is_from_bm25 = chunk_id in bm25_ids
            
            # Vector search is always used and weighted
            if is_from_dense:
                idx = dense_ids.index(chunk_id)
                score += config.dense_weight * (1 / (idx + 1))
            
            # Add BM25 score if enabled and found
            if config.use_bm25 and is_from_bm25:
                idx = bm25_ids.index(chunk_id)
                score += config.bm25_weight * (1 / (idx + 1))
            
            id_to_score[chunk_id] = {
                "score": score,
                "from_dense": is_from_dense,
                "from_bm25": is_from_bm25
            }
        
        # Sort by score and assign new reciprocal rank scores
        sorted_ids = sorted(
            id_to_score.keys(),
            key=lambda x: (id_to_score[x]["score"], x[0], x[1]),
            reverse=True
        )
        
        # Update scores based on final rank
        for rank, chunk_id in enumerate(sorted_ids):
            id_to_score[chunk_id]["score"] = 1 / (rank + 1)
        
        return id_to_score

    def search(
        self,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> Dict[str, Any]:
        """Perform all search conditions at once and return results for different configurations.
        Always returns all three types of results:
        1. Vector search top k results
        2. Vector + sparse search + rank fusion top k results
        3. Vector + sparse search + rank fusion + rerank top k results
        
        Args:
            query: Search query
            config: Optional search configuration override
            
        Returns:
            Dictionary containing results for different search configurations:
            {
                "vector_only": List of top-k results from vector search,
                "vector_sparse": List of top-k results from vector + sparse search,
                "full_hybrid": List of top-k results from vector + sparse + rerank
            }
        """
        config = config or self.config
        
        # Generate query embeddings
        dense_query = next(self.dense_model.embed([query]))
        sparse_query = next(self.bm25_model.embed([query]))
        
        # Get double the results for initial recall to ensure quality after filtering
        initial_limit = config.final_top_k * 2
        
        # 1. Vector search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=dense_query.tolist(),
            limit=initial_limit,
            with_payload=True
        )
        
        # Store vector-only results
        vector_results = []
        for result in results[:config.final_top_k]:
            content = result.payload.get("original_text") or result.payload.get("text", "")
            qa_id = result.payload.get("qa_id")
            vector_results.append({"content": content, "qa_id": qa_id})
        
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
            qa_id = result.payload.get("qa_id")
            vector_sparse_results.append({"content": content, "qa_id": qa_id})
        
        # 3. Full hybrid with reranking
        pairs = [[query, result.payload.get("original_text") or result.payload.get("text", "")] 
                for result in results[:initial_limit]]
        rerank_scores = self.reranker.predict(pairs)
        
        # Create reranked results
        reranked = list(zip(results[:initial_limit], rerank_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k reranked results
        full_hybrid_results = []
        for result, _ in reranked[:config.final_top_k]:
            content = result.payload.get("original_text") or result.payload.get("text", "")
            qa_id = result.payload.get("qa_id")
            full_hybrid_results.append({"content": content, "qa_id": qa_id})
        
        return {
            "vector_only": vector_results,
            "vector_sparse": vector_sparse_results,
            "full_hybrid": full_hybrid_results
        }