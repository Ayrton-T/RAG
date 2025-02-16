from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class SearchConfig:
    """Configuration for advanced hybrid search settings."""
    # Model names
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_model: str = "Qdrant/bm25"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Search parameters
    final_top_k: int = 10  # Final number of results to return
    
    # Scoring weights for hybrid search
    dense_weight: float = 0.8  # Semantic weight
    bm25_weight: float = 0.2   # BM25 weight
    
    # Vector names in Qdrant collection
    vector_names: Dict[str, str] = None
    
    def __post_init__(self):
        if self.vector_names is None:
            self.vector_names = {
                "dense": "all-MiniLM-L6-v2",
                "sparse": "bm25"
            }
    
    # Debug options
    return_source_info: bool = True  # Whether to return info about which search method found each result 