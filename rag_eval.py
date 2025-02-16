"""RAG evaluation script to compare different database configurations."""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.db.hybrid_db import HybridDatabase
from src.db.file_db import FileDB
from src.db.config import SearchConfig
from src.chat.chatbot import Chatbot
from src.chat.config import ChatConfig
from src.tool.search_tools import init_tools

import config  # set up the environment variables

def extract_qa_pairs(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Extract QA pairs from a conversation."""
    qa_pairs = []
    current_question = None
    
    for msg in messages:
        if msg["role"] == "user":
            current_question = msg["content"]
        elif msg["role"] == "assistant" and current_question is not None:
            qa_pairs.append({
                "question": current_question,
                "answer": msg["content"]
            })
            current_question = None
    
    return qa_pairs

def load_qa_pairs(data_dir: str = "data") -> List[Dict[str, str]]:
    """Load QA pairs from data directory."""
    qa_pairs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for file_path in data_path.glob("*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Handle conversation format
            if "messages" in data:
                qa_pairs.extend(extract_qa_pairs(data["messages"]))
            # Handle direct QA format (for backward compatibility)
            elif isinstance(data, list):
                for item in data:
                    if "question" in item and "answer" in item:
                        qa_pairs.append(item)
            elif isinstance(data, dict) and "question" in data and "answer" in data:
                qa_pairs.append(data)
    
    return qa_pairs

def build_database(qa_pairs: List[Dict[str, str]], config: SearchConfig) -> HybridDatabase:
    """Build database with given configuration. Only stores answers, not questions."""
    db = HybridDatabase("qa-knowledge", config)
    
    # Extract texts and metadata (only answers)
    texts = []
    metadatas = []
    
    for i, qa in enumerate(qa_pairs):
        # Only add answers to the database
        texts.append(qa["answer"])
        metadatas.append({
            "type": "answer",
            "qa_id": i,
            "original_text": qa["answer"]
        })
    
    # Add documents to database
    if texts:
        db.add_documents(texts, metadatas)
    
    return db

def evaluate_retrieval(
    chatbot: Chatbot,
    qa_pairs: List[Dict[str, str]],
    k: int,
    db: HybridDatabase
) -> Dict[str, float]:
    """Evaluate retrieval performance using both original and improved queries."""
    total = len(qa_pairs)
    # Results with improved queries
    successful_vector_improved = 0
    successful_hybrid_improved = 0
    successful_full_improved = 0
    # Results with original queries
    successful_vector_original = 0
    successful_hybrid_original = 0
    successful_full_original = 0
    improved_queries = []
    
    for i, qa in enumerate(tqdm(qa_pairs, desc="Evaluating queries", leave=False)):
        query = qa["question"]
        current_qa_id = str(i)
        
        # Test with original query first
        config = SearchConfig(final_top_k=k)
        original_results = db.search(query, config)
        
        # Check original query results
        retrieved_ids = {str(result["qa_id"]) for result in original_results["vector_only"]}
        if current_qa_id in retrieved_ids:
            successful_vector_original += 1
            
        retrieved_ids = {str(result["qa_id"]) for result in original_results["vector_sparse"]}
        if current_qa_id in retrieved_ids:
            successful_hybrid_original += 1
            
        retrieved_ids = {str(result["qa_id"]) for result in original_results["full_hybrid"]}
        if current_qa_id in retrieved_ids:
            successful_full_original += 1
        
        # Test with improved query
        improved_query = chatbot.improve_search_query(query)
        improved_queries.append(improved_query)
        improved_results = db.search(improved_query, config)
        
        # Check improved query results
        retrieved_ids = {str(result["qa_id"]) for result in improved_results["vector_only"]}
        if current_qa_id in retrieved_ids:
            successful_vector_improved += 1
            
        retrieved_ids = {str(result["qa_id"]) for result in improved_results["vector_sparse"]}
        if current_qa_id in retrieved_ids:
            successful_hybrid_improved += 1
            
        retrieved_ids = {str(result["qa_id"]) for result in improved_results["full_hybrid"]}
        if current_qa_id in retrieved_ids:
            successful_full_improved += 1
    
    return {
        "original": {
            "vector_only": {
                "success_rate": successful_vector_original / total,
                "error_rate": 1 - (successful_vector_original / total)
            },
            "vector_sparse": {
                "success_rate": successful_hybrid_original / total,
                "error_rate": 1 - (successful_hybrid_original / total)
            },
            "full_hybrid": {
                "success_rate": successful_full_original / total,
                "error_rate": 1 - (successful_full_original / total)
            }
        },
        "improved": {
            "vector_only": {
                "success_rate": successful_vector_improved / total,
                "error_rate": 1 - (successful_vector_improved / total)
            },
            "vector_sparse": {
                "success_rate": successful_hybrid_improved / total,
                "error_rate": 1 - (successful_hybrid_improved / total)
            },
            "full_hybrid": {
                "success_rate": successful_full_improved / total,
                "error_rate": 1 - (successful_full_improved / total)
            }
        },
        "improved_queries": improved_queries
    }

def run_evaluations() -> Tuple[Dict[str, List[float]], List[int], Dict[str, List[str]]]:
    """Run evaluations with different configurations."""
    # Load QA pairs
    qa_pairs = load_qa_pairs()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable must be set")
    
    # Initialize chatbot for query improvement
    chat_config = ChatConfig(
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        api_base=os.getenv("API_BASE", "https://api.openai.com/v1"),
        api_key=api_key
    )
    chatbot = Chatbot(chat_config)
    
    # Define k values to test
    k_values = [3, 5, 10]
    
    # Store results
    results = {
        "Vector Only (Original)": [],
        "Vector + BM25 (Original)": [],
        "Full Hybrid (Original)": [],
        "Vector Only (Improved)": [],
        "Vector + BM25 (Improved)": [],
        "Full Hybrid (Improved)": []
    }
    improved_queries = []
    
    print(f"Total number of QA pairs: {len(qa_pairs)}")
    
    # Create database with all features enabled
    db_config = SearchConfig()
    db = build_database(qa_pairs, db_config)
    file_db = FileDB()  # won't be used in this script
    init_tools(db, file_db)
    
    # Test different k values
    for k in tqdm(k_values, desc="Testing k values"):
        eval_results = evaluate_retrieval(
            chatbot=chatbot,
            qa_pairs=qa_pairs,
            k=k,
            db=db
        )
        
        # Store results for each configuration
        results["Vector Only (Original)"].append(eval_results["original"]["vector_only"]["error_rate"])
        results["Vector + BM25 (Original)"].append(eval_results["original"]["vector_sparse"]["error_rate"])
        results["Full Hybrid (Original)"].append(eval_results["original"]["full_hybrid"]["error_rate"])
        results["Vector Only (Improved)"].append(eval_results["improved"]["vector_only"]["error_rate"])
        results["Vector + BM25 (Improved)"].append(eval_results["improved"]["vector_sparse"]["error_rate"])
        results["Full Hybrid (Improved)"].append(eval_results["improved"]["full_hybrid"]["error_rate"])
        
        # Store improved queries
        improved_queries.extend(eval_results["improved_queries"])
    
    return results, k_values, improved_queries

def plot_results(results: Dict[str, List[float]], k_values: List[int]):
    """Plot error rates for different configurations."""
    plt.figure(figsize=(15, 10))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define colors and line styles
    colors = {
        "Vector Only": "#1f77b4",
        "Vector + BM25": "#ff7f0e",
        "Full Hybrid": "#2ca02c"
    }
    
    # Plot lines with consistent colors and different line styles for original vs improved
    for method in ["Vector Only", "Vector + BM25", "Full Hybrid"]:
        plt.plot(k_values, results[f"{method} (Original)"], 
                marker='o', linestyle='--', 
                color=colors[method], 
                label=f"{method} (Original)")
        plt.plot(k_values, results[f"{method} (Improved)"], 
                marker='s', linestyle='-', 
                color=colors[method], 
                label=f"{method} (Improved)")
    
    # Customize plot
    plt.xlabel("Top-k Results")
    plt.ylabel("Error Rate")
    plt.title("RAG Retrieval Error Rates\n(Original vs Query-Improved)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("rag_eval_results.png", bbox_inches='tight', dpi=300)
    plt.close()

def save_query_improvements(improved_queries: List[str], qa_pairs: List[Dict[str, str]]):
    """Save query improvements for analysis."""
    query_analysis = [
        {
            "original": qa["question"],
            "improved": improved,
            "answer": qa["answer"]
        }
        for qa, improved in zip(qa_pairs, improved_queries)
    ]
    
    with open("query_improvements.json", "w") as f:
        json.dump(query_analysis, f, indent=2)

def main():
    """Run the RAG evaluation experiment."""
    print("Starting RAG evaluation experiment...")
    
    # Run evaluations
    results, k_values, improved_queries = run_evaluations()
    
    # Plot results
    plot_results(results, k_values)
    
    # Save query improvements
    qa_pairs = load_qa_pairs()
    save_query_improvements(improved_queries, qa_pairs)
    
    # Print summary
    print("\nEvaluation Results:")
    print("-" * 50)
    for config_name, error_rates in results.items():
        print(f"\n{config_name}:")
        print(f"Best k: {k_values[np.argmin(error_rates)]}")
        print(f"Best error rate: {min(error_rates):.4f}")
        print(f"Average error rate: {np.mean(error_rates):.4f}")
    
    print("\nQuery improvements have been saved to 'query_improvements.json'")

if __name__ == "__main__":
    main()
