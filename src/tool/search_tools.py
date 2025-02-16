from typing import List, Dict, Any, Optional, Literal
import requests
from bs4 import BeautifulSoup
import json
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from ..db.hybrid_db import HybridDatabase
from ..db.file_db import FileDB
from ..db.config import SearchConfig

# Initialize global instances
ddg_search = DuckDuckGoSearchAPIWrapper(max_results=10)
qa_db: Optional[HybridDatabase] = None  # For QA pairs
file_db: Optional[FileDB] = None  # For uploaded files

def init_tools(qa_database: HybridDatabase, files_database: FileDB):
    """Initialize tools with both database instances.
    
    Args:
        qa_database: Database instance for QA pairs
        files_database: Database instance for uploaded files
    """
    global qa_db, file_db
    qa_db = qa_database
    file_db = files_database

def document_search(
    query: str | Dict,
    top_k: int = 5,
    search_type: Literal["vector_only", "vector_sparse", "full_hybrid"] = "full_hybrid"
) -> Dict[str, Any]:
    """Search through QA pairs using hybrid search.
    
    Args:
        query: Search query (string or dict with query parameters)
        top_k: Number of results to return
        search_type: Type of search to use
    
    Returns:
        Dictionary containing search results from QA database
    """
    if not qa_db:
        raise RuntimeError("QA database not initialized. Call init_tools first.")
    
    try:
        # Extract parameters if query is a dict
        if isinstance(query, dict):
            search_query = str(query.get('query', '')).strip()
            top_k = int(query.get('top_k', top_k))
            search_type = query.get('search_type', search_type)
        else:
            search_query = str(query).strip()
        
        # Validate query
        if not search_query:
            return {"results": [], "error": "Empty query"}
        
        # Create search config with top_k
        config = SearchConfig(final_top_k=top_k)
        
        # Search QA database
        results = qa_db.search(search_query, config)
        
        # Format results
        formatted_results = [
            {
                "content": r["content"],
                # "source": "qa_database",
                # "metadata": r.get("metadata", {})
            }
            for r in results[search_type]
        ]
        
        return {
            "results": formatted_results,
            # "total_found": len(formatted_results)
        }
        
    except Exception as e:
        return {
            "results": [],
            "error": f"QA search failed: {str(e)}"
        }

def web_search(query: str | Dict, num_results: int = 5) -> Dict[str, Any]:
    """Search the web using LangChain's DuckDuckGo integration."""
    try:
        # Extract parameters if query is a dict
        if isinstance(query, dict):
            search_query = str(query.get('query', '')).strip()
            num_results = int(query.get('num_results', num_results))
        else:
            search_query = str(query).strip()
        
        # Validate query
        if not search_query:
            return {"results": [], "error": "Empty query"}
        
        # Set max_results for this specific search
        ddg_search.max_results = num_results
        
        # Perform search with proper parameters
        results = ddg_search.run(search_query)  # Use run() instead of results()
        
        # Split results into individual snippets
        result_snippets = results.split('\n')
        
        # Format results
        formatted_results = [
            {
                "content": snippet.strip()
            }
            for snippet in result_snippets if snippet.strip()
        ][:num_results]
        
        return {
            "results": formatted_results,
            "query": search_query,
            "num_results": len(formatted_results)
        }
    except Exception as e:
        return {
            "error": f"Web search failed: {str(e)}",
            "results": []
        }

def web_scrape(url: str) -> Dict[str, Any]:
    """Scrape content from a webpage."""
    try:
        if not url.startswith(("https://", "http://")):
            url = "https://" + url
            
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Get main content
        text = soup.get_text()
        clean_text = text.splitlines()
        clean_text = [element.strip() for element in clean_text if element.strip()]
        clean_text = '\n'.join(clean_text)
        
        # Get title if available
        title = soup.title.string if soup.title else None
        
        return {
            "url": url,
            "title": title,
            "content": clean_text
        }
    except Exception as e:
        return {
            "error": f"Web scraping failed: {str(e)}",
            "url": url,
            "content": ""
        }

def file_search(
    query: str | Dict,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    search_type: Literal["vector_only", "vector_sparse", "full_hybrid"] = "full_hybrid"
) -> Dict[str, Any]:
    """Search through uploaded files using hybrid search with metadata filtering."""
    if not file_db:
        return {
            "results": [],
            "total_found": 0,
            "error": "File database not initialized. Call init_tools first."
        }
    
    try:
        # Extract parameters if query is a dict
        if isinstance(query, dict):
            search_query = str(query.get('query', '')).strip()
            top_k = int(query.get('top_k', top_k))
            search_type = query.get('search_type', search_type)
            filters = query.get('filters', filters)
        else:
            search_query = str(query).strip()
        
        # Validate query
        if not search_query:
            return {
                "results": [],
                "total_found": 0,
                "error": "Empty query"
            }
        
        # Create search config with top_k
        config = SearchConfig(final_top_k=top_k)
        
        # Process filters
        processed_filters = {}
        
        # Handle file types
        if 'file_type' in filters:
            file_types = filters['file_type']
            if isinstance(file_types, str):
                file_types = [file_types]
            
            # Map common extensions and normalize MIME types
            mime_map = {
                'txt': 'text/plain',
                'md': 'text/plain',
                'markdown': 'text/plain',
                'doc': 'application/msword',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'pdf': 'application/pdf'
            }
            
            normalized_types = []
            for ft in file_types:
                ft = ft.lower().strip('.')
                if ft in mime_map:
                    normalized_types.append(mime_map[ft])
                elif '/' in ft:  # Already a MIME type
                    normalized_types.append(ft)
                else:
                    normalized_types.append(f"text/{ft}")
            
            processed_filters['file_type'] = normalized_types
        
        # Handle file names
        if 'file_name' in filters:
            file_names = filters['file_name']
            if isinstance(file_names, str):
                file_names = [file_names]
            processed_filters['file_name'] = file_names
        
        results = file_db.search_with_filters(search_query, processed_filters, config)
        
        # Check for search errors
        if "error" in results:
            return {
                "results": [],
                "total_found": 0,
                "error": results["error"]
            }
        
        # Validate results
        if not results or search_type not in results:
            return {
                "results": [],
                "total_found": 0,
                "error": f"No results found for search type: {search_type}"
            }
        
        # Get results for the specified search type
        search_results = results[search_type]
        if not search_results:
            return {
                "results": [],
                "total_found": 0,
                "error": "No results found"
            }
        
        # Format results
        formatted_results = []
        for result in search_results:
            if not isinstance(result, dict) or "content" not in result:
                continue
                
            formatted_results.append({
                "content": result["content"],
                "source": "file_database",
                "metadata": result.get("metadata", {}),
                "file_info": {
                    "file_name": result.get("metadata", {}).get("file_name", ""),
                    "file_type": result.get("metadata", {}).get("file_type", ""),
                    "word_count": result.get("metadata", {}).get("word_count", 0),
                    "num_pages": result.get("metadata", {}).get("num_pages", 1)
                }
            })
        
        return {
            "results": formatted_results,
            "total_found": len(formatted_results)
        }
        
    except Exception as e:
        return {
            "results": [],
            "total_found": 0,
            "error": f"File search failed: {str(e)}"
        }

# Tool definitions in OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "document_search",
            "description": "Search through the QA database using hybrid search (vector + BM25).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search to use",
                        "enum": ["vector_only", "vector_sparse", "full_hybrid"],
                        "default": "full_hybrid"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search through uploaded files using hybrid search with metadata filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters",
                        "properties": {
                            "file_type": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file types to filter by"
                            },
                            "file_id": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file IDs to filter by"
                            },
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string"},
                                    "end": {"type": "string"}
                                },
                                "description": "Date range for filtering"
                            },
                            "min_tokens": {
                                "type": "integer",
                                "description": "Minimum number of tokens per chunk"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum number of tokens per chunk"
                            },
                            "page_range": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"}
                                },
                                "description": "Page range for filtering"
                            }
                        }
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search to use",
                        "enum": ["vector_only", "vector_sparse", "full_hybrid"],
                        "default": "full_hybrid"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo through LangChain integration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": "Fetch and extract content from a webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the webpage to scrape. Can include 'https://' or just the domain"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# Dictionary mapping tool names to their functions
available_tools = {
    "document_search": document_search,
    "file_search": file_search,
    "web_search": web_search,
    "web_scrape": web_scrape
} 