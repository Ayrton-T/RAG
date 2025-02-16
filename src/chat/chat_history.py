"""Chat history management module."""
import os
import json
from typing import List, Dict, Any, Deque
from collections import deque

class ChatHistory:
    """Maintains chat history with a fixed size."""
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.uploaded_files: List[Dict[str, Any]] = []  # Track uploaded files metadata
        
    def add_interaction(self, 
        query: str, 
        response: str, 
        search_results: Dict[str, Any] = None,
        tool_results: List[Dict[str, Any]] = None
    ):
        """Add a new interaction to the history."""
        interaction = {
            "query": query,
            "response": response,
            "timestamp": None,  # Could add timestamp if needed
            "search_results": search_results,
            "tool_results": tool_results
        }
        self.history.append(interaction)
    
    def add_uploaded_file(self, file_metadata: Dict[str, Any]):
        """Add metadata for an uploaded file."""
        self.uploaded_files.append(file_metadata)
    
    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Get history formatted for LLM context."""
        formatted_history = []
        for interaction in self.history:
            # Add user query
            formatted_history.append({
                "role": "user",
                "content": interaction["query"]
            })
            
            # Add search results if available
            if interaction["search_results"]:
                formatted_history.append({
                    "role": "system",
                    "content": f"Search results: {json.dumps(interaction['search_results'])}"
                })
            
            # Add tool results if available
            if interaction["tool_results"]:
                formatted_history.append({
                    "role": "system",
                    "content": f"Tool results: {json.dumps(interaction['tool_results'])}"
                })
            
            # Add assistant response
            formatted_history.append({
                "role": "assistant",
                "content": interaction["response"]
            })
        
        return formatted_history
    
    def save_history(self, filepath: str = "chat_history.json"):
        """Save chat history and uploaded files metadata to file."""
        data = {
            "history": list(self.history),
            "uploaded_files": self.uploaded_files
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_history(self, filepath: str = "chat_history.json"):
        """Load chat history and uploaded files metadata from file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                    # Handle both old format (list) and new format (dict)
                    if isinstance(data, list):
                        self.history = deque(data, maxlen=self.max_history)
                        self.uploaded_files = []
                    else:
                        self.history = deque(data.get("history", []), maxlen=self.max_history)
                        self.uploaded_files = data.get("uploaded_files", [])
                except Exception as e:
                    print(f"Error loading chat history: {str(e)}")