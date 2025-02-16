from dataclasses import dataclass
from typing import Optional
from ..prompts import SEARCH_SYSTEM_PROMPT, REASONING_SYSTEM_PROMPT

@dataclass
class ChatConfig:
    """Configuration for the chatbot."""
    # Model settings (OpenAI-compatible)
    model_name: str = "gpt-4o"  # Default to gpt-4o
    api_base: str = "https://api.openai.com/v1"  # Can be changed for other providers
    api_key: Optional[str] = None
    
    # Search settings
    min_iterations: int = 3
    max_iterations: int = 5
    temperature: float = 0.2
    
    # System prompts
    search_system_prompt: str = SEARCH_SYSTEM_PROMPT
    reasoning_system_prompt: str = REASONING_SYSTEM_PROMPT 