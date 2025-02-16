from typing import List, Dict, Any, Optional, Generator, Tuple, Literal
import time
import json
from dataclasses import dataclass
from openai import OpenAI
from .config import ChatConfig
from ..tool.search_tools import available_tools

@dataclass
class Reasoning:
    """Structure for reasoning steps."""
    title: str
    content: str
    next_action: Literal["continue", "final_answer"]
    tool: Optional[Literal["document_search", "web_search", "web_scrape"]] = None
    tool_input: Optional[str] = None

class LLMClient:
    """OpenAI-compatible LLM client wrapper."""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.api_base,
            api_key=config.api_key
        )
            
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        is_json: bool = False
    ) -> str:
        """Get chat completion from the LLM using OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"} if is_json else None
        )
        return response.choices[0].message.content

class Chatbot:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.llm = LLMClient(config)
        self.chat_history = None
        
    def set_chat_history(self, chat_history):
        """Set the chat history instance for context."""
        self.chat_history = chat_history
    
    def improve_search_query(self, query: str) -> str:
        """Use LLM to improve the search query."""
        messages = [
            {"role": "system", "content": self.config.search_system_prompt},
            {"role": "user", "content": query}
        ]
        
        improved_query = self.llm.chat_completion(messages, temperature=0.2)
        return improved_query.strip()
    
    def search_mode(self, query: str) -> Dict[str, Any]:
        """Search mode that forces document search."""
        # Improve the query first
        improved_query = self.improve_search_query(query)
        
        # Perform document search
        search_results = available_tools["document_search"](improved_query)
        
        return {
            "original_query": query,
            "improved_query": improved_query,
            "results": search_results
        }
    
    def reasoning_mode(
        self,
        query: str,
        search_results: Optional[Dict[str, Any]] = None
    ) -> Generator[Tuple[List[Tuple[str, str, float, Optional[str], Optional[str], Optional[Dict]]], Optional[float]], None, None]:
        """Reasoning mode that follows step-by-step thinking."""
        # Format chat history and uploaded files for context
        chat_context = ""
        files_metadata = ""
        search_context = ""
        
        if self.chat_history:
            # Get formatted chat history
            history = self.chat_history.get_formatted_history()
            if history:
                chat_context = "Previous relevant interactions:\n"
                for msg in history[-3:]:  # Only include last 3 interactions
                    chat_context += f"{msg['role'].title()}: {msg['content']}\n"
            
            # Get uploaded files metadata
            if self.chat_history.uploaded_files:
                files_metadata = "Available uploaded files:\n"
                for file in self.chat_history.uploaded_files:
                    files_metadata += (
                        f"- {file['file_name']} ({file['file_type']})\n"
                        f"  Pages: {file.get('num_pages', 1)}, "
                        f"Words: {file.get('word_count', 0)}\n"
                        f"  Description: {file.get('description', 'No description')}\n"
                    )
        
        # Add search results context if available
        if search_results:
            search_context = "\nInitial search results:\n"
            if "improved_query" in search_results:
                search_context += f"Improved query: {search_results['improved_query']}\n\n"
            if "results" in search_results and "results" in search_results["results"]:
                for i, result in enumerate(search_results["results"]["results"], 1):
                    content = result.get("content", "")
                    if content:
                        search_context += f"{i}. {content[:200]}...\n"
        
        # Format the system prompt with context
        system_prompt = self.config.reasoning_system_prompt.format(
            context=chat_context + search_context,
            files_metadata=files_metadata
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "I will now think through this step by step."}
        ]
        
        steps = []
        step_count = 1
        total_thinking_time = 0
        
        # Limit iterations to max_iterations
        while step_count <= self.config.max_iterations:
            # Get next reasoning step
            start_time = time.time()
            try:
                step_text = self.llm.chat_completion(messages, temperature=0.2, is_json=True)
                step_data = Reasoning(**json.loads(step_text))
            except Exception as e:
                step_data = Reasoning(
                    title="Error in Reasoning",
                    content=f"Failed to process step: {str(e)}",
                    next_action="final_answer"
                )
            
            end_time = time.time()
            thinking_time = end_time - start_time
            total_thinking_time += thinking_time
            
            # Handle tool usage if any
            tool_result = None
            if step_data.tool and step_data.tool_input:
                try:
                    tool_result = available_tools[step_data.tool](step_data.tool_input)
                except Exception as e:
                    tool_result = {"error": f"Tool execution failed: {str(e)}"}
            
            # Record step
            steps.append((
                f"Step {step_count}: {step_data.title}",
                step_data.content,
                thinking_time,
                step_data.tool,
                step_data.tool_input,
                tool_result
            ))
            
            # Add step to conversation history
            messages.append({"role": "assistant", "content": json.dumps(step_data.__dict__)})
            if tool_result:
                messages.append({"role": "system", "content": f"Tool result: {json.dumps(tool_result)}"})
            
            # Check if we should continue
            if step_data.next_action == "final_answer":
                if step_count < self.config.min_iterations:
                    messages.append({
                        "role": "system",
                        "content": f"Please continue reasoning. At least {self.config.min_iterations} steps are required."
                    })
                    step_count += 1
                    yield steps, None
                    continue
                break
            
            step_count += 1
            yield steps, None
        
        # Generate final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on the reasoning above."
        })
        
        start_time = time.time()
        final_answer = self.llm.chat_completion(messages, temperature=0.2)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append(("Final Answer", final_answer, thinking_time))
        
        yield steps, total_thinking_time 