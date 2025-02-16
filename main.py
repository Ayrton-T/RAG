import os
import json
from pathlib import Path
from typing import List, Dict, Any
from src.db.hybrid_db import HybridDatabase
from src.db.config import SearchConfig as DBConfig
from src.chat.config import ChatConfig
from src.chat.chatbot import Chatbot
from src.tool.search_tools import init_tools
from src.db.file_db import FileDB
from src.parser.processor import FileProcessor
from src.chat.chat_history import ChatHistory  # Added import

import config  # used to set up environment variables

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
                "answer": msg["content"],
                "type": "qa_pair"
            })
            current_question = None
    
    return qa_pairs

def load_assistant_responses(data_dir: str = "data") -> list:
    """Load assistant responses from data directory."""
    responses = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for file_path in data_path.glob("*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Handle conversation format
            if "messages" in data:
                responses.extend(extract_qa_pairs(data["messages"]))
            # Handle direct QA format (for backward compatibility)
            elif isinstance(data, list):
                for item in data:
                    if "question" in item and "answer" in item:
                        responses.append(item)
            elif isinstance(data, dict) and "question" in data and "answer" in data:
                responses.append(data)
    
    return responses

def build_database():
    """Build the database from assistant responses."""
    # Load responses
    responses = load_assistant_responses()
    
    # Initialize database
    db = HybridDatabase("assistant-knowledge")
    
    # Process and add documents
    texts = []
    metadatas = []
    
    for i, response in enumerate(responses):
        if isinstance(response, dict) and "question" in response and "answer" in response:
            # Only add answers to the database
            texts.append(response["answer"])
            metadatas.append({
                "type": "answer",
                "qa_id": i,
                "original_text": response["answer"]
            })
    
    # Add documents to database
    if texts:
        db.add_documents(texts, metadatas)
    
    return db

def build_file_database():
    """Build database for uploaded files."""
    return FileDB(collection_name="uploaded-files", upload_dir="uploads")

def main():
    # Build databases
    qa_db = build_database()
    file_db = build_file_database()
    
    # Initialize file processor with token-based chunking parameters
    file_processor = FileProcessor(
        tokens_per_chunk=800,  # Size of each chunk in tokens
        overlap_tokens=400,  # Overlap between chunks in tokens
        upload_dir="uploads",
        file_db=file_db
    )
    
    # Initialize tools with both databases
    init_tools(qa_db, file_db)
    
    # Create chatbot config
    api_base = os.getenv("API_BASE", "https://api.openai.com/v1")
    chat_config = ChatConfig(
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        api_base=api_base,
        api_key=os.getenv("API_KEY"),
        min_iterations=3,
        max_iterations=5
    )
    
    # Initialize chatbot and history
    chatbot = Chatbot(chat_config)
    chat_history = ChatHistory()
    
    # Try to load existing history
    chat_history.load_history()
    
    # Set chat history in chatbot
    chatbot.set_chat_history(chat_history)
    
    try:
        while True:
            # Optional file upload
            file_path = input("\nEnter path to upload a file (or press Enter to skip): ").strip()
            if file_path:
                try:
                    # Process the file
                    processed_data = file_processor.process_file(file_path)
                    
                    # Save to FileDB and get metadata
                    result = file_processor.save_processed_file(processed_data)
                    
                    # Add to chat history
                    chat_history.add_uploaded_file(result["metadata"])
                    print(f"\nSuccessfully uploaded: {result['metadata']['file_name']}")
                    
                    # Save history to persist uploaded files
                    chat_history.save_history()
                except Exception as e:
                    print(f"\nError uploading file: {str(e)}")
            
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            print("\nSearch Mode (Forcing document search):")
            search_results = chatbot.search_mode(query)
            print(f"Improved query: {search_results['improved_query']}")
            print("\nSearch Results:")
            for result in search_results['results']['results']:
                print(f"- {result['content'][:200]}...")
            
            tool_results = []
            print("\nReasoning Mode:")
            final_answer = None
            
            for steps, total_time in chatbot.reasoning_mode(query, search_results):
                if total_time is not None:
                    print("\nFinal Answer:")
                    final_answer = steps[-1][1]
                    print(final_answer)
                    print(f"\nTotal thinking time: {total_time:.2f} seconds")
                else:
                    print(f"\n{steps[-1][0]}:")
                    print(steps[-1][1])
                    if steps[-1][3]:  # If tool was used
                        print(f"Using tool: {steps[-1][3]}")
                        print(f"Tool input: {steps[-1][4]}")
                        tool_result = steps[-1][5]
                        print("Tool result:", json.dumps(tool_result, indent=2))
                        tool_results.append({
                            "tool": steps[-1][3],
                            "input": steps[-1][4],
                            "result": tool_result
                        })
            
            # Add interaction to history
            if final_answer:
                chat_history.add_interaction(
                    query=query,
                    response=final_answer,
                    search_results=search_results,
                    tool_results=tool_results
                )
                
                # Save history after each interaction
                chat_history.save_history()
    
    except KeyboardInterrupt:
        print("\nSaving chat history before exit...")
        chat_history.save_history()
        print("Chat history saved. Goodbye!")
    finally:
        # Clear chat history file
        if os.path.exists("chat_history.json"):
            try:
                os.remove("chat_history.json")
                print("Chat history cleared.")
            except Exception as e:
                print(f"Error clearing chat history: {str(e)}")

if __name__ == "__main__":
    main()
