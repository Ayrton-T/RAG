import os
import json
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from src.db.hybrid_db import HybridDatabase
from src.db.file_db import FileDB
from src.chat.config import ChatConfig
from src.chat.chatbot import Chatbot
from src.tool.search_tools import init_tools
from src.parser.processor import FileProcessor
from src.chat.chat_history import ChatHistory

import config  # used to set up environment variables

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
            # Handle direct QA format
            elif isinstance(data, list):
                for item in data:
                    if "question" in item and "answer" in item:
                        responses.append(item)
            elif isinstance(data, dict) and "question" in data and "answer" in data:
                responses.append(data)
    
    return responses

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

def format_search_results(search_results: Dict[str, Any]) -> str:
    """Format search results for display in a clean, structured way."""
    formatted = f"🔍 **Improved Query**:\n{search_results['improved_query']}\n\n"
    formatted += "📚 **Search Results**:\n"
    for i, result in enumerate(search_results['results']['results'], 1):
        content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        formatted += f"\n{i}. {content}\n"
    return formatted

def format_tool_result(tool: str, tool_input: str, result: Dict[str, Any]) -> str:
    """Format tool result for display in a clean, structured way."""
    formatted = f"🛠️ **Tool Used**: {tool}\n"
    formatted += f"📥 **Input**: {tool_input}\n\n"
    
    if isinstance(result, dict):
        if "error" in result:
            # Format error message to be more user-friendly
            error_msg = result["error"]
            if "Web search failed" in error_msg:
                formatted += "❌ The web search service is currently unavailable. Using internal knowledge base instead."
            else:
                formatted += f"❌ {error_msg}"
            return formatted
        
        formatted += "📤 **Results**:\n\n"
        
        if "results" in result:
            # Special handling for search results
            for i, item in enumerate(result["results"], 1):
                if isinstance(item, dict):
                    # Only display content, skip qa_id and other metadata
                    content = item.get("content", "")
                    if content:
                        formatted += f"{i}. {content}\n\n"
        else:
            # Format other dictionary results
            for key, value in result.items():
                if key == "qa_id":  # Skip qa_id
                    continue
                if isinstance(value, (list, dict)):
                    formatted += f"\n• {key}:\n"
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                # Skip qa_id and only show content for dict items
                                content = item.get("content", "")
                                if content:
                                    formatted += f"    - {content}\n"
                            else:
                                formatted += f"    - {item}\n"
                    else:  # dict
                        # Skip qa_id in nested dictionaries
                        content = value.get("content", "")
                        if content:
                            formatted += f"    - {content}\n"
                elif key != "qa_id":  # Skip qa_id in top-level
                    formatted += f"• {key}: {value}\n"
    else:
        formatted += str(result)
    
    return formatted

def format_final_answer(answer: str) -> tuple:
    """Format the final answer JSON into title and content."""
    try:
        # Try to parse as JSON
        answer_data = json.loads(answer)
        if isinstance(answer_data, dict):
            # Extract title and content
            title = answer_data.get("title", "")
            content = answer_data.get("content", "")
            return title, content
    except:
        # If not JSON or parsing fails, try to extract title and content from string
        lines = answer.split('\n', 1)
        if len(lines) > 1 and lines[0].startswith('###'):
            title = lines[0].replace('###', '').strip()
            content = lines[1].strip()
            return title, content
        return "", answer

def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        # Build database
        qa_db = build_database()
        file_db = build_file_database()
        
        # Initialize file processor
        file_processor = FileProcessor(
            tokens_per_chunk=800,  # Size of each chunk in tokens
            overlap_tokens=400,  # Overlap between chunks in tokens
            upload_dir="uploads",
            file_db=file_db
        )
        
        # Initialize tools with database
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
        st.session_state.chatbot = Chatbot(chat_config)
        st.session_state.chat_history = ChatHistory()
        st.session_state.file_processor = file_processor
        st.session_state.processed_files = set()  # Track processed files by hash
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and add it to the database."""
    try:
        # Save uploaded file temporarily
        temp_path = Path("uploads") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process the file
        processed_data = st.session_state.file_processor.process_file(temp_path)
        
        # Check if file already processed
        file_hash = processed_data["metadata"]["file_hash"]
        if file_hash in st.session_state.processed_files:
            os.remove(temp_path)
            return False, f"File '{uploaded_file.name}' has already been processed."
        
        # Save to FileDB and get metadata
        result = st.session_state.file_processor.save_processed_file(processed_data)
        
        # Add to chat history and processed files set
        st.session_state.chat_history.add_uploaded_file(result["metadata"])
        st.session_state.processed_files.add(file_hash)
        
        # Set chat history in chatbot
        st.session_state.chatbot.set_chat_history(st.session_state.chat_history)
        
        os.remove(temp_path)
        return True, f"Successfully processed: {uploaded_file.name}"
        
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        return False, f"Error processing file: {str(e)}"

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload section
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "pdf", "docx", "md"])
    
    if uploaded_file:
        with st.sidebar:
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    success, message = process_uploaded_file(uploaded_file)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # Display uploaded files
    if st.session_state.chat_history.uploaded_files:
        st.sidebar.header("📚 Uploaded Files")
        for file_meta in st.session_state.chat_history.uploaded_files:
            st.sidebar.markdown(
                f"""
                **{file_meta['file_name']}**  
                Type: {file_meta['file_type']}  
                Pages: {file_meta.get('num_pages', 1)}  
                Words: {file_meta.get('word_count', 0)}
                """
            )
    
    # Chat input
    with st.container():
        user_input = st.chat_input("Ask me anything!")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Create columns for search and reasoning
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 Search Process")
                with st.expander("Search Results", expanded=True):
                    # Perform search
                    with st.spinner("Searching..."):
                        search_results = st.session_state.chatbot.search_mode(user_input)
                        st.markdown(format_search_results(search_results))
            
            with col2:
                st.subheader("🤔 Reasoning Process")
                with st.expander("Step-by-Step Reasoning", expanded=True):
                    # Initialize reasoning container
                    reasoning_placeholder = st.empty()
                    
                    # Collect all steps
                    all_steps = []
                    final_answer = None
                    tool_results = []
                    
                    # Process reasoning steps with search results
                    for steps, total_time in st.session_state.chatbot.reasoning_mode(user_input, search_results):
                        if total_time is not None:
                            final_answer = steps[-1][1]
                            all_steps.append(f"⏱️ **Total Thinking Time**: {total_time:.2f} seconds")
                        else:
                            step_title, step_content, _, tool, tool_input, tool_result = steps[-1]
                            all_steps.append(f"### {step_title}")
                            all_steps.append(step_content)
                            
                            if tool:
                                all_steps.append("---")
                                all_steps.append(format_tool_result(tool, tool_input, tool_result))
                                tool_results.append({
                                    "tool": tool,
                                    "input": tool_input,
                                    "result": tool_result
                                })
                            all_steps.append("---")
                        
                        # Update reasoning display
                        reasoning_placeholder.markdown("\n\n".join(all_steps))
            
            # Display final answer using Streamlit components
            if final_answer:
                st.markdown("---")
                
                # Extract title and content
                title, content = format_final_answer(final_answer)
                
                # Create a container for the final answer
                final_answer_container = st.container()
                with final_answer_container:
                    # Use expander for the final answer
                    st.subheader("📝 Final Answer")
                    with st.expander("Result", expanded=True):
                        # Display title in a special container with custom styling
                        title_container = st.container()
                        title_container.write(f"#### {title}")
                        
                        # Add some space
                        st.write("")
                        
                        # Display content in a text area for better readability
                        st.write(content)
                
                # Add assistant message - combine title and content for chat history
                chat_content = f"{title}\n\n{content}"
                st.session_state.messages.append({"role": "assistant", "content": chat_content})
                
                # Add interaction to chat history
                st.session_state.chat_history.add_interaction(
                    query=user_input,
                    response=chat_content,
                    search_results=search_results,
                    tool_results=tool_results
                )
    
    # Display chat history
    st.markdown("---")
    st.subheader("💬 Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
