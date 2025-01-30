import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Perplexity-like Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .conversation {
        margin-bottom: 30px;
        padding: 20px;
        border-radius: 10px;
        background: #f7f7f7;
    }
    .sources {
        margin-top: 10px;
        padding: 10px;
        background: #fff;
        border-radius: 5px;
    }
    .citation {
        display: inline-block;
        padding: 2px 6px;
        margin: 0 2px;
        border-radius: 3px;
        background: #e0e0e0;
        font-size: 0.8em;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "sonar"
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'show_search' not in st.session_state:
    st.session_state.show_search = True

# Sidebar for model selection
with st.sidebar:
    st.title("Chat Settings")
    selected_model = st.selectbox(
        "Select Model",
        ["sonar-pro", "sonar-reasoning", "sonar"],
        format_func=lambda x: x.replace("-", " ").title(),
        index=["sonar-pro", "sonar-reasoning", "sonar"].index(st.session_state.selected_model)
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.conversations = []  # Clear conversations when model changes
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This is a Perplexity-like chat interface with:
        - Streaming responses
        - Citation support
        - Multiple model support
        - Conversation history
    """)

def get_ai_response_stream(query, response_placeholder):
    """Stream response from the Perplexity API"""
    try:
        client = OpenAI(
            api_key=os.getenv('PERPLEXITY_API_KEY'),
            base_url="https://api.perplexity.ai"
        )
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant. For every response:
1. Include detailed citations with source URLs
2. Use numbered citations in the text like [1], [2], etc.
3. At the end of your response, list all sources with their URLs"""
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        stream = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=messages,
            stream=True
        )
        
        # Initialize variables for tracking the response
        full_response = []
        collected_chunks = []
        
        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response.append(content)
                collected_chunks.append(content)
                response_placeholder.markdown("".join(full_response) + "▌")
        
        return "".join(full_response)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def extract_citations(text):
    """Extract citations from the text and return a list of sources"""
    import re
    
    # Find all citations in the format [number] Title URL
    citations = []
    lines = text.split('\n')
    for line in lines:
        if re.match(r'\[\d+\]', line):
            citations.append(line.strip())
    
    return citations

def display_conversation(query, response, citations):
    """Display a single conversation in Perplexity style"""
    with st.container():
        st.markdown(f"### Q: {query}")
        
        # Sources section
        if citations:
            st.markdown("#### Sources")
            for citation in citations:
                st.markdown(citation)
        
        # Answer section
        st.markdown("#### Answer")
        st.markdown(response)
        st.markdown("---")

# Main chat interface
st.title("Perplexity-like Chat")

# Display existing conversations
for conv in st.session_state.conversations:
    display_conversation(conv["query"], conv["response"], conv["citations"])

# Handle form submission through session state
if 'submitted_query' not in st.session_state:
    st.session_state.submitted_query = None

# Search form without floating container
if st.session_state.show_search:
    with st.form(key="search_form", clear_on_submit=True):
        query = st.text_input("Ask anything...", key=f"search_input_{st.session_state.input_key}")
        submit_button = st.form_submit_button("Send", use_container_width=True)
        
        if submit_button and query:
            st.session_state.submitted_query = query
            st.session_state.show_search = False
            st.rerun()

# Handle query processing
if st.session_state.submitted_query:
    query = st.session_state.submitted_query
    
    # Create placeholder for streaming response
    response_placeholder = st.empty()
    
    # Get streaming response
    response = get_ai_response_stream(query, response_placeholder)
    
    if response:
        # Extract citations after response is complete
        citations = extract_citations(response)
        
        # Add to conversation history
        st.session_state.conversations.append({
            "query": query,
            "response": response,
            "citations": citations
        })
        
        # Reset for next query
        st.session_state.submitted_query = None
        st.session_state.show_search = True
        st.session_state.input_key += 1
        st.rerun()