import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'new_search_clicked' not in st.session_state:
    st.session_state.new_search_clicked = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "sonar-pro"

# Configure sidebar
with st.sidebar:
    st.markdown("### Model Settings")
    st.markdown("""
    Choose the Perplexity model that best fits your needs:
    
    - **Sonar Pro**: Advanced model with enhanced reasoning and citation capabilities
    - **Sonar Reasoning**: Balanced model with good reasoning abilities
    - **Sonar**: Fast and efficient for general queries
    """)
    
    selected_model = st.selectbox(
        "Select Model",
        ["sonar-pro", "sonar-reasoning", "sonar"],
        format_func=lambda x: x.replace("-", " ").title(),
        index=["sonar-pro", "sonar-reasoning", "sonar"].index(st.session_state.selected_model)
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.conversations = []  # Clear conversations when model changes

# Add custom CSS
st.markdown("""
    <style>
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-uf99v8.ea3mdgi5 {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .search-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background-color: white;
        padding: 1rem 3rem;
        border-bottom: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .content-container {
        margin-top: 5rem;
        padding: 0 3rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px !important;
        font-family: "Source Sans Pro", sans-serif !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 2px rgba(66,153,225,0.5) !important;
        border-color: #63b3ed !important;
    }
    button[kind="secondary"] {
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid #e5e7eb !important;
    }
    .response-container {
        padding: 2rem 0;
        margin: 1rem 0;
        font-family: "Source Sans Pro", sans-serif !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    .sources-header {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        color: #0f172a !important;
    }
    .answer-header {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
        color: #0f172a !important;
    }
    .query-text {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #0f172a !important;
        margin-bottom: 1.5rem !important;
    }
    .think-prefix {
        color: #6c757d;
        font-family: monospace;
        font-size: 14px;
    }
    div[data-testid="stMarkdownContainer"] > p {
        font-family: "Source Sans Pro", sans-serif !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    .source-link {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: #f8f9fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        text-decoration: none;
        color: #1a202c;
        font-size: 14px;
    }
    .source-link:hover {
        background-color: #f1f5f9;
    }
    .sources-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .citation-number {
        font-size: 0.75rem;
        vertical-align: super;
        color: #2563eb;
        font-weight: 500;
        text-decoration: none;
    }
    .citations-list {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
        font-size: 14px;
        color: #4b5563;
    }
    </style>
    """, unsafe_allow_html=True)

def get_ai_response_stream(query, response_placeholder):
    """Get streaming response from Perplexity Sonar Reasoning API"""
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
3. At the end of your response, list all sources with their URLs in this format:
[1] Title https://example.com
[2] Title https://example.com
etc."""
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Create a stream of responses with citations enabled
        response_stream = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=messages,
            stream=True
        )
        
        # Initialize the placeholder for streaming content
        text_container = response_placeholder.empty()
        
        # Initialize variables for tracking the response
        full_response = []
        collected_chunks = []
        collected_messages = []
        
        # Update the response placeholder as chunks arrive
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk.choices[0].delta.content  # extract the message
                collected_messages.append(chunk_message)  # save the message
                
                # Format the response with proper styling
                if chunk_message.strip().startswith('<think>'):
                    chunk_message = chunk_message.replace('<think>', '<span class="think-prefix">&lt;think&gt;</span>')
                
                full_response.append(chunk_message)
                text = ''.join(full_response)
                
                # Update the display with the new content
                text_container.markdown(text + "▌", unsafe_allow_html=True)
        
        # Final update without the cursor
        text = ''.join(full_response)
        text_container.markdown(text, unsafe_allow_html=True)
        
        return ''.join(collected_messages)
    except Exception as e:
        response_placeholder.error(f"An error occurred: {str(e)}")
        return None

def extract_citations(text):
    """Extract citations and their corresponding numbers from the text"""
    import re
    citations = []
    
    # First, find all citation numbers in the text
    citation_pattern = r'\[(\d+)\]'
    citation_numbers = re.findall(citation_pattern, text)
    
    # Then find the corresponding URLs at the end of the text
    url_pattern = r'\[(\d+)\]\s*(?:(?:[^[\n]+?)?\s*)?(https?://[^\s\n]+)'
    matches = re.finditer(url_pattern, text)
    
    url_dict = {}
    for match in matches:
        num, url = match.groups()
        url_dict[num] = url.strip()
    
    # Create citations list maintaining the order they appear in the text
    seen = set()
    for num in citation_numbers:
        if num in url_dict and num not in seen:
            seen.add(num)
            citations.append({
                'number': num,
                'url': url_dict[num]
            })
    
    # Find where the URL section begins
    url_section_start = text.find('\n[1]')
    if url_section_start != -1:
        # Keep only the text before the URL section
        clean_text = text[:url_section_start].strip()
    else:
        clean_text = text
    
    return clean_text, citations

def format_text_with_citations(text, citations):
    """Format the text with clickable citation numbers"""
    import re
    formatted_text = text
    
    # Replace citation numbers with clickable links
    for citation in citations:
        num = citation['number']
        formatted_text = re.sub(
            f'\\[{num}\\]',
            f'<a href="{citation["url"]}" target="_blank" class="citation-number">[{num}]</a>',
            formatted_text
        )
    
    return formatted_text

def display_conversation(query, response_placeholder):
    """Display conversation in Perplexity style with Sources and Answer sections"""
    # Display the query
    response_placeholder.markdown(f'<div class="query-text">{query}</div>', unsafe_allow_html=True)
    
    # Display Answer section
    response_placeholder.markdown('<div class="answer-header">💡 Answer</div>', unsafe_allow_html=True)
    answer_container = response_placeholder.empty()
    
    # Get the streaming response
    response = get_ai_response_stream(query, answer_container)
    
    if response:
        # Extract and process citations
        clean_text, citations = extract_citations(response)
        
        if citations:
            # Format and display the text with citations
            formatted_text = format_text_with_citations(clean_text, citations)
            answer_container.markdown(formatted_text, unsafe_allow_html=True)
            
            # Display small references at the bottom
            references_html = '<div style="font-size: 12px; color: #666; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee;">'
            for citation in citations:
                references_html += f'<div style="margin: 4px 0;"><a href="{citation["url"]}" target="_blank" style="color: #666; text-decoration: none;">[{citation["number"]}] {citation["url"]}</a></div>'
            references_html += '</div>'
            response_placeholder.markdown(references_html, unsafe_allow_html=True)
        else:
            # If no citations, just display the text
            answer_container.markdown(clean_text, unsafe_allow_html=True)
        
        return response
    return None

def handle_new_search():
    st.session_state.new_search_clicked = True

# Search container with fixed position
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Create columns for the search interface
col1, col2 = st.columns([6, 1])

# Search input in the first column
with col1:
    query = st.text_input(
        "Enter your question:", 
        value="" if st.session_state.new_search_clicked else st.session_state.get('query', ''),
        placeholder="What would you like to know?",
        key="query",
        label_visibility="collapsed"
    )

# New Search button in the second column
with col2:
    if st.button("New Search", type="secondary", on_click=handle_new_search):
        st.session_state.conversations = []

st.markdown('</div>', unsafe_allow_html=True)

# Reset new_search_clicked state after rendering the input
if st.session_state.new_search_clicked:
    st.session_state.new_search_clicked = False

# Content container with proper spacing from fixed header
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Display conversation history and handle new queries
if query:
    # Create a new container for this conversation
    conversation_container = st.container()
    with conversation_container:
        with st.spinner(""):
            response = display_conversation(query, conversation_container)
            if response:
                st.session_state.conversations.append({
                    "query": query,
                    "response": response
                })

# Display previous conversations
for conv in reversed(st.session_state.conversations[:-1] if st.session_state.conversations else []):
    with st.container():
        st.markdown(f'<div class="query-text">{conv["query"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sources-header">🔍 Sources</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-header">💡 Answer</div>', unsafe_allow_html=True)
        st.markdown(conv["response"], unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
