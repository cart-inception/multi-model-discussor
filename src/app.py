import streamlit as st
import ollama
import threading
import time
import uuid
import sys
import os
import requests
import traceback
from pathlib import Path

# Add the project root to the Python path to allow importing modules from src
sys.path.append(str(Path(__file__).parent.parent))
# Import all needed components from advanced_config
from src.advanced_config import PersonaManager, advanced_config_main, gemini_config_ui, ollama_config_ui, persona_editor_ui

# Import Google Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class ModelAgent:
    def __init__(self, model_name, system_prompt, agent_name):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.conversation_history = []
        
    def generate_response(self, message, stream=False):
        """Generate a response from the model
        
        Args:
            message: The message to send to the model
            stream: If True, return a generator yielding tokens as they become available
                   If False, return the full response text
        """
        # Check if this is a Gemini model
        if self.model_name.startswith("gemini:"):
            # Delegate to GeminiModelAgent
            gemini_agent = GeminiModelAgent(self.model_name, self.system_prompt, self.agent_name)
            gemini_agent.conversation_history = self.conversation_history.copy()
            result = gemini_agent.generate_response(message, stream)
            
            # Update conversation history if non-streaming (streaming updates happen in the generator)
            if not stream:
                self.conversation_history = gemini_agent.conversation_history.copy()
                
            return result
        
        # Otherwise, use Ollama API
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history,
                {"role": "user", "content": message}
            ]
            
            if stream:
                # Return a streaming response generator with better error handling
                def stream_response():
                    full_response = ""
                    chunk_count = 0
                    
                    try:
                        stream = ollama.chat(
                            model=self.model_name,
                            messages=messages,
                            stream=True
                        )
                        
                        # Show debug info if needed
                        if "debug_mode" in st.session_state and st.session_state.debug_mode:
                            st.write(f"Debug: Started streaming with {self.model_name}")
                        
                        for chunk in stream:
                            chunk_count += 1
                            
                            # Debug the chunk format if debug mode enabled
                            if "debug_mode" in st.session_state and st.session_state.debug_mode and chunk_count <= 2:
                                st.write(f"Debug: Stream chunk format: {chunk}")
                            
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                full_response += content
                                yield content, full_response
                        
                        # Add to conversation history when complete
                        self.conversation_history.append({"role": "user", "content": message})
                        self.conversation_history.append({"role": "assistant", "content": full_response})
                        
                        # Debug completion
                        if "debug_mode" in st.session_state and st.session_state.debug_mode:
                            st.write(f"Debug: Completed streaming with {chunk_count} chunks")
                    
                    except Exception as e:
                        error_msg = f"Error in stream_response: {str(e)}"
                        if "debug_mode" in st.session_state and st.session_state.debug_mode:
                            st.error(f"Debug: {error_msg}")
                            st.write(f"Debug: Traceback: {traceback.format_exc()}")
                        yield error_msg, error_msg
                
                # Actually return the generator object
                return stream_response()
            else:
                # Return the complete response
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages
                )
                assistant_message = response['message']['content']
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message
        except Exception as e:
            error_msg = f"Error with {self.agent_name}: {str(e)}"
            if stream:
                # Return a single item generator for error cases
                def error_generator():
                    yield error_msg, error_msg
                return error_generator()
            return error_msg
            
class GeminiModelAgent:
    def __init__(self, model_name, system_prompt, agent_name):
        # Remove the gemini: prefix from the model name
        self.model_name = model_name.replace("gemini:", "")
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.conversation_history = []
        
        # Get API key from session state or environment
        api_key = st.session_state.get("gemini_api_key", os.environ.get("GOOGLE_API_KEY", ""))
        if not api_key:
            raise ValueError("Google API key not found. Please configure it in the Gemini settings.")
            
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
    def generate_response(self, message, stream=False):
        """Generate a response from the Gemini model
        
        Args:
            message: The message to send to the model
            stream: If True, return a generator yielding tokens as they become available
                   If False, return the full response text
        """
        try:
            # Create a new conversation model
            model = genai.GenerativeModel(self.model_name)
            
            # Format conversation history for Gemini
            # First convert to Gemini conversation format
            chat = model.start_chat(history=[])
            
            # Add system prompt as first message from user
            if self.system_prompt:
                chat.history.append({"role": "user", "parts": [self.system_prompt]})
                chat.history.append({"role": "model", "parts": ["I'll follow these instructions as I respond."]})
            
            # Add conversation history
            for msg in self.conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                chat.history.append({"role": role, "parts": [msg["content"]]})
            
            # Add the new message
            if stream:
                # Create a streaming response generator
                def stream_response():
                    full_response = ""
                    chunk_count = 0
                    
                    try:
                        # Send the message to the model
                        stream_resp = chat.send_message(message, stream=True)
                        
                        # Show debug info if needed
                        if "debug_mode" in st.session_state and st.session_state.debug_mode:
                            st.write(f"Debug: Started streaming with Gemini {self.model_name}")
                        
                        for chunk in stream_resp:
                            chunk_count += 1
                            
                            # Debug the chunk format if debug mode enabled
                            if "debug_mode" in st.session_state and st.session_state.debug_mode and chunk_count <= 2:
                                st.write(f"Debug: Gemini stream chunk format: {chunk}")
                            
                            if hasattr(chunk, 'text'):
                                content = chunk.text
                                full_response += content
                                yield content, full_response
                        
                        # Add to conversation history when complete
                        self.conversation_history.append({"role": "user", "content": message})
                        self.conversation_history.append({"role": "assistant", "content": full_response})
                        
                        # Debug completion
                        if "debug_mode" in st.session_state and st.session_state.debug_mode:
                            st.write(f"Debug: Completed Gemini streaming with {chunk_count} chunks")
                    
                    except Exception as e:
                        error_msg = f"Error in Gemini stream_response: {str(e)}"
                        if "debug_mode" in st.session_state and st.session_state.debug_mode:
                            st.error(f"Debug: {error_msg}")
                            st.write(f"Debug: Traceback: {traceback.format_exc()}")
                        yield error_msg, error_msg
                
                # Return the generator
                return stream_response()
            else:
                # Non-streaming response
                response = chat.send_message(message)
                assistant_message = response.text
                
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
                
        except Exception as e:
            error_msg = f"Error with Gemini ({self.agent_name}): {str(e)}"
            if stream:
                # Return a single item generator for error cases
                def error_generator():
                    yield error_msg, error_msg
                return error_generator()
            return error_msg

class DiscussionOrchestrator:
    def __init__(self):
        self.agents = {}
        self.discussion_history = {}
        
    def add_agent(self, agent_id, agent):
        self.agents[agent_id] = agent
        
    def start_discussion(self, topic, main_agent_id, discussion_id=None, rounds=2, stream_handler=None):
        """
        Start or continue a discussion with multiple rounds of interaction.
        
        Args:
            topic: The topic or question to discuss
            main_agent_id: The ID of the main synthesizing agent
            discussion_id: Optional ID for an existing discussion
            rounds: Number of discussion rounds between agents
            stream_handler: Optional callback function for streaming responses
                           Function signature: (agent_id, agent_name, content, is_complete)
            
        Returns:
            discussion_id: The ID of the discussion
        """
        if discussion_id is None:
            discussion_id = str(uuid.uuid4())
            self.discussion_history[discussion_id] = []
            
        main_agent = self.agents[main_agent_id]
        
        # Only do the initial prompt if this is a new discussion
        if not self.discussion_history[discussion_id]:
            # Format the topic as a request to discuss with other agents in a conversational way
            initial_prompt = f"""Let's have a natural, conversational discussion about this topic: {topic}.

As the discussion leader, please share your initial thoughts in a casual, engaging way.
Keep it friendly and conversational - imagine you're chatting with colleagues at a coffee shop.
Provide a balanced, nuanced perspective that introduces different angles to consider.
Raise several clear, distinct points that other participants can engage with specifically.
End with 2-3 thought-provoking questions that invite different perspectives and will spark an interactive discussion."""
            
            if stream_handler:
                # Stream the initial response
                stream_response = ""
                try:
                    # Get the streaming generator
                    stream_generator = main_agent.generate_response(initial_prompt, stream=True)
                    # Process each chunk
                    for chunk, full_response in stream_generator:
                        stream_handler(main_agent_id, main_agent.agent_name, chunk, False)
                        stream_response = full_response
                    # Mark as complete when done
                    stream_handler(main_agent_id, main_agent.agent_name, stream_response, True)
                    main_response = stream_response
                except Exception as e:
                    st.error(f"Error streaming initial response: {str(e)}")
                    # Fallback to non-streaming response
                    main_response = main_agent.generate_response(initial_prompt, stream=False)
            else:
                main_response = main_agent.generate_response(initial_prompt, stream=False)
                
            self.discussion_history[discussion_id].append({
                "agent": main_agent_id,
                "content": main_response,
                "timestamp": time.time(),
                "round": 0
            })
        
        current_round = max([msg.get('round', 0) for msg in self.discussion_history[discussion_id]]) if self.discussion_history[discussion_id] else 0
        
        # Run the specified number of discussion rounds
        for r in range(current_round, current_round + rounds):
            # Get the latest context to respond to
            latest_responses = {}
            for msg in self.discussion_history[discussion_id]:
                if msg.get('round') == r:
                    latest_responses[msg['agent']] = {
                        'content': msg['content'],
                        'agent_name': self.agents[msg['agent']].agent_name
                    }
            
            # Check if this is the final round - only the main agent should respond
            is_final_round = (r == current_round + rounds - 1)
            
            # Have each agent respond to the current round
            for agent_id, agent in self.agents.items():
                # Skip agents who have already contributed to this round
                if agent_id in [msg['agent'] for msg in self.discussion_history[discussion_id] if msg.get('round') == r+1]:
                    continue
                    
                # In the final round, only the main synthesizer agent should respond
                if is_final_round and agent_id != main_agent_id:
                    continue
                
                # Skip the main agent in intermediate rounds - it only responds at the beginning and end
                if agent_id == main_agent_id and r < current_round + rounds - 1:
                    continue
                
                # Build context from the latest responses in a more conversational format
                context_parts = [f"DISCUSSION TOPIC: {topic}"]
                context_parts.append("\nRecent messages in our conversation:")
                
                # Include user messages in latest responses if present
                user_messages = [msg for msg in self.discussion_history[discussion_id] 
                                if msg.get('agent') == 'user' and msg.get('round') == r]
                
                # Add any user messages to the latest responses
                for user_msg in user_messages:
                    if 'user' not in latest_responses:
                        latest_responses['user'] = {
                            'content': user_msg['content'],
                            'agent_name': 'Human User'
                        }
                
                # Sort responses by agent to create a more logical conversation flow
                sorted_responses = sorted(latest_responses.items(), key=lambda x: x[0] != "main")
                
                for resp_agent_id, resp_data in sorted_responses:
                    # Don't include the agent's own previous response in the context
                    if resp_agent_id != agent_id:
                        # Format as a chat message rather than a report
                        formatted_content = resp_data['content'].replace("\n\n", "\n").strip()
                        context_parts.append(f"\n{resp_data['agent_name']}: {formatted_content}")
                
                # Customize prompt based on agent role
                if agent_id == main_agent_id and is_final_round:
                    # Final synthesis by main agent - comprehensive summary
                    context_parts.insert(0, """Now that we've had this discussion, could you synthesize what we've talked about?

Please summarize the key points from our conversation, highlighting areas of agreement, disagreement, and interesting perspectives.
Identify how different participants built on or challenged each other's ideas throughout the conversation.
Highlight specific exchanges where viewpoints evolved or where there were productive disagreements.
Your synthesis should emphasize the interactive nature of the discussion rather than just listing each person's contributions.
Acknowledge any open questions that might remain for future discussions.""")
                else:
                    # Response from discussion agent - much more interactive
                    context_parts.insert(0, """You are participating in an interactive discussion. Your task is to respond to the specific points made by others in a way that advances the conversation.

IMPORTANT: DO NOT simply restate the topic or add general thoughts. Instead:
1. DIRECTLY ENGAGE with what others have said by name (e.g., "I agree with Sarah's point about X, but I think...")
2. CHALLENGE or BUILD UPON specific ideas from previous messages
3. INTRODUCE a new perspective only if it relates to what's already been discussed
4. ASK thought-provoking questions about others' contributions
5. ACKNOWLEDGE points of agreement AND disagreement with specific participants

Your response should feel like a natural part of an ongoing conversation where people are genuinely responding to each other, not just taking turns speaking.

If you disagree with something someone said, explain why specifically.
If you agree, add additional context, examples, or implications that develop the idea further.
Avoid broad generalizations that don't reference specific points already made.""")
                
                context = "\n\n".join(context_parts)
                
                if stream_handler and (is_final_round or r == 0):
                        # Stream responses for initial and final rounds
                    stream_response = ""
                    try:
                        # Get the streaming generator
                        stream_generator = agent.generate_response(context, stream=True)
                        # Process each chunk
                        for chunk, full_response in stream_generator:
                            stream_handler(agent_id, agent.agent_name, chunk, False)
                            stream_response = full_response
                        # Mark as complete when done
                        stream_handler(agent_id, agent.agent_name, stream_response, True)
                        agent_response = stream_response
                    except Exception as e:
                        st.error(f"Error streaming response: {str(e)}")
                        # Fallback to non-streaming response
                        agent_response = agent.generate_response(context, stream=False)
                else:
                    # Regular response for intermediate rounds
                    agent_response = agent.generate_response(context)
                
                self.discussion_history[discussion_id].append({
                    "agent": agent_id,
                    "content": agent_response,
                    "timestamp": time.time(),
                    "round": r+1
                })
        
        return discussion_id
    
    def get_discussion_history(self, discussion_id):
        return self.discussion_history.get(discussion_id, [])

# Utility functions
def refresh_available_models():
    """Check Ollama and Gemini for available models and update the session state"""
    available_models = []
    ollama_success = False
    gemini_success = False
    
    # Try to get Ollama models
    try:
        response = ollama.list()
        if 'models' in response:
            models = response['models']
            ollama_model_names = []
            
            # Debug the response format
            with st.expander("Debug: Ollama API Response Format", expanded=False):
                st.write("First model data structure:")
                if models and len(models) > 0:
                    st.write(f"Type: {type(models[0])}")
                    st.write(f"Example: {models[0]}")
                    
                    # If model is an object with __dict__ attribute, try to access model data
                    if hasattr(models[0], '__dict__'):
                        st.write("Model attributes:")
                        model_dict = models[0].__dict__
                        st.write(model_dict)
            
            for model in models:
                # Handle both dictionary and object formats
                if isinstance(model, dict) and 'name' in model:
                    ollama_model_names.append(model['name'])
                elif hasattr(model, 'model') and isinstance(model.model, str):
                    # Handle object format where model name is in a 'model' attribute
                    ollama_model_names.append(model.model)
                else:
                    # Try to extract name if it's an object with __dict__
                    if hasattr(model, '__dict__'):
                        model_dict = model.__dict__
                        if 'model' in model_dict and isinstance(model_dict['model'], str):
                            ollama_model_names.append(model_dict['model'])
                            continue
                    
                    st.warning(f"Skipping model with invalid format: {model}")
            
            # Add Ollama models to the available models
            available_models.extend(ollama_model_names)
            ollama_success = True
            
            # Add info about detected Ollama models
            st.info(f"Detected {len(ollama_model_names)} Ollama models")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error connecting to Ollama: {str(e)}")
        
        # Add a debug expander with technical details
        with st.expander("Debug Details"):
            st.code(error_details)
            
            # Check Ollama environment variable
            st.write(f"Ollama Host: {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}")
            
            # Attempt a basic connection test
            import requests
            try:
                # Try to connect to Ollama API
                host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
                st.write(f"Testing connection to: {host}/api/version")
                response = requests.get(f"{host}/api/version", timeout=5)
                st.write(f"Connection test status: {response.status_code}")
                st.write(f"Response: {response.text}")
            except Exception as conn_error:
                st.write(f"Connection test failed: {str(conn_error)}")
    
    # Try to get Gemini models if the API key is available
    if genai is not None and ("gemini_api_key" in st.session_state or "GOOGLE_API_KEY" in os.environ):
        try:
            # Get API key from session state or environment
            api_key = st.session_state.get("gemini_api_key", os.environ.get("GOOGLE_API_KEY", ""))
            
            if api_key:
                genai.configure(api_key=api_key)
                
                # List available models
                models = genai.list_models()
                
                # Filter to models that support text generation
                gemini_models = []
                for model in models:
                    if "generateContent" in model.supported_generation_methods:
                        # Add a prefix to distinguish from Ollama models
                        gemini_models.append(f"gemini:{model.name}")
                
                # Add Gemini models to available models
                available_models.extend(gemini_models)
                
                # Store in session state for reference
                st.session_state.gemini_models = gemini_models
                
                # Add info about detected Gemini models
                st.info(f"Detected {len(gemini_models)} Gemini models")
                
                gemini_success = True
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Error connecting to Google Gemini API: {str(e)}")
            
            # Add a debug expander with technical details
            with st.expander("Debug Details: Gemini"):
                st.code(error_details)
    
    # Update the session state with all available models
    st.session_state.available_models = available_models
    
    # Return True if either Ollama or Gemini was successful
    return ollama_success or gemini_success

# GUI Application
def main():
    st.set_page_config(
        page_title="Multi-Agent Discussion", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for improved UI
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        font-weight: 600;
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .stExpander {
        border-radius: 6px;
        border: 1px solid #E5E7EB;
    }
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1F2937;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF;
        color: #1E3A8A;
        font-weight: 500;
    }
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stTextArea label {
        font-weight: 500;
        color: #374151;
        font-size: 14px;
    }
    /* Fix for text input and text area to ensure text is visible */
    input, textarea {
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }
    /* Fix for text area in dark mode */
    textarea {
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }
    .stMarkdown p {
        line-height: 1.6;
    }
    .success {
        padding: 1rem;
        background-color: #ECFDF5;
        border-radius: 6px;
        color: #047857;
        margin-bottom: 1rem;
    }
    .info {
        padding: 1rem;
        background-color: #EFF6FF;
        border-radius: 6px;
        color: #1E40AF;
        margin-bottom: 1rem;
    }
    .warning {
        padding: 1rem;
        background-color: #FFFBEB;
        border-radius: 6px;
        color: #92400E;
        margin-bottom: 1rem;
    }
    /* Ensure code blocks have dark text */
    code {
        color: #111827 !important;
    }
    /* Ensure selectbox text is visible */
    .stSelectbox div [data-baseweb="select"] div [data-testid="stMarkdown"] p {
        color: #111827 !important;
    }
    /* Make sure monospace text in markdown is visible */
    .stMarkdown pre, .stMarkdown code {
        color: #111827 !important;
        background-color: #F1F5F9 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = DiscussionOrchestrator()
        
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
        refresh_available_models()
    
    if "current_discussion_id" not in st.session_state:
        st.session_state.current_discussion_id = None
        
    if "agents_configured" not in st.session_state:
        st.session_state.agents_configured = False
        
    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager(save_dir="personas")
    
    # Sidebar for navigation with improved styling
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/chat.png", width=80)
        st.title("Multi-Agent Discussion")
        st.markdown("---")
        pages = {
            "💬 Discussion": "main_discussion",
            "⚙️ Advanced Configuration": "advanced_config"
        }
        
        selection = st.radio("", list(pages.keys()))
        
        st.markdown("---")
        st.markdown("#### About")
        st.markdown("""
        Create engaging discussions between multiple AI personas with different viewpoints.
        """)
        
        # Add version info
        st.markdown("---")
        st.caption("v1.0.0 | [Documentation](https://github.com/yourusername/multi-agent-discussion)")
    
    # Display the selected page
    if pages[selection] == "main_discussion":
        display_main_discussion_ui()
    elif pages[selection] == "advanced_config":
        # Display header for Advanced Configuration
        st.markdown("# ⚙️ Advanced Configuration")
        st.markdown("Configure personas and model settings")
        
        # Apply custom CSS for better UI
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 500;
        }
        .config-card {
            background-color: #F9FAFB;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #E5E7EB;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tabs directly here to ensure they're all available
        tabs = st.tabs(["👤 Persona Editor", "🖥️ Ollama Configuration", "🧠 Google Gemini"])
        
        with tabs[0]:
            persona_editor_ui()
        
        with tabs[1]:
            ollama_config_ui()
            
        with tabs[2]:
            gemini_config_ui()
        
        # Debug info to ensure we can see what's happening with session state
        if "debug_mode" in st.session_state and st.session_state.debug_mode:
            with st.expander("🔍 Debug: Session State"):
                st.write("Session State Keys:", list(st.session_state.keys()))
                if "gemini_api_key" in st.session_state:
                    st.write("Gemini API Key is set")
                if "gemini_models" in st.session_state:
                    st.write("Gemini Models:", st.session_state.gemini_models)
                if "available_models" in st.session_state:
                    st.write("All Available Models:", st.session_state.available_models)

def display_main_discussion_ui():
    """Main discussion UI"""
    st.markdown("# Multi-Agent Discussion Platform")
    st.markdown("Create thoughtful discussions between AI agents with different perspectives")
    
    # Check if Ollama is available
    if not st.session_state.available_models:
        if not refresh_available_models():
            st.markdown("""
            <div class="info">
                <h4>📢 Setup Required</h4>
                <p>Please make sure Ollama is running. Go to Advanced Configuration to troubleshoot.</p>
            </div>
            """, unsafe_allow_html=True)
            return
    
    # Setup tab and Discussion tab
    tab1, tab2 = st.tabs(["🤖 Setup Agents", "💬 Discussion"])
    
    with tab1:
        st.markdown("## Configure Your Agents")
        
        # Model availability check in a card-like container
        connection_container = st.container()
        with connection_container:
            if not st.session_state.available_models:
                st.markdown("""
                <div class="warning">
                    <h4>⚠️ No Models Detected</h4>
                    <p>Make sure Ollama is running and models are available.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Refresh Ollama Models", key="refresh_main"):
                    if refresh_available_models():
                        st.markdown("""
                        <div class="success">
                            <h4>✅ Success!</h4>
                            <p>Found Ollama models. You can now proceed with agent configuration.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.experimental_rerun()
            else:
                st.markdown(f"""
                <div class="success">
                    <h4>✅ Connected to Ollama</h4>
                    <p>Found {len(st.session_state.available_models)} models available for use.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Refresh Models", key="refresh_additional"):
                    refresh_available_models()
                    st.experimental_rerun()
        
        # Main agent setup with card-like styling
        st.markdown("### 🎯 Main Synthesizer")
        st.markdown("This agent will lead the discussion and provide the final synthesis")
        
        main_agent_container = st.container()
        with main_agent_container:
            col1, col2 = st.columns([3, 7])
            
            with col1:
                main_agent_name = st.text_input("Name", "Synthesizer", 
                                              help="A distinctive name for your main agent")
                main_agent_model = st.selectbox(
                    "Model", 
                    st.session_state.available_models if st.session_state.available_models else ["No models available"],
                    help="The Ollama model that will power this agent"
                )
            
            with col2:
                main_agent_prompt = st.text_area(
                    "System Prompt",
                    """You are a thoughtful conversation facilitator in a casual group discussion.

Your role is to guide a natural, engaging conversation among different participants with diverse perspectives.
Speak in a conversational, friendly tone - like a good host at a coffee shop discussion.
Ask questions that draw out others' perspectives and encourage them to build on each other's ideas.
When synthesizing the discussion, highlight how ideas connected and evolved through conversation.
Reference specific points made by participants by name, showing how the conversation built toward insights.
Use natural language like 'I think,' 'it seems like,' and other conversational markers.
Your final summary should feel like a natural conclusion to a rich discussion among friends.""",
                    height=180,
                    help="Instructions that define how this agent will behave"
                )
        
        # Additional agents setup with improved visual design
        st.markdown("### 👥 Discussion Participants")
        st.markdown("Add up to 3 agents with different perspectives to join the conversation")
        
        # Get both built-in and custom personas with more conversational styles
        example_personas = {
            "Skeptical Analyst": """You are a skeptical but friendly analyst participating in a casual conversation.

Speak in a natural, conversational tone as if chatting with friends at a coffee shop.
Question assumptions in a friendly way, like 'That's interesting, but I wonder if...'
Ask for evidence or examples when others make claims, but do so conversationally.
Point out logical issues or alternative perspectives with a thoughtful, curious tone.
Use first-person language and occasionally reference personal experiences/examples.""",
            
            "Creative Thinker": """You are a creative, enthusiastic thinker participating in a casual conversation.

Speak in a natural, excited tone as if brainstorming with friends at a coffee shop.
Suggest novel ideas and unexpected connections with phrases like 'What if we...?' or 'This reminds me of...'
Build on others' ideas with 'Yes, and...' statements to extend the thinking.
Use metaphors, analogies and personal anecdotes to illustrate your points.
Ask open-ended questions that spark imagination and new possibilities.""",
            
            "Practical Expert": """You are a practical, experienced expert participating in a casual conversation.

Speak in a natural, down-to-earth tone as if advising friends at a coffee shop.
Share real-world insights based on what you've seen work and not work in practice.
Gently bring abstract discussions back to implementation with phrases like 'In my experience...'
Ask grounding questions like 'How would this actually work in practice?'
Use concrete examples and occasional personal anecdotes to illustrate practical concerns.""",
            
            "Ethical Advisor": """You are a thoughtful ethical advisor participating in a casual conversation.

Speak in a natural, conversational tone as if discussing important topics with friends.
Raise ethical considerations in a curious, non-judgmental way.
Consider diverse perspectives with phrases like 'I wonder how this might affect...'
Ask thought-provoking questions about broader impacts and unintended consequences.
Share personal reflections on values and societal implications in a relatable way."""
        }
        
        # Add custom personas from the persona manager
        custom_personas = st.session_state.persona_manager.get_all_personas()
        for name, persona in custom_personas.items():
            example_personas[f"{name} (Custom)"] = persona["system_prompt"]
        
        agent_configs = []
        
        # Add tabs for more organized agent configuration
        agent_tabs = st.tabs(["Agent 1", "Agent 2", "Agent 3"])
        
        for i, tab in enumerate(agent_tabs):
            with tab:
                st.markdown(f"#### Configure Agent {i+1}")
                
                col1, col2 = st.columns([3, 7])
                
                with col1:
                    agent_name = st.text_input(f"Name", f"Agent {i+1}", key=f"name_{i}")
                    agent_model = st.selectbox(
                        f"Model", 
                        st.session_state.available_models if st.session_state.available_models else ["No models available"],
                        key=f"model_{i}"
                    )
                    persona_select = st.selectbox(
                        f"Persona Template", 
                        ["Custom"] + list(example_personas.keys()),
                        key=f"persona_{i}"
                    )
                    
                    include_agent = st.checkbox(f"Include in discussion", value=(i==0), key=f"include_{i}")
                
                with col2:
                    if persona_select != "Custom":
                        agent_prompt = st.text_area(
                            f"System Prompt", 
                            example_personas[persona_select],
                            height=180,
                            key=f"prompt_{i}"
                        )
                    else:
                        agent_prompt = st.text_area(
                            f"System Prompt", 
                            f"""You are a participant in a casual, thoughtful group conversation.

Speak in a natural, conversational tone as if chatting with friends at a coffee shop.
Respond directly to points made by others, building on their ideas or offering alternative perspectives.
Use first-person language, occasional questions, and conversational phrases like "I think," "I wonder if," etc.
Share your unique perspective while engaging meaningfully with what others have said.
Feel free to use analogies, examples, or personal anecdotes to illustrate your points in a relatable way.""",
                            height=180,
                            key=f"prompt_{i}"
                        )
                
                if include_agent:
                    agent_configs.append({
                        "name": agent_name,
                        "model": agent_model,
                        "prompt": agent_prompt
                    })
        
        # Configuration button with improved styling
        st.markdown("### Ready to Start?")
        
        config_col1, config_col2 = st.columns([3, 1])
        with config_col1:
            st.markdown("Click the button to configure your agents and prepare for discussion")
        
        with config_col2:
            config_button = st.button("⚙️ Configure Agents", use_container_width=True)
        
        if config_button:
            if not st.session_state.available_models:
                st.markdown("""
                <div class="warning">
                    <h4>⚠️ No Models Available</h4>
                    <p>Make sure Ollama is running and properly configured.</p>
                </div>
                """, unsafe_allow_html=True)
                
                refresh_col1, refresh_col2 = st.columns([3, 1])
                with refresh_col2:
                    if st.button("🔄 Try Refreshing", key="refresh_on_config"):
                        if refresh_available_models():
                            st.markdown("""
                            <div class="success">
                                <h4>✅ Success!</h4>
                                <p>Found Ollama models. You can now proceed with agent configuration.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.experimental_rerun()
                        else:
                            st.markdown("""
                            <div class="warning">
                                <p>Please go to Advanced Configuration to troubleshoot your Ollama connection.</p>
                            </div>
                            """, unsafe_allow_html=True)
            elif len(agent_configs) == 0:
                st.markdown("""
                <div class="warning">
                    <h4>⚠️ Missing Agents</h4>
                    <p>Please include at least one discussion agent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    orchestrator = DiscussionOrchestrator()
                    
                    # Validate that selected models exist in available_models
                    if main_agent_model not in st.session_state.available_models:
                        st.markdown(f"""
                        <div class="warning">
                            <h4>⚠️ Model Not Available</h4>
                            <p>Main agent model '{main_agent_model}' is not available. Please select a different model.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        return
                        
                    # Check if main agent uses a Gemini model and API key is available
                    if main_agent_model.startswith("gemini:"):
                        api_key = st.session_state.get("gemini_api_key", os.environ.get("GOOGLE_API_KEY", ""))
                        if not api_key:
                            st.markdown(f"""
                            <div class="warning">
                                <h4>⚠️ Gemini API Key Missing</h4>
                                <p>Main agent uses a Gemini model but no API key is configured. 
                                Please configure your Google API key in the Advanced Configuration > Google Gemini tab.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            return
                    
                    # Create main agent
                    main_agent = ModelAgent(main_agent_model, main_agent_prompt, main_agent_name)
                    orchestrator.add_agent("main", main_agent)
                    
                    # Create discussion agents
                    for i, config in enumerate(agent_configs):
                        if config["model"] not in st.session_state.available_models:
                            st.markdown(f"""
                            <div class="warning">
                                <h4>⚠️ Model Not Available</h4>
                                <p>Agent {i+1} model '{config['model']}' is not available. Please select a different model.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            return
                        
                        # Check if this agent uses a Gemini model and API key is available
                        if config["model"].startswith("gemini:"):
                            api_key = st.session_state.get("gemini_api_key", os.environ.get("GOOGLE_API_KEY", ""))
                            if not api_key:
                                st.markdown(f"""
                                <div class="warning">
                                    <h4>⚠️ Gemini API Key Missing</h4>
                                    <p>Agent {i+1} uses a Gemini model but no API key is configured. 
                                    Please configure your Google API key in the Advanced Configuration > Google Gemini tab.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                return
                            
                        agent = ModelAgent(config["model"], config["prompt"], config["name"])
                        orchestrator.add_agent(f"agent_{i}", agent)
                    
                    # Test model connection based on provider (Ollama or Gemini)
                    if main_agent_model.startswith("gemini:"):
                        # Test Gemini connection
                        with st.spinner("Testing connection to Google Gemini API..."):
                            try:
                                # Get API key from session state
                                api_key = st.session_state.get("gemini_api_key", os.environ.get("GOOGLE_API_KEY", ""))
                                if not api_key:
                                    st.markdown("""
                                    <div class="warning">
                                        <h4>⚠️ Gemini API Key Missing</h4>
                                        <p>Please configure your Google API key in the Advanced Configuration > Google Gemini tab.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    return
                                
                                # Configure the API and test a simple call
                                import google.generativeai as genai
                                genai.configure(api_key=api_key)
                                
                                # The actual model name is without the gemini: prefix
                                actual_model_name = main_agent_model.replace("gemini:", "")
                                model = genai.GenerativeModel(actual_model_name)
                                
                                # Simple test query
                                response = model.generate_content("Respond with OK")
                                
                                st.markdown("""
                                <div class="success">
                                    <h4>✅ Gemini Connection Verified</h4>
                                    <p>Successfully connected to Google Gemini API.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as test_error:
                                st.markdown("""
                                <div class="warning">
                                    <h4>⚠️ Gemini Connection Error</h4>
                                    <p>Error testing Google Gemini API connection.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("Debug Details"):
                                    st.code(traceback.format_exc())
                                return
                    else:
                        # Test Ollama connection
                        with st.spinner("Testing connection to Ollama..."):
                            try:
                                # Simple test to ensure the model loads properly
                                test_result = ollama.chat(
                                    model=main_agent_model,
                                    messages=[
                                        {"role": "user", "content": "Respond with only the word 'OK' for connection test"}
                                    ]
                                )
                                st.markdown("""
                                <div class="success">
                                    <h4>✅ Ollama Connection Verified</h4>
                                    <p>Successfully connected to Ollama and models are responding.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as test_error:
                                st.markdown("""
                                <div class="warning">
                                    <h4>⚠️ Ollama Connection Error</h4>
                                    <p>Error testing Ollama connection.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("Debug Details"):
                                    st.code(traceback.format_exc())
                                return
                    
                    st.session_state.orchestrator = orchestrator
                    st.session_state.agents_configured = True
                    st.markdown("""
                    <div class="success">
                        <h4>✅ Agents Configured!</h4>
                        <p>Your agents are ready. Switch to the Discussion tab to start a conversation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning">
                        <h4>⚠️ Configuration Error</h4>
                        <p>Error configuring agents: {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Debug Details"):
                        st.code(traceback.format_exc())
    
    with tab2:
        st.markdown("## Start a Discussion")
        
        if not st.session_state.agents_configured:
            st.markdown("""
            <div class="info">
                <h4>👈 Setup Required</h4>
                <p>Please configure your agents in the Setup tab first.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Discussion setup in a card-like container
            setup_container = st.container()
            with setup_container:
                st.markdown("### Enter Topic or Question")
                
                topic = st.text_area(
                    "What would you like the agents to discuss?",
                    height=100,
                    placeholder="Enter a specific topic or question for the agents to discuss..."
                )
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    rounds = st.slider(
                        "Discussion Rounds", 
                        min_value=1, 
                        max_value=5, 
                        value=2,
                        help="Number of back-and-forth exchanges between agents"
                    )
                
                with col2:
                    continue_discussion = st.checkbox(
                        "Continue previous", 
                        value=False,
                        disabled=not st.session_state.current_discussion_id,
                        help="Continue from the last discussion"
                    )
                
                with col3:
                    # Initialize streaming UI controls
                    if "show_streaming" not in st.session_state:
                        st.session_state.show_streaming = True
                    
                    streaming_toggle = st.checkbox(
                        "Show streaming", 
                        value=st.session_state.show_streaming,
                        help="Show responses as they're generated"
                    )
                    st.session_state.show_streaming = streaming_toggle
            
            # Debug mode toggle in expandable section
            with st.expander("Advanced Options"):
                if "debug_mode" not in st.session_state:
                    st.session_state.debug_mode = False
                
                debug_toggle = st.checkbox(
                    "Debug mode", 
                    value=st.session_state.debug_mode,
                    help="Show technical details for troubleshooting"
                )
                st.session_state.debug_mode = debug_toggle
            
            # Start button prominently displayed
            start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
            with start_col2:
                start_btn = st.button("🚀 Start Discussion", use_container_width=True)
            
            # Initialize streaming placeholders if not present
            if "streaming_responses" not in st.session_state:
                st.session_state.streaming_responses = {}
            
            if start_btn:
                if not topic:
                    st.markdown("""
                    <div class="warning">
                        <h4>⚠️ Missing Topic</h4>
                        <p>Please enter a topic or question for the discussion.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Create placeholders for streaming responses
                    response_placeholders = {}
                    stream_container = st.empty()
                    
                    # Add debug container if debug mode is enabled
                    if st.session_state.debug_mode:
                        debug_container = st.container()
                        with debug_container:
                            st.markdown("#### Debug Information")
                            st.write("Ollama models:", st.session_state.available_models)
                            
                            # Test simple connection to Ollama
                            try:
                                test_response = ollama.list()
                                st.success("Ollama connection test successful")
                                st.json({"models_count": len(test_response.get('models', []))})
                            except Exception as conn_err:
                                st.error(f"Ollama connection test failed: {str(conn_err)}")
                    
                    # Stream handler function for updating UI in real-time
                    def handle_stream(agent_id, agent_name, content, is_complete):
                        if not st.session_state.show_streaming:
                            return
                        
                        try:    
                            with stream_container.container():
                                if agent_id not in response_placeholders:
                                    # First time seeing this agent in the stream
                                    st.markdown(f"### {agent_name} is responding...")
                                    response_placeholders[agent_id] = st.empty()
                                
                                # Update the placeholder with the latest content
                                if content:  # Check for empty content
                                    response_placeholders[agent_id].markdown(content)
                                
                                # Show completion status
                                if is_complete:
                                    st.markdown(f"""
                                    <div class="success">
                                        <p>✅ {agent_name} has finished responding.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except Exception as e:
                            # Fallback if streaming update fails
                            st.error(f"Error updating streaming UI: {str(e)}")
                    
                    with st.spinner("💭 Discussion in progress..."):
                        if (continue_discussion or st.session_state.get("continue_next_round", False)) and st.session_state.current_discussion_id:
                            # Reset the continue flag if it was set
                            if st.session_state.get("continue_next_round", False):
                                st.session_state.continue_next_round = False
                                
                            # Continue existing discussion
                            discussion_id = st.session_state.orchestrator.start_discussion(
                                topic, 
                                "main", 
                                discussion_id=st.session_state.current_discussion_id,
                                rounds=rounds,
                                stream_handler=handle_stream if st.session_state.show_streaming else None
                            )
                        else:
                            # Start new discussion
                            discussion_id = st.session_state.orchestrator.start_discussion(
                                topic, 
                                "main",
                                rounds=rounds,
                                stream_handler=handle_stream if st.session_state.show_streaming else None
                            )
                        st.session_state.current_discussion_id = discussion_id
                    
                    # Clear streaming container when done
                    stream_container.empty()
            
            if st.session_state.current_discussion_id:
                # Show horizontal divider
                st.markdown("---")
                
                # Improved discussion results display
                st.markdown("## 📄 Discussion Results")
                st.markdown(f"**Topic:** {st.session_state.orchestrator.get_discussion_history(st.session_state.current_discussion_id)[0]['content'][:100]}...")
                
                # Get discussion history and sort by round and timestamp
                discussion_history = st.session_state.orchestrator.get_discussion_history(
                    st.session_state.current_discussion_id
                )
                discussion_history.sort(key=lambda x: (x.get('round', 0), x['timestamp']))
                
                # Group entries by round
                rounds = {}
                for entry in discussion_history:
                    round_num = entry.get('round', 0)
                    if round_num not in rounds:
                        rounds[round_num] = []
                    rounds[round_num].append(entry)
                
                # Display discussion by rounds with improved styling
                for round_num in sorted(rounds.keys()):
                    if round_num == 0:
                        st.markdown("""
                        <div style="background-color: #EFF6FF; padding: 0.5rem 1rem; border-radius: 6px; margin-top: 1rem;">
                            <h3>📝 Initial Thoughts</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    elif round_num == max(rounds.keys()):
                        st.markdown("""
                        <div style="background-color: #ECFDF5; padding: 0.5rem 1rem; border-radius: 6px; margin-top: 1rem;">
                            <h3>🏁 Final Synthesis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #F3F4F6; padding: 0.5rem 1rem; border-radius: 6px; margin-top: 1rem;">
                            <h3>🔄 Round {round_num}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # In the final round, we should only have the synthesizer's response
                    is_final_round = (round_num == max(rounds.keys()))
                    
                    for entry in rounds[round_num]:
                        agent_id = entry["agent"]
                        
                        # Handle user messages specially
                        if agent_id == "user":
                            # Display user messages with a special style
                            st.markdown(f"""
                            <div style="background-color: #EBF8FF; padding: 1rem; border-radius: 6px; border-left: 4px solid #3182CE; margin: 1rem 0;">
                                <h4 style="color: #2C5282; margin-top: 0;">Your Input</h4>
                                <div style="margin-top: 0.5rem; color: #111827;">
                                    {entry["content"]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            continue
                            
                        # Regular agent messages
                        agent = st.session_state.orchestrator.agents[agent_id]
                        
                        # For final round, use a different style to emphasize it's the synthesis
                        if is_final_round:
                            st.markdown(f"""
                            <div style="background-color: #F0FDF4; padding: 1rem; border-radius: 6px; border-left: 4px solid #047857; margin: 1rem 0;">
                                <h4 style="color: #047857; margin-top: 0;">{agent.agent_name}'s Final Synthesis</h4>
                                <div style="margin-top: 0.5rem; color: #111827;">
                                    {entry["content"]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # For other rounds, use expandable sections with improved styling
                            with st.expander(f"💬 {agent.agent_name} says:", expanded=True):
                                st.markdown(entry["content"])
                
                # Add buttons for continuing or starting new discussions
                st.markdown("### What's Next?")
                
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button("🔄 Continue Discussion", key="continue_discussion_btn", use_container_width=True):
                        # Set flag in session state instead of immediate rerun
                        st.session_state.continue_discussion = True
                        # Add an input field for user contributions
                        user_input = st.text_area("Add your thoughts to the discussion (optional):", 
                                                  key="user_discussion_input", 
                                                  height=100)
                        
                        if st.button("Submit & Continue", key="submit_continue_btn"):
                            # If user added input, add it to the discussion history
                            if user_input.strip():
                                # Get the current discussion history
                                discussion_id = st.session_state.current_discussion_id
                                if discussion_id:
                                    # Add user contribution to history
                                    st.session_state.orchestrator.discussion_history[discussion_id].append({
                                        "agent": "user",
                                        "content": user_input,
                                        "timestamp": time.time(),
                                        "round": max([msg.get('round', 0) for msg in 
                                                    st.session_state.orchestrator.get_discussion_history(discussion_id)])
                                    })
                            # Continue with the next round
                            st.session_state.continue_next_round = True
                        
                with button_col2:
                    if st.button("🆕 Start New Discussion", key="new_discussion_btn", use_container_width=True):
                        st.session_state.current_discussion_id = None

if __name__ == "__main__":
    main()