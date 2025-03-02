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
from src.advanced_config import PersonaManager, advanced_config_main

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
Feel free to ask open questions that other participants can build upon."""
            
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

Please summarize the key points from our conversation, highlighting areas of agreement and interesting perspectives.
This should feel like a natural wrap-up to our coffee shop discussion - conversational but insightful.
Make it clear where different participants contributed valuable ideas, and how they built on each other.
Acknowledge any open questions that might remain for future discussions.""")
                else:
                    # Response from discussion agent - more conversational
                    context_parts.insert(0, """Here's how our conversation is going so far. 

Jump in naturally as if you're part of this ongoing coffee shop chat.
Respond directly to specific points others have made - agree, disagree, ask follow-up questions, or build on their ideas.
Keep your tone casual and conversational, but still thoughtful.
Feel free to use conversational language like "I see what you're saying about X, but have you considered Y?" or "That's an interesting point! I'd add that..."
Share personal perspectives or examples if relevant.""")
                
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
    """Check Ollama for available models and update the session state"""
    try:
        response = ollama.list()
        if 'models' not in response:
            st.error("Unexpected response from Ollama API: 'models' key missing")
            st.session_state.available_models = []
            return False
            
        models = response['models']
        model_names = []
        
        # Debug the response format
        with st.expander("Debug: Ollama API Response Format"):
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
                model_names.append(model['name'])
            elif hasattr(model, 'model') and isinstance(model.model, str):
                # Handle object format where model name is in a 'model' attribute
                model_names.append(model.model)
            else:
                # Try to extract name if it's an object with __dict__
                if hasattr(model, '__dict__'):
                    model_dict = model.__dict__
                    if 'model' in model_dict and isinstance(model_dict['model'], str):
                        model_names.append(model_dict['model'])
                        continue
                
                st.warning(f"Skipping model with invalid format: {model}")
                
        st.session_state.available_models = model_names
        
        # Log the detected models
        st.info(f"Detected {len(model_names)} models: {', '.join(model_names) if model_names else 'None'}")
        
        return True
    except KeyError as e:
        st.error(f"Error parsing Ollama response: {str(e)}")
        st.session_state.available_models = []
        return False
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.session_state.available_models = []
        
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
                
        return False

# GUI Application
def main():
    st.set_page_config(page_title="Multi-Agent Discussion", layout="wide")
    
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
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = {
        "Multi-Agent Discussion": "main_discussion",
        "Advanced Configuration": "advanced_config"
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display the selected page
    if pages[selection] == "main_discussion":
        display_main_discussion_ui()
    elif pages[selection] == "advanced_config":
        advanced_config_main()

def display_main_discussion_ui():
    """Main discussion UI"""
    st.title("Multi-Agent Discussion Platform")
    
    # Check if Ollama is available
    if not st.session_state.available_models:
        if not refresh_available_models():
            st.info("Please make sure Ollama is running. Go to Advanced Configuration to troubleshoot.")
            return
    
    # Setup tab and Discussion tab
    tab1, tab2 = st.tabs(["Setup Agents", "Discussion"])
    
    with tab1:
        st.header("Configure Your Agents")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.available_models:
                st.warning("No Ollama models detected. Make sure Ollama is running and models are available.")
        
        with col2:
            if st.button("Refresh Ollama Models"):
                if refresh_available_models():
                    st.success(f"Found {len(st.session_state.available_models)} models!")
                    st.experimental_rerun()
        
        # Main agent setup
        st.subheader("Main Agent (The one that will report back to you)")
        main_col1, main_col2 = st.columns(2)
        with main_col1:
            main_agent_name = st.text_input("Main Agent Name", "Synthesizer")
            main_agent_model = st.selectbox(
                "Main Agent Model", 
                st.session_state.available_models if st.session_state.available_models else ["No models available"]
            )
        with main_col2:
            main_agent_prompt = st.text_area(
                "Main Agent System Prompt",
                """You are a thoughtful conversation facilitator in a casual group discussion.

Your role is to guide a natural, engaging conversation among different participants with diverse perspectives.
Speak in a conversational, friendly tone - like a good host at a coffee shop discussion.
Ask questions that draw out others' perspectives and encourage them to build on each other's ideas.
When synthesizing the discussion, highlight how ideas connected and evolved through conversation.
Reference specific points made by participants by name, showing how the conversation built toward insights.
Use natural language like 'I think,' 'it seems like,' and other conversational markers.
Your final summary should feel like a natural conclusion to a rich discussion among friends."""
            )
        
        # Additional agents setup
        st.subheader("Discussion Agents")
        
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
        
        for i in range(3):  # Allow up to 3 additional agents
            st.markdown(f"#### Agent {i+1}")
            col1, col2 = st.columns(2)
            
            with col1:
                agent_name = st.text_input(f"Agent {i+1} Name", f"Agent {i+1}")
                agent_model = st.selectbox(
                    f"Agent {i+1} Model", 
                    st.session_state.available_models if st.session_state.available_models else ["No models available"],
                    key=f"model_{i}"
                )
                persona_select = st.selectbox(
                    f"Select a persona template or create custom", 
                    ["Custom"] + list(example_personas.keys()),
                    key=f"persona_{i}"
                )
            
            with col2:
                if persona_select != "Custom":
                    agent_prompt = st.text_area(
                        f"Agent {i+1} System Prompt", 
                        example_personas[persona_select],
                        key=f"prompt_{i}"
                    )
                else:
                    agent_prompt = st.text_area(
                        f"Agent {i+1} System Prompt", 
                        f"""You are a participant in a casual, thoughtful group conversation.

Speak in a natural, conversational tone as if chatting with friends at a coffee shop.
Respond directly to points made by others, building on their ideas or offering alternative perspectives.
Use first-person language, occasional questions, and conversational phrases like "I think," "I wonder if," etc.
Share your unique perspective while engaging meaningfully with what others have said.
Feel free to use analogies, examples, or personal anecdotes to illustrate your points in a relatable way.""",
                        key=f"prompt_{i}"
                    )
            
            include_agent = st.checkbox(f"Include Agent {i+1} in discussions", value=(i==0))
            
            if include_agent:
                agent_configs.append({
                    "name": agent_name,
                    "model": agent_model,
                    "prompt": agent_prompt
                })
        
        if st.button("Configure Agents"):
            if not st.session_state.available_models:
                st.error("No models available. Make sure Ollama is running.")
                # Add a refresh button right next to the error
                if st.button("Try Refreshing Models", key="refresh_on_config"):
                    if refresh_available_models():
                        st.success(f"Found {len(st.session_state.available_models)} models!")
                        st.experimental_rerun()
                    else:
                        st.warning("Please go to Advanced Configuration > Ollama Configuration to troubleshoot.")
            elif len(agent_configs) == 0:
                st.error("Please include at least one discussion agent.")
            else:
                try:
                    orchestrator = DiscussionOrchestrator()
                    
                    # Validate that selected models exist in available_models
                    if main_agent_model not in st.session_state.available_models:
                        st.error(f"Main agent model '{main_agent_model}' is not available. Please select a different model.")
                        return
                    
                    # Create main agent
                    main_agent = ModelAgent(main_agent_model, main_agent_prompt, main_agent_name)
                    orchestrator.add_agent("main", main_agent)
                    
                    # Create discussion agents
                    for i, config in enumerate(agent_configs):
                        if config["model"] not in st.session_state.available_models:
                            st.error(f"Agent {i+1} model '{config['model']}' is not available. Please select a different model.")
                            return
                            
                        agent = ModelAgent(config["model"], config["prompt"], config["name"])
                        orchestrator.add_agent(f"agent_{i}", agent)
                    
                    # Test the main agent with a simple query to verify Ollama connection
                    with st.spinner("Testing connection to Ollama..."):
                        try:
                            # Simple test to ensure the model loads properly
                            test_result = ollama.chat(
                                model=main_agent_model,
                                messages=[
                                    {"role": "user", "content": "Respond with only the word 'OK' for connection test"}
                                ]
                            )
                            st.success("Connection to Ollama verified successfully!")
                        except Exception as test_error:
                            st.error(f"Error testing Ollama connection: {str(test_error)}")
                            with st.expander("Debug Details"):
                                st.code(traceback.format_exc())
                            return
                    
                    st.session_state.orchestrator = orchestrator
                    st.session_state.agents_configured = True
                    st.success("Agents configured successfully!")
                except Exception as e:
                    st.error(f"Error configuring agents: {str(e)}")
                    with st.expander("Debug Details"):
                        st.code(traceback.format_exc())
    
    with tab2:
        st.header("Start a Discussion")
        
        if not st.session_state.agents_configured:
            st.warning("Please configure your agents in the Setup tab first.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                topic = st.text_area("Enter a topic or question for discussion:", height=100)
            
            with col2:
                rounds = st.number_input("Discussion rounds:", min_value=1, max_value=5, value=2, 
                                        help="Number of back-and-forth exchanges between agents")
                continue_discussion = st.checkbox("Continue previous discussion", value=False,
                                                disabled=not st.session_state.current_discussion_id)
                
                start_btn = st.button("Start Discussion")
                # Initialize streaming placeholders if not present
                if "streaming_responses" not in st.session_state:
                    st.session_state.streaming_responses = {}
                
                # Initialize streaming UI controls
                if "show_streaming" not in st.session_state:
                    st.session_state.show_streaming = True
                
                if "debug_mode" not in st.session_state:
                    st.session_state.debug_mode = False
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    streaming_toggle = st.checkbox("Show streaming responses", value=st.session_state.show_streaming)
                    st.session_state.show_streaming = streaming_toggle
                
                with col2:
                    debug_toggle = st.checkbox("Debug mode", value=st.session_state.debug_mode)
                    st.session_state.debug_mode = debug_toggle
                
                if start_btn:
                    if not topic:
                        st.error("Please enter a topic for discussion.")
                    else:
                        # Create placeholders for streaming responses
                        response_placeholders = {}
                        stream_container = st.empty()
                        
                        # Add debug container if debug mode is enabled
                        if st.session_state.debug_mode:
                            debug_container = st.container()
                            with debug_container:
                                st.subheader("Debug Information")
                                st.write("Ollama models:", st.session_state.available_models)
                                
                                # Test simple connection to Ollama
                                try:
                                    test_response = ollama.list()
                                    st.write("Ollama connection test successful")
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
                                        st.success(f"{agent_name} has finished responding.")
                            except Exception as e:
                                # Fallback if streaming update fails
                                st.error(f"Error updating streaming UI: {str(e)}")
                        
                        with st.spinner("Discussion in progress..."):
                            if continue_discussion and st.session_state.current_discussion_id:
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
                st.subheader("Discussion Results")
                
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
                
                # Display discussion by rounds
                for round_num in sorted(rounds.keys()):
                    if round_num == 0:
                        st.markdown("### 📝 Initial Thoughts")
                    elif round_num == max(rounds.keys()):
                        st.markdown("### 🏁 Final Synthesis")
                    else:
                        st.markdown(f"### 🔄 Round {round_num}")
                    
                    # In the final round, we should only have the synthesizer's response
                    is_final_round = (round_num == max(rounds.keys()))
                    
                    for entry in rounds[round_num]:
                        agent_id = entry["agent"]
                        agent = st.session_state.orchestrator.agents[agent_id]
                        
                        # For final round, use a different style to emphasize it's the synthesis
                        if is_final_round:
                            st.markdown(f"**{agent.agent_name}'s Final Synthesis:**")
                            st.markdown(entry["content"])
                        else:
                            # For other rounds, use expandable sections
                            with st.expander(f"{agent.agent_name} says:", expanded=True):
                                st.markdown(entry["content"])
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Continue Discussion"):
                        st.session_state.continue_discussion = True
                        st.experimental_rerun()
                        
                with col2:
                    if st.button("Start New Discussion"):
                        st.session_state.current_discussion_id = None
                        st.experimental_rerun()

if __name__ == "__main__":
    main()