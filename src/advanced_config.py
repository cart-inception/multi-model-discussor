import streamlit as st
import json
import os
from pathlib import Path

class PersonaManager:
    """Manages saving and loading custom personas for agent configuration"""
    
    def __init__(self, save_dir="personas"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.personas = self._load_personas()
    
    def _load_personas(self):
        """Load all saved personas from disk"""
        personas = {}
        for file_path in self.save_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    persona = json.load(f)
                    personas[persona["name"]] = persona
            except Exception as e:
                print(f"Error loading persona from {file_path}: {e}")
        return personas
    
    def save_persona(self, name, system_prompt, description=""):
        """Save a persona configuration to disk"""
        file_path = self.save_dir / f"{name.lower().replace(' ', '_')}.json"
        persona = {
            "name": name,
            "system_prompt": system_prompt,
            "description": description
        }
        
        with open(file_path, "w") as f:
            json.dump(persona, f, indent=2)
        
        # Update in-memory cache
        self.personas[name] = persona
        return persona
    
    def delete_persona(self, name):
        """Delete a saved persona"""
        if name in self.personas:
            file_path = self.save_dir / f"{name.lower().replace(' ', '_')}.json"
            if file_path.exists():
                file_path.unlink()
            del self.personas[name]
    
    def get_persona(self, name):
        """Get a specific persona by name"""
        return self.personas.get(name)
    
    def get_all_personas(self):
        """Get all available personas"""
        return self.personas

def persona_editor_ui():
    """UI for managing custom personas"""
    st.markdown("## 👤 Persona Editor")
    st.markdown("Create and manage custom personas for your discussion agents")
    
    # Initialize persona manager
    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager()
    
    persona_manager = st.session_state.persona_manager
    
    # Create two columns for better layout
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        # UI for creating or editing personas in a nice card
        st.markdown("""
        <div class="config-card">
            <h3 style="margin-top:0">Create or Edit Persona</h3>
        </div>
        """, unsafe_allow_html=True)
        
        persona_name = st.text_input("Persona Name", placeholder="Enter a distinctive name...")
        
        # Load existing persona if selected
        edit_mode = st.radio("Mode", ["Create New", "Edit Existing"], horizontal=True)
        existing_name = None
        
        if edit_mode == "Edit Existing" and persona_manager.get_all_personas():
            existing_name = st.selectbox(
                "Select persona to edit",
                options=list(persona_manager.get_all_personas().keys())
            )
            if existing_name:
                persona = persona_manager.get_persona(existing_name)
                persona_name = existing_name
                system_prompt = st.text_area(
                    "System Prompt", 
                    persona["system_prompt"], 
                    height=200,
                    placeholder="Enter instructions that define this persona's behavior..."
                )
                description = st.text_area(
                    "Description", 
                    persona.get("description", ""),
                    placeholder="Enter a brief description of this persona's characteristics..."
                )
        else:
            system_prompt = st.text_area(
                "System Prompt", 
                "You are an AI assistant with a unique perspective.", 
                height=200,
                placeholder="Enter instructions that define this persona's behavior..."
            )
            description = st.text_area(
                "Description", 
                "",
                placeholder="Enter a brief description of this persona's characteristics..."
            )
        
        # Example templates with better styling
        with st.expander("📋 Example Templates"):
            st.markdown("Select a template to use as a starting point:")
            
            example_templates = {
                "Expert": """You are an expert in {FIELD} with many years of experience.

You provide detailed, nuanced answers drawing on deep technical knowledge.
You prioritize accuracy and practical insights in your responses.
You often use specialized terminology from your field, but explain it clearly.
You cite research, evidence, or practical examples to support your points.
You acknowledge limitations and uncertainties when they exist.""",

                "Character": """You are {CHARACTER}, a fictional character from {SOURCE}.

Respond as this character would, using their typical speech patterns and catchphrases.
Express the attitudes, beliefs, and perspectives that define this character.
Reference events, relationships, and knowledge that would be familiar to this character.
Stay consistent with the character's personality traits and background story.
Avoid breaking character or acknowledging that you are an AI.""",

                "Contrarian": """You are a thoughtful but contrarian thinker who often challenges conventional wisdom.

Look for assumptions in the discussion that should be questioned or challenged.
Offer alternative perspectives that others might be overlooking.
Play devil's advocate when a discussion seems too one-sided.
Be constructively critical, not negative or dismissive.
Ground your contrarian views in logic and evidence when possible."""
            }
            
            template_choice = st.selectbox(
                "Select a template",
                ["Choose a template", "Expert", "Character", "Contrarian"],
                index=0
            )
            
            if template_choice != "Choose a template":
                system_prompt = example_templates[template_choice]
                st.info(f"Template inserted. Customize it by replacing placeholder text like {{FIELD}} with specific details.")
        
        # Action buttons with better styling
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            save_button = st.button("💾 Save Persona", use_container_width=True)
            if save_button:
                if not persona_name:
                    st.markdown("""
                    <div class="warning">
                        <h4>⚠️ Missing Name</h4>
                        <p>Please provide a name for the persona</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif not system_prompt:
                    st.markdown("""
                    <div class="warning">
                        <h4>⚠️ Missing Prompt</h4>
                        <p>System prompt cannot be empty</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    persona_manager.save_persona(persona_name, system_prompt, description)
                    st.markdown(f"""
                    <div class="success">
                        <h4>✅ Saved Successfully</h4>
                        <p>Persona '{persona_name}' has been saved and is ready to use</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with button_col2:
            if existing_name and edit_mode == "Edit Existing":
                delete_button = st.button("🗑️ Delete Persona", use_container_width=True)
                if delete_button:
                    persona_manager.delete_persona(existing_name)
                    st.markdown(f"""
                    <div class="success">
                        <h4>✅ Deleted</h4>
                        <p>Persona '{existing_name}' has been removed</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.experimental_rerun()
    
    with right_col:
        # Display all available personas with improved styling
        st.markdown("""
        <div class="config-card">
            <h3 style="margin-top:0">Available Personas</h3>
        </div>
        """, unsafe_allow_html=True)
        
        personas = persona_manager.get_all_personas()
        
        if not personas:
            st.markdown("""
            <div class="info" style="text-align: center;">
                <p>No custom personas created yet</p>
                <p>Create your first persona using the form on the left</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for name, persona in personas.items():
                with st.expander(f"👤 {name}"):
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <strong>Description:</strong><br>
                        {persona.get('description', 'No description provided')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<strong>System Prompt:</strong>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background-color: #F1F5F9; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; color: #111827;">
                    {persona["system_prompt"]}
                    </div>
                    """, unsafe_allow_html=True)

def gemini_config_ui():
    """UI for configuring Google Gemini API settings"""
    st.markdown("## 🧠 Google Gemini Configuration")
    st.markdown("Configure Google Gemini API settings and available models")
    
    # Configure Gemini API key in a card-like container
    st.markdown("""
    <div class="config-card">
        <h3 style="margin-top:0">API Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    server_container = st.container()
    with server_container:
        default_api_key = os.environ.get("GOOGLE_API_KEY", "")
        
        # Use password input for API key
        gemini_api_key = st.text_input(
            "Google API Key", 
            value=default_api_key,
            type="password",
            help="Your Google Gemini API key"
        )
        
        st.markdown("Set your Google API key to access Gemini models")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save API Key", key="gemini_save_key_btn", use_container_width=True):
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
                # Also save the API key to the session state
                if "gemini_api_key" not in st.session_state:
                    st.session_state.gemini_api_key = gemini_api_key
                else:
                    st.session_state.gemini_api_key = gemini_api_key
                    
                st.markdown(f"""
                <div class="success">
                    <h4>✅ API Key Saved</h4>
                    <p>Google API key has been saved for this session</p>
                </div>
                """, unsafe_allow_html=True)
                
                # If API key is valid, try to refresh models
                if gemini_api_key:
                    try:
                        # Import here to test the API key
                        import google.generativeai as genai
                        
                        # Configure the API key
                        genai.configure(api_key=gemini_api_key)
                        
                        # List available models to verify API key works
                        models = genai.list_models()
                        
                        # Add the Gemini models to available_models in session state
                        gemini_models = []
                        for model in models:
                            if "generateContent" in model.supported_generation_methods:
                                # Add a prefix to distinguish from Ollama models
                                gemini_models.append(f"gemini:{model.name}")
                        
                        # Store in session state
                        if "gemini_models" not in st.session_state:
                            st.session_state.gemini_models = gemini_models
                        else:
                            st.session_state.gemini_models = gemini_models
                            
                        # Add to available models
                        if "available_models" in st.session_state:
                            # Filter out any existing Gemini models
                            ollama_models = [m for m in st.session_state.available_models if not m.startswith("gemini:")]
                            # Add the updated Gemini models
                            st.session_state.available_models = ollama_models + gemini_models
                            
                        st.markdown(f"""
                        <div class="success">
                            <h4>✅ Models Refreshed</h4>
                            <p>Successfully retrieved {len(gemini_models)} Gemini models</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="warning">
                            <h4>⚠️ API Error</h4>
                            <p>Error connecting to Google Gemini API: {str(e)}</p>
                            <p>Please check your API key</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🔄 Refresh Models", key="gemini_refresh_btn", use_container_width=True):
                if not gemini_api_key:
                    st.markdown(f"""
                    <div class="warning">
                        <h4>⚠️ Missing API Key</h4>
                        <p>Please enter your Google API key first</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    try:
                        # Import here to test the API key
                        import google.generativeai as genai
                        
                        # Configure the API key
                        genai.configure(api_key=gemini_api_key)
                        
                        # List available models to verify API key works
                        models = genai.list_models()
                        
                        # Add the Gemini models to available_models in session state
                        gemini_models = []
                        for model in models:
                            if "generateContent" in model.supported_generation_methods:
                                # Add a prefix to distinguish from Ollama models
                                gemini_models.append(f"gemini:{model.name}")
                        
                        # Store in session state
                        if "gemini_models" not in st.session_state:
                            st.session_state.gemini_models = gemini_models
                        else:
                            st.session_state.gemini_models = gemini_models
                            
                        # Add to available models
                        if "available_models" in st.session_state:
                            # Filter out any existing Gemini models
                            ollama_models = [m for m in st.session_state.available_models if not m.startswith("gemini:")]
                            # Add the updated Gemini models
                            st.session_state.available_models = ollama_models + gemini_models
                            
                        st.markdown(f"""
                        <div class="success">
                            <h4>✅ Models Refreshed</h4>
                            <p>Successfully retrieved {len(gemini_models)} Gemini models</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="warning">
                            <h4>⚠️ API Error</h4>
                            <p>Error connecting to Google Gemini API: {str(e)}</p>
                            <p>Please check your API key</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Status indicator
    try:
        if gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            
            # Try to list models to verify connection
            models = genai.list_models()
            gemini_models = [m for m in models if "generateContent" in m.supported_generation_methods]
            
            if gemini_models:
                st.markdown(f"""
                <div class="success" style="display:flex; align-items:center;">
                    <div style="font-size:24px; margin-right:10px;">🟢</div>
                    <div>
                        <p style="margin:0; font-weight:500;">Connected to Google Gemini API</p>
                        <p style="margin:0; font-size:14px;">Found {len(gemini_models)} available models</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning" style="display:flex; align-items:center;">
                    <div style="font-size:24px; margin-right:10px;">🟡</div>
                    <div>
                        <p style="margin:0; font-weight:500;">Connected to Google Gemini API but no usable models found</p>
                        <p style="margin:0; font-size:14px;">Your API key may not have access to any models</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning" style="display:flex; align-items:center;">
                <div style="font-size:24px; margin-right:10px;">⚪</div>
                <div>
                    <p style="margin:0; font-weight:500;">Google Gemini API not configured</p>
                    <p style="margin:0; font-size:14px;">Please enter your API key above</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning" style="display:flex; align-items:center;">
            <div style="font-size:24px; margin-right:10px;">🔴</div>
            <div>
                <p style="margin:0; font-weight:500;">Cannot connect to Google Gemini API</p>
                <p style="margin:0; font-size:14px;">Error: {str(e)}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display available models with improved UI
    if "gemini_models" in st.session_state and st.session_state.gemini_models:
        st.markdown("""
        <div class="config-card" style="margin-top:20px;">
            <h3 style="margin-top:0">Available Gemini Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a grid of models with 2 columns
        col1, col2 = st.columns(2)
        columns = [col1, col2]
        
        for i, model in enumerate(st.session_state.gemini_models):
            # Extract the model name without prefix
            model_name = model.replace("gemini:", "")
            
            # Use alternating columns
            col = columns[i % 2]
            
            with col:
                with st.expander(f"🧠 {model_name}"):
                    model_display_name = model_name.split("/")[-1] if "/" in model_name else model_name
                    st.markdown(f"""
                    <div style="background-color: #F9FAFB; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
                        <p style="margin: 0;"><strong>Model:</strong> {model_display_name}</p>
                        <p style="margin: 0;"><strong>Full name:</strong> {model_name}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        if gemini_api_key:
            st.markdown("""
            <div class="info" style="text-align: center; padding: 20px;">
                <h4>No Gemini Models Found</h4>
                <p>No models are currently available with your API key.</p>
                <p>Make sure your API key has access to Gemini models.</p>
            </div>
            """, unsafe_allow_html=True)
        
    # API usage information
    st.markdown("""
    <div class="config-card" style="margin-top:20px;">
        <h3 style="margin-top:0">Google Gemini API Information</h3>
        <p>To use Google Gemini models, you need an API key from Google AI Studio:</p>
        <ol>
            <li>Visit <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></li>
            <li>Create or sign in with your Google account</li>
            <li>Get an API key and paste it in the configuration above</li>
        </ol>
        <p><strong>Note:</strong> API usage may incur charges based on your Google Cloud account settings.</p>
    </div>
    """, unsafe_allow_html=True)

def ollama_config_ui():
    """UI for configuring Ollama settings"""
    st.markdown("## 🖥️ Ollama Configuration")
    st.markdown("Configure Ollama server settings and manage models")
    
    # Import the refresh_available_models function
    from src.app import refresh_available_models
    
    # Configure Ollama host in a card-like container
    st.markdown("""
    <div class="config-card">
        <h3 style="margin-top:0">Server Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    server_container = st.container()
    with server_container:
        default_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ollama_host = st.text_input(
            "Ollama Host URL", 
            value=default_host,
            help="The URL where Ollama server is running (default: http://localhost:11434)"
        )
        
        st.markdown("Set the Ollama server URL and update model list")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Configuration", use_container_width=True):
                os.environ["OLLAMA_HOST"] = ollama_host
                st.markdown(f"""
                <div class="success">
                    <h4>✅ Configuration Saved</h4>
                    <p>Ollama host set to {ollama_host}</p>
                </div>
                """, unsafe_allow_html=True)
                # Refresh models after changing host
                refresh_available_models()
        
        with col2:
            if st.button("🔄 Refresh Models", use_container_width=True):
                if refresh_available_models():
                    st.markdown(f"""
                    <div class="success">
                        <h4>✅ Models Refreshed</h4>
                        <p>Successfully retrieved models from Ollama</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning">
                        <h4>⚠️ Refresh Failed</h4>
                        <p>Could not connect to Ollama. Check server settings.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Status indicator
    try:
        import ollama
        response = ollama.list()
        
        if 'models' in response and response['models']:
            st.markdown(f"""
            <div class="success" style="display:flex; align-items:center;">
                <div style="font-size:24px; margin-right:10px;">🟢</div>
                <div>
                    <p style="margin:0; font-weight:500;">Connected to Ollama</p>
                    <p style="margin:0; font-size:14px;">Server: {ollama_host}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning" style="display:flex; align-items:center;">
                <div style="font-size:24px; margin-right:10px;">🟡</div>
                <div>
                    <p style="margin:0; font-weight:500;">Connected to Ollama but no models found</p>
                    <p style="margin:0; font-size:14px;">You may need to pull models</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.markdown(f"""
        <div class="warning" style="display:flex; align-items:center;">
            <div style="font-size:24px; margin-right:10px;">🔴</div>
            <div>
                <p style="margin:0; font-weight:500;">Cannot connect to Ollama</p>
                <p style="margin:0; font-size:14px;">Check server URL and make sure Ollama is running</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display available models with improved UI
    st.markdown("""
    <div class="config-card" style="margin-top:20px;">
        <h3 style="margin-top:0">Available Models</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        import ollama
        response = ollama.list()
        
        if 'models' not in response:
            st.markdown("""
            <div class="warning">
                <h4>⚠️ API Error</h4>
                <p>Unexpected response from Ollama API: 'models' key missing</p>
            </div>
            """, unsafe_allow_html=True)
            return
            
        models = response['models']
        model_names = []
        model_tags = set()
        
        # Debug in expandable section
        with st.expander("🔍 Debug: Ollama Model Data", expanded=False):
            st.write("Response type:", type(response))
            st.write("Models type:", type(models))
            if models and len(models) > 0:
                st.write("First model type:", type(models[0]))
                st.write("First model data:", models[0])
        
        # Extract model names and tags
        for model in models:
            model_name = None
            
            # Handle dictionary format
            if isinstance(model, dict) and 'name' in model:
                model_name = model['name']
            
            # Handle object format with model attribute
            elif hasattr(model, 'model') and isinstance(model.model, str):
                model_name = model.model
            
            # Try to extract from __dict__ if available
            elif hasattr(model, '__dict__'):
                model_dict = model.__dict__
                if 'model' in model_dict and isinstance(model_dict['model'], str):
                    model_name = model_dict['model']
            
            if model_name:
                model_names.append(model_name)
                # Extract model family (part before the colon)
                parts = model_name.split(':', 1)
                if len(parts) > 0:
                    model_tags.add(parts[0])
            else:
                st.warning(f"Skipping model with invalid format: {model}")
        
        # Filter UI with improved styling
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Found {len(models)} models** on your Ollama server")
            filter_tag = st.selectbox(
                "Filter by model family", 
                ["All"] + sorted(list(model_tags)),
                help="Show only models from a specific family (e.g., llama, mistral)"
            )
        
        with col2:
            # Button to update available models in session state
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("📋 Update Selection List", use_container_width=True):
                if "available_models" in st.session_state:
                    # Make sure to preserve any Gemini models that might be in the session
                    gemini_models = []
                    if "gemini_models" in st.session_state and st.session_state.gemini_models:
                        gemini_models = st.session_state.gemini_models
                    
                    # Update the available models with both Ollama and Gemini models
                    st.session_state.available_models = model_names + gemini_models
                    
                    st.markdown(f"""
                    <div class="success">
                        <p>✅ Updated {len(model_names)} Ollama models for agent selection!</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Filter models based on tag
        filtered_models = []
        if filter_tag == "All":
            filtered_models = models
        else:
            for m in models:
                # Check each model format for filtering
                if isinstance(m, dict) and 'name' in m and m['name'].startswith(f"{filter_tag}:"):
                    filtered_models.append(m)
                elif hasattr(m, 'model') and isinstance(m.model, str) and m.model.startswith(f"{filter_tag}:"):
                    filtered_models.append(m)
                elif hasattr(m, '__dict__'):
                    m_dict = m.__dict__
                    if 'model' in m_dict and isinstance(m_dict['model'], str) and m_dict['model'].startswith(f"{filter_tag}:"):
                        filtered_models.append(m)
        
        # Display models in a grid layout
        if filtered_models:
            st.markdown("### Model Details")
            
            # Create a grid of models with 2 columns
            col1, col2 = st.columns(2)
            columns = [col1, col2]
            
            for i, model in enumerate(filtered_models):
                # Extract model information based on type
                model_name = None
                model_size = "Unknown"
                model_modified = "Unknown"
                
                # Dictionary format
                if isinstance(model, dict) and 'name' in model:
                    model_name = model['name']
                    model_size = model.get('size', 'Unknown')
                    model_modified = model.get('modified_at', 'Unknown')
                
                # Object format
                elif hasattr(model, 'model') and isinstance(model.model, str):
                    model_name = model.model
                    if hasattr(model, 'size'):
                        model_size = model.size
                    if hasattr(model, 'modified_at'):
                        model_modified = model.modified_at
                
                # Extract from __dict__
                elif hasattr(model, '__dict__'):
                    model_dict = model.__dict__
                    if 'model' in model_dict and isinstance(model_dict['model'], str):
                        model_name = model_dict['model']
                        model_size = model_dict.get('size', 'Unknown')
                        model_modified = model_dict.get('modified_at', 'Unknown')
                
                # Skip if no model name was found
                if not model_name:
                    continue
                
                # Convert size to human-readable format if it's a number
                if isinstance(model_size, (int, float)):
                    # Convert bytes to GB
                    model_size_gb = model_size / (1024 * 1024 * 1024)
                    model_size = f"{model_size_gb:.2f} GB"
                
                # Use alternating columns
                col = columns[i % 2]
                
                with col:
                    with st.expander(f"📦 {model_name}"):
                        st.markdown(f"""
                        <div style="background-color: #F9FAFB; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
                            <p style="margin: 0;"><strong>Size:</strong> {model_size}</p>
                            <p style="margin: 0;"><strong>Last Modified:</strong> {model_modified}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"↓ Pull Latest Version", key=f"pull_{model_name}"):
                            with st.spinner(f"Pulling latest version of {model_name}..."):
                                try:
                                    ollama.pull(model_name)
                                    st.markdown(f"""
                                    <div class="success">
                                        <p>✅ Successfully pulled latest version of {model_name}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Update available models after pulling
                                    try:
                                        # Import the function to refresh models
                                        from src.app import refresh_available_models
                                        refresh_available_models()
                                        st.success("Model list refreshed!")
                                    except Exception as pull_err:
                                        st.warning(f"Model pulled but failed to refresh model list: {str(pull_err)}")
                                except Exception as e:
                                    st.error(f"Error pulling model: {e}")
        else:
            if filter_tag == "All":
                st.markdown("""
                <div class="info" style="text-align: center; padding: 20px;">
                    <h4>No Models Found</h4>
                    <p>No models are currently available on your Ollama server.</p>
                    <p>You can pull models using the Ollama CLI with:</p>
                    <code>ollama pull llama3</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info" style="text-align: center; padding: 20px;">
                    <h4>No Models Found in "{filter_tag}" Family</h4>
                    <p>Try selecting a different model family or "All" to see all available models.</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="warning">
            <h4>⚠️ Connection Error</h4>
            <p>Error connecting to Ollama server: {e}</p>
            <p>Make sure Ollama is running and accessible at the configured host URL.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Troubleshooting tips in an expandable section
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            ## Troubleshooting Ollama Connection
            
            1. **Check if Ollama is running**
               - On Linux/Mac: `ps aux | grep ollama`
               - On Windows: Check Task Manager
            
            2. **Start Ollama if it's not running**
               - Run `ollama serve` in a terminal
               - On Windows, you may need to run the Ollama application
            
            3. **Verify the Ollama server URL**
               - Default is `http://localhost:11434`
               - If running on a different machine, use that IP address
               - Make sure to include the protocol (http://) and port
            
            4. **Check for firewall issues**
               - Make sure port 11434 is accessible
               - Temporarily disable firewall for testing
            
            5. **Test the API directly**
               - Try running: `curl http://localhost:11434/api/tags` in a terminal
               - Or open http://localhost:11434/api/tags in a browser
            
            6. **Restart Ollama**
               - Sometimes restarting the Ollama service can resolve connection issues
            """)

def advanced_config_main():
    """Main function for the advanced configuration UI"""
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
    /* Ensure code blocks have dark text */
    code {
        color: #111827 !important;
        background-color: #F1F5F9 !important;
    }
    /* Ensure selectbox text is visible */
    .stSelectbox div [data-baseweb="select"] div [data-testid="stMarkdown"] p {
        color: #111827 !important;
    }
    /* Fix monospace text background */
    .stMarkdown pre, .stMarkdown code {
        color: #111827 !important;
        background-color: #F1F5F9 !important;
    }
    /* Fix for system prompt display in expandable sections */
    .stExpander div [data-testid="stMarkdown"] div {
        color: #111827 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use tabs for better organization
    tabs = st.tabs(["👤 Persona Editor", "🖥️ Ollama Configuration", "🧠 Google Gemini"])
    
    with tabs[0]:
        persona_editor_ui()
    
    with tabs[1]:
        ollama_config_ui()
        
    with tabs[2]:
        gemini_config_ui()

if __name__ == "__main__":
    advanced_config_main()