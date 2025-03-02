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
    st.title("Persona Editor")
    
    # Initialize persona manager
    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager()
    
    persona_manager = st.session_state.persona_manager
    
    # UI for creating or editing personas
    st.header("Create or Edit Persona")
    persona_name = st.text_input("Persona Name")
    
    # Load existing persona if selected
    edit_existing = st.checkbox("Edit existing persona", value=False)
    existing_name = None
    
    if edit_existing and persona_manager.get_all_personas():
        existing_name = st.selectbox(
            "Select persona to edit",
            options=list(persona_manager.get_all_personas().keys())
        )
        if existing_name:
            persona = persona_manager.get_persona(existing_name)
            persona_name = existing_name
            system_prompt = st.text_area("System Prompt", persona["system_prompt"], height=200)
            description = st.text_area("Description", persona.get("description", ""))
    else:
        system_prompt = st.text_area(
            "System Prompt", 
            "You are an AI assistant with a unique perspective.", 
            height=200
        )
        description = st.text_area("Description (optional)", "")
    
    # Example templates to help users
    st.markdown("#### Example Templates:")
    example_templates = {
        "Expert": "You are an expert in {FIELD} with many years of experience. You provide detailed, nuanced answers drawing on deep technical knowledge. You prioritize accuracy and practical insights.",
        "Character": "You are {CHARACTER}, a fictional character from {SOURCE}. Respond as this character would, using their typical speech patterns, catchphrases, and attitudes. Stay true to the character's personality."
    }
    
    template_choice = st.selectbox("Insert template", ["Choose a template", "Expert", "Character"])
    if template_choice != "Choose a template":
        system_prompt = example_templates[template_choice]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Persona"):
            if not persona_name:
                st.error("Please provide a name for the persona")
            elif not system_prompt:
                st.error("System prompt cannot be empty")
            else:
                persona_manager.save_persona(persona_name, system_prompt, description)
                st.success(f"Persona '{persona_name}' saved successfully!")
    
    with col2:
        if existing_name and st.button("Delete Persona"):
            persona_manager.delete_persona(existing_name)
            st.success(f"Persona '{existing_name}' deleted")
            st.experimental_rerun()
    
    # Display all available personas
    st.header("Available Personas")
    personas = persona_manager.get_all_personas()
    
    if not personas:
        st.info("No custom personas created yet. Create one above!")
    else:
        for name, persona in personas.items():
            with st.expander(name):
                st.markdown(f"**Description:** {persona.get('description', 'No description')}")
                st.markdown("**System Prompt:**")
                st.text(persona["system_prompt"])

def ollama_config_ui():
    """UI for configuring Ollama settings"""
    st.title("Ollama Configuration")
    
    # Import the refresh_available_models function
    from src.app import refresh_available_models
    
    # Configure Ollama host
    st.header("Ollama Server Settings")
    
    default_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_host = st.text_input("Ollama Host URL", value=default_host)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Ollama Configuration"):
            os.environ["OLLAMA_HOST"] = ollama_host
            st.success(f"Ollama host set to {ollama_host}")
            # Refresh models after changing host
            refresh_available_models()
    
    with col2:
        if st.button("Refresh Models"):
            refresh_available_models()
            st.success("Models refreshed!")
    
    # Display available models and their details
    st.header("Available Models")
    
    try:
        import ollama
        response = ollama.list()
        
        if 'models' not in response:
            st.error("Unexpected response from Ollama API: 'models' key missing")
            return
            
        models = response['models']
        model_names = []
        model_tags = set()
        
        # Show the raw model data for debugging
        with st.expander("Debug: Ollama Model Data"):
            st.write("Response type:", type(response))
            st.write("Models type:", type(models))
            if models and len(models) > 0:
                st.write("First model type:", type(models[0]))
                st.write("First model data:", models[0])
        
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
        
        filter_tag = st.selectbox("Filter by model family", ["All"] + sorted(list(model_tags)))
        
        st.write(f"Found {len(models)} models on your Ollama server")
        
        # Button to update available models in session state
        if st.button("Update Models for Agent Selection"):
            if "available_models" in st.session_state:
                st.session_state.available_models = model_names
                st.success(f"Updated {len(model_names)} models for agent selection!")
        
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
        
        for model in filtered_models:
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
                
            with st.expander(model_name):
                st.write(f"**Size:** {model_size}")
                st.write(f"**Modified:** {model_modified}")
                
                if st.button(f"Pull latest version of {model_name}", key=f"pull_{model_name}"):
                    with st.spinner(f"Pulling latest version of {model_name}..."):
                        try:
                            ollama.pull(model_name)
                            st.success(f"Successfully pulled latest version of {model_name}")
                            
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
    
    except Exception as e:
        st.error(f"Error connecting to Ollama server: {e}")
        st.info("Make sure Ollama is running and accessible at the configured host URL")
        
        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            ## Troubleshooting Ollama Connection
            
            1. **Check if Ollama is running**
               - On Linux/Mac: `ps aux | grep ollama`
               - On Windows: Check Task Manager
            
            2. **Start Ollama if it's not running**
               - Run `ollama serve` in a terminal
            
            3. **Verify the Ollama server URL**
               - Default is `http://localhost:11434`
               - If running on a different machine, use that IP address
            
            4. **Check for firewall issues**
               - Make sure port 11434 is accessible
            
            5. **Test the API directly**
               - Try: `curl http://localhost:11434/api/tags`
            """)

def advanced_config_main():
    """Main function for the advanced configuration UI"""
    st.sidebar.title("Advanced Configuration")
    
    pages = {
        "Persona Editor": persona_editor_ui,
        "Ollama Configuration": ollama_config_ui
    }
    
    selection = st.sidebar.radio("Select Configuration", list(pages.keys()))
    
    # Display the selected page
    pages[selection]()

if __name__ == "__main__":
    advanced_config_main()