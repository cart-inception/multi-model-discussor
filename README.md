# Multi-Agent Discussion Platform

A powerful Streamlit application that facilitates rich, interactive discussions between multiple AI models, each with its own distinct personality and system prompt. This platform enables you to create a panel of AI personas that can discuss topics collaboratively, providing diverse perspectives and insights. Supports both local Ollama models and Google's Gemini API.



## Features

- **Multi-Agent Conversations**: Configure up to 4 AI agents (1 synthesizer + 3 discussion agents) with distinct personas
- **Natural Discussion Flow**: Agents build on each other's ideas in a realistic, conversational manner
- **Flexible Agent Configuration**: Choose from built-in personas or create custom ones with specific expertise
- **Interactive UI**: Watch responses stream in real-time and review discussion history
- **Sophisticated Orchestration**: Multi-round discussions with automatic context management
- **Advanced Persona Management**: Create, edit, and manage custom personas with the built-in editor
- **Multiple Model Support**:
  - **Ollama Integration**: Works with any model available in your local Ollama installation
  - **Google Gemini Integration**: Connect to Google's Gemini API models

## How It Works

The application uses a moderator-participant discussion model:

1. **Main Synthesizer Agent**: Acts as the discussion leader and synthesizes insights at the end
2. **Discussion Agents**: Contribute different perspectives based on their personas
3. **Multi-Round Format**: 
   - The synthesizer starts with initial thoughts
   - Discussion agents respond to the topic and each other
   - After configured rounds, the synthesizer provides a final summary

The orchestrator manages the conversation flow, ensuring each agent has appropriate context from previous messages and maintains a cohesive discussion.

## Requirements

- **Python 3.8+**
- **Streamlit 1.26.0+**: For the interactive web interface
- **At least one of the following model backends**:
  - **Ollama**: Running either locally or on a remote server, with local Ollama models (llama3, mistral, phi3, etc.)
  - **Google Gemini API key**: For accessing Google's Gemini models

## Detailed Setup Guide

### 1. Install Python Dependencies

First, ensure you have Python 3.8 or newer installed. Then install the required packages:

```bash
# Clone this repository
git clone https://github.com/yourusername/multi-agent-discussion.git
cd multi-agent-discussion

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Your Model Backend

The application supports two model backends:

#### Option A: Ollama (Local Models)

1. Install Ollama following the [official instructions](https://ollama.ai/download) for your platform
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull at least one model to use with the application:
   ```bash
   ollama pull llama3  # or any other model you prefer
   ```

You can also configure Ollama to run on a different machine by setting the `OLLAMA_HOST` environment variable or configuring it through the application's Advanced Configuration page.

#### Option B: Google Gemini API

1. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Store your API key in the application:
   - Option 1: Set an environment variable before starting the app:
     ```bash
     export GOOGLE_API_KEY=your_api_key_here
     ```
   - Option 2: Enter your API key in the Advanced Configuration > Google Gemini section of the application

Note: You can set up both backends to access a wider variety of models.

### 3. Launch the Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Your default web browser should open automatically to the application (typically at http://localhost:8501).

## Detailed Usage Guide

### Setting Up a Discussion

1. **Configure Your Agents**:
   - In the "Setup Agents" tab, set up your main synthesizer agent that will lead the discussion
   - Add 1-3 discussion agents with different personas
   - For each agent, select:
     - A name (e.g., "Creative Thinker", "Skeptical Analyst")
     - An Ollama model to power the agent (different models can provide more diverse perspectives)
     - A system prompt defining the agent's persona (use built-in templates or create custom ones)

2. **Start a Discussion**:
   - Switch to the "Discussion" tab
   - Enter a topic or question for discussion (more specific topics generally yield better results)
   - Choose the number of discussion rounds (2-3 rounds work well for most topics)
   - Click "Start Discussion" to begin

3. **View Results**:
   - Watch as agents respond in real-time (if streaming is enabled)
   - Review the complete discussion, organized by rounds
   - The final synthesis provides a comprehensive summary of the key insights

### Working with Personas

The application comes with several built-in personas and allows you to create custom ones:

1. Go to "Advanced Configuration" > "Persona Editor"
2. To create a new persona:
   - Enter a distinctive name
   - Craft a system prompt that defines the persona's characteristics, expertise, and communication style
   - Add an optional description for easy reference
   - Click "Save Persona"

3. To edit or delete existing personas:
   - Check "Edit existing persona"
   - Select the persona from the dropdown
   - Modify the system prompt or description
   - Click "Save Persona" to update or "Delete Persona" to remove

Personas are saved as JSON files in the `personas/` directory and will persist between sessions.

### Configuring Model Backends

#### Ollama Configuration

If you're running Ollama on a different machine or want to manage your models:

1. Go to "Advanced Configuration" > "Ollama Configuration"
2. Set your Ollama host URL (default is http://localhost:11434)
3. View available models and pull new ones if needed
4. Filter models by family (llama, mistral, phi, etc.)

#### Google Gemini Configuration

To set up or manage your Gemini API access:

1. Go to "Advanced Configuration" > "Google Gemini"
2. Enter your Google API key
3. Click "Save API Key" to store it for the session
4. View available Gemini models that will appear in agent configuration

## Advanced Features

### Streaming Responses

By default, the application streams responses in real-time. You can toggle this feature on/off:

- Enable streaming to see the thought process develop
- Disable streaming for faster results with complex topics

### Continuing Discussions

After a discussion completes:

1. Click "Continue Discussion" to add more rounds with the same agents and topic
2. Or "Start New Discussion" to begin fresh with a different topic

### Debug Mode

For troubleshooting:

1. Enable "Debug mode" in the Discussion tab
2. This shows additional information about model responses and connection status

## Code Architecture

The application consists of several key components:

- **`app.py`**: Main application file containing:
  - `ModelAgent` class: Handles individual agent interactions with Ollama
  - `DiscussionOrchestrator` class: Manages multi-agent conversations and rounds
  - Streamlit UI implementation for the main discussion interface

- **`advanced_config.py`**: Handles advanced configuration features:
  - `PersonaManager` class: Manages saving/loading custom personas
  - UI components for persona editing and Ollama configuration

- **`personas/` directory**: Stores custom persona configurations as JSON files

## Tips for Best Results

- **Model Diversity**: 
  - Use different models for different agents when possible
  - Mix Ollama and Gemini models for varied thinking styles
- **Agent Balance**: Include agents with complementary perspectives (e.g., creative + practical)
- **Specific Topics**: Focused questions yield more coherent discussions than broad topics
- **Persona Design**: Create personas with clear, distinct viewpoints and expertise areas
- **Round Count**: 2-3 rounds typically produce the best balance of depth and coherence
- **System Prompts**: Refine agent prompts to guide the style and focus of contributions

## Troubleshooting

### Ollama Connection Issues

If the application can't connect to Ollama:

1. Verify Ollama is running with `ollama serve`
2. Check the Ollama host URL in Advanced Configuration
3. Ensure no firewall is blocking port 11434
4. Test the connection directly with `curl http://localhost:11434/api/tags`

### Gemini API Issues

If you have problems with Google Gemini models:

1. Verify your API key is correct and properly configured
2. Check your API key permissions and quotas in Google AI Studio
3. Try refreshing the models list in the Gemini Configuration section
4. Check for any error messages in the API response

### Model Loading Problems

If models aren't appearing or loading:

1. Go to Advanced Configuration > Ollama Configuration or Google Gemini
2. Click "Refresh Models" to update the available model list
3. For Ollama models that are missing, pull them using Ollama or the UI
4. Verify model names in the application match those in your model providers

### UI Responsiveness

For better performance:

1. Reduce the number of discussion agents (1-2 works well for most topics)
2. Use smaller/faster models for quicker responses
3. Consider disabling streaming for faster completion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using [Streamlit](https://streamlit.io/) for the interactive UI
- Powered by [Ollama](https://ollama.ai/) for local LLM inference
- Integrated with [Google Gemini API](https://ai.google.dev/) for cloud-based models
- Inspired by various multi-agent AI collaboration frameworks