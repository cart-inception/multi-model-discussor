# Multi-Agent Discussion Platform

A Streamlit application that facilitates discussions between multiple Ollama AI models, each with its own distinct personality and system prompt.

## Features

- Configure multiple AI agents with different personas and personalities
- Run multi-round discussions between AI models
- Create and manage custom personas
- Configure Ollama settings
- Interactive UI with discussion visualization

## Requirements

- Python 3.8+
- Streamlit
- Ollama (running locally or on a remote server)
- Local Ollama models (e.g., llama3, mistral, phi3, etc.)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/multi-agent-discussion.git
cd multi-agent-discussion
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running (see [Ollama installation guide](https://ollama.ai/download))

## Usage

1. Start the Streamlit app:
```
streamlit run src/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Configure your agents:
   - Set up the main "synthesizer" agent that will consolidate insights
   - Add 1-3 discussion agents with different personas
   - Choose from built-in personas or create custom ones in the Advanced Configuration section

4. Start a discussion by entering a topic or question

5. View the results of the multi-agent discussion

## Creating Custom Personas

1. Navigate to the "Advanced Configuration" section
2. Select "Persona Editor"
3. Create a new persona with a name, system prompt, and optional description
4. Save your persona to use it in discussions

## Configuring Ollama

1. Navigate to the "Advanced Configuration" section
2. Select "Ollama Configuration"
3. Configure your Ollama host URL (default: http://localhost:11434)
4. View available models and pull new ones if needed

## Project Structure

```
multi-agent-discussion/
├── src/
│   ├── app.py                 # Main application file
│   └── advanced_config.py     # Advanced configuration UI
├── personas/                  # Directory for saved personas (created automatically)
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Tips for Best Results

- Use different model types for different agents to get diverse perspectives
- Create personas with clear, distinct viewpoints and expertise
- Use concrete, specific topics to get more focused discussions
- Experiment with different numbers of discussion rounds
- For complex topics, continue the discussion for multiple rounds

## License

This project is licensed under the MIT License - see the LICENSE file for details.