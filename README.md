
# Python AI Chatbot App

A simple AI-powered chatbot app using Hugging Face Transformers and Gradio. The app lets you chat with state-of-the-art large language models (LLMs) through a clean web interface.

## Features
- Chat with LLMs from Hugging Face
- Web interface powered by Gradio
- Easy deployment with Docker

## Requirements
- Python 3.10+
- pip
- (Optional) Docker

## Installation

### 1. Create a Virtual Environment (Recommended)
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the App
```bash
python chatbot.py
```

Once started, open your browser and go to `http://127.0.0.1:7860`.

## Run with Docker
```bash
docker build -t python-ai-chatbot-app .
docker run -p 7860:7860 python-ai-chatbot-app
```

## Project Structure
- `chatbot.py` : Main application file
- `requirements.txt` : Python dependencies
- `Dockerfile` : Docker image configuration
- `.gitignore` : Files and folders ignored by git

## Notes
- The model files will be downloaded automatically on first run. This process can take a long time (sometimes 10-30 minutes or more) depending on your internet speed and the model size.
- Memory requirements depend on the selected model.

## Contributing & Contact
Feel free to open issues or pull requests for suggestions and contributions.

---

This project is for learning and experimentation purposes.
