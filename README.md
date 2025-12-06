# Python AI Chatbot App

A simple, local chatbot application that demonstrates how to run pre-trained models from Hugging Face with a lightweight Gradio web UI. It is intended for experimentation and learning — not production deployment.

**Key points:** the app loads a tokenizer and model from Hugging Face, accepts a user message and a selected model, and returns the model's generated reply in the web UI.

<img width="2492" height="922" alt="image" src="https://github.com/user-attachments/assets/53660b16-d583-4c5d-88c4-01febcd37a9b" />

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Run with Docker](#run-with-docker)
- [Models and Recommendations](#models-and-recommendations)
- [Notes about first run](#notes-about-first-run)
- [Troubleshooting](#troubleshooting)
- [Files in this repository](#files-in-this-repository)
- [Development tips](#development-tips)
- [Contributing](#contributing)
  
## Features
- **Interactive chat** using Hugging Face models.
- **Web UI** powered by Gradio for quick local testing.
- **Dockerfile** included for containerized runs.

## Requirements
- Python 3.10+
- pip
- (Optional) Docker

The project includes a `requirements.txt` (transformers, torch, numpy, gradio). Use a virtual environment to keep dependencies isolated.

## Quick Start

### 1. Create a virtual environment (recommended)
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run locally
```bash
python main.py
```

Open `http://127.0.0.1:7860` in your browser. If you want a public link from Gradio, edit the launch call and set `share=True`.

## Run with Docker
The included `Dockerfile` can be used to build a container. The Dockerfile should use an official Python base image. Example build/run commands:

```bash
docker build -t python-ai-chatbot-app .
docker run -p 7860:7860 python-ai-chatbot-app
```

## Models and Recommendations
- The app supports both causal and seq2seq models; some models are much faster to download and run locally. For quick experiments use smaller models:
	- `distilgpt2` — very small and fast
	- `gpt2` — small, general-purpose
	- `google/flan-t5-small` — small seq2seq instruction model (uses Seq2Seq loader)
- Larger models (Mixtral, Mistral, Llama, Falcon 7B+) require more RAM and may not fit on a typical laptop.

## Notes about first run
- The first time you run a model it will download weights from Hugging Face. This can take time (10–30 minutes or more depending on model size and internet speed).
- If you switch models frequently, expect additional download time for each new model.

## Troubleshooting

- **App starts but UI shows repeated or confusing text**: this often means the prompt format is confusing the model (e.g., including `User:`/`Model:` markers). Try using a simpler prompt (only the user message) or use a dialogue-specialized model like `microsoft/DialoGPT`.
- **HFValidationError: Repo id must use alphanumeric chars...**: this means a user message (for example "how are you") was passed where a model id was expected. Check that the Gradio `inputs` order matches the function signature `generate_response(prompt, model_name)`. Also validate the `model_name` value before passing it to `from_pretrained()`.
- **transformers errors about unknown model type (KeyError: 'mistral')**: update `transformers` to a recent release that knows the model type, e.g. `pip install --upgrade transformers`.
- **Model too large / out of memory**: use a smaller model or run on a machine with more RAM / GPU.

## Files in this repository
- `chatbot.py` — main logic: loads model/tokenizer, builds prompt, calls the model, and returns a response.
- `main.py` — small launcher that starts the Gradio app.
- `requirements.txt` — pinned dependencies used by this project.
- `Dockerfile` — container configuration (may need small fixes depending on your environment).
- `.gitignore` — project ignore rules.

## Development tips
- Keep the `inputs` list in `get_interface()` and the function signature in sync: e.g. `inputs=[Textbox, Dropdown]` with `def generate_response(prompt, model_name):`.
- Validate `model_name` before calling `AutoModel.from_pretrained(...)` to avoid accidental user text being used as a repo id.
- Cache loaded models/tokenizers between calls (the app can be slow if a model is reloaded on every request).

## Contributing
If you find bugs or want to propose improvements, open an issue or a pull request. Suggested improvements: better model selection UI, caching/memory optimizations, and support for locally cached models.

---

This README is written to be clear and concise for a developer preparing the repo for GitHub. If you want, I can also add a short `USAGE.md` or expand the Troubleshooting section with exact log samples from your environment.


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
