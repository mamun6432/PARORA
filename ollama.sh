#!/bin/bash

# The official installer (https://ollama.com/download) runs Ollama automatically
# as a background service — no need to call `ollama serve` manually.

ollama pull qwen2.5:7b   # used by server.py / app_lite.py
ollama pull llama3.2     # used by app.py (the Docker default)
echo "Ollama is ready at http://localhost:11434"