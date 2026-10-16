# Hybrid RAG PDF Q&A Chatbot

Upload PDFs and ask questions about them. Combines vector search (Chroma + nomic-embed-text) with keyword search (BM25), and answers with a local LLM (qwen2.5) through Ollama.

## Run
1. Install Ollama and pull the models:
   ```
   ollama pull qwen2.5:latest
   ollama pull nomic-embed-text:latest
   ```
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`

No local machine? Run it in Google Colab with a T4 GPU (install Ollama, pull the models, then start Streamlit behind a cloudflared tunnel).
