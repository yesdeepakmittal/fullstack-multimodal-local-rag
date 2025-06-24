# Fullstack Multimodal Local RAG

A fullstack Retrieval-Augmented Generation (RAG) application that supports multimodal (text and image) data, running entirely locally.

## Features

- Multimodal RAG: Handles both text and images for retrieval and generation.
- Local Deployment: No cloud dependencies; runs on your machine.
- Integrates with Ollama, Gemini, and a local VectorDB.
- PDF parsing and semantic chunking.

|App       | Techstacks                                  | Youtube Link                        |
|----------|---------------------------------------------|-------------------------------------|
| **Fullstack Multimodal RAG** | Frontend<br>Ollama<br>VectorDB | [![Multimodal RAG Overview](https://img.shields.io/badge/Multimodal%20RAG%20Overview-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=kcn6uI87nGc&list=PL0x86ZW374m3uIp_WWOg-jjf-EyXr5KEn)<br>[![PDF Parsing for Multimodal RAG](https://img.shields.io/badge/PDF%20Parsing-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=LIus-y2bJH4&list=PL0x86ZW374m3uIp_WWOg-jjf-EyXr5KEn)<br> |

## Quick Start

1. **Clone the repository**
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install and start Ollama as a Docker Container:**
   ```bash
   docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   ```
5. **Pull required models:**
   ```bash
   docker exec -it ollama ollama run deepseek-r1:1.5b          
   docker exec -it ollama ollama run nomic-embed-text # For embeddings
   ```
6. **Start OpenSearch:**  
   ```bash
   docker compose -f docker-compose.yml
   ```
   Make sure your OpenSearch instance is running on `localhost:9200` (or update the connection settings in the code)

7. **Create `.env` file & Save API KEY**
   ```bash
   GEMINI_API_KEY=
   ```
8. **Ingest Chunks**
   ```bash
   python ingestion.py
   ```
9. **Run the app**
   ```bash
   python app.py
   ```

## Project Structure

- `app.py` – Main application entry point
- `chunking.py` – PDF and data chunking logic
- `generation.py` – RAG response generation
- `retrieval.py` – Search and retrieval utilities
- `ingestion.py` - Ingest Chunks to VectorDB
- `helper.py` – Supporting modules

## Used Links
- PDF file Used - https://arxiv.org/pdf/2312.10997
- Ollama Embedding Model - https://www.ollama.com/library/nomic-embed-text
- Ollama Text Model - https://www.ollama.com/library/deepseek-r1
- Run Ollama as a Docker Container - https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image