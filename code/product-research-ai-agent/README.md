# Patent Innovation Predictor

A powerful agentic AI system for patent trend analysis and future technology prediction using Ollama, OpenSearch, and CrewAI.

[![Product Research AI Agent](https://img.shields.io/badge/Product%20Research%20AI%20Agent-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=uhWXMrpS8Gg&list=PL0x86ZW374m3yeuDQ0iJ7KWRoYo1RqhvF)

## Overview

This system analyzes patent data to identify trends and predict future innovations in specific technology areas (with a focus on lithium battery technology). It uses a multi-agent approach with specialized roles for research direction, patent retrieval, data analysis, and innovation forecasting.

## System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     User Interface Layer                      │
└───────────────────────────────────────────────────────────────┘
                │                │                │
                ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                 Agent Orchestration Layer                     │
│  ┌──────────────────┐   ┌────────────────┐  ┌───────────────┐│
│  │ Research Director│   │Patent Retriever│  │Data Analyst   ││
│  └──────────────────┘   └────────────────┘  └───────────────┘│
│                                                               │
│  ┌──────────────────┐                                         │
│  │Innovation        │                                         │
│  │Forecaster        │                                         │
│  └──────────────────┘                                         │
└───────────────────────────────────────────────────────────────┘
                │                │                │
                ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                Knowledge Processing Layer                     │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐ │
│  │ Semantic      │    │ Hybrid        │    │ Iterative     │ │
│  │ Search        │    │ Search        │    │ Search        │ │
│  └───────────────┘    └───────────────┘    └───────────────┘ │
└───────────────────────────────────────────────────────────────┘
                │                │                │
                ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                     Data Storage Layer                        │
│  ┌───────────────────────────────────────────────────────────┐│
│  │                       OpenSearch                          ││
│  └───────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) for running local LLM models
- OpenSearch instance running (default: `localhost:9200`)
- Access to patent data (pre-loaded in OpenSearch)

## Installation

1. **Clone the repository:**
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
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
   Make sure your OpenSearch instance is running on `localhost:9200` (or update the connection settings in the code).

## Configuration

1. **Environment variables (optional):**  
   Create a `.env` file for any API keys or configuration:
   ```
   # Optional API keys
   SERPAPI_API_KEY=your_key_here
   ```

2. **OpenSearch setup:**  
   The system will automatically create necessary indices if they don't exist.

## Usage

Run the main application:
```bash
python agentic_rag.py
```
