# AI Podcast Generator

## Overview
The AI Podcast Generator is a web application that converts blog content into podcast episodes. It uses advanced AI tools for summarizing blog content and generating speech.

[![Podcast Generator AI Agent](https://img.shields.io/badge/Podcast%20Generator%20AI%20Agent-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=Vdt0eUz6jVM&list=PL0x86ZW374m3yeuDQ0iJ7KWRoYo1RqhvF)

## Features
- Extracts blog content using ```FirecrawlScrapeWebsiteTool```.
- Summarizes blog content using ```Gemini LLM```.
- Converts summaries into speech using ```ElevenLabs```.
- Provides a user-friendly interface via ```Gradio```.

## Requirements
- Python 3.11
- Docker (optional for containerized deployment)

## Installation

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/awesome-genai-apps.git
   cd awesome-genai-apps/code/ai-podcast-ai-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file .
   - Add your API keys for Gemini, Firecrawl, and ElevenLabs.

4. Run the application:
   ```bash
   python app.py
   ```

### Docker Setup
1. Build the Docker image:
   ```bash
   docker build -t ai-podcast-generator .
   ```

2. Run the container using either of below ways:
   ```bash
   docker run -it \
       --name ai-podcast-generator \
       -p 7860:7860 \
       -e GEMINI_API_KEY=your_key \
       -e FIRECRAWL_API_KEY=your_key \
       -e ELEVENLABS_API_KEY=your_key \
       ai-podcast-generator
   ```

   ```
   docker run -it \
        --name ai-podcast-generator \
        -p 7860:7860 \
        --env-file .env \
        ai-podcast-generator
   ```

3. Access the application at [http://localhost:7860](http://localhost:7860).

## Usage
1. Enter a blog URL in the input field.
2. Click "Generate Podcast".
3. View the blog summary and download the generated podcast audio.

