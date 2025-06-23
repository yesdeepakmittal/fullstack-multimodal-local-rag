def process_images_with_captions(raw_chunks, use_gemini=True):
    """
    Extract images from raw document chunks, identify captions, and generate descriptions.

    Args:
        raw_chunks: List of document elements from unstructured.partition_pdf
        use_gemini: Whether to use Gemini for image captioning (default: True)

    Returns:
        List of dictionaries with image data, captions, and generated descriptions
        encountered_errors: List of dictionaries containing any errors encountered during processing
    """
    import base64
    import os

    import google.generativeai as genai
    from dotenv import load_dotenv
    from unstructured.documents.elements import FigureCaption, Image

    load_dotenv()

    # Configure Gemini API
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        genai.configure(api_key=api_key)

    # Extract images and their captions
    processed_images = []
    encountered_errors = []

    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Image):
            # Check if next element is a figure caption
            if idx + 1 < len(raw_chunks) and isinstance(
                raw_chunks[idx + 1], FigureCaption
            ):
                caption = raw_chunks[idx + 1].text
            else:
                caption = "No caption available"

            # Store image data
            image_data = {
                # "index": idx,
                "caption": caption,
                "image_text": chunk.text if hasattr(chunk, "text") else "",
                "base64_image": chunk.metadata.image_base64,
                "content": (
                    chunk.text if hasattr(chunk, "text") else ""
                ),  # Fallback content
                "content_type": "image",
                "filename": (
                    chunk.metadata.filename if hasattr(chunk, "metadata") else ""
                ),
            }

            error_data = {
                "error": None,
                "error_message": None,
            }

            # Generate description if requested
            if use_gemini:
                try:
                    image_binary = base64.b64decode(chunk.metadata.image_base64)

                    # Use Gemini model for image description
                    model = genai.GenerativeModel("gemini-1.5-flash")

                    prompt = (
                        f"Generate a comprehensive and detailed description of this image from a technical document about Retrieval-Augmented Generation (RAG).\n\n"
                        f"CONTEXT INFORMATION:\n"
                        f"- Caption: {caption}\n"
                        f"- Text extracted from image: {chunk.text if hasattr(chunk, 'text') else 'No text'}\n\n"
                        f"DESCRIPTION REQUIREMENTS:\n"
                        f"1. Begin with a clear overview of what the image shows (diagram, chart, architecture, etc.)\n"
                        f"2. If it's a diagram or flowchart: describe components, connections, data flow direction, and system architecture\n"
                        f"3. If it's a chart or graph: explain axes, trends, key data points, and significance\n"
                        f"4. Explain technical terminology and abbreviations that appear in the image\n"
                        f"5. Interpret how this visual relates to RAG concepts and implementation\n"
                        f"6. Include any numerical data, performance metrics, or comparative results shown\n"
                        f"7. Target length: 150-300 words for complex diagrams, 100-150 words for simpler images\n\n"
                        f"Focus on providing information that would be valuable in a technical context for someone implementing or researching RAG systems."
                    )

                    response = model.generate_content(
                        [prompt, {"mime_type": "image/jpeg", "data": image_binary}]
                    )

                    image_data["content"] = response.text
                except Exception as e:
                    print(f"Warning: Error generating description: {str(e)}")

                    error_data["error"] = str(e)
                    error_data["error_message"] = (
                        "Error generating description with Gemini."
                    )
                    encountered_errors.append(error_data)

            processed_images.append(image_data)

    print(f"Processed {len(processed_images)} images with captions and descriptions")
    print(f"Errors encountered: {len(encountered_errors)}")
    return processed_images, encountered_errors


def process_tables_with_descriptions(raw_chunks, use_gemini=True, use_ollama=False):
    """Process tables with descriptions using Gemini or Ollama"""
    import os

    import google.generativeai as genai
    import requests
    from dotenv import load_dotenv
    from unstructured.documents.elements import Table

    load_dotenv()

    # Configure Gemini API
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        genai.configure(api_key=api_key)

    # Extract tables and generate descriptions
    processed_tables = []
    encountered_errors = []

    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Table):
            # Store table data
            table_data = {
                # "index": idx,
                "table_as_html": chunk.metadata.text_as_html,
                "table_text": chunk.text,
                "content": chunk.text,  # Fallback content
                "content_type": "table",
                "filename": (
                    chunk.metadata.filename if hasattr(chunk, "metadata") else ""
                ),
            }

            # Generate description with Gemini if requested
            if use_gemini:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")

                    prompt = (
                        f"Generate a comprehensive and detailed description of the following table from a technical document about Retrieval-Augmented Generation (RAG).\n\n"
                        f"TABLE HTML:\n{chunk.metadata.text_as_html}\n\n"
                        f"DESCRIPTION REQUIREMENTS:\n"
                        f"1. Provide an overview of the table's purpose and what it represents in the context of RAG.\n"
                        f"2. Explain the significance of each column and row, including any key metrics or data points.\n"
                        f"3. Describe any trends, comparisons, or notable findings presented in the table.\n"
                        f"4. If applicable, explain how this data supports or illustrates RAG concepts or implementations.\n"
                        f"5. Target length: 150-300 words.\n\n"
                        f"6. Do not include information like 'This table shows' or 'The table contains', but rather directly explain the content and significance of the table.\n"
                        f"Directly return the summary without additional commentary or preamble."
                    )

                    response = model.generate_content([prompt])
                    table_data["content"] = response.text
                except Exception as e:
                    encountered_errors.append(
                        {
                            "error": str(e),
                            "error_message": "Error generating description with Gemini.",
                        }
                    )

            elif use_ollama:
                # If using Ollama instead of Gemini
                try:
                    url = "http://localhost:11434/api/generate"
                    data = {
                        "model": "deepseek-r1:1.5b",
                        "prompt": (
                            "Analyze the following table and provide a detailed summary of its contents, "
                            "including the structure, key data points, and any notable trends or insights."
                            f"Here is the table in HTML format: {chunk.metadata.text_as_html}"
                            "Directly analyze the table and provide a detailed summary without any additional text."
                        ),
                        "max_tokens": 1000,
                        "stream": False,
                        "temperature": 0.2,
                    }

                    response = requests.post(url, json=data)
                    response.raise_for_status()

                    table_data["content"] = response.json().get(
                        "response", "No response from model"
                    )
                except Exception as e:
                    encountered_errors.append(
                        {
                            "error": str(e),
                            "error_message": "Error generating description with Ollama.",
                        }
                    )

            processed_tables.append(table_data)

    print(f"Processed {len(processed_tables)} tables with descriptions")
    print(f"Errors encountered: {len(encountered_errors)}")
    return processed_tables, encountered_errors


def create_semantic_chunks(chunks):
    """
    Create semantic chunks from a PDF document based on title structure.

    Args:
        chunks: List of document elements from unstructured.partition_pdf

    Returns:
        List of semantic chunks
    """
    from unstructured.documents.elements import CompositeElement

    # Convert to more usable format
    processed_chunks = []

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, CompositeElement):
            chunk_data = {
                # "index": idx,
                # "text": chunk.text,
                "content": chunk.text,
                "content_type": "text",
                "filename": (
                    chunk.metadata.filename if hasattr(chunk, "metadata") else ""
                ),
            }
            processed_chunks.append(chunk_data)

    print(f"Created {len(processed_chunks)} semantic chunks from document")
    return processed_chunks
