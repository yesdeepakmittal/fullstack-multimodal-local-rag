import requests


def get_embedding(prompt, model="nomic-embed-text"):
    """
    Get the embedding for the given prompt using the specified model.

    Args:
        prompt (str): The prompt to embed.
        model (str): The model to use for embedding. Default is "nomic-embed-text".

    Returns:
        list: The embedding vector.
    """
    url = "http://localhost:11434/api/embeddings/"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "model": model}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("embedding", [])
    else:
        raise Exception(
            f"Error fetching embedding: {response.status_code}, {response.text}"
        )


if __name__ == "__main__":
    sample_prompt = "The sky is blue because of Rayleigh scattering."
    try:
        embedding = get_embedding(sample_prompt)
        print("Embedding Dimesion:", len(embedding))
        print("Embedding:", embedding)
    except Exception as e:
        print(f"Failed to get embedding: {e}")
