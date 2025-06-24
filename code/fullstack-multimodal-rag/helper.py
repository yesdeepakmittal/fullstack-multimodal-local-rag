import requests
from opensearchpy import OpenSearch


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


def get_opensearch_client(host, port):
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    if client.ping():
        print("Connected to OpenSearch!")
        info = client.info()
        print(f"Cluster name: {info['cluster_name']}")
        print(f"OpenSearch version: {info['version']['number']}")
    else:
        print("Connection failed!")
        raise ConnectionError("Failed to connect to OpenSearch.")
    return client


# token count using Tiktoken
def get_token_count(text, model="gpt-3.5-turbo"):
    """
    Get the token count for the given text using the specified model.

    Args:
        text (str): The text to count tokens for.
        model (str): The model to use for token counting. Default is "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the text.
    """
    import tiktoken

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


if __name__ == "__main__":
    # Example usage
    try:
        embedding = get_embedding("This is a test prompt.")
        print("Embedding:", embedding)
    except Exception as e:
        print("Failed to get embedding:", e)
