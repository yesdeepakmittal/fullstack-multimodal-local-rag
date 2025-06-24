from opensearchpy import OpenSearch


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


def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index to create
    """
    # Delete the index if it exists (to ensure proper mapping)
    if client.indices.exists(index=index_name):
        print(
            f"Deleting existing index '{index_name}' to recreate with proper mappings..."
        )
        client.indices.delete(index=index_name)

    # Get dimension from a sample embedding
    from embedding import get_embedding

    sample_embedding = get_embedding("Sample text for dimension detection")
    dimension = len(sample_embedding)
    print(f"Using embedding dimension: {dimension}")

    # Define mappings with vector field for embeddings
    mappings = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "abstract": {"type": "text"},
                "publication_date": {
                    "type": "date",
                    "format": "yyyy-MM-dd||yyyy||epoch_millis||strict_date_optional_time",
                },
                "patent_id": {"type": "keyword"},
                "pdf": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "embedding": {"type": "knn_vector", "dimension": dimension},
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",  # Use cosine similarity for embeddings
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


if __name__ == "__main__":
    host = "localhost"
    port = 9200
    client = get_opensearch_client(host, port)

    # List all indices
    indices = client.cat.indices(format="json")
    print("Available indices:")
    for index in indices:
        print(f"  - {index['index']}")
