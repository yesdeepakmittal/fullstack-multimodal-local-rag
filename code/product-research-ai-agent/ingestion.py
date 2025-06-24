import json
import os

import tiktoken

from embedding import get_embedding
from opensearch_client import create_index_if_not_exists, get_opensearch_client


def load_patent_data(dir_path):
    """
    Load patent data from JSON files in the specified directory.

    Args:
        dir_path (str): Path to the directory containing JSON files.

    Returns:
        list: A list of dictionaries containing patent data.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory '{dir_path}' does not exist.")

    all_files = os.listdir(dir_path)
    chunks = []

    for file in all_files:
        if file.endswith(".json"):
            file_path = os.path.join(dir_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            title = data.get("title")
            pdf = data.get("pdf")
            publication_date = data.get("publication_date")
            patent_id = data.get("search_parameters", {}).get("patent_id", None)
            abstract = data.get("abstract", "")
            token_count = len(
                tiktoken.encoding_for_model("gpt-3.5-turbo").encode(abstract)
            )
            embedding = get_embedding(abstract)

            chunks.append(
                {
                    "title": title,
                    "pdf": pdf,
                    "publication_date": publication_date,
                    "patent_id": patent_id,
                    "abstract": abstract,
                    "token_count": token_count,
                    "embedding": embedding,
                }
            )

    return chunks


# Index opensearch data
def index_patent_data(client, index_name, patent_data):
    """
    Index patent data into OpenSearch.

    Args:
        client: OpenSearch client instance.
        index_name (str): Name of the index to store the patent data.
        patent_data (list): List of dictionaries containing patent data.
    """
    for patent in patent_data:
        client.index(index=index_name, body=patent)
    print(f"Indexed {len(patent_data)} patents into '{index_name}' index.")


if __name__ == "__main__":
    dir_path = "results"

    host = "localhost"
    port = 9200
    client = get_opensearch_client(host, port)
    index_name = "patents"
    create_index_if_not_exists(client, index_name)

    try:
        patent_data = load_patent_data(dir_path)
        print(f"Loaded {len(patent_data)} patents from '{dir_path}'")

        index_patent_data(client, index_name, patent_data)
        print(f"Indexed {len(patent_data)} patents into '{index_name}' index.")
    except Exception as e:
        print(f"Error: {e}")
