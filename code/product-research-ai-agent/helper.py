import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SERPAPI_API_KEY")
if not api_key:
    raise ValueError("SERPAPI_API_KEY environment variable is not set.")


def get_serpapi_url(data):
    """
    Constructs the SerpApi URL from the provided data.
    Args:
        data (dict): The data containing the SerpApi link.
    Returns:
        str: The complete SerpApi URL with the API key.
    """
    if "serpapi_link" not in data:
        raise ValueError("The provided data does not contain 'serpapi_link'.")

    # Get the URL from the data
    serpapi_url = data["serpapi_link"]

    # Add API key to the URL if not already present
    if "api_key=" not in serpapi_url:
        separator = "&" if "?" in serpapi_url else "?"
        serpapi_url = f"{serpapi_url}{separator}api_key={api_key}"

    return serpapi_url


def get_data_from_serpapi(serpapi_url):
    """
    Fetches data from the given SerpApi URL.

    Args:
        serpapi_url (str): The SerpApi URL to fetch data from.

    Returns:
        dict: The parsed JSON response from SerpApi.

    Raises:
        HTTPError: If the HTTP request returns an error status code.
    """
    # Pass the API key as a parameter
    params = {"api_key": api_key}
    response = requests.get(serpapi_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
