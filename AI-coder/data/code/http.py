def fetch_url(url: str) -> str:
    import requests
    return requests.get(url).text
