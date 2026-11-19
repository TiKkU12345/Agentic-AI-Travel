"""Small HTTP helper with retries. Tests replace get_json and post_json."""
import requests

from travel_planner import config

HEADERS = {"User-Agent": "agentic-travel-planner/1.0 (student project)"}


def get_json(url, params=None):
    last_error = None
    for _ in range(config.HTTP_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=config.HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
    raise RuntimeError(f"request to {url.split('/')[2]} failed ({type(last_error).__name__})")


def post_json(urls, data):
    """POST to the first working URL in urls (Overpass mirrors) and return the JSON body."""
    last_error = None
    for url in ([urls] if isinstance(urls, str) else urls):
        try:
            response = requests.post(url, data=data, headers=HEADERS, timeout=config.OVERPASS_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
    raise RuntimeError(f"OpenStreetMap (Overpass) request failed ({type(last_error).__name__})")
