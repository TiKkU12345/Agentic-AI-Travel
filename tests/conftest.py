"""Shared test helpers: fake Open-Meteo responses and a scripted chat model."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_planner import global_data, http_client, locations, weather  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def make_fake_http_get(rain_on_index=None, calls=None):
    """Return a requests.get replacement that echoes the requested date range."""
    def fake_get(url, params, timeout):
        if calls is not None:
            calls.append((url, params))
        start = date.fromisoformat(params["start_date"])
        count = (date.fromisoformat(params["end_date"]) - start).days + 1
        return FakeResponse({"daily": {
            "time": [(start + timedelta(days=i)).isoformat() for i in range(count)],
            "temperature_2m_max": [31.0 + i for i in range(count)],
            "temperature_2m_min": [24.0] * count,
            "precipitation_sum": [9.0 if i == rain_on_index else 0.0 for i in range(count)],
            "weather_code": [63 if i == rain_on_index else 1 for i in range(count)],
        }})
    return fake_get


@pytest.fixture(autouse=True)
def clear_caches():
    for module in (weather, locations, global_data):
        module.clear_cache()
    yield
    for module in (weather, locations, global_data):
        module.clear_cache()


GEOCODING = {
    "tokyo": {"name": "Tokyo", "latitude": 35.6895, "longitude": 139.69171, "country": "Japan", "country_code": "JP"},
    "paris": {"name": "Paris", "latitude": 48.85341, "longitude": 2.3488, "country": "France", "country_code": "FR"},
}
PLACES_ELEMENTS = [
    {"type": "node", "id": 1, "lat": 35.71, "lon": 139.79, "tags": {
        "name": "Senso-ji", "tourism": "attraction", "wikidata": "Q1", "wikipedia": "ja:x", "website": "http://x"}},
    {"type": "way", "id": 2, "center": {"lat": 35.71, "lon": 139.77}, "tags": {
        "name": "Tokyo National Museum", "tourism": "museum", "wikidata": "Q2"}},
    {"type": "node", "id": 3, "lat": 35.71, "lon": 139.77, "tags": {
        "name": "Ueno Park", "leisure": "park", "wikidata": "Q3"}},
    {"type": "node", "id": 4, "lat": 35.67, "lon": 139.70, "tags": {
        "name": "Meiji Shrine", "historic": "monument", "wikidata": "Q4", "wikipedia": "x"}},
    {"type": "node", "id": 5, "lat": 35.70, "lon": 139.79, "tags": {"name": "Senso-ji", "tourism": "attraction"}},
]
HOTEL_ELEMENTS = [
    {"type": "node", "id": 101, "lat": 35.6, "lon": 139.7, "tags": {"name": "Hotel Sakura", "tourism": "hotel", "stars": "4"}},
    {"type": "node", "id": 102, "lat": 35.6, "lon": 139.7, "tags": {"name": "Budget Hostel", "tourism": "hostel"}},
    {"type": "way", "id": 103, "center": {"lat": 35.6, "lon": 139.7}, "tags": {
        "name": "Shinjuku Inn", "tourism": "hotel", "stars": "3S", "internet_access": "wlan"}},
]


@pytest.fixture
def mock_world(monkeypatch):
    """Fake geocoding and Overpass so global-mode code runs without network access."""
    state = {"hotels": list(HOTEL_ELEMENTS), "queries": []}

    def fake_get_json(url, params=None):
        hit = GEOCODING.get(str(params["name"]).lower())
        return {"results": [hit]} if hit else {}

    def fake_post_json(urls, data):
        state["queries"].append(data["data"])
        return {"elements": state["hotels"] if "guest_house" in data["data"] else PLACES_ELEMENTS}

    monkeypatch.setattr(http_client, "get_json", fake_get_json)
    monkeypatch.setattr(http_client, "post_json", fake_post_json)
    return state


@pytest.fixture
def fake_http():
    return make_fake_http_get()
