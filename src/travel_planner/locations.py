"""Resolve any place name to coordinates: dataset cities first, then Open-Meteo geocoding."""
from dataclasses import dataclass

from travel_planner import config, http_client
from travel_planner.data_loader import UnknownCityError, get_city_coordinates, supported_cities


@dataclass(frozen=True)
class Location:
    name: str
    country: str
    country_code: str
    latitude: float
    longitude: float
    in_dataset: bool
    cost_index: float = 1.0


_CACHE = {}


def clear_cache():
    _CACHE.clear()


def resolve_location(name):
    """Return a Location for a city. Dataset cities use the provided data; others are geocoded online."""
    if not name or not str(name).strip():
        raise UnknownCityError("City name is empty.")
    cleaned = str(name).strip()
    key = config.CITY_ALIASES.get(cleaned.lower(), cleaned).lower()
    if key in _CACHE:
        return _CACHE[key]
    dataset = {c.lower(): c for c in supported_cities()}
    if key in dataset:
        city = dataset[key]
        coords = get_city_coordinates()[city]
        location = Location(city, "India", "IN", coords["latitude"], coords["longitude"], True, 1.0)
    else:
        payload = http_client.get_json(config.OPEN_METEO_GEOCODING_URL,
                                       {"name": cleaned, "count": 1, "language": "en", "format": "json"})
        results = payload.get("results") or []
        if not results:
            raise UnknownCityError(f"Could not find a place called '{cleaned}'. Check the spelling or try a nearby city.")
        hit = results[0]
        code = (hit.get("country_code") or "").upper()
        location = Location(hit["name"], hit.get("country", ""), code, hit["latitude"], hit["longitude"], False,
                            config.COST_INDEX.get(code, config.DEFAULT_COST_INDEX))
    _CACHE[key] = location
    return location
