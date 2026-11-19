"""Load and index the JSON datasets, and normalise city names."""
import json
from functools import lru_cache

from travel_planner import config


class UnknownCityError(ValueError):
    """Raised when a city is not present in the datasets."""


@lru_cache(maxsize=None)
def _read_json(filename):
    path = config.DATA_DIR / filename
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def get_flights(use_extended=False):
    """Return all flights; optionally include the synthetic extension file."""
    flights = list(_read_json("flights.json"))
    if use_extended:
        flights += _read_json("flights_extended.json")
    return flights


def get_hotels():
    return list(_read_json("hotels.json"))


def get_places():
    return list(_read_json("places.json"))


def get_city_coordinates():
    return dict(_read_json("city_coordinates.json"))


def supported_cities():
    """Cities that have hotels, places and coordinates."""
    return sorted({h["city"] for h in get_hotels()} & {p["city"] for p in get_places()}
                  & set(get_city_coordinates()))


def normalize_city(name):
    """Map user input (case, aliases) to a dataset city name or raise UnknownCityError."""
    if not name or not str(name).strip():
        raise UnknownCityError("City name is empty.")
    cleaned = str(name).strip()
    cities = supported_cities()
    lookup = {c.lower(): c for c in cities}
    key = cleaned.lower()
    key = config.CITY_ALIASES.get(key, key).lower()
    if key in lookup:
        return lookup[key]
    raise UnknownCityError(f"'{cleaned}' is not supported. Available cities: {', '.join(cities)}.")


def flights_by_id(use_extended=False):
    return {f["flight_id"]: f for f in get_flights(use_extended)}


def hotels_by_id():
    return {h["hotel_id"]: h for h in get_hotels()}


def places_by_id():
    return {p["place_id"]: p for p in get_places()}
