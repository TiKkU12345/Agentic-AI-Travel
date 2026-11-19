"""Central configuration: paths, constants and planning assumptions."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

MIN_TRIP_DAYS = 1
MAX_TRIP_DAYS = 30
MAX_FLIGHT_STOPS = 3
MAX_PLACES_PER_DAY = 2
RAIN_THRESHOLD_MM = 5.0
FORECAST_HORIZON_DAYS = 15

DAILY_EXPENSE_BY_STYLE = {"budget": 800, "standard": 1200, "premium": 2000}
VALID_STYLES = tuple(DAILY_EXPENSE_BY_STYLE)

CITY_ALIASES = {
    "new delhi": "Delhi",
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "panaji": "Goa",
}

INDOOR_PLACE_TYPES = {"museum", "temple", "market", "monument", "gallery", "aquarium"}
OUTDOOR_PLACE_TYPES = {"beach", "park", "lake", "fort", "zoo", "theme_park", "viewpoint", "attraction",
                       "historic_site"}
OSM_PLACE_TYPES = ("attraction", "museum", "gallery", "zoo", "theme_park", "viewpoint", "aquarium", "fort",
                   "monument", "historic_site", "beach", "park")

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HTTP_TIMEOUT_SECONDS = 10
HTTP_RETRIES = 2

# Groq retired llama-3.3-70b-versatile on 2026-08-16; its recommended replacements are used instead.
RETIRED_LLM_MODELS = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b",
                      "meta-llama/llama-4-scout-17b-16e-instruct"}
FALLBACK_LLM_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL") or FALLBACK_LLM_MODELS[0]

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
OVERPASS_TIMEOUT_SECONDS = 40
PLACE_SEARCH_RADIUS_M = 12000
HOTEL_SEARCH_RADIUS_M = 8000
MAX_OSM_RESULTS = 40

# Rough price level relative to India (1.0), used to scale estimated hotel and daily costs outside the dataset.
COST_INDEX = {"IN": 1.0, "NP": 0.8, "LK": 1.0, "VN": 1.0, "ID": 1.0, "TH": 1.2, "MY": 1.3, "CN": 1.5,
              "KR": 2.3, "JP": 2.6, "SG": 3.0, "AE": 2.5, "TR": 1.6, "ES": 2.3, "IT": 2.6, "DE": 2.8,
              "FR": 3.0, "GB": 3.2, "US": 3.5, "CA": 2.8, "AU": 3.0}
DEFAULT_COST_INDEX = 2.0
# Estimated nightly price in rupees at cost index 1.0, by star level.
BASE_HOTEL_PRICE_BY_STARS = {1: 1200, 2: 2200, 3: 3800, 4: 6500, 5: 11000}
