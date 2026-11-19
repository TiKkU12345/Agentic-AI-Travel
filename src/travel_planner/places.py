"""Places discovery: dataset attractions for dataset cities, OpenStreetMap attractions elsewhere."""
from travel_planner import config, global_data
from travel_planner.data_loader import get_places
from travel_planner.locations import resolve_location


def place_setting(place_type):
    """Classify a place type as indoor, outdoor or unknown (used for weather-aware planning)."""
    if place_type in config.INDOOR_PLACE_TYPES:
        return "indoor"
    if place_type in config.OUTDOOR_PLACE_TYPES:
        return "outdoor"
    return "unknown"


def places_for_location(location):
    if location.in_dataset:
        return [{**p, "score": p["rating"], "source": "dataset"} for p in get_places() if p["city"] == location.name]
    return global_data.fetch_places(location)


def discover_places(city, types=None, min_rating=0.0, top_n=10):
    """List attractions in a city, best first, optionally filtered by type and minimum rating.

    Dataset cities have ratings. Other places come from OpenStreetMap, which has no ratings, so they are
    ranked by a notability score (1 to 5) and min_rating is ignored. In the dataset the type field can
    disagree with the name (a place named 'Famous Fort' typed 'lake'); the type is treated as correct.
    """
    location = resolve_location(city)
    wanted = {t.lower() for t in (types or [])}
    results = []
    for place in places_for_location(location):
        if location.in_dataset and place["rating"] < min_rating:
            continue
        if wanted and place["type"].lower() not in wanted:
            continue
        results.append({**place, "setting": place_setting(place["type"])})
    results.sort(key=lambda p: (-p["score"], p["name"], p["place_id"]))
    notes = [] if results else [f"No places in {location.name} match the filters."]
    if results and not location.in_dataset:
        notes.append("Places are from OpenStreetMap and ranked by notability (how well documented they are), "
                     "not by visitor ratings.")
    return {"city": location.name, "places": results[:top_n], "notes": notes}


def available_place_types():
    return sorted({p["type"] for p in get_places()} | set(config.OSM_PLACE_TYPES))
