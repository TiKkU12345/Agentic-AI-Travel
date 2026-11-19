"""Live and estimated data for places outside the built-in dataset.

Attractions and hotels come from OpenStreetMap (Overpass API, no key). OpenStreetMap has no user ratings
or room prices, so attractions are ranked by a notability score and hotel and flight prices are estimates.
Ids encode their source so a plan can be re-validated: "EST|origin|destination" and "OSM|type|number".
"""
import hashlib
import math

from travel_planner import config, http_client
from travel_planner.locations import resolve_location

EST_PREFIX = "EST|"
OSM_PREFIX = "OSM|"
_PLACE_CACHE, _HOTEL_CACHE = {}, {}


def clear_cache():
    _PLACE_CACHE.clear()
    _HOTEL_CACHE.clear()


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(a))


def estimate_flight(origin, destination):
    """Fare and duration estimated from great-circle distance. Not a real quote."""
    km = haversine_km(origin.latitude, origin.longitude, destination.latitude, destination.longitude)
    domestic = origin.country_code == destination.country_code
    price = (1800 + 4.0 * km) if domestic else (6000 + 5.5 * km)
    return {
        "flight_id": f"{EST_PREFIX}{origin.name}|{destination.name}",
        "airline": "Typical airline (estimated fare)",
        "from": origin.name, "to": destination.name,
        "departure_clock": None, "arrival_clock": None,
        "duration_hours": round(0.8 + km / 800, 1),
        "price": int(round(price / 50.0) * 50),
        "estimated": True, "distance_km": round(km),
    }


def flight_from_id(flight_id):
    """Rebuild an estimated flight from its id (names are re-geocoded, results are cached)."""
    parts = flight_id.split("|")
    if len(parts) != 3:
        raise ValueError(f"'{flight_id}' is not a valid estimated flight id.")
    return estimate_flight(resolve_location(parts[1]), resolve_location(parts[2]))


def _hash_fraction(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:6], 16) / 0xFFFFFF


def _run_overpass(query):
    payload = http_client.post_json(config.OVERPASS_URLS, {"data": query})
    return payload.get("elements", [])


def _osm_id(element):
    return f"{OSM_PREFIX}{element['type']}|{element['id']}"


def _place_type(tags):
    tourism, historic = tags.get("tourism"), tags.get("historic")
    if tourism in ("museum", "gallery", "zoo", "theme_park", "viewpoint", "aquarium", "attraction"):
        return tourism
    if historic in ("castle", "fort", "palace"):
        return "fort"
    if historic in ("monument", "memorial"):
        return "monument"
    if historic:
        return "historic_site"
    if tags.get("natural") == "beach":
        return "beach"
    if tags.get("leisure") == "park":
        return "park"
    return "attraction"


def _notability(tags):
    """1 to 5 score from how well documented a place is on OpenStreetMap (a proxy, not a rating)."""
    score = 1.0 + 2.0 * ("wikidata" in tags) + 1.0 * ("wikipedia" in tags)
    score += 0.5 * sum(1 for key in ("website", "opening_hours", "image", "wikimedia_commons") if key in tags)
    return round(min(score, 5.0), 1)


def fetch_places(location):
    """Attractions near a location from OpenStreetMap, best first, one entry per name."""
    key = (round(location.latitude, 3), round(location.longitude, 3))
    if key in _PLACE_CACHE:
        return _PLACE_CACHE[key]
    around = f"(around:{config.PLACE_SEARCH_RADIUS_M},{location.latitude},{location.longitude})"
    query = ("[out:json][timeout:30];("
             f'nwr["tourism"~"^(attraction|museum|gallery|zoo|theme_park|viewpoint|aquarium)$"]["name"]{around};'
             f'nwr["historic"~"^(castle|fort|monument|archaeological_site|palace|memorial|ruins)$"]["name"]{around};'
             f'nwr["natural"="beach"]["name"]{around};'
             f'nwr["leisure"="park"]["name"]["wikidata"]{around};'
             ");out center 150;")
    best = {}
    for element in _run_overpass(query):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        place = {"place_id": _osm_id(element), "name": name, "city": location.name, "type": _place_type(tags),
                 "rating": None, "score": _notability(tags), "source": "openstreetmap"}
        if name.lower() not in best or place["score"] > best[name.lower()]["score"]:
            best[name.lower()] = place
    places = sorted(best.values(), key=lambda p: (-p["score"], p["name"]))[:config.MAX_OSM_RESULTS]
    _PLACE_CACHE[key] = places
    return places


def _hotel_stars(tags):
    raw = str(tags.get("stars", ""))
    if raw[:1].isdigit() and 1 <= int(raw[:1]) <= 5:
        return int(raw[:1]), False
    return {"hostel": 1, "guest_house": 2, "motel": 2}.get(tags.get("tourism"), 3), True


def _hotel_amenities(tags):
    found = []
    if tags.get("internet_access") in ("wlan", "wifi", "yes"):
        found.append("wifi")
    if tags.get("swimming_pool") == "yes" or tags.get("pool") == "yes":
        found.append("pool")
    if tags.get("parking") or tags.get("parking:fee"):
        found.append("parking")
    if tags.get("breakfast") == "yes":
        found.append("breakfast")
    return found


def _estimated_price(stars, cost_index, seed):
    base = config.BASE_HOTEL_PRICE_BY_STARS[stars] * cost_index
    return int(round(base * (0.85 + 0.30 * _hash_fraction(seed)) / 50.0) * 50)


def fetch_hotels(location):
    """Hotels near a location from OpenStreetMap with estimated nightly prices (OSM has no prices)."""
    key = (round(location.latitude, 3), round(location.longitude, 3))
    if key in _HOTEL_CACHE:
        return _HOTEL_CACHE[key]
    around = f"(around:{config.HOTEL_SEARCH_RADIUS_M},{location.latitude},{location.longitude})"
    query = ("[out:json][timeout:30];("
             f'nwr["tourism"~"^(hotel|guest_house|hostel|motel|resort)$"]["name"]{around};'
             ");out center 120;")
    hotels, seen = [], set()
    for element in _run_overpass(query):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        stars, stars_estimated = _hotel_stars(tags)
        hotels.append({"hotel_id": _osm_id(element), "name": name, "city": location.name, "stars": stars,
                       "price_per_night": _estimated_price(stars, location.cost_index, name),
                       "amenities": _hotel_amenities(tags), "estimated_price": True,
                       "stars_estimated": stars_estimated, "source": "openstreetmap"})
    if not hotels:
        hotels.append({"hotel_id": f"{EST_PREFIX}stay|{location.name}", "name": f"Typical 3-star hotel in {location.name}",
                       "city": location.name, "stars": 3,
                       "price_per_night": _estimated_price(3, location.cost_index, location.name),
                       "amenities": [], "estimated_price": True, "stars_estimated": True, "source": "estimated"})
    hotels.sort(key=lambda h: (-h["stars"], h["name"]))
    hotels = hotels[:config.MAX_OSM_RESULTS]
    _HOTEL_CACHE[key] = hotels
    return hotels


def place_index(location):
    return {p["place_id"]: p for p in fetch_places(location)}


def hotel_index(location):
    return {h["hotel_id"]: h for h in fetch_hotels(location)}
