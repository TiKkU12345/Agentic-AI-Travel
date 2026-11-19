"""Hotel recommendation: dataset hotels for dataset cities, OpenStreetMap hotels with estimated prices elsewhere."""
from travel_planner import global_data
from travel_planner.data_loader import get_hotels
from travel_planner.locations import resolve_location

SORT_KEYS = {
    "value": lambda h: (-h["value_score"], h["price_per_night"]),
    "price": lambda h: (h["price_per_night"], -h["stars"]),
    "stars": lambda h: (-h["stars"], h["price_per_night"]),
}


def hotels_for_location(location):
    if location.in_dataset:
        return [{**h, "source": "dataset"} for h in get_hotels() if h["city"] == location.name]
    return global_data.fetch_hotels(location)


def search_hotels(city, min_stars=0, max_price_per_night=None, required_amenities=None,
                  sort_by="value", top_n=3):
    """Recommend hotels in a city.

    value_score = stars per 1000 rupees of nightly price (higher is better value).
    sort_by: "value" (default), "price" (cheapest first) or "stars" (highest rated first).
    Outside the dataset, hotel names come from OpenStreetMap and prices are estimates.
    """
    location = resolve_location(city)
    required = {a.lower() for a in (required_amenities or [])}
    results = []
    for hotel in hotels_for_location(location):
        if hotel["stars"] < min_stars:
            continue
        if max_price_per_night is not None and hotel["price_per_night"] > max_price_per_night:
            continue
        if not required.issubset({a.lower() for a in hotel["amenities"]}):
            continue
        results.append({**hotel, "value_score": round(hotel["stars"] * 1000 / hotel["price_per_night"], 2)})
    results.sort(key=SORT_KEYS.get(sort_by, SORT_KEYS["value"]))
    notes = [] if results else [f"No hotel in {location.name} matches the filters; relax stars, price or amenities."]
    if results and not location.in_dataset:
        notes.append("Hotel names are from OpenStreetMap; nightly prices are estimates from star level and "
                     "destination cost level, not live rates. Stars marked stars_estimated were not listed.")
    return {"city": location.name, "sort_by": sort_by if sort_by in SORT_KEYS else "value",
            "hotels": results[:top_n], "notes": notes}
