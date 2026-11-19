"""Flight search: direct and connecting routes, ranked by fewest stops then price or duration."""
from collections import defaultdict
from datetime import datetime

from travel_planner import config
from travel_planner import global_data
from travel_planner.data_loader import get_flights
from travel_planner.locations import resolve_location


def enrich_flight(flight):
    """Add readable clock times and duration (hours) to a raw flight record."""
    departure = datetime.fromisoformat(flight["departure_time"])
    arrival = datetime.fromisoformat(flight["arrival_time"])
    return {
        **flight,
        "departure_clock": departure.strftime("%H:%M"),
        "arrival_clock": arrival.strftime("%H:%M"),
        "duration_hours": round((arrival - departure).total_seconds() / 3600, 1),
    }


def _build_graph(flights):
    graph = defaultdict(lambda: defaultdict(list))
    for flight in flights:
        graph[flight["from"]][flight["to"]].append(enrich_flight(flight))
    return graph


def _city_paths(graph, origin, destination, max_legs):
    """All simple city paths from origin to destination using at most max_legs flights."""
    paths, stack = [], [[origin]]
    while stack:
        path = stack.pop()
        current = path[-1]
        if current == destination and len(path) > 1:
            paths.append(path)
            continue
        if len(path) - 1 >= max_legs:
            continue
        for nxt in graph[current]:
            if nxt not in path:
                stack.append(path + [nxt])
    return paths


def _best_leg(candidates, preference):
    if preference == "fastest":
        return min(candidates, key=lambda f: (f["duration_hours"], f["price"]))
    return min(candidates, key=lambda f: (f["price"], f["duration_hours"]))


def search_flights(origin, destination, preference="cheapest", max_stops=config.MAX_FLIGHT_STOPS,
                   max_price=None, top_n=3, use_extended=False):
    """Find flight options between two cities.

    Options are ranked by number of stops first, then by price (preference="cheapest")
    or total flight time (preference="fastest"). Dataset flight dates are fixed 2025 samples,
    so only route, price and time of day are used, not the traveller's calendar dates.
    """
    origin_loc, destination_loc = resolve_location(origin), resolve_location(destination)
    origin, destination = origin_loc.name, destination_loc.name
    if origin == destination:
        raise ValueError("Origin and destination must be different cities.")
    if preference not in ("cheapest", "fastest"):
        preference = "cheapest"
    if not (origin_loc.in_dataset and destination_loc.in_dataset):
        return _estimated_options(origin_loc, destination_loc, preference, max_price)
    graph = _build_graph(get_flights(use_extended))
    options = []
    for path in _city_paths(graph, origin, destination, max_stops + 1):
        legs = [_best_leg(graph[a][b], preference) for a, b in zip(path, path[1:])]
        options.append({
            "route": " > ".join(path),
            "stops": len(path) - 2,
            "legs": legs,
            "total_price": sum(leg["price"] for leg in legs),
            "total_duration_hours": round(sum(leg["duration_hours"] for leg in legs), 1),
        })
    if max_price is not None:
        options = [o for o in options if o["total_price"] <= max_price]
    metric = "total_duration_hours" if preference == "fastest" else "total_price"
    options.sort(key=lambda o: (o["stops"], o[metric], o["total_price"]))

    notes = []
    direct = any(o["stops"] == 0 for o in options)
    if not options:
        notes.append(f"No flight route found from {origin} to {destination} within {max_stops} stop(s)"
                     + (f" and under Rs {max_price}." if max_price is not None else "."))
    elif not direct:
        notes.append("No direct flight in the dataset; showing connecting routes. Layover timing is "
                     "indicative only because dataset flights have fixed sample dates.")
    return {"origin": origin, "destination": destination, "preference": preference,
            "direct_available": direct, "options": options[:top_n], "notes": notes}


def _estimated_options(origin, destination, preference, max_price):
    """Flight option for routes outside the dataset: one estimated direct fare from distance."""
    leg = global_data.estimate_flight(origin, destination)
    option = {"route": f"{origin.name} > {destination.name}", "stops": 0, "legs": [leg],
              "total_price": leg["price"], "total_duration_hours": leg["duration_hours"]}
    options = [option] if max_price is None or option["total_price"] <= max_price else []
    notes = ["This route is not in the flight dataset, so the fare is an estimate from distance "
             f"({leg['distance_km']} km), not a real quote. Connecting flights are not modelled."]
    if not options:
        notes.append(f"The estimated fare is above the Rs {max_price} limit.")
    return {"origin": origin.name, "destination": destination.name, "preference": preference,
            "direct_available": True, "options": options, "notes": notes}
