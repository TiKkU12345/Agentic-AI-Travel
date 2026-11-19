"""Plan validation, deterministic (no-LLM) planner and human-readable rendering.

finalize_plan is the guardrail used for every plan: it re-reads flights, hotels and places by id (from the
datasets, or from OpenStreetMap and the fare estimator for places outside them), and recomputes dates,
weather and budget, so an LLM cannot invent prices.
"""
import re
from datetime import timedelta

from travel_planner import config, global_data
from travel_planner.budget import estimate_budget
from travel_planner.data_loader import flights_by_id, hotels_by_id, places_by_id
from travel_planner.flights import enrich_flight, search_flights
from travel_planner.hotels import search_hotels
from travel_planner.locations import resolve_location
from travel_planner.places import discover_places, place_setting
from travel_planner.schemas import TripDraft, TripPlan
from travel_planner.weather import get_weather


class PlanValidationError(ValueError):
    """Raised when a plan references data that does not exist or does not fit together."""


def _leg_from_id(flight_id, flight_index, label):
    if flight_id.startswith(global_data.EST_PREFIX):
        try:
            return global_data.flight_from_id(flight_id)
        except (ValueError, RuntimeError) as error:
            raise PlanValidationError(f"{label} flight '{flight_id}' could not be rebuilt: {error}")
    if flight_id in flight_index:
        return enrich_flight(flight_index[flight_id])
    raise PlanValidationError(f"{label} flight '{flight_id}' does not exist in the dataset.")


def _resolve_chain(flight_ids, expected_from, expected_to, flight_index, label):
    legs = [_leg_from_id(flight_id, flight_index, label) for flight_id in flight_ids]
    if not legs:
        raise PlanValidationError(f"No {label.lower()} flight ids were given.")
    connected = all(a["to"] == b["from"] for a, b in zip(legs, legs[1:]))
    if legs[0]["from"] != expected_from or legs[-1]["to"] != expected_to or not connected:
        raise PlanValidationError(
            f"{label} flights do not form one connected route {expected_from} to {expected_to}.")
    return {
        "route": " > ".join([legs[0]["from"]] + [leg["to"] for leg in legs]),
        "stops": len(legs) - 1,
        "legs": legs,
        "total_price": sum(leg["price"] for leg in legs),
        "total_duration_hours": round(sum(leg["duration_hours"] for leg in legs), 1),
        "estimated": any(leg.get("estimated") for leg in legs),
    }


def _hotel_lookup(location):
    return hotels_by_id() if location.in_dataset else global_data.hotel_index(location)


def _place_lookup(location):
    if location.in_dataset:
        return {pid: {**place, "score": place["rating"], "source": "dataset"}
                for pid, place in places_by_id().items() if place["city"] == location.name}
    return global_data.place_index(location)


def distribute_places(ordered_places, num_days, rainy_flags):
    """Spread places over days, lightest on arrival and departure days, weather-aware.

    ordered_places must already be in priority order. On rainy days indoor places are preferred,
    on dry days outdoor places. Returns a list of place_id lists, one per day.
    """
    capacity = [1 if i in (0, num_days - 1) else config.MAX_PLACES_PER_DAY for i in range(num_days)]
    slots = [0] * num_days
    order = sorted(range(num_days), key=lambda i: (-capacity[i], i))
    to_place = len(ordered_places)
    while to_place > 0 and any(slots[i] < capacity[i] for i in range(num_days)):
        for i in order:
            if to_place > 0 and slots[i] < capacity[i]:
                slots[i] += 1
                to_place -= 1
    remaining, plan = list(ordered_places), []
    for i in range(num_days):
        wanted = "indoor" if rainy_flags[i] else "outdoor"
        chosen = []
        for _ in range(slots[i]):
            pick = next((p for p in remaining if p.get("setting") == wanted), remaining[0])
            chosen.append(pick["place_id"])
            remaining.remove(pick)
        plan.append(chosen)
    return plan


def _date_range_label(start, end):
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.strftime('%b %d')}-{end.strftime('%d, %Y')}"
    return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"


def place_label(place):
    """Name with type and rating (dataset) or type only (OpenStreetMap has no ratings)."""
    detail = f"{place['type']}, rating {place['rating']}" if place.get("rating") is not None else place["type"]
    return f"{place['name']} ({detail})"


def _default_reasoning(parts, use_extended):
    """Fact-based explanations, used for any reasoning key the agent did not supply."""
    outbound, hotel, destination, origin = (parts[k] for k in ("outbound", "hotel", "destination", "origin"))
    options = search_flights(origin, destination, use_extended=use_extended, top_n=50)["options"]
    hotels = search_hotels(destination, top_n=100)["hotels"]
    rank = next((i + 1 for i, h in enumerate(hotels) if h["hotel_id"] == hotel["hotel_id"]), None)
    value = next((h["value_score"] for h in hotels if h["hotel_id"] == hotel["hotel_id"]), None)
    kind = "estimated fare" if outbound.get("estimated") else "dataset fare"
    return {
        "flight": (f"{outbound['route']} costs Rs {outbound['total_price']:,} ({kind}) with {outbound['stops']} stop(s) "
                   f"and {outbound['total_duration_hours']} h in the air. It was one of {len(options)} route option(s) "
                   f"for {origin} to {destination}, ranked by fewest stops, then price."),
        "hotel": (f"{hotel['name']} is {hotel['stars']}-star at Rs {hotel['price_per_night']:,} per night"
                  + (f", ranked {rank} of {len(hotels)} in {destination} by stars per Rs 1000 "
                     f"(score {value})." if rank else ".")),
        "itinerary": ("Places are picked by rating or notability, with indoor places on rainy days, outdoor places "
                      "on dry days, and lighter arrival and departure days."),
        "budget": "Flights, hotel nights and per-day food and local travel, computed by the budget tool.",
    }


WEATHER_WORDS = re.compile(r"\b(rain\w*|drizzle|sunny|clear sky|cloud\w*|storm\w*|hot|humid|weather|forecast)\b", re.I)


def _clean_note(text):
    """Keep an agent's day tip only if it makes no weather claim (weather comes from the API, not the LLM)."""
    text = (text or "").strip()
    if not text or WEATHER_WORDS.search(text):
        return ""
    return text if text[-1] in ".!?" else text + "."


def _clock_note(flight, index, verb):
    leg = flight["legs"][index]
    clock = leg.get("arrival_clock") if verb == "lands" else leg.get("departure_clock")
    return f" {verb} around {clock}" if clock else ""


def _humanize(text, names):
    """Replace internal hotel and place ids in agent-written text with their names."""
    for identifier in sorted(names, key=len, reverse=True):
        text = text.replace(identifier, names[identifier])
    return text


def _budget_reasoning(budget, limit, warnings):
    """Budget explanation built from the computed numbers, never from the agent's own arithmetic."""
    text = (f"Flights Rs {budget['flights']:,} + hotel Rs {budget['hotel']:,} + food and local travel "
            f"Rs {budget['food_and_local_travel']:,} = Rs {budget['total']:,}.")
    if limit:
        budget["limit"], budget["within_limit"] = limit, budget["total"] <= limit
        if budget["within_limit"]:
            text += f" This is within your budget of Rs {limit:,}, with Rs {limit - budget['total']:,} to spare."
        else:
            over = budget["total"] - limit
            warnings.append(f"Estimated total Rs {budget['total']:,} is above your budget of Rs {limit:,} by Rs {over:,}.")
            text += f" This is Rs {over:,} above your budget of Rs {limit:,}."
    return text


def finalize_plan(draft, use_extended=False, today=None, http_get=None):
    """Validate a TripDraft against the data and build the final TripPlan."""
    warnings = []
    origin_loc, destination_loc = resolve_location(draft.origin), resolve_location(draft.destination)
    origin, destination = origin_loc.name, destination_loc.name
    if origin == destination:
        raise PlanValidationError("Origin and destination must be different cities.")
    start, end = draft.start_date, draft.end_date
    num_days = (end - start).days + 1
    if not config.MIN_TRIP_DAYS <= num_days <= config.MAX_TRIP_DAYS:
        raise PlanValidationError(
            f"Trip length is {num_days} day(s); it must be {config.MIN_TRIP_DAYS} to {config.MAX_TRIP_DAYS} days.")
    nights = num_days - 1

    flight_index = flights_by_id(use_extended)
    outbound = _resolve_chain(draft.outbound_flight_ids, origin, destination, flight_index, "Outbound")
    return_flight = None
    if draft.return_flight_ids:
        return_flight = _resolve_chain(draft.return_flight_ids, destination, origin, flight_index, "Return")
    elif draft.round_trip:
        warnings.append("No return flight found or selected; the budget covers the outbound flight only.")
    for label, flight in (("Outbound", outbound), ("Return", return_flight)):
        if flight and flight["stops"] > 0:
            warnings.append(f"{label} route is a connecting itinerary ({flight['route']}); layover timing is "
                            "indicative because dataset flights have fixed sample dates.")
    flights_estimated = outbound["estimated"] or bool(return_flight and return_flight["estimated"])
    if flights_estimated:
        warnings.append("Flight fares for this route are estimates from distance, not real quotes. "
                        "Check live fares before booking.")

    hotel = _hotel_lookup(destination_loc).get(draft.hotel_id)
    if hotel is None:
        raise PlanValidationError(f"Hotel '{draft.hotel_id}' does not exist for {destination}.")
    if hotel["city"] != destination:
        raise PlanValidationError(f"Hotel '{hotel['name']}' is in {hotel['city']}, not {destination}.")
    hotel = {**hotel, "source": hotel.get("source", "dataset")}
    if hotel["source"] != "dataset":
        warnings.append("Hotel names come from OpenStreetMap; nightly prices are estimates from star level and "
                        "destination cost level, not live rates.")

    weather = get_weather(destination, start, end, today=today, http_get=http_get)
    weather_by_date = {d["date"]: d for d in weather["days"]}
    warnings += weather["notes"]

    place_index = _place_lookup(destination_loc)
    flags = [bool((weather_by_date.get((start + timedelta(days=i)).isoformat()) or {}).get("rainy"))
             for i in range(num_days)]
    draft_days = {d.day: d for d in draft.days}
    used, day_place_ids = set(), []
    for offset in range(num_days):
        valid = []
        for place_id in (draft_days[offset + 1].place_ids if offset + 1 in draft_days else []):
            place = place_index.get(place_id)
            if place is None or place["city"] != destination:
                warnings.append(f"Day {offset + 1}: place '{place_id}' is not an attraction in {destination}; removed.")
            elif place_id not in used:
                used.add(place_id)
                valid.append(place_id)
        day_place_ids.append(valid)

    rebalanced = False
    if any(not ids for ids in day_place_ids) and any(len(ids) > 1 for ids in day_place_ids):
        chosen = [{**place_index[pid], "setting": place_setting(place_index[pid]["type"])}
                  for ids in day_place_ids for pid in ids]
        day_place_ids = distribute_places(chosen, num_days, flags)
        rebalanced = True

    itinerary = []
    for offset in range(num_days):
        day_number, day_date = offset + 1, start + timedelta(days=offset)
        places = [{**place_index[pid], "setting": place_setting(place_index[pid]["type"])}
                  for pid in day_place_ids[offset]]
        notes = []
        if offset == 0:
            notes.append(f"Arrive in {destination};{_clock_note(outbound, -1, 'lands') or ' flight time is flexible'}.")
        if offset == num_days - 1 and return_flight:
            notes.append("Check out and fly home;" + (_clock_note(return_flight, 0, "departs") or " flight time is flexible") + ".")
        tip = _clean_note(draft_days[day_number].notes) if day_number in draft_days and places and not rebalanced else ""
        if tip:
            notes.append(tip)
        day_weather = weather_by_date.get(day_date.isoformat())
        if day_weather and day_weather["rainy"] and any(p["setting"] == "outdoor" for p in places):
            notes.append("Rain expected: carry an umbrella and keep an indoor backup for outdoor stops.")
        if not places:
            notes.append("Free day: explore on your own or rest.")
        itinerary.append({"day": day_number, "date": day_date.isoformat(), "weather": day_weather,
                          "places": places, "notes": " ".join(notes)})
    if rebalanced:
        warnings.append("Places were spread evenly across the days so no day is left empty while another is busy.")
    open_days = sum(1 for d in itinerary if not d["places"])
    if open_days:
        warnings.append(f"Only {len(place_index)} attraction(s) are available for {destination}, so "
                        f"{open_days} day(s) are left open.")
    if any(p.get("source") == "openstreetmap" for d in itinerary for p in d["places"]):
        warnings.append("Attractions come from OpenStreetMap and are ranked by notability, not visitor ratings.")

    flight_per_person = outbound["total_price"] + (return_flight["total_price"] if return_flight else 0)
    budget = estimate_budget(flight_per_person, hotel["price_per_night"], nights, draft.travelers, draft.style,
                             cost_index=destination_loc.cost_index)
    sources = {"flights": "estimated from distance" if flights_estimated else "dataset",
               "hotel": "dataset" if hotel["source"] == "dataset" else "OpenStreetMap (price estimated)",
               "places": "dataset" if destination_loc.in_dataset else "OpenStreetMap",
               "weather": weather["source"]}

    parts = {"outbound": outbound, "hotel": hotel, "origin": origin, "destination": destination}
    defaults = _default_reasoning(parts, use_extended)
    reasoning = {**defaults, **{k: v for k, v in draft.reasoning.items() if v}}
    if rebalanced:
        reasoning["itinerary"] = defaults["itinerary"]  # the agent's text described the plan before it was re-spread
    reasoning["budget"] = _budget_reasoning(budget, draft.budget_limit, warnings)
    names = {hotel["hotel_id"]: hotel["name"], **{pid: place["name"] for pid, place in place_index.items()}}
    reasoning = {key: _humanize(text, names) for key, text in reasoning.items()}
    return TripPlan(
        title=f"Your {num_days}-Day Trip to {destination} ({_date_range_label(start, end)})",
        origin=origin, destination=destination, start_date=start, end_date=end, num_days=num_days,
        nights=nights, travelers=draft.travelers, style=draft.style, outbound=outbound,
        return_flight=return_flight, hotel=hotel, itinerary=itinerary, weather_source=weather["source"],
        budget=budget, data_sources=sources, reasoning=reasoning, warnings=warnings)


def _pick_hotel(destination, style, cheapest=False):
    if cheapest or style == "budget":
        return search_hotels(destination, sort_by="price", top_n=1)["hotels"][0]
    if style == "premium":
        return search_hotels(destination, sort_by="stars", top_n=1)["hotels"][0]
    candidates = search_hotels(destination, min_stars=3, sort_by="value", top_n=1)["hotels"]
    return candidates[0] if candidates else search_hotels(destination, sort_by="value", top_n=1)["hotels"][0]


def plan_without_llm(origin, destination, start_date, end_date, travelers=1, style="standard",
                     interests=None, max_budget=None, round_trip=True, use_extended=False,
                     today=None, http_get=None):
    """Deterministic planner that calls the same tools without an LLM (demo mode and fallback)."""
    origin, destination = resolve_location(origin).name, resolve_location(destination).name
    out_options = search_flights(origin, destination, "cheapest", use_extended=use_extended)["options"]
    if not out_options:
        raise PlanValidationError(f"No flight route found from {origin} to {destination}.")
    outbound_ids = [leg["flight_id"] for leg in out_options[0]["legs"]]
    return_ids = []
    if round_trip:
        back = search_flights(destination, origin, "cheapest", use_extended=use_extended)["options"]
        return_ids = [leg["flight_id"] for leg in back[0]["legs"]] if back else []

    num_days = (end_date - start_date).days + 1
    weather = get_weather(destination, start_date, end_date, today=today, http_get=http_get)
    rainy = {d["date"]: d["rainy"] for d in weather["days"]}
    flags = [rainy.get((start_date + timedelta(days=i)).isoformat(), False) for i in range(num_days)]

    everything = discover_places(destination, top_n=50)["places"]
    if not everything:
        raise PlanValidationError(f"No attractions found for {destination}.")
    wanted = {t.lower() for t in (interests or [])}
    matched = [p for p in everything if p["type"] in wanted]
    ordered = matched + [p for p in everything if p not in matched]
    extra_warnings = []
    if wanted and len(matched) < min(len(everything), 4):
        found = "None of the places" if not matched else f"Only {len(matched)} place(s)"
        extra_warnings.append(f"{found} in {destination} match your interests ({', '.join(sorted(wanted))}); "
                              "other top places were added.")
    day_places = distribute_places(ordered, num_days, flags)
    days = [{"day": i + 1, "place_ids": ids} for i, ids in enumerate(day_places)]

    def build(hotel):
        draft = TripDraft(origin=origin, destination=destination, start_date=start_date, end_date=end_date,
                          travelers=travelers, style=style, round_trip=round_trip,
                          outbound_flight_ids=outbound_ids, return_flight_ids=return_ids,
                          hotel_id=hotel["hotel_id"], budget_limit=max_budget, days=days)
        return finalize_plan(draft, use_extended, today, http_get)

    plan = build(_pick_hotel(destination, style))
    if max_budget and plan.budget["total"] > max_budget:
        plan = build(_pick_hotel(destination, style, cheapest=True))
        if plan.budget["within_limit"]:
            extra_warnings.append("Switched to the cheapest hotel to stay within your budget.")
    plan.warnings += extra_warnings
    return plan


def render_markdown(plan):
    """Human-readable itinerary in the layout requested by the project brief."""
    def money(value):
        return f"Rs {value:,}"

    def describe_flight(flight):
        lines = []
        for leg in flight["legs"]:
            if leg.get("estimated"):
                lines.append(f"- {leg['from']} to {leg['to']}, about {leg['duration_hours']} h "
                             f"(estimated fare {money(leg['price'])})")
            else:
                lines.append(f"- {leg['airline']} {leg['flight_id']}: {leg['from']} to {leg['to']}, departs "
                             f"{leg['departure_clock']}, arrives {leg['arrival_clock']} ({money(leg['price'])})")
        return lines

    hotel, budget = plan.hotel, plan.budget
    price_note = " (estimated)" if hotel.get("estimated_price") else ""
    lines = [f"# {plan.title}", f"{plan.travelers} traveller(s), {plan.style} style, {plan.nights} night(s).", ""]
    lines += ["## Flight Selected", f"**Outbound ({plan.outbound['route']}, {plan.outbound['stops']} stop(s)):**"]
    lines += describe_flight(plan.outbound)
    if plan.return_flight:
        lines += [f"**Return ({plan.return_flight['route']}, {plan.return_flight['stops']} stop(s)):**"]
        lines += describe_flight(plan.return_flight)
    lines += ["", "## Hotel Recommendation",
              f"- {hotel['name']} ({money(hotel['price_per_night'])}/night{price_note}, {hotel['stars']}-star)"]
    if hotel["amenities"]:
        lines.append(f"- Amenities: {', '.join(hotel['amenities'])}")
    lines += ["", "## Day-wise Itinerary"]
    for day in plan.itinerary:
        weather = day["weather"]
        weather_text = (f"{weather['condition']}, {weather['temp_min_c']} to {weather['temp_max_c']} C, "
                        f"rain {weather['precipitation_mm']} mm") if weather else "weather unavailable"
        stops = ", ".join(place_label(p) for p in day["places"]) or "Free day"
        lines += [f"**Day {day['day']} ({day['date']})** - {weather_text}", f"- {stops}"]
        if day["notes"]:
            lines.append(f"- {day['notes']}")
        lines.append("")
    lines += ["## Estimated Total Budget",
              f"- Flights: {money(budget['flights'])}",
              f"- Hotel ({budget['nights']} night(s), {budget['rooms']} room(s)): {money(budget['hotel'])}",
              f"- Food and local travel: {money(budget['food_and_local_travel'])}",
              f"- **Total cost: {money(budget['total'])}**"]
    if "limit" in budget:
        lines.append(f"- Your budget: {money(budget['limit'])} ({'within budget' if budget['within_limit'] else 'over budget'})")
    lines += ["", "## Why we selected this"]
    lines += [f"- **{key.title()}:** {text}" for key, text in plan.reasoning.items()]
    if plan.warnings:
        lines += ["", "## Notes"] + [f"- {w}" for w in plan.warnings]
    return "\n".join(lines)
