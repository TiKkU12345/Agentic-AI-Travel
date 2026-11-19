"""LangChain tools: thin wrappers that expose the planning functions to the agent as JSON."""
import json
from typing import Optional

from langchain_core.tools import tool

from travel_planner import budget as budget_module
from travel_planner import flights as flights_module
from travel_planner import hotels as hotels_module
from travel_planner import places as places_module
from travel_planner.locations import resolve_location
from travel_planner import weather as weather_module


def _as_json(callable_, *args, **kwargs):
    """Run a function and return JSON text; errors become {"error": ...} so the agent can recover."""
    try:
        return json.dumps(callable_(*args, **kwargs), default=str)
    except (ValueError, KeyError, RuntimeError) as error:
        return json.dumps({"error": str(error)})


def _compact_flights(result):
    """Trim flight search output to the fields the agent needs, to save tokens."""
    for option in result["options"]:
        option["legs"] = [{k: leg[k] for k in ("flight_id", "airline", "from", "to", "departure_clock",
                                               "arrival_clock", "duration_hours", "price")}
                          for leg in option["legs"]]
    return result


def make_tools(use_extended=False):
    """Build the five agent tools. use_extended switches on the synthetic extra flights."""

    @tool("search_flights")
    def search_flights(origin: str, destination: str, preference: str = "cheapest",
                       max_price: Optional[int] = None) -> str:
        """Find flight options between two cities (direct or connecting).

        origin/destination: any city in the world, such as Delhi, Goa or Tokyo. Dataset cities (Delhi, Mumbai,
        Goa, Bangalore, Chennai, Hyderabad, Kolkata, Jaipur) use real dataset flights; other routes return an
        estimated fare from distance.
        preference: "cheapest" or "fastest". max_price: optional cap on total price in rupees.
        Returns options ranked by fewest stops, then price or duration. Each option lists legs with
        flight_id values that must be reused in the final plan. Call it once for the outbound route
        and once with origin and destination swapped for the return.
        """
        return _as_json(lambda: _compact_flights(flights_module.search_flights(
            origin, destination, preference, max_price=max_price, use_extended=use_extended)))

    @tool("recommend_hotels")
    def recommend_hotels(city: str, min_stars: int = 0, max_price_per_night: Optional[int] = None,
                         required_amenities: Optional[list[str]] = None, sort_by: str = "value") -> str:
        """Recommend hotels in a city.

        sort_by: "value" (stars per rupee, default), "price" (cheapest first) or "stars" (highest first).
        required_amenities: optional list such as ["wifi", "pool"]. Returns hotel_id, name, stars,
        price_per_night, amenities and value_score.
        """
        return _as_json(hotels_module.search_hotels, city, min_stars, max_price_per_night,
                        required_amenities, sort_by)

    @tool("discover_places")
    def discover_places(city: str, types: Optional[list[str]] = None, min_rating: float = 0.0) -> str:
        """List attractions in a city, best rated first.

        types: optional filter from beach, park, lake, fort, museum, temple, market, monument, gallery,
        viewpoint, zoo, aquarium, theme_park, attraction, historic_site.
        Returns place_id, name, type, rating (dataset cities only) and setting (indoor or outdoor). Places outside
        the dataset come from OpenStreetMap and have no rating. Use place_id values in the final plan.
        """
        return _as_json(places_module.discover_places, city, types, min_rating)

    @tool("get_weather")
    def get_weather(city: str, start_date: str, end_date: str) -> str:
        """Daily weather for a city between two ISO dates (YYYY-MM-DD), via the free Open-Meteo API.

        Returns condition, temperature range, rainfall and a rainy flag for each day. Use it to put
        indoor places on rainy days. The source field says if it is a live forecast or last year's data.
        """
        return _as_json(weather_module.get_weather, city, start_date, end_date)

    @tool("estimate_budget")
    def estimate_budget(flight_cost_per_person: int, hotel_price_per_night: int, nights: int,
                        travelers: int = 1, style: str = "standard", destination: Optional[str] = None) -> str:
        """Estimate the total trip cost in rupees.

        flight_cost_per_person: outbound plus return flight cost for one traveller.
        nights: number of hotel nights (trip days minus 1). style: "budget", "standard" or "premium"
        (sets daily food and local travel spend). destination: pass the destination city so daily costs
        match its price level. Returns flights, hotel, food_and_local_travel, total.
        """
        def compute():
            index = resolve_location(destination).cost_index if destination else 1.0
            return budget_module.estimate_budget(flight_cost_per_person, hotel_price_per_night, nights,
                                                 travelers, style, cost_index=index)
        return _as_json(compute)

    return [search_flights, recommend_hotels, discover_places, get_weather, estimate_budget]
