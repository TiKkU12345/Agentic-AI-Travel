"""Budget estimation: flights + hotel + daily local expenses."""
import math

from travel_planner import config


def estimate_budget(flight_cost_per_person, hotel_price_per_night, nights, travelers=1, style="standard",
                    cost_index=1.0):
    """Estimate total trip cost in rupees.

    flight_cost_per_person: total flight cost for one traveller (outbound plus return).
    Rooms are assumed to hold 2 people each. Local expenses (food and local travel) are charged
    per person per day, using the style rate (budget 800, standard 1200, premium 2000) times cost_index,
    a price level relative to India that is 1.0 for dataset cities.
    """
    if style not in config.DAILY_EXPENSE_BY_STYLE:
        style = "standard"
    travelers, nights = max(1, int(travelers)), max(0, int(nights))
    days = nights + 1
    rooms = math.ceil(travelers / 2)
    daily = round(config.DAILY_EXPENSE_BY_STYLE[style] * cost_index)
    flights = round(flight_cost_per_person * travelers)
    hotel = round(hotel_price_per_night * nights * rooms)
    local = round(daily * days * travelers)
    return {"flights": flights, "hotel": hotel, "food_and_local_travel": local,
            "total": flights + hotel + local, "travelers": travelers, "rooms": rooms,
            "nights": nights, "days": days, "daily_expense_per_person": daily, "style": style}
