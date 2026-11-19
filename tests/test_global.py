"""Tests for places outside the dataset: geocoding, estimated flights, OpenStreetMap places and hotels."""
from datetime import date

import pytest

from conftest import make_fake_http_get
from travel_planner import config, global_data, http_client
from travel_planner.budget import estimate_budget
from travel_planner.data_loader import UnknownCityError
from travel_planner.flights import search_flights
from travel_planner.hotels import search_hotels
from travel_planner.locations import resolve_location
from travel_planner.places import discover_places
from travel_planner.planner import PlanValidationError, finalize_plan, plan_without_llm, render_markdown
from travel_planner.schemas import TripDraft
from travel_planner.tools import make_tools

TODAY = date(2026, 11, 5)


def test_dataset_cities_never_hit_the_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network used for a dataset city")
    monkeypatch.setattr(http_client, "get_json", boom)
    location = resolve_location("bengaluru")
    assert location.name == "Bangalore" and location.in_dataset and location.cost_index == 1.0


def test_other_cities_are_geocoded_once_and_unknown_ones_rejected(mock_world, monkeypatch):
    tokyo = resolve_location("tokyo")
    assert (tokyo.name, tokyo.country_code, tokyo.in_dataset) == ("Tokyo", "JP", False)
    assert tokyo.cost_index == config.COST_INDEX["JP"]
    with pytest.raises(UnknownCityError, match="Could not find"):
        resolve_location("Atlantis")
    monkeypatch.setattr(http_client, "get_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached")))
    assert resolve_location("Tokyo") is tokyo


def test_flight_to_foreign_city_is_an_estimate_from_distance(mock_world):
    result = search_flights("Delhi", "Tokyo")
    leg = result["options"][0]["legs"][0]
    delhi, tokyo = resolve_location("Delhi"), resolve_location("Tokyo")
    km = global_data.haversine_km(delhi.latitude, delhi.longitude, tokyo.latitude, tokyo.longitude)
    assert leg["flight_id"] == "EST|Delhi|Tokyo" and leg["estimated"] is True
    assert leg["price"] == int(round((6000 + 5.5 * km) / 50.0) * 50)
    assert 5700 < km < 6100 and result["notes"]


def test_domestic_estimate_is_cheaper_than_international(mock_world):
    domestic = global_data.estimate_flight(resolve_location("Delhi"), resolve_location("Mumbai"))
    abroad = global_data.estimate_flight(resolve_location("Delhi"), resolve_location("Paris"))
    assert domestic["price"] < abroad["price"]


def test_dataset_pair_still_uses_dataset_even_when_route_is_missing(mock_world):
    option = search_flights("Delhi", "Goa")["options"][0]
    assert option["stops"] == 3 and all(leg["flight_id"].startswith("FL") for leg in option["legs"])


def test_flight_price_filter_applies_to_estimates(mock_world):
    assert search_flights("Delhi", "Tokyo", max_price=1000)["options"] == []


def test_osm_places_are_deduplicated_ranked_and_unrated(mock_world):
    places = discover_places("Tokyo")["places"]
    assert [p["name"] for p in places][:2] == ["Senso-ji", "Meiji Shrine"]
    assert len([p for p in places if p["name"] == "Senso-ji"]) == 1
    assert all(p["rating"] is None and p["source"] == "openstreetmap" for p in places)
    types = {p["name"]: (p["type"], p["setting"]) for p in places}
    assert types["Tokyo National Museum"] == ("museum", "indoor")
    assert types["Ueno Park"] == ("park", "outdoor") and types["Meiji Shrine"][0] == "monument"
    assert [p["name"] for p in discover_places("Tokyo", types=["museum"])["places"]] == ["Tokyo National Museum"]


def test_osm_hotels_have_estimated_prices_scaled_by_cost_level(mock_world):
    hotels = {h["name"]: h for h in search_hotels("Tokyo", top_n=10)["hotels"]}
    assert hotels["Hotel Sakura"]["stars"] == 4 and hotels["Shinjuku Inn"]["stars"] == 3
    assert hotels["Budget Hostel"]["stars"] == 1 and hotels["Budget Hostel"]["stars_estimated"] is True
    assert hotels["Shinjuku Inn"]["amenities"] == ["wifi"]
    base = config.BASE_HOTEL_PRICE_BY_STARS[4] * config.COST_INDEX["JP"]
    assert 0.8 * base <= hotels["Hotel Sakura"]["price_per_night"] <= 1.2 * base
    assert all(h["estimated_price"] for h in hotels.values())


def test_generic_hotel_fallback_when_osm_has_none(mock_world):
    mock_world["hotels"].clear()
    hotel = search_hotels("Tokyo")["hotels"][0]
    assert hotel["hotel_id"] == "EST|stay|Tokyo" and hotel["stars"] == 3


def test_budget_daily_costs_follow_destination_price_level():
    dataset = estimate_budget(10000, 3000, nights=2, travelers=1, style="standard")
    japan = estimate_budget(10000, 3000, nights=2, travelers=1, style="standard", cost_index=2.6)
    assert dataset["daily_expense_per_person"] == 1200 and japan["daily_expense_per_person"] == 3120


def test_finalize_accepts_estimated_flights_and_osm_ids(mock_world):
    draft = TripDraft(origin="Delhi", destination="Tokyo", start_date=date(2026, 11, 10), end_date=date(2026, 11, 12),
                      outbound_flight_ids=["EST|Delhi|Tokyo"], return_flight_ids=["EST|Tokyo|Delhi"],
                      hotel_id="OSM|node|101", days=[{"day": 1, "place_ids": ["OSM|node|1"]},
                                                     {"day": 2, "place_ids": ["OSM|way|2", "OSM|node|3"]}])
    calls = []
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get(calls=calls))
    assert calls[0][1]["latitude"] == pytest.approx(35.6895)  # weather uses geocoded coordinates
    assert plan.data_sources["flights"] == "estimated from distance"
    assert plan.data_sources["places"] == "OpenStreetMap" and plan.hotel["estimated_price"] is True
    assert plan.budget["daily_expense_per_person"] == round(1200 * config.COST_INDEX["JP"])
    assert any("estimates from distance" in w for w in plan.warnings)
    assert any("OpenStreetMap" in w for w in plan.warnings)
    assert plan.itinerary[0]["notes"].startswith("Arrive in Tokyo; flight time is flexible")


def test_finalize_rejects_unknown_or_foreign_osm_ids(mock_world):
    base = dict(origin="Delhi", destination="Tokyo", start_date=date(2026, 11, 10), end_date=date(2026, 11, 12),
                outbound_flight_ids=["EST|Delhi|Tokyo"], return_flight_ids=[])
    with pytest.raises(PlanValidationError, match="does not exist"):
        finalize_plan(TripDraft(hotel_id="OSM|node|999", **base), today=TODAY, http_get=make_fake_http_get())
    with pytest.raises(PlanValidationError, match="connected route"):
        finalize_plan(TripDraft(hotel_id="OSM|node|101", **{**base, "outbound_flight_ids": ["EST|Delhi|Paris"]}),
                      today=TODAY, http_get=make_fake_http_get())


def test_plan_without_llm_for_foreign_destination_and_render(mock_world):
    plan = plan_without_llm("Delhi", "Tokyo", date(2026, 11, 10), date(2026, 11, 12), interests=["museum"],
                            today=TODAY, http_get=make_fake_http_get(rain_on_index=1))
    text = render_markdown(plan)
    assert "estimated fare" in text and "(estimated" in text
    assert "Tokyo National Museum (museum)" in text and "rating None" not in text
    assert plan.budget["total"] == plan.budget["flights"] + plan.budget["hotel"] + plan.budget["food_and_local_travel"]


def test_search_flights_tool_returns_estimated_id_to_the_agent(mock_world):
    tool = next(t for t in make_tools() if t.name == "search_flights")
    assert "EST|Delhi|Tokyo" in tool.invoke({"origin": "Delhi", "destination": "Tokyo"})
