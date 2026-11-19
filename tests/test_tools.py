"""Unit tests for flights, hotels, places, weather and budget modules."""
from datetime import date

import pytest
import requests

from conftest import make_fake_http_get
from travel_planner import config
from travel_planner.budget import estimate_budget
from travel_planner.data_loader import UnknownCityError, normalize_city
from travel_planner.flights import search_flights
from travel_planner.hotels import search_hotels
from travel_planner.places import discover_places
from travel_planner.weather import clear_cache, get_weather


def test_city_aliases_and_unknown_city():
    assert normalize_city(" bengaluru ") == "Bangalore"
    assert normalize_city("new delhi") == "Delhi"
    with pytest.raises(UnknownCityError):
        normalize_city("Paris")


def test_direct_flight_is_ranked_before_connections():
    result = search_flights("Bangalore", "Goa")
    assert result["direct_available"] is True
    assert result["options"][0]["stops"] == 0
    assert result["options"][0]["total_price"] == 5356  # cheaper of FL0020 (6624) and FL0021 (5356)


def test_fastest_preference_picks_shortest_duration():
    result = search_flights("Bangalore", "Goa", preference="fastest")
    assert result["options"][0]["legs"][0]["flight_id"] == "FL0020"  # 1 hour


def test_missing_route_falls_back_to_connections_and_extended_data_adds_direct():
    original = search_flights("Delhi", "Goa")
    assert original["direct_available"] is False
    assert original["options"][0]["stops"] == 3
    extended = search_flights("Delhi", "Goa", use_extended=True)
    assert extended["options"][0]["stops"] == 0


def test_max_price_filter_can_remove_all_options():
    result = search_flights("Bangalore", "Goa", max_price=100)
    assert result["options"] == [] and result["notes"]


def test_hotel_filters_and_sorting():
    best_value = search_hotels("Goa")["hotels"][0]
    assert best_value["hotel_id"] == "HOT0014"
    cheapest = search_hotels("Goa", sort_by="price")["hotels"][0]
    assert cheapest["price_per_night"] == 1232
    assert all(h["stars"] >= 5 for h in search_hotels("Goa", min_stars=5)["hotels"])
    with_pool_gym = search_hotels("Goa", required_amenities=["pool", "gym"], top_n=10)["hotels"]
    assert all({"pool", "gym"} <= set(h["amenities"]) for h in with_pool_gym)
    assert search_hotels("Goa", max_price_per_night=100)["hotels"] == []


def test_places_sorted_by_rating_and_type_filter():
    places = discover_places("Kolkata")["places"]
    ratings = [p["rating"] for p in places]
    assert ratings == sorted(ratings, reverse=True)
    temples = discover_places("Kolkata", types=["temple"])["places"]
    assert temples and all(p["type"] == "temple" for p in temples)
    assert discover_places("Goa", types=["beach"])["places"] == []


def test_budget_matches_brief_example_shape():
    result = estimate_budget(4800, 3200, nights=2, travelers=1, style="budget")
    assert result["flights"] == 4800 and result["hotel"] == 6400
    assert result["food_and_local_travel"] == 800 * 3
    assert result["total"] == 4800 + 6400 + 2400


def test_budget_scales_with_travelers_and_rooms():
    result = estimate_budget(5000, 2000, nights=2, travelers=3)
    assert result["rooms"] == 2 and result["flights"] == 15000 and result["hotel"] == 8000


def test_weather_uses_live_forecast_inside_window():
    calls = []
    result = get_weather("Goa", "2026-11-10", "2026-11-12", today=date(2026, 11, 5),
                         http_get=make_fake_http_get(calls=calls))
    assert result["source"] == "live_forecast" and len(result["days"]) == 3
    assert calls[0][0] == config.OPEN_METEO_FORECAST_URL
    assert calls[0][1]["latitude"] == 15.2993


def test_weather_far_future_uses_last_year_archive():
    calls = []
    result = get_weather("Goa", "2027-02-12", "2027-02-14", today=date(2026, 11, 5),
                         http_get=make_fake_http_get(calls=calls))
    assert result["source"] == "historical_same_dates_last_year"
    assert calls[0][1]["start_date"] == "2026-02-12"
    assert result["days"][0]["date"] == "2027-02-12" and result["notes"]


def test_weather_marks_rainy_days_and_handles_api_failure():
    rainy = get_weather("Goa", "2026-11-10", "2026-11-12", today=date(2026, 11, 5),
                        http_get=make_fake_http_get(rain_on_index=1))
    assert [d["rainy"] for d in rainy["days"]] == [False, True, False]

    clear_cache()

    def broken(url, params, timeout):
        raise requests.ConnectionError("offline")
    failed = get_weather("Goa", "2026-11-10", "2026-11-12", today=date(2026, 11, 5), http_get=broken)
    assert failed["status"] == "error" and failed["days"] == []


def test_weather_results_are_cached():
    calls = []
    fake = make_fake_http_get(calls=calls)
    get_weather("Goa", "2026-11-10", "2026-11-12", today=date(2026, 11, 5), http_get=fake)
    get_weather("Goa", "2026-11-10", "2026-11-12", today=date(2026, 11, 5), http_get=fake)
    assert len(calls) == 1


def test_long_trip_uses_live_forecast_first_then_last_years_weather():
    calls = []
    result = get_weather("Goa", "2026-11-15", "2026-11-25", today=date(2026, 11, 5),
                         http_get=make_fake_http_get(calls=calls))
    assert result["source"] == "live_forecast+historical_same_dates_last_year"
    assert len(result["days"]) == 11 and result["days"][0]["date"] == "2026-11-15"
    assert result["days"][-1]["date"] == "2026-11-25"
    assert calls[0][1]["end_date"] == "2026-11-20" and calls[1][1]["start_date"] == "2025-11-21"
    assert any("2026-11-21 onwards" in n for n in result["notes"])


def test_one_failed_weather_segment_keeps_the_other():
    import requests as req
    good = make_fake_http_get()

    def flaky(url, params, timeout):
        if "archive" in url:
            raise req.ConnectionError("offline")
        return good(url, params, timeout)
    result = get_weather("Goa", "2026-11-15", "2026-11-25", today=date(2026, 11, 5), http_get=flaky)
    assert result["status"] == "ok" and len(result["days"]) == 6
    assert any("Weather unavailable for 2026-11-21" in n for n in result["notes"])
