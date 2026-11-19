"""Tests for plan validation, the deterministic planner and rendering."""
from datetime import date

import pytest

from conftest import make_fake_http_get
from travel_planner.planner import PlanValidationError, distribute_places, finalize_plan, plan_without_llm, render_markdown
from travel_planner.schemas import TripDraft

TODAY = date(2026, 11, 5)


def make_draft(**overrides):
    base = dict(origin="Bangalore", destination="Goa", start_date=date(2026, 11, 10), end_date=date(2026, 11, 12),
                outbound_flight_ids=["FL0021"], return_flight_ids=["FL0023"], hotel_id="HOT0014",
                days=[{"day": 1, "place_ids": ["PLC0012"]}, {"day": 2, "place_ids": ["PLC0014", "PLC0011"]}])
    base.update(overrides)
    return TripDraft(**base)


def test_finalize_recomputes_budget_from_dataset_not_from_llm():
    plan = finalize_plan(make_draft(), today=TODAY, http_get=make_fake_http_get())
    assert plan.budget["flights"] == 5356 + 4537
    assert plan.budget["hotel"] == 1232 * 2
    assert plan.budget["total"] == 5356 + 4537 + 2464 + 1200 * 3
    assert plan.nights == 2 and plan.num_days == 3
    assert plan.title.startswith("Your 3-Day Trip to Goa")


def test_finalize_rejects_bad_references():
    kwargs = dict(today=TODAY, http_get=make_fake_http_get())
    with pytest.raises(PlanValidationError, match="does not exist"):
        finalize_plan(make_draft(outbound_flight_ids=["FL9999"]), **kwargs)
    with pytest.raises(PlanValidationError, match="connected route"):
        finalize_plan(make_draft(outbound_flight_ids=["FL0023"]), **kwargs)
    with pytest.raises(PlanValidationError, match="not Goa"):
        finalize_plan(make_draft(hotel_id="HOT0001"), **kwargs)
    with pytest.raises(PlanValidationError, match="1 to 30"):
        finalize_plan(make_draft(end_date=date(2026, 11, 9)), **kwargs)
    with pytest.raises(PlanValidationError, match="1 to 30"):
        finalize_plan(make_draft(end_date=date(2026, 12, 10)), **kwargs)


def test_finalize_removes_places_from_other_cities_and_duplicates():
    draft = make_draft(days=[{"day": 1, "place_ids": ["PLC0012", "PLC0001"]}, {"day": 2, "place_ids": ["PLC0012"]}])
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    assert [p["place_id"] for p in plan.itinerary[0]["places"]] == ["PLC0012"]
    assert plan.itinerary[1]["places"] == []
    assert any("PLC0001" in w for w in plan.warnings)


def test_missing_return_flight_is_warned():
    plan = finalize_plan(make_draft(return_flight_ids=[]), today=TODAY, http_get=make_fake_http_get())
    assert plan.return_flight is None
    assert any("No return flight" in w for w in plan.warnings)


def test_weather_failure_does_not_break_the_plan():
    import requests

    def broken(url, params, timeout):
        raise requests.ConnectionError("offline")
    plan = finalize_plan(make_draft(), today=TODAY, http_get=broken)
    assert plan.weather_source == "unavailable"
    assert all(day["weather"] is None for day in plan.itinerary)


def test_distribute_places_prefers_indoor_on_rainy_days_and_never_repeats():
    places = [{"place_id": f"P{i}", "setting": s, "rating": 5 - i * 0.1}
              for i, s in enumerate(["outdoor", "indoor", "outdoor", "indoor", "outdoor"])]
    plan = distribute_places(places, 3, [False, True, False])
    flat = [pid for day in plan for pid in day]
    assert len(flat) == len(set(flat))
    middle = {p["place_id"]: p["setting"] for p in places}
    assert all(middle[pid] == "indoor" for pid in plan[1])
    assert len(plan[0]) <= 1 and len(plan[2]) <= 1 and len(plan[1]) <= 2


def test_distribute_handles_more_days_than_places():
    places = [{"place_id": f"P{i}", "setting": "outdoor", "rating": 4.0} for i in range(5)]
    plan = distribute_places(places, 7, [False] * 7)
    assert sum(len(d) for d in plan) == 5 and len(plan) == 7


def test_plan_without_llm_end_to_end_and_render():
    plan = plan_without_llm("Bangalore", "Goa", date(2026, 11, 10), date(2026, 11, 12), interests=["fort"],
                            today=TODAY, http_get=make_fake_http_get(rain_on_index=1))
    assert plan.outbound["stops"] == 0 and plan.return_flight["stops"] == 0
    assert plan.itinerary[1]["weather"]["rainy"] is True
    text = render_markdown(plan)
    for heading in ("Flight Selected", "Hotel Recommendation", "Day-wise Itinerary",
                    "Estimated Total Budget", "Why we selected this"):
        assert heading in text
    assert f"Total cost: Rs {plan.budget['total']:,}" in text


def test_budget_cap_switches_to_cheapest_hotel_or_warns():
    plan = plan_without_llm("Bangalore", "Goa", date(2026, 11, 10), date(2026, 11, 12), style="premium",
                            max_budget=1000, today=TODAY, http_get=make_fake_http_get())
    assert any("above your budget" in w for w in plan.warnings)
    assert plan.hotel["price_per_night"] == 1232


def test_plan_json_round_trip():
    plan = plan_without_llm("Bangalore", "Goa", date(2026, 11, 10), date(2026, 11, 12), today=TODAY,
                            http_get=make_fake_http_get())
    assert '"budget"' in plan.to_json()


def _goa_draft(days):
    return make_draft(start_date=date(2026, 11, 20), end_date=date(2026, 11, 26), days=days)


def test_open_day_count_uses_only_the_destination_cities_places():
    draft = _goa_draft([{"day": 1, "place_ids": ["PLC0012"]}])
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    assert any("Only 5 attraction(s)" in w for w in plan.warnings)


def test_weather_claims_in_agent_notes_are_dropped_and_free_day_text_is_not_duplicated():
    draft = _goa_draft([{"day": 1, "place_ids": ["PLC0012"], "notes": "Start early as light drizzle begins"},
                        {"day": 2, "place_ids": ["PLC0014"], "notes": "Try the sunset view"},
                        {"day": 4, "notes": "Leisure day"}])
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    notes = [d["notes"] for d in plan.itinerary]
    assert all("drizzle" not in n for n in notes)
    assert "Try the sunset view." in notes[1]
    assert all(n.count("Free day") <= 1 and "Leisure day" not in n for n in notes)


def test_places_are_spread_when_agent_front_loads_the_trip():
    draft = _goa_draft([{"day": 1, "place_ids": ["PLC0012", "PLC0014"]}, {"day": 2, "place_ids": ["PLC0013", "PLC0011"]},
                        {"day": 3, "place_ids": ["PLC0015"]}])
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    counts = [len(d["places"]) for d in plan.itinerary]
    assert sum(counts) == 5 and max(counts) == 1 and all(counts[1:6])
    assert any("spread evenly" in w for w in plan.warnings)


def test_balanced_agent_plan_is_left_untouched():
    draft = make_draft(days=[{"day": 1, "place_ids": ["PLC0012"]}, {"day": 2, "place_ids": ["PLC0014", "PLC0011"]},
                             {"day": 3, "place_ids": ["PLC0013"]}])
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    assert [len(d["places"]) for d in plan.itinerary] == [1, 2, 1]
    assert not any("spread evenly" in w for w in plan.warnings)


def test_trip_length_is_not_limited_to_a_week():
    for end, days in ((date(2026, 11, 11), 2), (date(2026, 11, 17), 8), (date(2026, 12, 9), 30)):
        plan = finalize_plan(make_draft(end_date=end, days=[{"day": 1, "place_ids": ["PLC0012"]}]),
                             today=TODAY, http_get=make_fake_http_get())
        assert plan.num_days == days and len(plan.itinerary) == days
        assert plan.budget["nights"] == days - 1 and plan.budget["food_and_local_travel"] == 1200 * days


def test_long_trip_with_few_places_spreads_them_and_leaves_free_days():
    plan = plan_without_llm("Bangalore", "Goa", date(2026, 11, 10), date(2026, 12, 9), today=TODAY,
                            http_get=make_fake_http_get())
    assert plan.num_days == 30
    assert sum(len(d["places"]) for d in plan.itinerary) == 5
    assert any("Only 5 attraction(s)" in w for w in plan.warnings)
    assert "Day 30" in render_markdown(plan)


def test_config_limits_are_sane():
    from travel_planner import config
    assert config.MIN_TRIP_DAYS == 1 and config.MAX_TRIP_DAYS == 30


def test_ids_in_agent_reasoning_become_names_and_budget_text_uses_computed_numbers():
    draft = make_draft(budget_limit=30000, days=[{"day": 1, "place_ids": ["PLC0012"]}, {"day": 2, "place_ids": ["PLC0014", "PLC0011"]},
                                                 {"day": 3, "place_ids": ["PLC0013"]}],
                       reasoning={"hotel": "HOT0014 is cheapest", "itinerary": "Day 1: PLC0014 and PLC0012",
                                  "budget": "Total is Rs 1 only"})
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    assert "Royal Heritage is cheapest" in plan.reasoning["hotel"] and "HOT0014" not in plan.reasoning["hotel"]
    assert "PLC0014" not in plan.reasoning["itinerary"] and "Beautiful Park" in plan.reasoning["itinerary"]
    total = plan.budget["total"]
    assert f"Rs {total:,}" in plan.reasoning["budget"] and "Rs 1 only" not in plan.reasoning["budget"]
    assert plan.budget["within_limit"] is True and "to spare" in plan.reasoning["budget"]


def test_over_budget_is_reported_with_the_exact_gap():
    plan = finalize_plan(make_draft(budget_limit=5000), today=TODAY, http_get=make_fake_http_get())
    gap = plan.budget["total"] - 5000
    assert plan.budget["within_limit"] is False
    assert any(f"above your budget of Rs 5,000 by Rs {gap:,}" in w for w in plan.warnings)


def test_rebalanced_plan_does_not_keep_stale_itinerary_reasoning():
    draft = _goa_draft([{"day": 1, "place_ids": ["PLC0012", "PLC0014"]}, {"day": 2, "place_ids": ["PLC0013", "PLC0011"]}])
    draft.reasoning["itinerary"] = "Day 1 has two forts, Day 2 has two museums"
    plan = finalize_plan(draft, today=TODAY, http_get=make_fake_http_get())
    assert "two forts" not in plan.reasoning["itinerary"] and "rating or notability" in plan.reasoning["itinerary"]


def test_budget_text_appears_in_markdown_when_a_limit_is_given():
    plan = plan_without_llm("Bangalore", "Goa", date(2026, 11, 10), date(2026, 11, 12), max_budget=60000,
                            today=TODAY, http_get=make_fake_http_get())
    assert "within budget" in render_markdown(plan)
