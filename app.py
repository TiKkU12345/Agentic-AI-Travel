"""Streamlit interface for the agentic travel planner."""
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from travel_planner import config
from travel_planner.agent import AgentError, MissingApiKeyError, run_agent
from travel_planner.data_loader import supported_cities
from travel_planner.places import available_place_types
from travel_planner.planner import PlanValidationError, place_label, plan_without_llm, render_markdown

load_dotenv()
st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("AI Travel Planning Assistant")
st.caption("A LangChain agent that searches flights, hotels, places and live weather, then builds a day-wise "
           "itinerary with a budget and the reasoning behind each choice.")

MODE_AGENT, MODE_RULES = "AI agent (Groq LLM)", "Rule-based planner (no API key)"
CITIES = supported_cities()
default_start = date.today() + timedelta(days=10)
EXAMPLE_QUERY = (f"Plan a 3 day trip from Bangalore to Goa from {default_start:%Y-%m-%d} to "
                 f"{default_start + timedelta(days=2):%Y-%m-%d} for 2 people. We like beaches, budget under 30000.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Planner", [MODE_AGENT, MODE_RULES])
    use_extended = st.checkbox("Fill missing flight routes with synthetic flights",
                               help="The provided data covers only 25 of 56 routes, so some trips need connections. "
                                    "This adds clearly marked synthetic direct flights.")
    api_key, model_name = "", None
    if mode == MODE_AGENT:
        model_name = st.text_input("Groq model", value=config.DEFAULT_LLM_MODEL,
                                   help="If this model is retired, the app tries the fallback models automatically.")
        if not os.getenv("GROQ_API_KEY"):
            api_key = st.text_input("Groq API key", type="password")
    st.caption("Any city in the world works. Exact dataset records exist for: " + ", ".join(CITIES) +
               ". Other places use live OpenStreetMap data with estimated flight and hotel prices.")


def show_plan(plan, trace=None, attempts=1, model=None, recovered=False):
    st.header(plan.title)
    budget = plan.budget
    cols = st.columns(4)
    cols[0].metric("Total cost", f"Rs {budget['total']:,}")
    cols[1].metric("Flights", f"Rs {budget['flights']:,}")
    cols[2].metric("Hotel", f"Rs {budget['hotel']:,}")
    cols[3].metric("Food and local travel", f"Rs {budget['food_and_local_travel']:,}")
    for warning in plan.warnings:
        st.warning(warning)

    tabs = st.tabs(["Itinerary", "Budget", "Why this plan", "JSON", "Agent steps"])
    with tabs[0]:
        st.subheader("Flights")
        for label, flight in (("Outbound", plan.outbound), ("Return", plan.return_flight)):
            if flight:
                st.markdown(f"**{label}: {flight['route']}** ({flight['stops']} stop(s), Rs {flight['total_price']:,})")
                for leg in flight["legs"]:
                    if leg.get("estimated"):
                        st.write(f"{leg['from']} to {leg['to']}, about {leg['duration_hours']} h "
                                 f"(estimated fare Rs {leg['price']:,})")
                    else:
                        st.write(f"{leg['airline']} {leg['flight_id']}: {leg['from']} to {leg['to']}, departs "
                                 f"{leg['departure_clock']}, arrives {leg['arrival_clock']} (Rs {leg['price']:,})")
        hotel = plan.hotel
        st.subheader("Hotel")
        price_note = " (estimated)" if hotel.get("estimated_price") else ""
        amenities = f" Amenities: {', '.join(hotel['amenities'])}." if hotel["amenities"] else ""
        st.write(f"**{hotel['name']}**, {hotel['stars']}-star, Rs {hotel['price_per_night']:,} per night{price_note}.{amenities}")
        st.subheader("Day-wise plan")
        for day in plan.itinerary:
            weather = day["weather"]
            heading = f"Day {day['day']} ({day['date']})"
            if weather:
                heading += f": {weather['condition']}, {weather['temp_min_c']} to {weather['temp_max_c']} C"
            with st.expander(heading, expanded=plan.num_days <= 10):
                for place in day["places"]:
                    st.write(f"- {place_label(place)}")
                if day["notes"]:
                    st.caption(day["notes"])
        st.download_button("Download itinerary (Markdown)", render_markdown(plan), "itinerary.md")
    with tabs[1]:
        breakdown = pd.DataFrame({"Rs": [budget["flights"], budget["hotel"], budget["food_and_local_travel"]]},
                                 index=["Flights", "Hotel", "Food and local travel"])
        st.bar_chart(breakdown)
        st.caption(f"{budget['travelers']} traveller(s), {budget['rooms']} room(s), {budget['nights']} night(s), "
                   f"Rs {budget['daily_expense_per_person']:,} per person per day for food and local travel "
                   f"({budget['style']} style).")
    with tabs[2]:
        for key, text in plan.reasoning.items():
            st.markdown(f"**{key.title()}:** {text}")
        st.caption("Data sources: " + ", ".join(f"{k} = {v}" for k, v in plan.data_sources.items()))
    with tabs[3]:
        st.json(plan.model_dump(mode="json"), expanded=False)
        st.download_button("Download plan (JSON)", plan.to_json(), "trip_plan.json")
    with tabs[4]:
        if recovered:
            st.info("The model sent its final answer through an invalid tool call. The answer was recovered and "
                    "validated, but the intermediate tool steps are not available for this run.")
        elif trace:
            st.caption(f"Model: {model or 'custom'}. The agent needed {attempts} attempt(s) to produce a valid plan.")
            for number, step in enumerate(trace, 1):
                with st.expander(f"{number}. {step['tool']}"):
                    st.json(step["args"])
                    st.code(step["result_preview"], language="json")
        else:
            st.info("Rule-based planner: no LLM steps. It calls the same tools in a fixed order.")


def run_planner():
    if mode == MODE_AGENT:
        conversation = st.session_state["conversation"]
        result = run_agent(conversation[-1]["content"], history=conversation[:-1], use_extended=use_extended,
                           api_key=api_key or None, model_name=model_name)
        if result.plan is None:
            conversation.append({"role": "assistant", "content": result.message})
            return {"message": result.message}
        return {"plan": result.plan, "trace": result.trace, "attempts": result.attempts, "model": result.model,
                "recovered": result.recovered}
    params = st.session_state["rule_params"]
    return {"plan": plan_without_llm(use_extended=use_extended, **params), "trace": None, "attempts": 1}


input_box, chat_box, reply_box, result_box = st.container(), st.container(), st.container(), st.container()
submitted = replied = False

with input_box:
    if mode == MODE_AGENT:
        st.text_area("Describe your trip", value=EXAMPLE_QUERY, key="query", height=100)
        submitted = st.button("Plan my trip", type="primary")
        if submitted:
            st.session_state["conversation"] = [{"role": "user", "content": st.session_state["query"]}]
    else:
        with st.form("rules"):
            c1, c2, c3 = st.columns(3)
            origin = c1.text_input("From (any city)", value="Bangalore")
            destination = c2.text_input("To (any city)", value="Goa")
            travelers = c3.number_input("Travellers", 1, 10, 1)
            c4, c5, c6 = st.columns(3)
            start = c4.date_input("Start date", default_start)
            end = c5.date_input("End date", default_start + timedelta(days=2))
            style = c6.selectbox("Style", ["budget", "standard", "premium"], index=1)
            interests = st.multiselect("Interests (place types)", available_place_types())
            c7, c8 = st.columns(2)
            max_budget = c7.number_input("Budget cap in Rs (0 = none)", 0, 500000, 0, step=1000)
            round_trip = c8.checkbox("Include return flight", value=True)
            submitted = st.form_submit_button("Plan my trip", type="primary")
        st.session_state["rule_params"] = dict(
            origin=origin, destination=destination, start_date=start, end_date=end, travelers=int(travelers),
            style=style, interests=interests, max_budget=int(max_budget) or None, round_trip=round_trip)

previous = st.session_state.get("result")
awaiting_answer = mode == MODE_AGENT and bool(previous) and "message" in previous and not submitted
if awaiting_answer:
    with reply_box:
        with st.form("reply", clear_on_submit=True):
            answer = st.text_input("Your answer to the assistant", placeholder="Type your answer here")
            replied = st.form_submit_button("Send answer")
    if replied and answer.strip():
        st.session_state["conversation"].append({"role": "user", "content": answer.strip()})
    else:
        replied = False

if submitted or replied:
    with st.spinner("Planning your trip..."):
        try:
            st.session_state["result"] = run_planner()
        except (PlanValidationError, MissingApiKeyError, AgentError, ValueError) as error:
            st.session_state["result"] = {"error": str(error)}
        except RuntimeError as error:  # a web service (geocoding, OpenStreetMap) could not be reached
            st.session_state["result"] = {"error": f"Could not reach an online data service: {error}. Check your internet connection and try again."}
        except Exception as error:  # network or LLM provider failures
            st.session_state["result"] = {"error": f"Unexpected problem: {error}"}
    st.rerun()

if mode == MODE_AGENT:
    with chat_box:
        for message in st.session_state.get("conversation", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])

outcome = st.session_state.get("result")
with result_box:
    if outcome and "error" in outcome:
        st.error(outcome["error"])
    elif outcome and "plan" in outcome:
        show_plan(outcome["plan"], outcome["trace"], outcome["attempts"], outcome.get("model"),
                  outcome.get("recovered", False))
