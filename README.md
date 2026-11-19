# Agentic AI Travel Planning Assistant (LangChain)

A LangChain tool-calling agent that turns a free-text trip request into a validated, day-wise itinerary:
flight, hotel, places, weather per day, budget breakdown and the reasoning behind each choice. It has a
Streamlit app and a CLI, and it also works without an LLM through a rule-based planner that calls the same tools.

## How it works

```
User request
   |
   v
LangChain agent (create_agent + Groq llama-3.3-70b)  <->  5 tools
   |                                                      search_flights    flights.json (direct + connecting)
   |                                                      recommend_hotels   hotels.json
   |                                                      discover_places    places.json
   |                                                      get_weather        Open-Meteo API (free, no key)
   |                                                      estimate_budget    flights + hotel + daily expenses
   v
JSON draft (ids only)  ->  finalize_plan (guardrail)  ->  TripPlan  ->  Markdown + JSON
```

The agent never writes prices or names. It returns dataset ids, and `finalize_plan` re-reads flights, hotels
and places from the JSON files, checks that the flight legs form one connected route, that the hotel is in
the destination and that the trip is 3 to 7 days, then recomputes weather and budget itself. If validation
fails, the error is sent back to the agent for up to two repairs. If the request is missing information
(origin, destination, dates), the agent asks a clarifying question instead.

## Project structure

| Path | Purpose |
| :--- | :--- |
| `src/travel_planner/flights.py` | Route graph search, direct then connecting, cheapest or fastest |
| `src/travel_planner/hotels.py` | Filter by city, stars, price, amenities; rank by value, price or stars |
| `src/travel_planner/places.py` | Attractions by city, type and rating |
| `src/travel_planner/weather.py` | Open-Meteo forecast, archive fallback, retries, caching |
| `src/travel_planner/budget.py` | Flights + hotel nights + per-day local expenses |
| `src/travel_planner/tools.py` | LangChain `@tool` wrappers (JSON in and out, errors returned to the agent) |
| `src/travel_planner/agent.py` | System prompt, agent creation, JSON parsing, repair loop, step trace |
| `src/travel_planner/planner.py` | `finalize_plan` guardrail, rule-based planner, Markdown renderer |
| `src/travel_planner/schemas.py` | Pydantic `TripDraft` and `TripPlan` |
| `app.py`, `cli.py` | Streamlit UI and command line |
| `tests/` | 30 tests (tools, planner, agent with a scripted chat model) |
| `scripts/generate_extended_flights.py` | Optional synthetic flights for missing routes |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your GROQ_API_KEY in .env
streamlit run app.py
```

Command line:

```bash
python cli.py "Plan a 3 day trip from Bangalore to Goa, Feb 12 to Feb 14, for 2 people, love beaches"
python cli.py --no-llm --origin Bangalore --destination Goa --start 2027-02-12 --end 2027-02-14 --interests beach park
python -m pytest
```

Tested with Python 3.12, langchain 1.4.9, langchain-groq 1.1.3, streamlit 1.66. Get a free Groq key at console.groq.com.

## Data facts you should know (they shape the design)

1. **Only 25 of the 56 possible city-to-city routes exist in `flights.json`** (30 flights, 8 cities). Delhi to Goa,
   the brief's own example, has no direct flight; the shortest path is 3 stops. The agent therefore searches
   connecting routes (up to 3 stops, fewest stops first). The sidebar option "Fill missing flight routes with
   synthetic flights" adds 31 generated direct flights (`flights_extended.json`, each marked `synthetic: true`).
2. **Flights are one-way and dated in 2025.** Only route, price and time of day are used, not the calendar date.
   Layover timing on connecting routes is therefore indicative only. A round trip searches the reverse route
   too, so the brief's one-way example total is not what this app computes.
3. **Places have no coordinates and only 5 per city.** Distance-based routing is impossible. Places are ranked
   by rating, spread over days (lighter arrival and departure days), and days beyond the available places are
   shown as free days. The `type` field sometimes contradicts the name (a "Famous Fort" typed `lake`); the
   `type` is treated as correct.
4. **Hotels have stars, not guest ratings.** "Best value" means stars per Rs 1000 of nightly price.
5. **City coordinates for weather** are not in the datasets, so `data/city_coordinates.json` adds them.

## Assumptions

| Item | Rule |
| :--- | :--- |
| Trip length | 3 to 7 days; dates inclusive; nights = days - 1 |
| Rooms | 2 travellers per room |
| Food and local travel | Rs 800 / 1200 / 2000 per person per day (budget / standard / premium) |
| Hotel choice | budget: cheapest; standard: best value with 3+ stars; premium: most stars |
| Weather window | Live forecast for the next 15 days; otherwise the same dates last year from the archive, labelled as such |
| Rainy day | 5 mm or more precipitation; indoor places (museum, temple, market, monument) are preferred |

## Mapping to the project brief

| Brief requirement | Where |
| :--- | :--- |
| Flight, hotel, places tools from JSON | `flights.py`, `hotels.py`, `places.py` |
| Open-Meteo weather | `weather.py` |
| Budget estimation tool | `budget.py` |
| ReAct / tool-calling agent | `agent.py` (`create_agent`, tool calling) |
| Day-wise plan, accommodation, weather, budget | `TripPlan`, `render_markdown` |
| Filtering, ranking, optimisation | cheapest/fastest, value/price/stars, rating, budget cap |
| "Why we selected this" | `reasoning` field, shown in the "Why this plan" tab |
| JSON plus readable output | JSON tab and download, Markdown itinerary |
| Streamlit interface | `app.py` |
| Error handling | tool errors returned to the agent, weather failures degrade gracefully, validation guardrail |

## Limitations

* The LLM step was tested with a scripted fake chat model, not a live Groq call, so prompt quality with the real
  model should be checked on your machine. The tools, guardrail, planner and UI are fully tested.
* Live weather was tested against mocked Open-Meteo responses; the request format follows the API docs.
* The datasets are random samples (generic place names, repeated hotel names), so itineraries read generically.
* Prices are dataset prices, not real fares.

`docs/sample_itinerary_offline.md` is a sample produced without network access, so its weather rows say
"weather unavailable". On a machine with internet, each day shows condition, temperature and rainfall.
