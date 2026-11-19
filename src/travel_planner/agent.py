"""The LangChain tool-calling agent that plans a trip, plus parsing, validation and retry logic."""
import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from langchain.agents import create_agent

from travel_planner import config
from travel_planner.data_loader import supported_cities
from travel_planner.planner import PlanValidationError, finalize_plan
from travel_planner.schemas import TripDraft, TripPlan
from travel_planner.tools import make_tools

SYSTEM_PROMPT = """You are a travel planning agent for domestic trips in India. Today's date is {today}.
You may only use the five tools provided. Never invent flights, hotels, places or prices.

Workflow:
1. Read the request (the traveller decides the trip length: any length from 1 to 30 days is fine, never ask
   them to shorten it): origin, destination, start and end date (inclusive), number of travellers, budget,
   interests and style (budget, standard or premium). If the year is missing use the next future date.
   Any city in the world can be used. These cities have real records in the built-in dataset: {cities}. For
   other places the tools use live OpenStreetMap data and estimated flight and hotel prices; the final
   answer must still use ids exactly as the tools return them.
   Do not call tools and reply with one short plain-text question (no JSON) when: origin, destination or dates
   are missing; the trip is longer than 30 days; the stated trip length and the date range disagree (ask which to
   use); or a tool says it could not find a place (ask the user for a nearby or differently spelled city). If a
   country is given instead of a city, ask which city. The user's answer will arrive as the next message, so
   use it together with the earlier messages.
2. search_flights for the outbound route, then again with origin and destination swapped for the return.
   Prefer the cheapest option unless the user wants the fastest.
3. recommend_hotels in the destination (sort_by "value" by default, "price" for budget style, "stars" for premium).
4. discover_places in the destination, using the user's interests as types when given.
5. get_weather for the trip dates. Put indoor places on rainy days and outdoor places on dry days.
6. estimate_budget with the chosen flights and hotel (nights = trip days minus 1) and destination set to the destination city. If the user gave a budget
   and the total is higher, choose cheaper options and check again.
7. Reply with ONLY one JSON object, no other text, in exactly this shape:
{"origin": "Delhi", "destination": "Goa", "start_date": "2027-02-12", "end_date": "2027-02-14",
 "travelers": 1, "style": "standard", "round_trip": true, "budget_limit": 30000,
 "outbound_flight_ids": ["FL0001"], "return_flight_ids": ["FL0002"], "hotel_id": "HOT0011",
 "days": [{"day": 1, "place_ids": ["PLC0011"], "notes": "short tip"}],
 "reasoning": {"flight": "why this flight", "hotel": "why this hotel", "itinerary": "why these places",
               "budget": "how the budget fits"}}

The JSON is plain text in your reply. Never call a tool named json, or any tool other than the five above,
to deliver it.

Rules for the JSON: budget_limit is the user's budget in rupees, or null if none was given; reasoning text uses
names, never ids or arithmetic of your own; copy every flight_id, hotel_id and place_id exactly from tool results; list flight ids in
travel order (all legs of a connecting route); use at most 2 places per day and never repeat a place; list only
days that have places (days left out become free days), so long trips stay short; each reasoning value is one or two sentences with real numbers from the tools. Day notes are optional short tips
about the places only (under 12 words): never mention weather, flights or free days in a note; leave it empty instead.
"""


class MissingApiKeyError(RuntimeError):
    """Raised when no Groq API key is available."""


class AgentError(RuntimeError):
    """Raised when the agent cannot produce a valid plan."""


@dataclass
class AgentResult:
    plan: Optional[TripPlan]
    draft: Optional[TripDraft]
    trace: list = field(default_factory=list)
    message: str = ""
    attempts: int = 1
    model: Optional[str] = None
    recovered: bool = False


def build_system_prompt(today=None):
    prompt = SYSTEM_PROMPT.replace("{today}", (today or date.today()).isoformat())
    return prompt.replace("{cities}", ", ".join(supported_cities()))


def build_agent(use_extended=False, llm=None, api_key=None, model_name=None, today=None):
    """Create the tool-calling agent. Pass llm to inject a different chat model (used in tests)."""
    if llm is None:
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise MissingApiKeyError("GROQ_API_KEY is not set. Add it to .env or enter it in the app sidebar.")
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model_name or config.DEFAULT_LLM_MODEL, temperature=0, api_key=api_key)
    return create_agent(llm, make_tools(use_extended), system_prompt=build_system_prompt(today))


def _message_text(message):
    content = message.content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content or ""


def extract_json(text):
    """Pull the first JSON object out of model text, tolerating code fences and chatter."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    if not candidate.strip():
        raise ValueError("No JSON object found in the model reply.")
    return json.loads(candidate)


def extract_trace(messages):
    """Pair each tool call with its result so the UI can show how the agent reasoned."""
    results = {getattr(m, "tool_call_id", None): m for m in messages if m.type == "tool"}
    steps = []
    for message in messages:
        if message.type != "ai":
            continue
        for call in getattr(message, "tool_calls", None) or []:
            result = results.get(call.get("id"))
            preview = _message_text(result)[:700] if result is not None else ""
            steps.append({"tool": call["name"], "args": call["args"], "result_preview": preview})
    return steps


def model_candidates(preferred=None):
    """Models to try in order: the preferred one (unless retired), then the fallbacks."""
    ordered = [preferred or config.DEFAULT_LLM_MODEL, *config.FALLBACK_LLM_MODELS]
    seen, result = set(), []
    for name in ordered:
        if name and name not in config.RETIRED_LLM_MODELS and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def is_model_unavailable(error):
    """True when the provider says the model does not exist, is retired or is not accessible."""
    text = str(error).lower()
    return any(key in text for key in ("model_not_found", "model_decommissioned", "decommissioned",
                                        "does not exist or you do not have access"))


def is_tool_use_failed(error):
    """True when the provider rejected the model's tool call (gpt-oss sometimes sends its JSON as a 'json' tool)."""
    text = str(error).lower()
    return "tool_use_failed" in text or "not in request.tools" in text


def _find_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def recover_failed_generation(error):
    """Get the JSON the model tried to send from a tool_use_failed error, or None if it cannot be read.

    The provider echoes the rejected output as "failed_generation", either in the error body or inside
    the error text as a Python string literal. A call like {"name": "json", "arguments": {...}} is unwrapped.
    """
    raw = _find_key(getattr(error, "body", None), "failed_generation")
    if raw is None:
        match = re.search(r"""['"]failed_generation['"]: ('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")""", str(error), re.DOTALL)
        if match:
            try:
                raw = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                raw = None
    if not raw:
        return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(payload, dict) and "arguments" in payload and "name" in payload:
            payload = payload["arguments"]
            payload = json.loads(payload) if isinstance(payload, str) else payload
        return json.dumps(payload) if isinstance(payload, dict) else None
    except ValueError:
        return None


def _run_conversation(agent, messages, use_extended, today, http_get, max_repairs):
    last_error = None
    for attempt in range(1, max_repairs + 2):
        recovered = False
        try:
            result = agent.invoke({"messages": messages}, config={"recursion_limit": 40})
            history = result["messages"]
            final_text = _message_text(history[-1])
            trace = extract_trace(history)
        except Exception as error:
            final_text = recover_failed_generation(error) if is_tool_use_failed(error) else None
            if final_text is None:
                raise
            history, trace, recovered = list(messages), [], True
        if "{" not in final_text:
            return AgentResult(plan=None, draft=None, trace=trace, message=final_text.strip(), attempts=attempt)
        try:
            draft = TripDraft.model_validate(extract_json(final_text))
            plan = finalize_plan(draft, use_extended, today, http_get)
            return AgentResult(plan=plan, draft=draft, trace=trace, attempts=attempt, recovered=recovered)
        except ValueError as error:
            last_error = error
            messages = history + [{"role": "user", "content": (
                f"Your JSON was rejected: {error}. Use only ids returned by the tools, keep the trip "
                "between 1 and 30 days, and reply with the corrected JSON object only.")}]
    raise AgentError(f"The agent could not produce a valid plan after {max_repairs + 1} attempts: {last_error}")


def run_agent(query, use_extended=False, llm=None, api_key=None, model_name=None, today=None,
              http_get=None, max_repairs=2, history=None):
    """Run the agent on a free-text request and return a validated plan (or a clarifying question).

    history: earlier turns as [{"role": "user" | "assistant", "content": ...}]. When the agent asked a
    clarifying question, pass the conversation so far plus the user's answer as query.

    If the provider reports that a model is retired or unavailable, the next candidate model is tried.
    """
    messages = list(history or []) + [{"role": "user", "content": query}]
    candidates = [None] if llm is not None else model_candidates(model_name)
    for index, candidate in enumerate(candidates):
        agent = build_agent(use_extended, llm, api_key, candidate, today)
        try:
            result = _run_conversation(agent, messages, use_extended, today, http_get, max_repairs)
            result.model = candidate
            return result
        except Exception as error:
            if (is_model_unavailable(error) or is_tool_use_failed(error)) and index < len(candidates) - 1:
                continue
            raise
