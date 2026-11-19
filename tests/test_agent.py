"""Agent tests with a scripted chat model (no network, no API key)."""
import json
from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from conftest import make_fake_http_get
from travel_planner.agent import AgentError, MissingApiKeyError, build_agent, extract_json, run_agent

TODAY = date(2026, 11, 5)


class ScriptedChatModel(GenericFakeChatModel):
    """Replays prepared AIMessages and ignores bind_tools, so create_agent can wrap it."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


GOOD_JSON = json.dumps({
    "origin": "Bangalore", "destination": "Goa", "start_date": "2026-11-10", "end_date": "2026-11-12",
    "travelers": 1, "style": "standard", "round_trip": True,
    "outbound_flight_ids": ["FL0021"], "return_flight_ids": ["FL0023"], "hotel_id": "HOT0014",
    "days": [{"day": 1, "place_ids": ["PLC0012"], "notes": "Easy start"}, {"day": 2, "place_ids": ["PLC0014"]}],
    "reasoning": {"flight": "Cheapest direct option."}})


def scripted(*replies):
    return ScriptedChatModel(messages=iter(replies))


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Result {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_agent_runs_tools_and_returns_validated_plan():
    llm = scripted(call("search_flights", {"origin": "Bangalore", "destination": "Goa"}, "c1"),
                   call("recommend_hotels", {"city": "Goa"}, "c2"),
                   AIMessage(content=GOOD_JSON))
    result = run_agent("3 days Bangalore to Goa", llm=llm, today=TODAY, http_get=make_fake_http_get())
    assert result.plan.budget["flights"] == 5356 + 4537
    assert [step["tool"] for step in result.trace] == ["search_flights", "recommend_hotels"]
    assert result.plan.reasoning["flight"] == "Cheapest direct option."
    assert "FL0021" in result.trace[0]["result_preview"]


def test_agent_asks_clarifying_question_when_no_json():
    llm = scripted(AIMessage(content="Which city are you travelling from?"))
    result = run_agent("plan a trip to Goa", llm=llm, today=TODAY)
    assert result.plan is None and "travelling from" in result.message


def test_agent_repairs_invalid_json_once():
    bad = GOOD_JSON.replace("HOT0014", "HOT9999")
    llm = scripted(AIMessage(content=bad), AIMessage(content=GOOD_JSON))
    result = run_agent("trip", llm=llm, today=TODAY, http_get=make_fake_http_get())
    assert result.attempts == 2 and result.plan.hotel["hotel_id"] == "HOT0014"


def test_agent_gives_up_after_repeated_invalid_output():
    bad = GOOD_JSON.replace("HOT0014", "HOT9999")
    llm = scripted(AIMessage(content=bad), AIMessage(content=bad), AIMessage(content=bad))
    with pytest.raises(AgentError):
        run_agent("trip", llm=llm, today=TODAY, http_get=make_fake_http_get())


def test_tool_errors_are_returned_to_the_agent_not_raised(mock_world):
    llm = scripted(call("search_flights", {"origin": "Atlantis", "destination": "Goa"}, "c1"),
                   AIMessage(content=GOOD_JSON))
    result = run_agent("trip", llm=llm, today=TODAY, http_get=make_fake_http_get())
    assert "Could not find a place" in result.trace[0]["result_preview"]


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        build_agent()


def test_falls_back_to_next_model_when_first_is_unavailable(monkeypatch):
    from travel_planner import agent as agent_module
    tried = []

    class FlakyAgent:
        def __init__(self, model):
            self.model = model

        def invoke(self, payload, config=None):
            if self.model == "openai/gpt-oss-120b":
                raise RuntimeError("Error code: 404 - model_not_found")
            return {"messages": [AIMessage(content=GOOD_JSON)]}

    def fake_build(use_extended, llm, api_key, model_name, today):
        tried.append(model_name)
        return FlakyAgent(model_name)

    monkeypatch.setattr(agent_module, "build_agent", fake_build)
    result = run_agent("trip", today=TODAY, http_get=make_fake_http_get())
    assert tried == ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
    assert result.model == "qwen/qwen3.6-27b" and result.plan is not None


def test_retired_model_names_are_never_tried():
    from travel_planner.agent import model_candidates
    assert "llama-3.3-70b-versatile" not in model_candidates("llama-3.3-70b-versatile")
    assert model_candidates("llama-3.3-70b-versatile")[0] == "openai/gpt-oss-120b"
    assert model_candidates("my/custom-model")[0] == "my/custom-model"


def test_system_prompt_lists_supported_cities():
    from travel_planner.agent import build_system_prompt
    prompt = build_system_prompt(TODAY)
    assert "Goa" in prompt and "Jaipur" in prompt and "2026-11-05" in prompt
    assert "Any city in the world" in prompt


def test_agent_continues_conversation_after_clarifying_question():
    history = [{"role": "user", "content": "Plan a trip to Japan"},
               {"role": "assistant", "content": "Japan is not supported. Which of these cities?"}]
    llm = scripted(AIMessage(content=GOOD_JSON))
    result = run_agent("Goa from Bangalore, Nov 10 to Nov 12", llm=llm, today=TODAY,
                       http_get=make_fake_http_get(), history=history)
    assert result.plan.destination == "Goa"


GENERATION = json.dumps({"name": "json", "arguments": json.loads(GOOD_JSON)}, indent=1)


def _provider_error_text(generation):
    """Text of a Groq tool_use_failed error, as str(error) shows it (a Python dict repr)."""
    body = {"error": {"message": "Tool call validation failed: attempted to call tool 'json' which was not "
                                 "in request.tools", "type": "invalid_request_error", "code": "tool_use_failed",
                      "failed_generation": generation}}
    return "Error code: 400 - " + repr(body)


class FailingAgent:
    def __init__(self, error):
        self.error = error

    def invoke(self, payload, config=None):
        raise self.error


def test_recovers_json_from_a_tool_use_failed_error_text(monkeypatch):
    from travel_planner import agent as agent_module
    text = _provider_error_text(GENERATION.replace("Easy start", "traveler's note \u202f"))
    monkeypatch.setattr(agent_module, "build_agent", lambda *a, **k: FailingAgent(RuntimeError(text)))
    result = run_agent("trip", today=TODAY, http_get=make_fake_http_get())
    assert result.recovered is True and result.plan.hotel["hotel_id"] == "HOT0014"
    assert result.trace == []


def test_recovers_json_from_the_error_body(monkeypatch):
    from travel_planner import agent as agent_module

    class ApiError(Exception):
        body = {"error": {"code": "tool_use_failed", "failed_generation": GENERATION}}
    monkeypatch.setattr(agent_module, "build_agent", lambda *a, **k: FailingAgent(ApiError("400 tool_use_failed")))
    assert run_agent("trip", today=TODAY, http_get=make_fake_http_get()).plan.destination == "Goa"


def test_unreadable_tool_use_failed_error_falls_back_to_the_next_model(monkeypatch):
    from travel_planner import agent as agent_module
    tried = []

    def fake_build(use_extended, llm, api_key, model_name, today):
        tried.append(model_name)
        if len(tried) == 1:
            return FailingAgent(RuntimeError("Error code: 400 - tool_use_failed garbage"))
        class OkAgent:
            def invoke(self, payload, config=None):
                return {"messages": [AIMessage(content=GOOD_JSON)]}
        return OkAgent()
    monkeypatch.setattr(agent_module, "build_agent", fake_build)
    result = run_agent("trip", today=TODAY, http_get=make_fake_http_get())
    assert len(tried) == 2 and result.plan is not None and result.recovered is False


def test_prompt_forbids_json_tool_calls():
    from travel_planner.agent import build_system_prompt
    assert "Never call a tool named json" in build_system_prompt(TODAY)
