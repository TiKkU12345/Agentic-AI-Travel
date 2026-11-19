"""Pydantic models: the draft an agent returns, and the final validated trip plan."""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DayDraft(BaseModel):
    day: int
    place_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class TripDraft(BaseModel):
    """What the planning agent must return. All facts are referenced by dataset ids."""
    origin: str
    destination: str
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=10)
    style: Literal["budget", "standard", "premium"] = "standard"
    round_trip: bool = True
    outbound_flight_ids: list[str]
    return_flight_ids: list[str] = Field(default_factory=list)
    hotel_id: str
    budget_limit: Optional[int] = Field(default=None, ge=0)
    days: list[DayDraft] = Field(default_factory=list)
    reasoning: dict[str, str] = Field(default_factory=dict)


class TripPlan(BaseModel):
    """Final plan: every number and name is resolved from the datasets, not from the LLM."""
    title: str
    origin: str
    destination: str
    start_date: date
    end_date: date
    num_days: int
    nights: int
    travelers: int
    style: str
    outbound: dict
    return_flight: Optional[dict] = None
    hotel: dict
    itinerary: list[dict]
    weather_source: str
    budget: dict
    data_sources: dict[str, str] = Field(default_factory=dict)
    reasoning: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def to_json(self):
        return self.model_dump_json(indent=2)
