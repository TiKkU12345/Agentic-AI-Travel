"""Weather lookup via the free Open-Meteo API (no API key needed).

Dates inside the forecast window use the live forecast. Dates outside it use the same calendar
dates one year earlier from Open-Meteo's historical archive and are labelled as such.
"""
from datetime import date, timedelta

import requests

from travel_planner import config
from travel_planner.locations import resolve_location

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light showers",
    81: "Showers", 82: "Heavy showers", 85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}
_CACHE = {}


def clear_cache():
    _CACHE.clear()


def describe_weather_code(code):
    return WMO_CODES.get(int(code), "Unknown") if code is not None else "Unknown"


def _shift_back_one_year(day):
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day - timedelta(days=365)


def _fetch(url, params, http_get):
    last_error = None
    for _ in range(config.HTTP_RETRIES + 1):
        try:
            response = http_get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
    raise RuntimeError(f"could not reach Open-Meteo ({type(last_error).__name__}); showing the plan without weather")


def _parse_daily(payload, requested_dates):
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        raise RuntimeError("Open-Meteo returned no daily data.")
    codes = daily.get("weather_code") or [None] * len(times)
    rain = daily.get("precipitation_sum") or [0] * len(times)
    tmax = daily.get("temperature_2m_max") or [None] * len(times)
    tmin = daily.get("temperature_2m_min") or [None] * len(times)
    days = []
    for index, requested in enumerate(requested_dates):
        if index >= len(times):
            break
        precip = rain[index] if rain[index] is not None else 0.0
        days.append({
            "date": requested.isoformat(),
            "temp_max_c": tmax[index],
            "temp_min_c": tmin[index],
            "precipitation_mm": precip,
            "condition": describe_weather_code(codes[index]),
            "rainy": precip >= config.RAIN_THRESHOLD_MM,
        })
    return days


def _load_segment(location, dates, source, http_get):
    """Fetch one run of consecutive dates from the forecast or the archive (cached)."""
    if source == "live_forecast":
        url, query_start, query_end = config.OPEN_METEO_FORECAST_URL, dates[0], dates[-1]
    elif source == "historical_actual":
        url, query_start, query_end = config.OPEN_METEO_ARCHIVE_URL, dates[0], dates[-1]
    else:
        url, query_start, query_end = (config.OPEN_METEO_ARCHIVE_URL, _shift_back_one_year(dates[0]),
                                       _shift_back_one_year(dates[-1]))
    key = (location.name, query_start, query_end, source)
    if key not in _CACHE:
        params = {
            "latitude": location.latitude, "longitude": location.longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "start_date": query_start.isoformat(), "end_date": query_end.isoformat(), "timezone": "auto",
        }
        _CACHE[key] = _parse_daily(_fetch(url, params, http_get), dates)
    return _CACHE[key]


def get_weather(city, start_date, end_date, today=None, http_get=None):
    """Daily weather for a city between two dates (ISO strings or date objects).

    The live forecast covers about 15 days. Later days of a long trip use the same calendar dates from
    last year, and the result says so, so a 30 day trip starting next week gets a mix of both.
    """
    http_get = http_get or requests.get
    location = resolve_location(city)
    start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    if end < start:
        raise ValueError("end_date must not be before start_date.")
    today = today or date.today()
    requested = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    last_live_day = today + timedelta(days=config.FORECAST_HORIZON_DAYS)

    if end < today:
        segments = [("historical_actual", requested)]
    elif today <= start <= last_live_day:
        live = [d for d in requested if d <= last_live_day]
        later = [d for d in requested if d > last_live_day]
        segments = [("live_forecast", live)] + ([("historical_same_dates_last_year", later)] if later else [])
    else:
        segments = [("historical_same_dates_last_year", requested)]

    days, sources, notes = [], [], []
    for source, dates in segments:
        try:
            days += _load_segment(location, dates, source, http_get)
            sources.append(source)
            if source == "historical_same_dates_last_year":
                notes.append(f"{dates[0].isoformat()} onwards is beyond the live forecast window; showing what the "
                             "weather was on the same dates last year as a typical guide.")
        except RuntimeError as error:
            notes.append(f"Weather unavailable for {dates[0].isoformat()} to {dates[-1].isoformat()}: {error}.")
    return {"city": location.name, "status": "ok" if days else "error",
            "source": "+".join(sources) if sources else "unavailable", "days": days, "notes": notes}
