"""Create data/flights_extended.json: synthetic flights for city pairs missing from flights.json.

The provided flights.json covers only 25 of the 56 directed routes between the 8 cities.
This optional file fills the gaps so any city pair has a direct flight. Every generated record
is marked "synthetic": true and is only used when the user switches the option on.
"""
import json
import random
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AIRLINES = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Go First"]


def main():
    rng = random.Random(42)
    flights = json.loads((DATA_DIR / "flights.json").read_text(encoding="utf-8"))
    cities = sorted({f["from"] for f in flights} | {f["to"] for f in flights})
    existing = {(f["from"], f["to"]) for f in flights}
    extra = []
    for origin, destination in permutations(cities, 2):
        if (origin, destination) in existing:
            continue
        departure = datetime(2025, rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23), rng.choice([0, 15, 30, 45]))
        arrival = departure + timedelta(hours=rng.randint(1, 4))
        extra.append({
            "flight_id": f"FX{len(extra) + 1:04d}",
            "airline": rng.choice(AIRLINES),
            "from": origin,
            "to": destination,
            "departure_time": departure.isoformat(timespec="seconds"),
            "arrival_time": arrival.isoformat(timespec="seconds"),
            "price": rng.randint(2500, 9000),
            "synthetic": True,
        })
    (DATA_DIR / "flights_extended.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    print(f"Wrote {len(extra)} synthetic flights")


if __name__ == "__main__":
    main()
