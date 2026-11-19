"""Command-line interface: plan a trip with the LLM agent or the rule-based planner."""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

from travel_planner.agent import AgentError, MissingApiKeyError, run_agent
from travel_planner.planner import PlanValidationError, plan_without_llm, render_markdown


def build_parser():
    parser = argparse.ArgumentParser(description="Agentic AI travel planner")
    parser.add_argument("query", nargs="?", help="Free-text request for the LLM agent")
    parser.add_argument("--no-llm", action="store_true", help="Use the rule-based planner (no API key)")
    parser.add_argument("--origin")
    parser.add_argument("--destination")
    parser.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--travelers", type=int, default=1)
    parser.add_argument("--style", choices=["budget", "standard", "premium"], default="standard")
    parser.add_argument("--interests", nargs="*", default=[], help="Place types, e.g. beach museum")
    parser.add_argument("--max-budget", type=int)
    parser.add_argument("--one-way", action="store_true")
    parser.add_argument("--extended", action="store_true", help="Add synthetic flights for missing routes")
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON")
    return parser


def main():
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.no_llm:
            if not all([args.origin, args.destination, args.start, args.end]):
                parser.error("--no-llm needs --origin, --destination, --start and --end")
            plan = plan_without_llm(args.origin, args.destination, args.start, args.end, args.travelers,
                                    args.style, args.interests, args.max_budget, not args.one_way, args.extended)
        else:
            if not args.query:
                parser.error("give a query, or use --no-llm with the trip options")
            result = run_agent(args.query, use_extended=args.extended)
            if result.plan is None:
                print(result.message)
                return 0
            plan = result.plan
    except (PlanValidationError, MissingApiKeyError, AgentError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(plan.to_json() if args.json else render_markdown(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
