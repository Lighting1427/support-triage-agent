import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import TypeAdapter

from triage_agent.config import Settings
from triage_agent.data import load_tickets
from triage_agent.evaluation import GoldenCase, evaluate, llm_action_was_correct
from triage_agent.pipeline import build_triage_graph, triage_ticket

GOLDEN_SET = Path(__file__).resolve().parent.parent / "evals" / "golden_set.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Repeat each ticket N times.")
    args = parser.parse_args()

    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    settings = Settings.from_env()
    dataset = load_tickets(settings.data_dir / "tickets.json")
    tickets = {t.ticket_id: t for t in dataset.tickets}
    cases = TypeAdapter(list[GoldenCase]).validate_python(
        json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    )
    graph = build_triage_graph(settings, dataset.reference_time)

    passed = total = llm_correct = final_correct = overrides = 0
    for run in range(1, args.runs + 1):
        for case in cases:
            outcome = triage_ticket(
                graph, tickets[case.ticket_id], dataset.reference_time, settings.recursion_limit
            )
            results = evaluate(case, outcome)
            print(f"\n{case.ticket_id} (run {run}/{args.runs})")
            for r in results:
                print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:<15} {r.detail}")
            print(f"  policy overrides: {', '.join(outcome.policy_overrides) or '-'}")

            passed += sum(r.passed for r in results)
            total += len(results)
            llm_correct += llm_action_was_correct(case, outcome)
            final_correct += outcome.action is case.expected_action
            overrides += len(outcome.policy_overrides)

    n = len(cases) * args.runs
    print("\n" + "=" * 50)
    print(f"Checks passed          : {passed}/{total} ({passed / total:.0%})")
    print(f"Action correct (LLM)   : {llm_correct}/{n}   <- before policy guardrails")
    print(f"Action correct (final) : {final_correct}/{n}   <- after policy guardrails")
    print(f"Policy overrides       : {overrides} in {n} tickets")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
