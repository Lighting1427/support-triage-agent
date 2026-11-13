import argparse
import json
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv

from triage_agent.config import Settings
from triage_agent.data import load_tickets
from triage_agent.models import TicketOutcome
from triage_agent.pipeline import build_triage_graph, triage_ticket


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage the sample support tickets.")
    parser.add_argument("--ticket", action="append", help="Ticket ID to run (repeatable).")
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log tool calls.")
    return parser.parse_args(argv)


def format_outcome(outcome: TicketOutcome) -> str:
    """แปลงผลของ ticket หนึ่งใบเป็นข้อความอ่านง่ายสำหรับ console"""
    header = f"{'=' * 20} {outcome.ticket_id} {'=' * 20}"
    if outcome.result is None:
        return f"{header}\nAction      : {outcome.action} (agent failed: {outcome.error})\n"

    r = outcome.result
    info = r.extracted
    lines = [
        header,
        f"Urgency     : {r.urgency.upper()} - {r.urgency_reasoning}",
        f"Action      : {r.action} -> {', '.join(r.route_to) or '-'}",
        f"Issues      : {', '.join(info.issue_types)} (area: {info.product_area})",
        f"Sentiment   : {info.sentiment.label} ({info.sentiment.trend}), language: {info.language}",
        f"Risks       : {', '.join(info.business_risks) or '-'}",
        f"Deadline    : {info.customer_deadline or '-'}",
        f"KB articles : {', '.join(r.kb_article_ids) or '-'}",
        f"Confidence  : {r.confidence:.2f}",
        f"Tools used  : {' -> '.join(outcome.tool_calls) or '-'}",
        f"Policy      : {', '.join(outcome.policy_overrides) or 'no overrides'}",
        "Key facts   :",
        *(f"  - {fact}" for fact in info.key_facts),
        "Internal note:",
        textwrap.indent(r.internal_note, "  "),
        "Draft reply :",
        textwrap.indent(r.draft_reply, "  "),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv() 
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    settings = Settings.from_env()
    dataset = load_tickets(settings.data_dir / "tickets.json")
    graph = build_triage_graph(settings, dataset.reference_time)

    wanted = set(args.ticket or [])
    outcomes = [
        triage_ticket(graph, ticket, dataset.reference_time, settings.recursion_limit)
        for ticket in dataset.tickets
        if not wanted or ticket.ticket_id in wanted
    ]

    if args.json:
        payload = [o.model_dump(mode="json") for o in outcomes]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("\n".join(format_outcome(o) for o in outcomes))
    return 0