import sys

from dotenv import load_dotenv

from triage_agent.config import Settings
from triage_agent.data import KnowledgeBase, MockDataStore, load_tickets
from triage_agent.graph import build_graph
from triage_agent.llm import create_investigator
from triage_agent.tools import build_tools


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    dataset = load_tickets(settings.data_dir / "tickets.json")

    tools = build_tools(
        MockDataStore(settings.data_dir),
        KnowledgeBase.from_file(settings.data_dir / "knowledge_base.json"),
        today=dataset.reference_time.date(),
    )
    graph = build_graph(create_investigator(settings, tools), tools)

    wanted = set(sys.argv[1:])
    for ticket in dataset.tickets:
        if wanted and ticket.ticket_id not in wanted:
            continue
        print(f"\n{'#' * 30} {ticket.ticket_id} {'#' * 30}")
        state = graph.invoke(
            {"ticket": ticket, "reference_time": dataset.reference_time},
            config={"recursion_limit": settings.recursion_limit},
        )
        for message in state["messages"][1:]:
            message.pretty_print()


if __name__ == "__main__":
    main()
