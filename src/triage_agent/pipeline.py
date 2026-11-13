import logging
from datetime import datetime

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.state import CompiledStateGraph

from triage_agent.config import Settings
from triage_agent.data import KnowledgeBase, MockDataStore
from triage_agent.graph import build_graph
from triage_agent.llm import create_decider, create_investigator
from triage_agent.models import NextAction, Ticket, TicketOutcome
from triage_agent.policy import make_apply_policy_node
from triage_agent.tools import build_tools

logger = logging.getLogger(__name__)


def build_triage_graph(settings: Settings, reference_time: datetime) -> CompiledStateGraph:
    store = MockDataStore(settings.data_dir)
    kb = KnowledgeBase.from_file(settings.data_dir / "knowledge_base.json")
    tools = build_tools(store, kb, today=reference_time.date())
    return build_graph(
        investigator=create_investigator(settings, tools),
        decider=create_decider(settings),
        tools=tools,
        apply_policy=make_apply_policy_node(store, kb.article_ids),
    )


def summarize_tool_calls(messages: list[AnyMessage]) -> list[str]:
    """ดึงรายการ tool ที่ agent เรียกตามลำดับ: ["get_customer_profile(customer_id='C-001')", ...]"""
    return [
        f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})"
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def triage_ticket(
    graph: CompiledStateGraph, ticket: Ticket, reference_time: datetime, recursion_limit: int
) -> TicketOutcome:
    try:
        state = graph.invoke(
            {"ticket": ticket, "reference_time": reference_time},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as exc:
        logger.exception("Triage failed for %s", ticket.ticket_id)
        return TicketOutcome(
            ticket_id=ticket.ticket_id,
            action=NextAction.ESCALATE_TO_HUMAN,
            error=f"{type(exc).__name__}: {exc}",
        )

    final = state["final"]
    return TicketOutcome(
        ticket_id=ticket.ticket_id,
        action=final.action,
        result=final,
        llm_result=state["triage"],
        policy_overrides=state.get("policy_overrides", []),
        tool_calls=summarize_tool_calls(state.get("messages", [])),
    )
