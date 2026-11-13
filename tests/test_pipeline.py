from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from triage_agent.graph import build_graph
from triage_agent.main import format_outcome
from triage_agent.models import IssueType, NextAction
from triage_agent.pipeline import triage_ticket
from triage_agent.policy import make_apply_policy_node


class _BrokenGraph:
    def invoke(self, *args, **kwargs):
        raise RuntimeError("insufficient_quota")


def test_failed_run_falls_back_to_a_human(tickets, now):
    outcome = triage_ticket(_BrokenGraph(), tickets["T-1001"], now, recursion_limit=5)

    assert outcome.action is NextAction.ESCALATE_TO_HUMAN
    assert outcome.result is None
    assert "insufficient_quota" in outcome.error
    assert "agent failed" in format_outcome(outcome)


def test_successful_run_reports_both_decisions_and_tool_calls(
    store, kb, tools, tickets, now, make_result
):
    billing_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_billing_transactions", "args": {"customer_id": "C-001"}, "id": "c1"}
        ],
    )
    graph = build_graph(
        investigator=GenericFakeChatModel(messages=iter([billing_call, "Evidence summary"])),
        decider=RunnableLambda(lambda _: make_result(issue_types=[IssueType.BILLING])),
        tools=list(tools.values()),
        apply_policy=make_apply_policy_node(store, kb.article_ids),
    )

    outcome = triage_ticket(graph, tickets["T-1001"], now, recursion_limit=20)

    assert outcome.tool_calls == ["get_billing_transactions(customer_id='C-001')"]
    assert outcome.llm_result.action is NextAction.AUTO_RESPOND  # LLM ตัดสินใจผิด
    assert outcome.action is NextAction.ROUTE_TO_SPECIALIST  # policy แก้ให้
    assert outcome.policy_overrides == ["billing_never_auto_responds"]

    text = format_outcome(outcome)
    assert "route_to_specialist -> billing" in text
    assert "billing_never_auto_responds" in text