import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from triage_agent.graph import build_graph
from triage_agent.main import format_outcome, main, render_outcomes
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
    assert outcome.llm_result.action is NextAction.AUTO_RESPOND  # Incorrect LLM decision.
    assert outcome.action is NextAction.ROUTE_TO_SPECIALIST  # Corrected by policy.
    assert outcome.policy_overrides == ["billing_never_auto_responds"]

    text = format_outcome(outcome)
    assert "route_to_specialist -> billing" in text
    assert "billing_never_auto_responds" in text


def test_json_output_is_valid_and_keeps_thai_readable(tickets, now):
    outcome = triage_ticket(_BrokenGraph(), tickets["T-1002"], now, recursion_limit=5)
    outcome = outcome.model_copy(update={"error": "ระบบล่ม"})

    text = render_outcomes([outcome], as_json=True)

    assert json.loads(text)[0]["ticket_id"] == "T-1002"
    assert "ระบบล่ม" in text


def test_cli_writes_utf8_output_directly_without_calling_api(
    monkeypatch, tmp_path, capsys, dataset, tickets, now
):
    outcome = triage_ticket(_BrokenGraph(), tickets["T-1002"], now, recursion_limit=5)
    outcome = outcome.model_copy(update={"error": "ระบบล่ม — we’re checking"})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("triage_agent.main.load_tickets", lambda _: dataset)
    monkeypatch.setattr("triage_agent.main.build_triage_graph", lambda *_: object())
    monkeypatch.setattr("triage_agent.main.triage_ticket", lambda *_: outcome)
    output = tmp_path / "results.json"

    exit_code = main(["--ticket", "T-1002", "--json", "--output", str(output)])

    assert exit_code == 0
    saved = json.loads(output.read_bytes().decode("utf-8"))
    assert saved[0]["error"] == "ระบบล่ม — we’re checking"
    assert f"Saved 1 results to {output}" in capsys.readouterr().out
