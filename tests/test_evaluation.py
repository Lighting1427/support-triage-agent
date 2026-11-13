import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from triage_agent.evaluation import GoldenCase, evaluate, llm_action_was_correct
from triage_agent.models import IssueType, NextAction, Team, TicketOutcome

GOLDEN_SET = Path(__file__).resolve().parent.parent / "evals" / "golden_set.json"


@pytest.fixture(scope="module")
def cases() -> dict[str, GoldenCase]:
    raw = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    return {c.ticket_id: c for c in TypeAdapter(list[GoldenCase]).validate_python(raw)}


@pytest.fixture
def ideal_t3_outcome(make_result) -> TicketOutcome:
    result = make_result(
        issue_types=[IssueType.BUG, IssueType.FEATURE_REQUEST],
        kb_article_ids=["KB-008", "KB-009"],
        route_to=[Team.ENGINEERING, Team.PRODUCT],
    )
    return TicketOutcome(
        ticket_id="T-1003",
        action=result.action,
        result=result,
        llm_result=result,
        tool_calls=[
            "get_customer_profile(customer_id='C-003')",
            "search_knowledge_base(query='x')",
        ],
    )


def test_golden_set_covers_every_sample_ticket(cases, tickets):
    assert set(cases) == set(tickets)


def test_ideal_outcome_passes_every_check(cases, ideal_t3_outcome):
    results = evaluate(cases["T-1003"], ideal_t3_outcome)

    assert [r.name for r in results if not r.passed] == []


def test_wrong_action_missing_team_and_wrong_language_are_reported(cases, ideal_t3_outcome):
    wrong = ideal_t3_outcome.result.model_copy(
        update={
            "action": NextAction.ESCALATE_TO_HUMAN,
            "route_to": [Team.PRODUCT],
            "draft_reply": "สวัสดีครับ", 
        }
    )
    outcome = ideal_t3_outcome.model_copy(update={"result": wrong})

    failed = {r.name for r in evaluate(cases["T-1003"], outcome) if not r.passed}

    assert failed == {"action", "teams", "reply_language"}


def test_thai_ticket_requires_thai_reply(cases, ideal_t3_outcome):
    results = {r.name: r for r in evaluate(cases["T-1002"], ideal_t3_outcome)}

    assert results["reply_language"].passed is False


def test_failed_agent_fails_every_check(cases):
    outcome = TicketOutcome(ticket_id="T-1001", action=NextAction.ESCALATE_TO_HUMAN, error="boom")

    assert not any(r.passed for r in evaluate(cases["T-1001"], outcome))


def test_llm_action_is_judged_before_policy(cases, ideal_t3_outcome):
    assert llm_action_was_correct(cases["T-1003"], ideal_t3_outcome)

    llm_wrong = ideal_t3_outcome.llm_result.model_copy(
        update={"action": NextAction.ESCALATE_TO_HUMAN}
    )
    assert not llm_action_was_correct(
        cases["T-1003"], ideal_t3_outcome.model_copy(update={"llm_result": llm_wrong})
    )