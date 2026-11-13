import pytest

from triage_agent.models import BusinessRisk, IssueType, NextAction, Team, Urgency
from triage_agent.nodes import compute_facts
from triage_agent.policy import (
    _ACTION_LEVEL,
    DEFAULT_RULES,
    PolicyContext,
    apply_rules,
    make_apply_policy_node,
)

KNOWN_KB = frozenset({"KB-002", "KB-003", "KB-007", "KB-008"})


@pytest.fixture
def ctx_for(store, tickets, now):
    """สร้าง PolicyContext ของ ticket ที่ระบุ (ใช้ข้อมูลลูกค้าจาก mock data)"""

    def _ctx(ticket_id: str) -> PolicyContext:
        ticket = tickets[ticket_id]
        return PolicyContext(
            facts=compute_facts(ticket, now),
            customer=store.get_customer(ticket.customer_id),
            known_kb_ids=KNOWN_KB,
        )

    return _ctx


def test_safe_how_to_answer_passes_unchanged(make_result, ctx_for):
    result = make_result()

    final, applied = apply_rules(result, ctx_for("T-1003"))

    assert final == result
    assert applied == []


def test_invented_kb_ids_are_dropped_and_ungrounded_reply_goes_to_a_human(make_result, ctx_for):
    final, applied = apply_rules(make_result(kb_article_ids=["KB-999"]), ctx_for("T-1003"))

    assert final.kb_article_ids == []
    assert final.action is NextAction.ROUTE_TO_SPECIALIST
    assert Team.GENERAL_SUPPORT in final.route_to
    assert applied == ["drop_unknown_kb_articles", "auto_respond_requires_kb_evidence"]


def test_critical_is_always_escalated(make_result, ctx_for):
    result = make_result(urgency=Urgency.CRITICAL, route_to=[Team.ENGINEERING])

    final, applied = apply_rules(result, ctx_for("T-1003"))

    assert final.action is NextAction.ESCALATE_TO_HUMAN
    assert "critical_requires_human" in applied


def test_billing_issue_never_auto_responds_and_goes_to_billing(make_result, ctx_for):
    result = make_result(issue_types=[IssueType.BILLING], kb_article_ids=["KB-002"])

    final, _ = apply_rules(result, ctx_for("T-1001"))

    assert final.action is NextAction.ROUTE_TO_SPECIALIST
    assert Team.BILLING in final.route_to


def test_chargeback_threat_is_escalated(make_result, ctx_for):
    result = make_result(
        urgency=Urgency.HIGH,
        issue_types=[IssueType.BILLING],
        business_risks=[BusinessRisk.CHARGEBACK_OR_LEGAL_THREAT],
    )

    final, applied = apply_rules(result, ctx_for("T-1001"))

    assert final.action is NextAction.ESCALATE_TO_HUMAN
    assert Team.BILLING in final.route_to
    assert "chargeback_or_legal_threat_requires_human" in applied


def test_low_confidence_is_escalated(make_result, ctx_for):
    final, _ = apply_rules(make_result(confidence=0.4), ctx_for("T-1003"))

    assert final.action is NextAction.ESCALATE_TO_HUMAN


def test_urgent_enterprise_outage_pages_oncall_and_account_manager(make_result, ctx_for):
    result = make_result(
        urgency=Urgency.CRITICAL,
        issue_types=[IssueType.SERVICE_OUTAGE],
        action=NextAction.ESCALATE_TO_HUMAN,
        route_to=[],
    )

    final, _ = apply_rules(result, ctx_for("T-1002"))  # C-002 = enterprise

    assert {Team.SRE_ONCALL, Team.ACCOUNT_MANAGER} <= set(final.route_to)


def test_account_manager_is_not_added_for_non_enterprise(make_result, ctx_for):
    result = make_result(urgency=Urgency.HIGH, action=NextAction.ESCALATE_TO_HUMAN)

    final, _ = apply_rules(result, ctx_for("T-1001"))  # C-001 = free plan

    assert Team.ACCOUNT_MANAGER not in final.route_to


def test_handoff_without_team_goes_to_general_support(make_result, ctx_for):
    result = make_result(action=NextAction.ROUTE_TO_SPECIALIST, route_to=[])

    final, applied = apply_rules(result, ctx_for("T-1003"))

    assert final.route_to == [Team.GENERAL_SUPPORT]
    assert applied == ["handoff_needs_a_team"]


@pytest.mark.parametrize(
    ("ticket_id", "reply", "needs_review"),
    [
        ("T-1003", "Hi! Use the web app instead.", False),
        ("T-1003", "สวัสดีครับ ลองใช้เว็บแทนครับ", True),
        ("T-1003", "Hi! ลองใช้เว็บแทนครับ", True),
        ("T-1002", "ลองใช้เว็บแทนครับ", False),
        ("T-1002", "Hi! Use the web app instead.", True),
        ("T-1003", "   ", True),
        ("T-1002", "", True),
    ],
)
def test_wrong_language_blocks_auto_response(make_result, ctx_for, ticket_id, reply, needs_review):
    result = make_result(draft_reply=reply, route_to=[Team.ENGINEERING])

    final, applied = apply_rules(result, ctx_for(ticket_id))

    if needs_review:
        assert final.action is NextAction.ROUTE_TO_SPECIALIST
        assert set(final.route_to) == {Team.ENGINEERING, Team.GENERAL_SUPPORT}
        assert applied == ["wrong_reply_language_needs_review"]
    else:
        assert final == result
        assert applied == []
    assert final.draft_reply == reply
    assert result.action is NextAction.AUTO_RESPOND


@pytest.mark.parametrize("action", [NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN])
def test_wrong_language_preserves_existing_human_review(make_result, ctx_for, action):
    result = make_result(action=action, draft_reply="สวัสดีครับ", route_to=[Team.ENGINEERING])

    final, applied = apply_rules(result, ctx_for("T-1003"))

    assert final == result
    assert applied == []


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda r: r.__name__)
@pytest.mark.parametrize("action", list(NextAction))
def test_no_rule_ever_makes_a_decision_less_safe(make_result, ctx_for, rule, action):
    result = make_result(
        urgency=Urgency.CRITICAL,
        issue_types=[IssueType.BILLING, IssueType.SERVICE_OUTAGE],
        business_risks=[BusinessRisk.CHARGEBACK_OR_LEGAL_THREAT],
        action=action,
        route_to=[Team.ENGINEERING],
        confidence=0.3,
    )

    updated = rule(result, ctx_for("T-1002")) or result

    assert _ACTION_LEVEL[updated.action] >= _ACTION_LEVEL[action]
    assert set(result.route_to) <= set(updated.route_to)


def test_apply_policy_node_stores_final_result_and_rule_names(store, make_result, tickets, now):
    node = make_apply_policy_node(store, KNOWN_KB)
    ticket = tickets["T-1001"]
    llm_result = make_result(issue_types=[IssueType.BILLING])

    update = node(
        {
            "ticket": ticket,
            "reference_time": now,
            "facts": compute_facts(ticket, now),
            "triage": llm_result,
        }
    )

    assert update["final"].action is NextAction.ROUTE_TO_SPECIALIST
    assert update["policy_overrides"] == ["billing_never_auto_responds"]
