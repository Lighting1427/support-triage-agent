from collections.abc import Callable, Sequence
from dataclasses import dataclass

from triage_agent.data import MockDataStore
from triage_agent.language import reply_matches_language
from triage_agent.models import (
    BusinessRisk,
    Customer,
    IssueType,
    NextAction,
    Plan,
    Team,
    TicketFacts,
    TriageResult,
    Urgency,
)
from triage_agent.state import TriageState

MIN_CONFIDENCE = 0.6

_ACTION_LEVEL = {
    NextAction.AUTO_RESPOND: 0,
    NextAction.ROUTE_TO_SPECIALIST: 1,
    NextAction.ESCALATE_TO_HUMAN: 2,
}
_URGENT = {Urgency.CRITICAL, Urgency.HIGH}


@dataclass(frozen=True)
class PolicyContext:
    facts: TicketFacts
    customer: Customer | None
    known_kb_ids: frozenset[str]


Rule = Callable[[TriageResult, PolicyContext], TriageResult | None]


def _at_least(result: TriageResult, action: NextAction) -> TriageResult:
    if _ACTION_LEVEL[result.action] >= _ACTION_LEVEL[action]:
        return result  # ปลอดภัยพอแล้ว ไม่ลดระดับลง
    return result.model_copy(update={"action": action})


def _with_teams(result: TriageResult, *teams: Team) -> TriageResult:
    missing = [t for t in teams if t not in result.route_to]
    if not missing:
        return result
    return result.model_copy(update={"route_to": [*result.route_to, *missing]})


def drop_unknown_kb_articles(result: TriageResult, ctx: PolicyContext) -> TriageResult | None:
    valid = [a for a in result.kb_article_ids if a in ctx.known_kb_ids]
    if valid == result.kb_article_ids:
        return None
    return result.model_copy(update={"kb_article_ids": valid})


def critical_requires_human(result: TriageResult, ctx: PolicyContext) -> TriageResult | None:
    if result.urgency is Urgency.CRITICAL:
        return _at_least(result, NextAction.ESCALATE_TO_HUMAN)
    return None


def chargeback_or_legal_threat_requires_human(
    result: TriageResult, ctx: PolicyContext
) -> TriageResult | None:
    if BusinessRisk.CHARGEBACK_OR_LEGAL_THREAT in result.extracted.business_risks:
        return _at_least(result, NextAction.ESCALATE_TO_HUMAN)
    return None


def low_confidence_requires_human(result: TriageResult, ctx: PolicyContext) -> TriageResult | None:
    if result.confidence < MIN_CONFIDENCE:
        return _at_least(result, NextAction.ESCALATE_TO_HUMAN)
    return None


def billing_never_auto_responds(result: TriageResult, ctx: PolicyContext) -> TriageResult | None:
    if IssueType.BILLING in result.extracted.issue_types:
        return _with_teams(_at_least(result, NextAction.ROUTE_TO_SPECIALIST), Team.BILLING)
    return None


def urgent_outage_pages_oncall(result: TriageResult, ctx: PolicyContext) -> TriageResult | None:
    if IssueType.SERVICE_OUTAGE in result.extracted.issue_types and result.urgency in _URGENT:
        return _with_teams(result, Team.SRE_ONCALL)
    return None


def urgent_enterprise_notifies_account_manager(
    result: TriageResult, ctx: PolicyContext
) -> TriageResult | None:
    is_enterprise = ctx.customer is not None and ctx.customer.plan is Plan.ENTERPRISE
    if is_enterprise and result.urgency in _URGENT:
        return _with_teams(result, Team.ACCOUNT_MANAGER)
    return None


def auto_respond_requires_kb_evidence(
    result: TriageResult, ctx: PolicyContext
) -> TriageResult | None:
    if result.action is NextAction.AUTO_RESPOND and not result.kb_article_ids:
        return _with_teams(_at_least(result, NextAction.ROUTE_TO_SPECIALIST), Team.GENERAL_SUPPORT)
    return None


def wrong_reply_language_needs_review(
    result: TriageResult, ctx: PolicyContext
) -> TriageResult | None:
    # คำตอบที่จะส่งอัตโนมัติต้องเป็นภาษาเดียวกับที่ลูกค้าเขียนและต้องไม่ว่าง
    # ส่วน action อื่นมีคนอ่านก่อนส่งอยู่แล้ว จึงไม่ต้องบังคับ
    if result.action is NextAction.AUTO_RESPOND and not reply_matches_language(
        result.draft_reply, ctx.facts.reply_language
    ):
        return _with_teams(_at_least(result, NextAction.ROUTE_TO_SPECIALIST), Team.GENERAL_SUPPORT)
    return None


def handoff_needs_a_team(result: TriageResult, ctx: PolicyContext) -> TriageResult | None:
    if result.action is not NextAction.AUTO_RESPOND and not result.route_to:
        return _with_teams(result, Team.GENERAL_SUPPORT)
    return None


DEFAULT_RULES: tuple[Rule, ...] = (
    drop_unknown_kb_articles,
    critical_requires_human,
    chargeback_or_legal_threat_requires_human,
    low_confidence_requires_human,
    billing_never_auto_responds,
    urgent_outage_pages_oncall,
    urgent_enterprise_notifies_account_manager,
    auto_respond_requires_kb_evidence,
    wrong_reply_language_needs_review,
    handoff_needs_a_team,
)


def apply_rules(
    result: TriageResult, ctx: PolicyContext, rules: Sequence[Rule] = DEFAULT_RULES
) -> tuple[TriageResult, list[str]]:
    """รันกฎตามลำดับ คืนผลสุดท้าย + ชื่อกฎที่ "เปลี่ยนค่าจริง" """
    applied: list[str] = []
    for rule in rules:
        updated = rule(result, ctx)
        if updated is not None and updated != result:
            result = updated
            applied.append(rule.__name__)
    return result, applied


def make_apply_policy_node(
    store: MockDataStore, known_kb_ids: frozenset[str]
) -> Callable[[TriageState], dict]:
    """สร้าง node apply_policy โดย inject แหล่งข้อมูลที่ตรวจสอบแล้วเข้ามา"""

    def apply_policy(state: TriageState) -> dict:
        ctx = PolicyContext(
            facts=state["facts"],
            customer=store.get_customer(state["ticket"].customer_id),
            known_kb_ids=known_kb_ids,
        )
        final, applied = apply_rules(state["triage"], ctx)
        return {"final": final, "policy_overrides": applied}

    return apply_policy
