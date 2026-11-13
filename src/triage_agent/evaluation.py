from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from triage_agent.language import contains_thai, reply_matches_language
from triage_agent.models import IssueType, NextAction, Team, TicketOutcome, Urgency


class GoldenCase(BaseModel):
    ticket_id: str
    allowed_urgency: list[Urgency]
    expected_action: NextAction
    required_teams: list[Team]
    required_issue_types: list[IssueType]
    kb_any_of: list[str]
    required_tools: list[str]
    reply_language: str
    note: str = ""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


Check = Callable[[GoldenCase, TicketOutcome], CheckResult]


def _tool_names(outcome: TicketOutcome) -> set[str]:
    return {call.split("(", 1)[0] for call in outcome.tool_calls}


def _requires_all(name: str, required: list, actual: list | set) -> CheckResult:
    missing = sorted(set(required) - set(actual))
    return CheckResult(name, not missing, f"missing {missing}" if missing else "ok")


def check_urgency(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    urgency = outcome.result.urgency
    return CheckResult("urgency", urgency in case.allowed_urgency, f"got {urgency}")


def check_action(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    action = outcome.result.action
    return CheckResult("action", action is case.expected_action, f"got {action}")


def check_teams(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    return _requires_all("teams", case.required_teams, outcome.result.route_to)


def check_issue_types(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    actual = outcome.result.extracted.issue_types
    return _requires_all("issue_types", case.required_issue_types, actual)


def check_kb_articles(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    cited = outcome.result.kb_article_ids
    return CheckResult("kb_articles", bool(set(cited) & set(case.kb_any_of)), f"cited {cited}")


def check_tools_used(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    return _requires_all("tools_used", case.required_tools, _tool_names(outcome))


def check_reply_language(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    has_thai = contains_thai(outcome.result.draft_reply)
    passed = reply_matches_language(outcome.result.draft_reply, case.reply_language)
    return CheckResult("reply_language", passed, f"thai characters: {has_thai}")


def check_internal_note_language(case: GoldenCase, outcome: TicketOutcome) -> CheckResult:
    # Internal teams (SRE, billing) need English notes regardless of the customer's language.
    has_thai = contains_thai(outcome.result.internal_note)
    return CheckResult("internal_note_en", not has_thai, f"thai characters: {has_thai}")


DEFAULT_CHECKS: tuple[Check, ...] = (
    check_urgency,
    check_action,
    check_teams,
    check_issue_types,
    check_kb_articles,
    check_tools_used,
    check_reply_language,
    check_internal_note_language,
)


def evaluate(
    case: GoldenCase, outcome: TicketOutcome, checks: tuple[Check, ...] = DEFAULT_CHECKS
) -> list[CheckResult]:
    if outcome.result is None:
        names = [check.__name__.removeprefix("check_") for check in checks]
        return [CheckResult(name, False, "agent failed") for name in names]
    return [check(case, outcome) for check in checks]


def llm_action_was_correct(case: GoldenCase, outcome: TicketOutcome) -> bool:
    """Check the LLM's action before policy guardrails are applied."""
    return outcome.llm_result is not None and outcome.llm_result.action is case.expected_action
