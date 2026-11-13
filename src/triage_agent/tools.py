import json
from datetime import date, timedelta
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool

from triage_agent.data import KnowledgeBase, MockDataStore
from triage_agent.models import ChargeStatus, ComponentStatus, Health, Transaction

KB_TOP_K = 3
DUPLICATE_WINDOW = timedelta(hours=24)
GLOBAL_REGION = "global"
_ACTIVE_CHARGE_STATUSES = {ChargeStatus.CAPTURED, ChargeStatus.PENDING}


def months_between(start: date, end: date) -> int:
    """Count full months: Mar 12 to Nov 12 is 8 months; Mar 12 to Nov 11 is 7."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def find_possible_duplicates(transactions: list[Transaction]) -> list[str]:
    """Flag later charges with the same amount within DUPLICATE_WINDOW of an earlier charge."""
    active = sorted(
        (t for t in transactions if t.status in _ACTIVE_CHARGE_STATUSES),
        key=lambda t: t.created_at,
    )
    duplicates: list[str] = []
    for i, later in enumerate(active):
        for earlier in active[:i]:
            same_amount = earlier.amount == later.amount
            close_in_time = later.created_at - earlier.created_at <= DUPLICATE_WINDOW
            if same_amount and close_in_time:
                duplicates.append(later.transaction_id)
                break
    return duplicates


def total_by_status(transactions: list[Transaction], status: ChargeStatus) -> Decimal:
    return sum((t.amount for t in transactions if t.status is status), start=Decimal("0"))


def disagrees_with_status_page(component: ComponentStatus) -> bool:
    """Detect an unhealthy component whose public status page still reports operational."""
    return component.health is not Health.OPERATIONAL and (
        component.public_status_page is Health.OPERATIONAL
    )


def _to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _error(message: str) -> str:
    return _to_json({"error": message})


def build_tools(store: MockDataStore, kb: KnowledgeBase, today: date) -> list[BaseTool]:
    @tool
    def get_customer_profile(
        customer_id: Annotated[str, "Customer ID from the ticket, e.g. 'C-001'."],
    ) -> str:
        """Get the customer's plan, seats, region, preferred language, account manager
        and previous support tickets. Call this first for every ticket."""
        customer = store.get_customer(customer_id)
        if customer is None:
            return _error(f"No customer found with id '{customer_id}'.")
        return _to_json(
            {
                "customer": customer.model_dump(mode="json"),
                "tenure_months": months_between(customer.customer_since, today),
            }
        )

    @tool
    def search_knowledge_base(
        query: Annotated[
            str,
            "Short ENGLISH keyword query, e.g. 'duplicate charge refund'. "
            "Translate non-English tickets to English first.",
        ],
    ) -> str:
        """Search help-center articles (FAQ, policies, known issues).
        Call it once per distinct issue in the ticket. Only cite article IDs it returns."""
        hits = kb.search(query, top_k=KB_TOP_K)
        return _to_json(
            {
                "query": query,
                "results": [
                    {
                        "article_id": article.article_id,
                        "title": article.title,
                        "content": article.content,
                        "score": score,
                    }
                    for article, score in hits
                ],
            }
        )

    @tool
    def get_billing_transactions(
        customer_id: Annotated[str, "Customer ID from the ticket, e.g. 'C-001'."],
    ) -> str:
        """Get recent charges (pending/captured/failed/refunded), the plan shown in the
        billing system, provisioning errors and possible duplicate charges.
        Use for any payment, charge, refund or upgrade problem."""
        account = store.get_billing(customer_id)
        if account is None:
            return _error(f"No billing account found for customer '{customer_id}'.")
        transactions = account.transactions
        return _to_json(
            {
                "subscription_plan": account.subscription_plan,
                "last_provisioning_error": account.last_provisioning_error,
                "transactions": [t.model_dump(mode="json") for t in transactions],
                "captured_total": total_by_status(transactions, ChargeStatus.CAPTURED),
                "pending_total": total_by_status(transactions, ChargeStatus.PENDING),
                "possible_duplicate_transaction_ids": find_possible_duplicates(transactions),
            }
        )

    @tool
    def check_service_status(
        region: Annotated[
            str, "Region from the customer profile, e.g. 'asia-southeast' or 'us-east'."
        ],
    ) -> str:
        """Check INTERNAL monitoring (health, 5xx error rate, open incidents) for a region
        plus global components. More reliable than the public status page.
        Use for errors, outages, slowness or access problems."""
        query = region.strip().lower()
        statuses = store.service_statuses()
        regional = [s for s in statuses if s.region.startswith(query)]
        if not regional:
            known = sorted({s.region for s in statuses} - {GLOBAL_REGION})
            return _error(f"Unknown region '{region}'. Known regions: {', '.join(known)}.")
        components = regional + [s for s in statuses if s.region == GLOBAL_REGION]
        return _to_json(
            {
                "components": [c.model_dump(mode="json") for c in components],
                "status_page_mismatch": any(disagrees_with_status_page(c) for c in components),
            }
        )

    return [
        get_customer_profile,
        search_knowledge_base,
        get_billing_transactions,
        check_service_status,
    ]
