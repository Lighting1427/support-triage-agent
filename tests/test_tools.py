import json
from datetime import UTC, date, datetime
from decimal import Decimal

from langchain_core.utils.function_calling import convert_to_openai_tool

from triage_agent.models import ChargeStatus, Transaction
from triage_agent.tools import find_possible_duplicates, months_between


def call(tools: dict, name: str, **arguments) -> dict:
    """เรียก tool ด้วยชื่อ แล้วแปลง JSON ที่ LLM จะได้เห็น กลับเป็น dict เพื่อ assert"""
    return json.loads(tools[name].invoke(arguments))


def test_customer_profile_includes_plan_and_tenure(tools):
    result = call(tools, "get_customer_profile", customer_id="C-002")

    assert result["customer"]["plan"] == "enterprise"
    assert result["customer"]["seats"] == 45
    assert result["tenure_months"] == 8


def test_customer_profile_unknown_id_returns_error(tools):
    result = call(tools, "get_customer_profile", customer_id="C-999")

    assert "No customer found" in result["error"]


def test_billing_flags_duplicate_charges_for_ticket_1(tools):
    result = call(tools, "get_billing_transactions", customer_id="C-001")

    assert result["subscription_plan"] == "free"
    assert result["captured_total"] == "59.98"
    assert result["pending_total"] == "29.99"
    assert result["possible_duplicate_transaction_ids"] == ["TX-9002", "TX-9003"]
    assert "webhook" in result["last_provisioning_error"]


def test_billing_healthy_account_has_no_duplicates(tools):
    result = call(tools, "get_billing_transactions", customer_id="C-003")

    assert result["possible_duplicate_transaction_ids"] == []


def test_status_detects_status_page_mismatch_for_ticket_2(tools):
    result = call(tools, "check_service_status", region="asia-southeast")

    web_app = next(c for c in result["components"] if c["component"] == "web-app")
    assert web_app["health"] == "outage"
    assert web_app["public_status_page"] == "operational"
    assert result["status_page_mismatch"] is True


def test_status_accepts_partial_region_and_adds_global_components(tools):
    result = call(tools, "check_service_status", region="Asia")
    regions = {c["region"] for c in result["components"]}

    assert regions == {"asia-southeast", "global"}


def test_status_unknown_region_lists_known_regions(tools):
    result = call(tools, "check_service_status", region="europe")

    assert "asia-southeast" in result["error"]


def test_kb_tool_returns_article_content(tools):
    result = call(tools, "search_knowledge_base", query="duplicate charge refund")

    assert result["results"][0]["article_id"] == "KB-003"
    assert result["results"][0]["content"]


def _tx(tx_id: str, hour: int, status: ChargeStatus, amount: str = "29.99") -> Transaction:
    return Transaction(
        transaction_id=tx_id,
        created_at=datetime(2026, 11, 12, hour, tzinfo=UTC),
        amount=Decimal(amount),
        status=status,
        description="Pro plan",
        card_last4="0000",
    )


def test_duplicates_ignore_failed_and_refunded_charges():
    transactions = [
        _tx("A", 1, ChargeStatus.CAPTURED),
        _tx("B", 2, ChargeStatus.FAILED),
        _tx("C", 3, ChargeStatus.REFUNDED),
    ]
    assert find_possible_duplicates(transactions) == []


def test_duplicates_ignore_different_amounts():
    transactions = [
        _tx("A", 1, ChargeStatus.CAPTURED, "29.99"),
        _tx("B", 2, ChargeStatus.CAPTURED, "9.99"),
    ]
    assert find_possible_duplicates(transactions) == []


def test_months_between_counts_only_full_months():
    assert months_between(date(2026, 3, 12), date(2026, 11, 12)) == 8
    assert months_between(date(2026, 3, 12), date(2026, 11, 11)) == 7


def test_tool_definitions_are_strict_mode_compatible(tools):
    for t in tools.values():
        params = convert_to_openai_tool(t)["function"]["parameters"]
        assert set(params.get("required", [])) == set(params["properties"]), t.name


def test_tool_definitions_sent_to_llm_are_english_only(tools):
    for t in tools.values():
        definition = json.dumps(convert_to_openai_tool(t), ensure_ascii=False)
        assert definition.isascii(), t.name
