import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from triage_agent.models import (
    BillingAccount,
    ComponentStatus,
    Customer,
    KBArticle,
    TicketDataset,
    TriageResult,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("filename", "schema"),
    [
        ("customers.json", list[Customer]),
        ("billing.json", list[BillingAccount]),
        ("service_status.json", list[ComponentStatus]),
        ("knowledge_base.json", list[KBArticle]),
    ],
)
def test_data_file_matches_schema(filename, schema):
    TypeAdapter(schema).validate_python(_load(filename))


def test_tickets_reference_known_customers():
    dataset = TicketDataset.model_validate(_load("tickets.json"))
    customer_ids = {c["customer_id"] for c in _load("customers.json")}

    for ticket in dataset.tickets:
        assert ticket.customer_id in customer_ids


def test_messages_are_in_chronological_order():
    dataset = TicketDataset.model_validate(_load("tickets.json"))

    for ticket in dataset.tickets:
        times = [m.sent_at for m in ticket.messages]
        assert times == sorted(times), ticket.ticket_id
        assert times[-1] <= dataset.reference_time


def _all_object_schemas(schema: dict) -> list[dict]:
    """ดึง object schema ทั้งหมด ทั้งตัวหลักและ model ที่ซ้อนอยู่ใน $defs"""
    return [schema, *schema.get("$defs", {}).values()]


def test_output_schema_is_strict_mode_compatible():
    for obj in _all_object_schemas(TriageResult.model_json_schema()):
        if obj.get("type") != "object":
            continue
        properties = obj["properties"]
        assert set(obj.get("required", [])) == set(properties), obj.get("title")
        assert all("default" not in p for p in properties.values()), obj.get("title")


def test_output_schema_sent_to_llm_is_english_only():
    text = json.dumps(TriageResult.model_json_schema(), ensure_ascii=False)
    assert text.isascii()
