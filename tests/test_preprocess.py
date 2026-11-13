import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from triage_agent.nodes import compute_facts, preprocess
from triage_agent.prompts import _ago, render_ticket


@pytest.mark.parametrize(
    ("ticket_id", "expected"),
    [
        # T1: ลูกค้าส่ง 4 ข้อความใน 3 ชม. ยังไม่มีใครตอบเลย
        ("T-1001", {"customer_message_count": 4, "unanswered_customer_messages": 4,
                    "minutes_waiting_for_reply": 180, "has_support_replied": False}),
        # T2: 4 ข้อความใน 2 ชม. ยังไม่มีใครตอบ
        ("T-1002", {"customer_message_count": 4, "unanswered_customer_messages": 4,
                    "minutes_waiting_for_reply": 120, "has_support_replied": False}),
        # T3: support เคยตอบแล้ว จึงนับเฉพาะ 3 ข้อความหลังจากนั้น (รอมา 1 วัน)
        ("T-1003", {"customer_message_count": 4, "unanswered_customer_messages": 3,
                    "minutes_waiting_for_reply": 1440, "has_support_replied": True}),
    ],
)  
def test_compute_facts(tickets, now, ticket_id, expected):
    facts = compute_facts(tickets[ticket_id], now)
    assert facts.model_dump(include=set(expected)) == expected


@pytest.mark.parametrize(
    ("minutes", "text"),
    [(45, "45m ago"), (180, "3h 0m ago"), (2880, "2d 0h ago")],
)
def test_ago_formats_relative_time(minutes, text):
    assert _ago(minutes) == text


def test_render_ticket_wraps_customer_text_and_keeps_thai(tickets, now):
    ticket = tickets["T-1002"]
    text = render_ticket(ticket, compute_facts(ticket, now), now)

    assert text.startswith('<ticket id="T-1002" customer_id="C-002"')
    assert "</ticket>" in text
    assert "ระบบเข้าไม่ได้ครับ" in text 
    assert "[2026-11-12 05:00 UTC | 2h 0m ago]" in text
    assert "minutes_waiting_for_reply: 120" in text


def test_preprocess_node_returns_facts_and_first_two_messages(tickets, now):
    update = preprocess({"ticket": tickets["T-1001"], "reference_time": now})

    assert update["facts"].minutes_waiting_for_reply == 180
    system, human = update["messages"]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert "T-1001" in human.content