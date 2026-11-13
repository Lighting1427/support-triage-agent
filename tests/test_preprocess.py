import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from triage_agent.models import Author, Message
from triage_agent.nodes import compute_facts, preprocess
from triage_agent.prompts import _ago, render_ticket


@pytest.mark.parametrize(
    ("ticket_id", "expected"),
    [
        # T1: Four customer messages over three hours, with no support reply.
        (
            "T-1001",
            {
                "customer_message_count": 4,
                "unanswered_customer_messages": 4,
                "minutes_waiting_for_reply": 180,
                "has_support_replied": False,
                "reply_language": "en",
            },
        ),
        # T2: Four messages over two hours, with no support reply.
        (
            "T-1002",
            {
                "customer_message_count": 4,
                "unanswered_customer_messages": 4,
                "minutes_waiting_for_reply": 120,
                "has_support_replied": False,
                "reply_language": "th",
            },
        ),
        # T3: Count only the three messages since support last replied; waiting one day.
        (
            "T-1003",
            {
                "customer_message_count": 4,
                "unanswered_customer_messages": 3,
                "minutes_waiting_for_reply": 1440,
                "has_support_replied": True,
                "reply_language": "en",
            },
        ),
    ],
)
def test_compute_facts(tickets, now, ticket_id, expected):
    facts = compute_facts(tickets[ticket_id], now)
    assert facts.model_dump(include=set(expected)) == expected


@pytest.mark.parametrize(
    ("customer_text", "support_text", "language"),
    [
        ("Please help with dark mode.", "สวัสดีครับ", "en"),
        ("ช่วยตรวจสอบให้หน่อยครับ", "We are checking.", "th"),
        ("เปิด app แล้วขึ้น error 500 ครับ", "We are checking.", "th"),
    ],
)
def test_reply_language_uses_customer_text_only(
    tickets, now, customer_text, support_text, language
):
    ticket = tickets["T-1003"].model_copy(
        update={
            "messages": [
                Message(author=Author.CUSTOMER, sent_at=now, text=customer_text),
                Message(author=Author.SUPPORT, sent_at=now, text=support_text),
            ]
        }
    )

    assert compute_facts(ticket, now).reply_language == language


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
    assert "reply_language: th" in text


def test_preprocess_node_returns_facts_and_first_two_messages(tickets, now):
    update = preprocess({"ticket": tickets["T-1001"], "reference_time": now})

    assert update["facts"].minutes_waiting_for_reply == 180
    system, human = update["messages"]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert "T-1001" in human.content
