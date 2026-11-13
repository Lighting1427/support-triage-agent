import logging
from collections.abc import Callable
from datetime import datetime

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from triage_agent.models import Author, Ticket, TicketFacts, TriageResult
from triage_agent.prompts import (
    DECIDE_SYSTEM_PROMPT,
    INVESTIGATE_SYSTEM_PROMPT,
    render_evidence,
    render_ticket,
)
from triage_agent.state import TriageState

logger = logging.getLogger(__name__)

ChatRunnable = Runnable[LanguageModelInput, BaseMessage]
DecisionRunnable = Runnable[LanguageModelInput, TriageResult]

def _minutes(later: datetime, earlier: datetime) -> int:
    return int((later - earlier).total_seconds() // 60)


def compute_facts(ticket: Ticket, now: datetime) -> TicketFacts:
    """คำนวณข้อเท็จจริงเรื่องเวลาและจำนวนข้อความ """
    customer_messages = [m for m in ticket.messages if m.author is Author.CUSTOMER]
    support_times = [m.sent_at for m in ticket.messages if m.author is Author.SUPPORT]
    last_support_reply = max(support_times, default=None)

    unanswered = [
        m for m in customer_messages if last_support_reply is None or m.sent_at > last_support_reply
    ]

    return TicketFacts(
        customer_message_count=len(customer_messages),
        minutes_since_first_message=_minutes(now, ticket.messages[0].sent_at),
        unanswered_customer_messages=len(unanswered),
        minutes_waiting_for_reply=_minutes(now, unanswered[0].sent_at) if unanswered else 0,
        has_support_replied=last_support_reply is not None,
    )


def preprocess(state: TriageState) -> dict:
    facts = compute_facts(state["ticket"], state["reference_time"])
    return {
        "facts": facts,
        "messages": [
            SystemMessage(INVESTIGATE_SYSTEM_PROMPT),
            HumanMessage(render_ticket(state["ticket"], facts, state["reference_time"])),
        ],
    }


def make_investigate_node(investigator: ChatRunnable) -> Callable[[TriageState], dict]:
    """สร้าง node investigate โดย inject model เข้ามา (test ส่ง fake model แทนได้)"""

    def investigate(state: TriageState) -> dict:
        # ส่ง messages ทั้งหมด (= context ทั้งหมด) ให้ LLM แล้วต่อท้ายด้วยคำตอบใหม่
        response = investigator.invoke(state["messages"])
        for call in getattr(response, "tool_calls", []):
            logger.info("tool call: %s(%s)", call["name"], call["args"])
        return {"messages": [response]}

    return investigate


def make_decide_node(decider: DecisionRunnable) -> Callable[[TriageState], dict]:
    """สร้าง node decide: context ใหม่ที่สะอาด = system prompt + ticket + หลักฐาน"""

    def decide(state: TriageState) -> dict:
        ticket_text = render_ticket(state["ticket"], state["facts"], state["reference_time"])
        # messages[2:] = ข้าม system prompt กับ ticket ของช่วง investigate เหลือแต่หลักฐาน
        evidence_text = render_evidence(state["messages"][2:])
        result = decider.invoke(
            [
                SystemMessage(DECIDE_SYSTEM_PROMPT),
                HumanMessage(f"{ticket_text}\n\n{evidence_text}"),
            ]
        )
        # validate อีกชั้น เผื่อ model ส่ง dict กลับมาแทน object
        return {"triage": TriageResult.model_validate(result)}

    return decide


def handle_tool_error(error: Exception) -> str:
    """bug ที่ไม่คาดคิดใน tool: log รายละเอียดไว้ให้ dev แต่บอก LLM แค่ข้อความกลางๆ

    """
    logger.error("Tool failed unexpectedly: %r", error)
    return "Tool failed unexpectedly. Do not retry it; continue with the evidence you have."