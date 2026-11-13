from datetime import date
from pathlib import Path

import pytest

from triage_agent.data import KnowledgeBase, MockDataStore, load_tickets
from triage_agent.models import (
    ExtractedInfo,
    IssueType,
    NextAction,
    Sentiment,
    SentimentLabel,
    SentimentTrend,
    TriageResult,
    Urgency,
)
from triage_agent.tools import build_tools

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIXED_TODAY = date(2026, 11, 12)  

@pytest.fixture(scope="session") 
def store() -> MockDataStore:
    return MockDataStore(DATA_DIR)


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_file(DATA_DIR / "knowledge_base.json")


@pytest.fixture(scope="session")
def tools(store, kb) -> dict:
    return {t.name: t for t in build_tools(store, kb, today=FIXED_TODAY)}


@pytest.fixture(scope="session")
def dataset():
    return load_tickets(DATA_DIR / "tickets.json")


@pytest.fixture(scope="session")
def now(dataset):
    return dataset.reference_time 


@pytest.fixture(scope="session")
def tickets(dataset) -> dict:
    return {t.ticket_id: t for t in dataset.tickets}

@pytest.fixture
def make_result():
    """โรงงานสร้าง TriageResult สำหรับ test: ค่าเริ่มต้นเป็นเคส "ง่ายและปลอดภัย" แก้เฉพาะที่สนใจ

    ตัวอย่าง: make_result(urgency=Urgency.CRITICAL, issue_types=[IssueType.SERVICE_OUTAGE])
    """

    def _make(*, issue_types=(IssueType.HOW_TO,), business_risks=(), **overrides) -> TriageResult:
        fields = {
            "urgency": Urgency.LOW,
            "urgency_reasoning": "How-to question with no time pressure.",
            "extracted": ExtractedInfo(
                product_area="web_app",
                issue_types=list(issue_types),
                sentiment=Sentiment(label=SentimentLabel.NEUTRAL, trend=SentimentTrend.STABLE),
                language="en",
                key_facts=[],
                customer_deadline=None,
                business_risks=list(business_risks),
            ),
            "kb_article_ids": ["KB-007"],
            "action": NextAction.AUTO_RESPOND,
            "route_to": [],
            "draft_reply": "Hi! Here is how to do it.",
            "internal_note": "Answered from KB-007.",
            "confidence": 0.9,
        }
        return TriageResult(**(fields | overrides))

    return _make