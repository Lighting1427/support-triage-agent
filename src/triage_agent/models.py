from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base ของข้อมูลที่ห้ามแก้หลังโหลดแล้ว (immutable) กัน โค้ดอื่นเผลอแก้ค่า"""

    model_config = ConfigDict(frozen=True)


# ----------------------------------
# Input: ticket ลูกค้าส่งเข้ามา
# ----------------------------------


class Author(StrEnum):
    CUSTOMER = "customer"
    SUPPORT = "support"


class Message(_Frozen):
    author: Author = Author.CUSTOMER
    sent_at: datetime
    text: str


class Ticket(_Frozen):
    """แชทของ ticket หนึ่งใบ มีเฉพาะสิ่งที่ลูกค้าส่งมาเท่านั้น

    จะไม่ใส่ข้อมูลลูกค้า (plan, seats, region) ไว้ที่นี่
    agent ต้องเรียก tool เพื่อหาเอง
    """

    ticket_id: str
    customer_id: str
    channel: str = "email"
    messages: list[Message] = Field(min_length=1)


class TicketDataset(_Frozen):
    """ticket หลายใบ + เวลา "Now" แบบตายตัว เพื่อให้ demo และ test ได้แบบเดิม"""

    reference_time: datetime
    tickets: list[Ticket]


class Plan(StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class PastTicket(_Frozen):
    ticket_id: str
    subject: str
    urgency: str
    resolved: bool


class Customer(_Frozen):
    customer_id: str
    name: str
    plan: Plan
    region: str
    seats: int = Field(default=1, ge=1)
    customer_since: date
    preferred_language: str = "en"
    activity_level: str  # เช่น "daily", "weekly", "inactive"
    account_manager: str | None = None
    support_history: list[PastTicket] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ChargeStatus(StrEnum):
    PENDING = "pending"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"


class Transaction(_Frozen):
    transaction_id: str
    created_at: datetime
    amount: Decimal
    currency: str = "USD"
    status: ChargeStatus
    description: str
    card_last4: str


class BillingAccount(_Frozen):
    customer_id: str
    subscription_plan: Plan
    last_provisioning_error: str | None = None
    transactions: list[Transaction] = Field(default_factory=list)


class Health(StrEnum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OUTAGE = "outage"


class ComponentStatus(_Frozen):
    """ข้อมูลภายใน อาจไม่ตรงกับ status ที่ลูกค้าเห็น"""

    region: str
    component: str
    health: Health
    error_rate_5xx: float = Field(ge=0, le=1)
    degraded_since: datetime | None = None
    public_status_page: Health
    open_incident_id: str | None = None


class KBArticle(_Frozen):
    article_id: str
    title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    content: str


class TicketFacts(_Frozen):
    customer_message_count: int
    minutes_since_first_message: int
    unanswered_customer_messages: int
    minutes_waiting_for_reply: int
    has_support_replied: bool
    reply_language: Literal["th", "en"]  # ตรวจจากตัวอักษรที่ลูกค้าเขียนจริง ไม่ให้ LLM เดาเอง


class Urgency(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NextAction(StrEnum):
    AUTO_RESPOND = "auto_respond"
    ROUTE_TO_SPECIALIST = "route_to_specialist"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class Team(StrEnum):
    GENERAL_SUPPORT = "general_support"
    BILLING = "billing"
    SRE_ONCALL = "sre_oncall"
    ENGINEERING = "engineering"
    PRODUCT = "product"
    ACCOUNT_MANAGER = "account_manager"


class IssueType(StrEnum):
    BILLING = "billing"
    SERVICE_OUTAGE = "service_outage"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    HOW_TO = "how_to"
    OTHER = "other"


class BusinessRisk(StrEnum):
    CHARGEBACK_OR_LEGAL_THREAT = "chargeback_or_legal_threat"
    CHURN_RISK = "churn_risk"
    CUSTOMER_REVENUE_AT_RISK = "customer_revenue_at_risk"


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class SentimentTrend(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    ESCALATING = "escalating"


class Sentiment(BaseModel):
    label: SentimentLabel = Field(description="Customer sentiment in the latest messages.")
    trend: SentimentTrend = Field(description="How sentiment changed across the conversation.")


class ExtractedInfo(BaseModel):
    product_area: str = Field(description="Main product area, e.g. 'billing', 'web_app'.")
    issue_types: list[IssueType] = Field(description="Every distinct issue in the ticket.")
    sentiment: Sentiment
    language: str = Field(description="ISO 639-1 code of the customer's language, e.g. 'th'.")
    key_facts: list[str] = Field(description="Concrete facts a human agent needs to know.")
    customer_deadline: str | None = Field(
        description="Any deadline the customer mentioned, or null."
    )
    business_risks: list[BusinessRisk] = Field(
        description="Business risks stated or clearly implied in the ticket. Empty if none."
    )


class TriageResult(BaseModel):
    urgency: Urgency
    urgency_reasoning: str = Field(description="One or two sentences justifying the urgency.")
    extracted: ExtractedInfo
    kb_article_ids: list[str] = Field(
        description="IDs of relevant KB articles returned by the search tool. Never invent IDs."
    )
    action: NextAction
    route_to: list[Team] = Field(
        description=(
            "Teams to hand off to or notify. Can be non-empty with auto_respond, "
            "e.g. engineering for a confirmed bug or product for a feature request."
        )
    )
    draft_reply: str = Field(
        description=(
            "Reply to the customer in their language. Sent as-is for auto_respond, "
            "otherwise a suggested reply for the human agent."
        )
    )
    internal_note: str = Field(description="Short summary for the human support agent.")
    confidence: float = Field(ge=0, le=1, description="Confidence in this decision, 0 to 1.")


class TicketOutcome(BaseModel):
    """ผลลัพธ์ของ ticket หนึ่งใบ ทั้งกรณีสำเร็จและกรณี agent พัง"""

    ticket_id: str
    action: NextAction
    result: TriageResult | None = None
    llm_result: TriageResult | None = None
    policy_overrides: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    error: str | None = None
