from datetime import datetime
from typing import Annotated, NotRequired, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from triage_agent.models import Ticket, TicketFacts, TriageResult

class TriageState(TypedDict):
    ticket: Ticket
    reference_time: datetime 
    facts: NotRequired[TicketFacts]
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    triage: NotRequired[TriageResult] 
    final: NotRequired[TriageResult]  
    policy_overrides: NotRequired[list[str]]  