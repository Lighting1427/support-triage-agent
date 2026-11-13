from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from triage_agent.models import Ticket, TicketFacts, TriageResult

class TriageInput(TypedDict):
    ticket: Ticket
    reference_time: datetime  

class TriageState(TriageInput, total=False):
    facts: TicketFacts
    messages: Annotated[list[AnyMessage], add_messages]

    triage: TriageResult 
    final: TriageResult 
    policy_overrides: list[str] 