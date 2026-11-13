from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from triage_agent.config import Settings
from triage_agent.models import TriageResult
from triage_agent.nodes import ChatRunnable, DecisionRunnable


def create_investigator(settings: Settings, tools: Sequence[BaseTool]) -> ChatRunnable:
    llm = ChatOpenAI(model=settings.openai_model)  
    return llm.bind_tools(
        tools,
        strict=True, 
        parallel_tool_calls=False,
    )


def create_decider(settings: Settings) -> DecisionRunnable:
    llm = ChatOpenAI(model=settings.openai_model)
    return llm.with_structured_output(TriageResult, method="json_schema", strict=True)