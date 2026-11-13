from itertools import repeat

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from triage_agent.graph import build_graph
from triage_agent.models import IssueType, NextAction, Team
from triage_agent.policy import make_apply_policy_node


def _ask_for_tool(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


@pytest.fixture
def run(store, kb, tickets, now, make_result):
    """รัน graph กับ ticket T-1001 ด้วย fake investigator + fake decider

    คืน (state, prompts) โดย prompts = message ที่ decider ได้รับ ใช้ตรวจว่า decide เห็นหลักฐานครบ
    """

    def _run(investigator_responses, tools, decision=None):
        prompts = []

        def fake_decider(messages):
            prompts.append(messages)
            return decision or make_result()

        graph = build_graph(
            investigator=GenericFakeChatModel(messages=investigator_responses),
            decider=RunnableLambda(fake_decider),
            tools=tools,
            apply_policy=make_apply_policy_node(store, kb.article_ids),
        )
        state = graph.invoke(
            {"ticket": tickets["T-1001"], "reference_time": now},
            config={"recursion_limit": 20},
        )
        return state, prompts

    return _run


def test_full_flow_investigates_decides_and_applies_policy(run, tools, make_result):
    llm_decision = make_result(issue_types=[IssueType.BILLING], kb_article_ids=["KB-003"])

    state, prompts = run(
        iter([_ask_for_tool("get_billing_transactions", customer_id="C-001"), "Evidence summary"]),
        list(tools.values()),
        decision=llm_decision,
    )

    kinds = [type(m).__name__ for m in state["messages"]]
    assert kinds == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]

    decide_input = prompts[0][1].content
    assert "<ticket" in decide_input
    assert "possible_duplicate_transaction_ids" in decide_input
    assert "Evidence summary" in decide_input
    assert "reply_language: en" in decide_input

    assert state["triage"] == llm_decision
    assert state["final"].action is NextAction.ROUTE_TO_SPECIALIST
    assert Team.BILLING in state["final"].route_to
    assert state["policy_overrides"] == ["billing_never_auto_responds"]


def test_goes_straight_to_decide_when_llm_calls_no_tool(run, tools):
    state, prompts = run(iter(["Nothing to check"]), list(tools.values()))

    assert len(state["messages"]) == 3
    assert len(prompts) == 1
    assert "final" in state


def test_graph_holds_wrong_language_auto_reply_for_review(run, tools, make_result):
    decision = make_result(draft_reply="สวัสดีครับ ลองใช้เว็บแทนครับ")

    state, prompts = run(iter(["Evidence summary"]), list(tools.values()), decision)

    assert "reply_language: en" in prompts[0][1].content
    assert state["triage"] == decision
    assert state["triage"].action is NextAction.AUTO_RESPOND
    assert state["final"].action is NextAction.ROUTE_TO_SPECIALIST
    assert state["final"].route_to == [Team.GENERAL_SUPPORT]
    assert state["policy_overrides"] == ["wrong_reply_language_needs_review"]


def test_crashing_tool_becomes_safe_error_message(run):
    @tool
    def flaky_tool() -> str:
        """Always crashes."""
        raise RuntimeError("database password is hunter2")

    state, _ = run(iter([_ask_for_tool("flaky_tool"), "Done"]), [flaky_tool])

    error = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "failed unexpectedly" in error.content
    assert "hunter2" not in error.content


def test_recursion_limit_stops_an_agent_that_never_stops_calling_tools(run, tools):
    endless = (_ask_for_tool("get_customer_profile", customer_id="C-001") for _ in repeat(None))

    with pytest.raises(GraphRecursionError):
        run(endless, list(tools.values()))
