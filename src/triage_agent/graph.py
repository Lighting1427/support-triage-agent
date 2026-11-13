from collections.abc import Callable, Sequence

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from triage_agent.nodes import (
    ChatRunnable,
    DecisionRunnable,
    handle_tool_error,
    make_decide_node,
    make_investigate_node,
    preprocess,
)
from triage_agent.state import TriageState


def build_graph(
    *,
    investigator: ChatRunnable,
    decider: DecisionRunnable,
    tools: Sequence[BaseTool],
    apply_policy: Callable[[TriageState], dict],
) -> CompiledStateGraph:
    graph = StateGraph(TriageState)

    graph.add_node("preprocess", preprocess)
    graph.add_node("investigate", make_investigate_node(investigator))
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_error))
    graph.add_node("decide", make_decide_node(decider))
    graph.add_node("apply_policy", apply_policy)

    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "investigate")
    graph.add_conditional_edges("investigate", tools_condition, {"tools": "tools", END: "decide"})
    graph.add_edge("tools", "investigate")
    graph.add_edge("decide", "apply_policy")
    graph.add_edge("apply_policy", END)

    return graph.compile()
