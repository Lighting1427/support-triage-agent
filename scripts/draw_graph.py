from datetime import date
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import RunnableLambda

from triage_agent.data import KnowledgeBase, MockDataStore
from triage_agent.graph import build_graph
from triage_agent.policy import make_apply_policy_node
from triage_agent.tools import build_tools

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    store = MockDataStore(DATA_DIR)
    kb = KnowledgeBase.from_file(DATA_DIR / "knowledge_base.json")
    tools = build_tools(store, kb, today=date.today())
    graph = build_graph(
        investigator=GenericFakeChatModel(messages=iter([])),
        decider=RunnableLambda(lambda _: None),
        tools=tools,
        apply_policy=make_apply_policy_node(store, kb.article_ids),
    )
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()