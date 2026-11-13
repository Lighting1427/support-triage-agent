import json
from datetime import date
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool

from triage_agent.data import KnowledgeBase, MockDataStore
from triage_agent.tools import build_tools

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SAMPLE_CALLS = [
    ("get_customer_profile", {"customer_id": "C-002"}),
    ("get_billing_transactions", {"customer_id": "C-001"}),
    ("check_service_status", {"region": "Asia"}),
    ("search_knowledge_base", {"query": "dark mode macos system default"}),
    ("check_service_status", {"region": "europe"}), 
]


def main() -> None:
    store = MockDataStore(DATA_DIR)
    kb = KnowledgeBase.from_file(DATA_DIR / "knowledge_base.json")
    tools = {t.name: t for t in build_tools(store, kb, today=date(2026, 11, 12))}

    print("=" * 30, "1) TOOL DEFINITIONS (ส่งให้ LLM ทุกครั้ง)", "=" * 30)
    for t in tools.values():
        print(json.dumps(convert_to_openai_tool(t), indent=2))

    print("=" * 30, "2) TOOL RESULTS (LLM ได้อ่านหลังเรียก tool)", "=" * 30)
    for name, arguments in SAMPLE_CALLS:
        print(f"\n>>> {name}({arguments})")
        print(json.dumps(json.loads(tools[name].invoke(arguments)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()