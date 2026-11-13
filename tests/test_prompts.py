from langchain_core.messages import AIMessage, ToolMessage

from triage_agent.prompts import render_evidence


def test_render_evidence_lists_calls_results_and_summary_in_order():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_billing_transactions", "args": {"customer_id": "C-001"}, "id": "c1"}
            ],
        ),
        ToolMessage(
            content='{"captured_total": "59.98"}',
            name="get_billing_transactions",
            tool_call_id="c1",
        ),
        AIMessage(content="Two duplicate captured charges."),
    ]

    text = render_evidence(messages)

    assert text.startswith("<evidence>") and text.endswith("</evidence>")
    call = text.index('Tool call: get_billing_transactions({"customer_id": "C-001"})')
    result = text.index('Result of get_billing_transactions:\n{"captured_total": "59.98"}')
    summary = text.index("Investigator summary:\nTwo duplicate captured charges.")
    assert call < result < summary