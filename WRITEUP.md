# Write-up: Support Ticket Triage Agent

## Architecture decisions

- **LangGraph with explicit nodes.** The graph separates LLM steps from deterministic
  code, so each concern can be tested and changed alone: `preprocess` (code) →
  `investigate` ⇄ `tools` (ReAct) → `decide` (LLM) → `apply_policy` (code).
- **Investigate and decide are separate LLM calls.** Investigate only gathers evidence
  with tools. Decide gets a fresh context (ticket + evidence) and returns a strict
  `TriageResult` via structured outputs. Simpler prompts and a guaranteed schema cost
  one extra call per ticket; the two steps could also use different model sizes.
- **Tools return verified, pre-computed facts.** Duplicate charges, totals, tenure and a
  status-page mismatch flag are computed in code, because money and dates must not
  depend on LLM arithmetic. Tools are read-only: the agent decides, humans act.
- **Business rules live in code, not in the prompt.** Policy rules can only make a
  decision safer (billing never auto-responds, critical always escalates, invented KB
  IDs are dropped). They use system data (CRM plan, real KB IDs), not LLM claims, and
  every override is recorded.
- **BM25 retrieval.** The KB is small, the LLM writes English queries and can retry, and
  the retrieval tests rank the right article first. Upgrade to hybrid search
  (BM25 + embeddings, Reciprocal Rank Fusion) when the KB grows or recall drops.
- **Fail-safe by default.** Any failure (API error, loop limit, invalid output) returns
  `escalate_to_human` with the error. Models are injected, so all tests run with fakes.

## What could go wrong

| Risk | Handling now | Next step |
|---|---|---|
| Under-triage (urgent ticket handled as routine) | Impact-based rubric; guardrails escalate critical, billing and legal threats | Track critical recall as the main metric |
| Hallucinated facts, KB articles or promises | Facts come from tools; unknown KB IDs dropped; prompt forbids unsupported promises | LLM-as-judge on every draft reply |
| Prompt injection inside a ticket | Ticket text wrapped in `<ticket>` and treated as data; tools are read-only | Injection cases in the golden set |
| Tool errors, API outages, endless loops | Errors returned to the model as guidance; `recursion_limit`; fail-safe escalation | Retries with backoff; alerting |
| Wrong language / lost nuance | Reply language detected in code from the customer's text; a wrong-language reply is never auto-sent | Language-detection library; multilingual KB search |
| Stale data (status page says "operational") | Prompt and tool favour internal monitoring and flag mismatches | Open the incident automatically |
| Sensitive data sent to the LLM | Tool crashes never expose internals to the model | Redact card numbers and emails first |

## Evaluating in production

- **Offline (before every prompt or model change).** Golden set of labelled tickets
  (`scripts/run_evals.py`): urgency, action, teams, issue types, KB citations, tools used
  and reply language. Run several times to measure consistency, and gate CI on it.
  Weight under-triage far more than over-triage. Grow the set from real tickets and
  every production failure.
- **Reply quality.** LLM-as-judge with a rubric (accurate, grounded in KB, no
  unsupported promises, right language and tone), calibrated against human ratings.
- **Shadow mode first.** The agent triages live tickets but humans decide. Compare per
  field before enabling auto-respond, and then only for low-risk categories.
- **Online metrics.** Human override rate per field, policy override rate (a high rate
  means the prompt needs work), auto-reply reopen rate and CSAT, time to first
  response, escalation precision, and cost/latency per ticket.
- **Observability.** Trace every run (tool calls, evidence, both decisions) with
  LangSmith, and monitor drift in the urgency and action distribution.
