# Support Ticket Triage Agent
# Support Ticket Triage Agent

A LangGraph agent that triages customer-support tickets with an OpenAI GPT model.
For each ticket it classifies urgency, extracts key information, searches a knowledge
base, checks customer/billing/system data with tools, and decides the next action:
**auto-respond**, **route to a specialist**, or **escalate to a human**.

| Requirement | Where it lives |
|---|---|
| 1. Classify urgency | `TriageResult.urgency` — rubric in `DECIDE_SYSTEM_PROMPT` |
| 2. Extract product, issue type, sentiment | `TriageResult.extracted` (`models.py`) |
| 3. Search a knowledge base | `search_knowledge_base` tool (BM25 over `data/knowledge_base.json`) |
| 4. Decide next action | `decide` node + deterministic guardrails in `policy.py` |
| 5. At least 2 tools | 4 tools in `tools.py` |
| System prompt | `src/triage_agent/prompts.py` (`INVESTIGATE_SYSTEM_PROMPT`, `DECIDE_SYSTEM_PROMPT`) |
| Tool definitions (schema + implementation) | `src/triage_agent/tools.py` — print the exact schemas with `python scripts/inspect_tools.py` |
| Write-up | [`WRITEUP.md`](WRITEUP.md) |

## Quick start

Requires Python 3.11+ and an OpenAI API key.

```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then set OPENAI_API_KEY in .env
python -m triage_agent
```

```cmd
:: Windows (cmd)
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
:: then set OPENAI_API_KEY in .env
python -m triage_agent
```

## Usage

| Command | What it does |
|---|---|
| `python -m triage_agent` | Triage all sample tickets, readable console output |
| `python -m triage_agent --ticket T-1002` | Triage one ticket (repeatable flag) |
| `python -m triage_agent --json` | Machine-readable output (`TicketOutcome` list) |
| `python -m triage_agent --json --output results.json` | Save results as a UTF-8 file (use this instead of shell redirection, which can garble Thai text on Windows) |
| `python -m triage_agent -v` | Also log every tool call as it happens |
| `python scripts/run_evals.py --runs 3` | Score the agent against `evals/golden_set.json` |
| `python scripts/draw_graph.py` | Print the graph as Mermaid (no API key needed) |
| `python scripts/inspect_tools.py` | Print tool schemas and sample tool outputs (no API key needed) |
| `pytest` | Full test suite, no API key needed (LLMs replaced by fakes) |

Configuration (`.env`):

Set your API key and model in `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-6-luna
```

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for the agent and evals |
| `OPENAI_MODEL` | `gpt-6-luna` | Model name passed to OpenAI; must support tool calling and structured outputs |
| `TRIAGE_DATA_DIR` | `./data` | Folder with the mock data |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | off | Optional: full traces in LangSmith |

## Architecture

```mermaid
flowchart LR
    S([start]) --> P["preprocess<br/>code"]
    P --> I["investigate<br/>LLM + tools"]
    I -- tool call --> T["tools<br/>code"]
    T --> I
    I -- evidence summary --> D["decide<br/>LLM, structured output"]
    D --> A["apply_policy<br/>code guardrails"]
    A --> E([end])
```

| Node | Type | Responsibility |
|---|---|---|
| `preprocess` | code | Compute wait time, unanswered messages and reply language; render the ticket for the LLM |
| `investigate` ⇄ `tools` | LLM + code | ReAct loop: the model chooses tools, tools return verified facts |
| `decide` | LLM | Reads ticket + evidence in a fresh context, returns a strict `TriageResult` |
| `apply_policy` | code | Business rules that can only make a decision safer (e.g. billing never auto-responds) |

Tools (all read-only, mocked from `data/`):

| Tool | Returns |
|---|---|
| `get_customer_profile` | Plan, seats, region, language, account manager, history, tenure |
| `search_knowledge_base` | Top 3 KB articles (BM25) |
| `get_billing_transactions` | Charges, totals, possible duplicates, provisioning errors |
| `check_service_status` | Internal monitoring per region, plus a status-page mismatch flag |

Any failure (API error, loop limit, invalid output) returns `escalate_to_human` with the
error, so a ticket is never dropped. See [`WRITEUP.md`](WRITEUP.md) for the reasoning.

## Project structure

```
data/                  mock tickets, customers, billing, service status, knowledge base
evals/golden_set.json  expected triage per sample ticket
scripts/               run_evals.py, draw_graph.py, inspect_tools.py
src/triage_agent/
  models.py            Pydantic models: input, tool data, LLM output schema, app output
  language.py          Thai/English script checks shared by policy and evals
  data.py              mock data store + BM25 knowledge base
  tools.py             the 4 tools and their deterministic logic
  prompts.py           system prompts and ticket/evidence rendering
  state.py             LangGraph state
  nodes.py             preprocess, investigate, decide nodes
  policy.py            guardrail rules + apply_policy node
  graph.py             graph wiring
  llm.py               OpenAI model clients
  pipeline.py          composition root + fail-safe runner
  evaluation.py        golden-set checks
  main.py              CLI
tests/                 unit + graph tests with fake LLMs
```

## Expected triage for the sample tickets

From `evals/golden_set.json` (what a correct agent should decide):

| Ticket | Urgency | Action | Teams | Why |
|---|---|---|---|---|
| T-1001 | high (critical accepted) | escalate_to_human | billing | 2 duplicate captured charges + 1 pending; a webhook timeout left the plan on Free; chargeback threat and a 2-hour deadline. The reply must not promise a refund. |
| T-1002 | critical | escalate_to_human | sre_oncall, account_manager | 38% 5xx errors in `asia-southeast` while the public status page says operational; enterprise account with 45 seats. Reply in Thai. |
| T-1003 | low | auto_respond | engineering, product | Known macOS v4.1 bug (workaround: web app), a dark toggle is on the roadmap, and scheduled dark mode is a feature request. |

## Results

Golden-set eval, 3 runs per ticket with `gpt-6-luna` (`python scripts/run_evals.py --runs 3`):

| Metric | Result |
|---|---|
| Checks passed | 72/72 |
| Correct action, LLM alone | 9/9 |
| Correct action, after guardrails | 9/9 |
| Policy overrides | 0 of 9 |

Full outputs: [`docs/agent_output.json`](docs/agent_output.json),
[`docs/sample_output.txt`](docs/sample_output.txt), [`docs/eval_results.txt`](docs/eval_results.txt).
The guardrails did not need to fire on these tickets; their behaviour is covered by
unit tests in `tests/test_policy.py`.

What the evals caught while building (and how each was fixed):

- T-1003 rated medium: the urgency rubric itself was ambiguous → clarified the rubric.
- T-1003 not auto-answered: the KB did not say whether theme scheduling exists →
  added the fact to KB-007 (a knowledge gap, not a model error).
- Wrong reply language: language was left to the LLM → detected in code, plus a
  guardrail that never auto-sends a reply in the wrong language.
- Internal notes written in Thai → prompt rule plus a new eval check.

## Assumptions

- Customer details are not part of the ticket; the agent must fetch them with tools.
- Mock data was designed to match each ticket's story (e.g. charge timestamps match the
  customer's messages; internal monitoring disagrees with the status page for T-1002).
- T-1003's first support reply ("Settings > Appearance") is inferred from the customer's
  second message and included in the conversation.
- The sample data uses a fixed "now": `reference_time = 2026-11-12T07:00:00Z`.
  Wait times are calculated from each message's timestamp against this value,
  not the computer's current clock. For example, a message sent at 04:00 UTC
  has a 3-hour wait at 07:00 UTC, regardless of when you run the demo.
  This keeps time calculations consistent; model responses can still vary.
- Reply language supports Thai and English: Thai characters in customer messages
  select `th`; otherwise `en`. Support messages are excluded. This is a script
  heuristic, not general language detection; mixed customer messages select Thai.
- An automatic reply with mismatched language (or no text) is held for review by
  general support. Drafts already routed to humans remain drafts for human review.
- The garbled emojis in the brief were replaced with 🙂 and 😄.
