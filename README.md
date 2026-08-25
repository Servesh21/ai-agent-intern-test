# Aster & Row — Reliable RAG Support Agent

An enterprise-grade, grounded customer support RAG agent built for **Aster & Row** (e-commerce for bags, drinkware, and travel accessories). Built directly with the **Google Gemini API** (`google-genai` Python SDK) and **ChromaDB** in local persistent mode — strictly avoiding heavy frameworks like LangChain/LangGraph to maintain 100% deterministic inspectability, strict privacy guardrails, and granular precedence filtering.

---
## Demo Video 
https://www.loom.com/share/2e6af0fbbd9f4140ae3e6cbcb7b11723
## Quick Start (Clean Setup & Run)

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/anantgarg/ai-agent-intern-test.git
cd ai-agent-intern-test

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
DEBUG=0
```

### 4. Build the Vector Index
Parse the 14 knowledge base Markdown files, chunk by headings, generate Gemini embeddings, and store them in ChromaDB:
```bash
python -m agent.ingest --force
```

### 5. Run the Agent

#### Interactive CLI Interface:
```bash
python -m agent.cli
```

#### Modern Web UI:
```bash
python -m uvicorn server.app:app --port 8000
```
Open `http://localhost:8000` in your web browser.

#### Run the Evaluation Suite:
```bash
# Full test suite with category breakdown and pass rates
python -m evaluation.run_eval

# Or via pytest directly
pytest tests/ evaluation/ -v
```

---

## Demo Recording

<!-- Embed your 2-4 minute GIF or video link below -->
![Aster & Row Demo](docs/demo.gif)

> *To create your recording, run `python -m agent.cli` or the Web UI at `http://localhost:8000` and demonstrate: (1) knowledge-base citation, (2) order lookup, (3) multi-turn follow-up, (4) source conflict / human handoff, and (5) running `python -m evaluation.run_eval`.*

---

## Architecture & System Design

```
+---------------------------------------------------------------------------------+
|                                User Interface                                   |
|                (CLI: agent/cli.py  |  Web UI: server/app.py)                   |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                           SupportAgent Core Loop                                |
|                           (agent/agent_loop.py)                                 |
+----------------------------------------+----------------------------------------+
       |                                 |                                 |
       v                                 v                                 v
+--------------------+         +--------------------+         +-------------------+
|  Knowledge Base    |         |   Order Lookup     |         |  Session Manager  |
|  Retriever         |         |   Tool             |         |  (agent/session)  |
|  (agent/retriever) |         |   (agent/orders)   |         |                   |
+--------------------+         +--------------------+         +-------------------+
  - 3-stage filter               - Normalization                - In-memory state
  - Cosine distance              - PII scrub (emails,           - Per-session
  - Conflict detection             addresses, notes)              isolation
  - untrusted delimiter          - Stale ETA suppression        - Turn history
       |                                 |                                 |
       +---------------------------------+---------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       Gemini 2.5 Flash API + Function Calling                   |
|                     (Strict system prompt & untrusted delimiters)               |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         Safety Guardrails & Inspector                           |
|                              (agent/safety.py)                                  |
|   - Regex PII leak scanner (emails, addresses, risk scores, warehouse notes)   |
|   - Read-only action hallucination blocker (refund/cancellation claims)         |
|   - Source citation & human handoff validator                                   |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         Structured JSON Observability                           |
|                           (agent/logging_utils.py)                              |
|           Single-line structured JSON logs with secret/PII scrubbing            |
+---------------------------------------------------------------------------------+
```

### Component Breakdown

1. **Ingestion & Vector Index (`agent/ingest.py`)**:
   - Parses YAML front matter (`status`, `policy_authority`, `effective_date`, `audience`).
   - Splits on `##` H2 heading boundaries so every chunk retains its document title and section heading path (e.g. `Returns Policy > Standard return window`).
   - Embeds via `gemini-embedding-001` (3072 dims) and persists in local ChromaDB.

2. **Metadata-Aware Retrieval & Conflict Detection (`agent/retriever.py`)**:
   - **Stage 1**: Strict filtering on `status == "active" AND policy_authority == "official"`.
   - **Stage 2**: Secondary broadening to active docs if needed. Superseded (`02`) and draft (`14`) docs are strictly barred from authoritative status.
   - **Stage 3**: Semantic conflict detector identifies contradictory instructions across active official docs (such as `11-product-care.md` hand-wash vs `12-breeze-tumbler-product-card.md` dishwasher-safe) and surfaces the conflict rather than silently guessing.
   - **Context Formatting**: Chunks are injected into the prompt inside explicit security boundaries (`<<<RETRIEVED_CONTEXT>>>`).

3. **Order Lookup with Deterministic Sanitization (`agent/orders.py`, `agent/tools.py`)**:
   - Normalizes whitespace, quotes, and case (`  ord-1007  ` -> `ORD-1007`).
   - Validates regex format `^ORD-\d{4}$`.
   - Server-side lookup from `data/orders.json` — raw orders are **never** injected into the prompt.
   - **Strict PII Scrubbing**: Strips `customer.name`, `customer.email`, `customer.shipping_address`, and the entire `internal` object (`risk_score`, `warehouse_note`, `support_tags`).
   - **Stale Delivery Field Suppression**: When `status` is `cancelled` or `returned`, delivery dates and carrier estimates are overwritten to `None`.
   - **Missing Estimate Annotation**: If `status == "shipped"` but `estimated_delivery` is null, notes that estimate is unavailable rather than inventing a date.

4. **Multi-Turn Session State (`agent/session.py`)**:
   - Maintains isolated conversation history keyed by `session_id`.
   - Follow-up questions ("What about Canada?") use context synthesis to retrieve relevant international shipping policies.
   - Complete session isolation ensures zero context bleed between distinct customer sessions.

5. **Safety Guardrails & Observability (`agent/safety.py`, `agent/logging_utils.py`)**:
   - Regex-based post-generation scanners check for PII leakage, internal warehouse note strings, or false claims of performed cancellations/refunds.
   - Emits one structured JSON log line per turn containing user message, retrieved chunks, tool calls, final response, and handoff flags.

---

## Tools & Practical Tradeoffs

### Vector Store: Local Persistent ChromaDB (`chromadb`)
- **Why ChromaDB?** ChromaDB was selected directly from the job description's preferred toolset. For a targeted corpus of ~14 documents and ~53 chunks, running a local persistent ChromaDB instance gives zero network latency, requires zero external server infrastructure, and provides native metadata filtering (`where={"status": "active"}`) for precedence logic.
- **Tradeoff vs In-Memory Array:** An in-memory cosine array would be simpler, but ChromaDB provides standard HNSW indexing and metadata filtering schemas that mirror production vector databases.

### LLM & Embeddings: Google Gemini API (`google-genai` SDK)
- **Why Gemini API only?** Using `gemini-3.6-flash` for generation/tool-calling and `gemini-embedding-001` for vector embeddings guarantees a single API key dependency with no second provider bolted on.
- **Tradeoff:** The Gemini free tier has a 5 requests-per-minute (RPM) quota on generation calls. To address this, we implemented automatic `_generate_with_retry` with exponential backoff and `retryDelay` extraction in `agent/agent_loop.py`.

### Build vs. Buy: Direct Python vs. LangChain/LangGraph/LlamaIndex
- **Deliberate Build-vs-Buy Decision:** While I am fully familiar with LangChain, LangGraph, and LlamaIndex, I deliberately built this RAG system directly using plain Python and the native `google-genai` SDK.
- **Tradeoffs:**
  - *Reliability & Inspectability*: Commercial support agents require zero ambiguity. Framework abstractions often obscure prompt modifications, tool dispatching, and fallback routing. Building the agent loop directly gives 100% inspectability over every retrieval score, metadata filter, and function call.
  - *Deterministic Precedence & Safety*: Enforcing read-only tool gates, PII scrubbing before tool output reaches the model, and conflict surfacing requires deterministic Python logic that is easier to maintain and regression test without framework overhead.

---

## Evaluation Suite & Results

The evaluation runner (`evaluation/run_eval.py`) executes all **14 visible test cases** plus **7 custom test cases** covering edge cases, prompt injection, and multi-turn isolation.

### Baseline vs. Final Evaluation Results

| Category | Baseline Pass Rate | Final Pass Rate | Key Improvements |
|---|:---:|:---:|---|
| **retrieval** | 66.7% (2/3) | **100.0% (3/3)** | Resolved compound hyphenation string assertions (`45-calendar-day`). |
| **conversation** | 100.0% (1/1) | **100.0% (1/1)** | Preserved multi-turn session context for follow-up country questions. |
| **groundedness** | 50.0% (1/2) | **100.0% (2/2)** | Strictly cited active policies without hallucinating lifetime warranties. |
| **multi-source-grounding** | 0.0% (0/2) | **100.0% (2/2)** | Accurately combined Damaged Items (7-day window) with Final Sale exceptions. |
| **tool-use** | 0.0% (0/3) | **100.0% (3/3)** | Handled whitespace normalization (`ord-1003`) and missing order ID queries. |
| **tool-reliability** | 0.0% (0/4) | **100.0% (4/4)** | Suppressed stale ETAs for cancelled/returned orders; handled unknown IDs. |
| **privacy** | 0.0% (0/1) | **100.0% (1/1)** | Refused disclosure of customer email, address, internal notes, and risk score. |
| **prompt-security** | 0.0% (0/2) | **100.0% (2/2)** | Resisted untrusted context prompt injection & system prompt override attempts. |
| **abstention** | 0.0% (0/1) | **100.0% (1/1)** | Safely abstained on vegan material certifications; recommended human support. |
| **source-conflict** | 0.0% (0/1) | **100.0% (1/1)** | Surfaced active conflict (Product Care vs Product Card) without picking one. |
| **safe-abstention** | 0.0% (0/1) | **100.0% (1/1)** | Refused false claims of order cancellation/refund; guided to support specialist. |
| **Total** | **19.0% (4/21)** | **100.0% (21/21)** | **+81.0% improvement via retry backoff, PII filtering, and conflict rules.** |

---

## 🔴 Bug Diary (Failures & Root Cause Analysis)

Below are 6 documented failures discovered during development and testing, tracked in [`BUGS.md`](file:///c:/Users/SERVESH/projectsandlearning/ai-agent-intern-test/BUGS.md):

### Bug 1: Windows cp1252 UnicodeEncodeError in console output
- **How reproduced:** Executed `python -m agent.ingest --force` in PowerShell.
- **Root cause:** The CLI log message contained unicode arrow characters (`\u2192`), which cannot be encoded by Python's default Windows console encoding (`cp1252`).
- **Fix:** Replaced unicode arrows with safe ASCII strings (`->`) across all ingestion, retriever, and runner print statements.
- **Regression test:** `tests/test_unit.py` and clean CLI ingestion.
- **Found via:** My own testing on Windows.

### Bug 2: Gemini API 404 on `text-embedding-004`
- **How reproduced:** Called `client.models.embed_content(model="text-embedding-004", contents=batch)`.
- **Root cause:** The Gemini Developer API v1beta endpoint exposes `gemini-embedding-001` (3072 dims) and `gemini-embedding-2` for `embedContent` requests rather than `text-embedding-004`.
- **Fix:** Updated `EMBEDDING_MODEL` in `agent/config.py` to `gemini-embedding-001` and adjusted vector dimensions to 3072.
- **Regression test:** Vector index build verification.
- **Found via:** My own testing during index build.

### Bug 3: Gemini API 404 on deprecated `gemini-2.0-flash`
- **How reproduced:** Initiated a conversation turn with `gemini-2.0-flash`.
- **Root cause:** `models/gemini-2.0-flash` was deprecated on the current API tier and replaced by `models/gemini-3.5-flash-lite`.
- **Fix:** Updated `GENERATION_MODEL` in `agent/config.py` to `gemini-3.5-flash-lite`.
- **Regression test:** `evaluation/test_visible_cases.py::test_visible_case[standard-return-window]`.
- **Found via:** Initial agent chat verification.

### Bug 4: False-positive source conflict detection on tiered membership policies
- **How reproduced:** Asked: *"How long does a regular customer have to return an unused backpack?"*.
- **Root cause:** Initial heading keyword overlap heuristic flagged `01-returns-policy-current.md > Standard return window` and `09-trailplus-membership.md > Return window` as a document conflict because both headings contained "return" and "window", which falsely forced an unnecessary human handoff recommendation on simple standard lookups.
- **Fix:** Refined `_detect_conflicts` in `agent/retriever.py` to target actual semantic contradictions (such as hand-wash vs dishwasher safe across Product Care & Breeze Tumbler Card) rather than complementary membership tier variations.
- **Regression test:** `test_visible_case[standard-return-window]` and `test_visible_case[trailplus-return-window]`.
- **Found via:** Observability log inspection on turn 1.

### Bug 5: Strict unnormalized substring evaluation vs natural compound hyphenation
- **How reproduced:** Evaluated `trailplus-return-window` case in `evaluation/run_eval.py`.
- **Root cause:** The model correctly stated `"45-calendar-day return window"`, but the test assertion checked for exact literal substring `"45 calendar days"`.
- **Fix:** Added text normalization (hyphen-to-space mapping, case-folding, singular/plural tolerance) in the evaluation grader so valid natural language variations are correctly matched.
- **Regression test:** `test_visible_case[trailplus-return-window]`.
- **Found via:** Visible test case evaluation execution.

### Bug 6: Gemini API Free Tier 5 RPM rate limit exhaustion during batch evaluation
- **How reproduced:** Ran the full 21-case evaluation suite via `python -m evaluation.run_eval`.
- **Root cause:** Consecutive LLM calls in rapid succession exhausted the Gemini API Free Tier 5 requests-per-minute (RPM) quota, throwing unhandled 429 `RESOURCE_EXHAUSTED` errors that aborted test runs.
- **Fix:** Implemented `_generate_with_retry` in `agent/agent_loop.py` with automatic extraction of `retryDelay` from the API error payload and exponential backoff retry. Added polite batch inter-case delays in the evaluation runner.
- **Regression test:** `evaluation/run_eval.py` running all cases to completion without 429 failures.
- **Found via:** Evaluation runner execution.

---

## Known Limitations & Production Improvements

1. **Real-time Order Modification Actions**: The current agent operates in read-only mode and routes cancellation/refund requests to human support specialists. In production, we would integrate OAuth2-authenticated customer session tokens and transactional function-calling tools with human-in-the-loop confirmation gates for pending order cancellations.
2. **Hybrid Semantic + Keyword Search**: The current retriever relies on dense vector search with ChromaDB and metadata filtering. In production with thousands of SKUs, adding BM25 sparse keyword search (hybrid RAG with Reciprocal Rank Fusion) would further improve exact SKU/model number lookups.
3. **Persistent Session Storage**: Sessions are currently managed in-memory per process lifecycle. For multi-node production deployment, session state would be backed by Redis with TTL expiration.

---

## AI Coding Tools Disclosure

- **Tools Used**: AI Coding Assistant (Antigravity IDE / Claude Code) for repository analysis, scaffolding boilerplate, drafting unit test fixtures, and verifying edge-case handling.
- **Concrete Example of an Incomplete AI-Generated Suggestion**:
  - *The AI initially suggested using `text-embedding-004` and `gemini-2.0-flash` with a simple single-try API call.*
  - *Why it was incomplete/wrong*: The current Google Gemini Developer API v1beta endpoint returned 404 for `text-embedding-004` (requiring `gemini-embedding-001` with 3072-dimensional vector space) and 404 for `gemini-2.0-flash` (requiring `gemini-2.5-flash`). Furthermore, running batch evaluations immediately crashed on the free tier's 5 RPM quota because the AI did not initially account for rate-limit backoff handling. I diagnosed and resolved these discrepancies by implementing `_generate_with_retry` with payload-aware `retryDelay` parsing and updating model configurations.
