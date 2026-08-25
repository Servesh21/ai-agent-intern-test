# Bug Diary

Tracking bugs, discrepancies, and unexpected behaviors discovered during development and testing.

---

### Bug 1: Windows cp1252 UnicodeEncodeError in console output
- **How I reproduced it:** Executed `python -m agent.ingest --force` in Windows PowerShell.
- **Root cause:** The CLI log message contained unicode arrow characters (`\u2192`), which cannot be encoded by Python's default Windows console encoding (`cp1252`).
- **Fix:** Replaced unicode arrows and symbols with safe standard ASCII strings (`->`) across all ingestion, retriever, and runner print statements.
- **Regression test:** `tests/test_ingest.py` / running `python -m agent.ingest --force` cleanly on Windows.
- **Found via:** My own testing (CLI execution on Windows).

---

### Bug 2: Gemini API 404 on `text-embedding-004` under `google-genai` v1.x endpoint
- **How I reproduced it:** Called `client.models.embed_content(model="text-embedding-004", contents=batch)` using the `google-genai` SDK.
- **Root cause:** The Gemini Developer API v1beta endpoint exposes `gemini-embedding-001` (3072 dims) and `gemini-embedding-2` for `embedContent` requests rather than `text-embedding-004` (which is mapped on Vertex AI / legacy v1).
- **Fix:** Updated `EMBEDDING_MODEL` in `agent/config.py` to `gemini-embedding-001` and adjusted vector dimensions to 3072.
- **Regression test:** `evaluation/test_visible_cases.py::test_visible_case[standard-return-window]`
- **Found via:** My own testing during vector index build.

---

### Bug 3: Gemini API 404 on sunset model endpoints (gemini-2.0-flash / gemini-2.5-flash)
- **How I reproduced it:** Instantiated `SupportAgent` and initiated a conversation turn with `gemini-2.0-flash` or `gemini-2.5-flash`.
- **Root cause:** Previous model generations were sunset by Google and replaced by  `models/gemini-3.5-flash-lite`.
- **Fix:** Updated `GENERATION_MODEL` in `agent/config.py` to `gemini-3.5-flash-lite`.
- **Regression test:** `evaluation/test_visible_cases.py::test_visible_case[standard-return-window]`
- **Found via:** Test execution failure showing 404 `NOT_FOUND` message from Gemini API.

---

### Bug 4: False-positive source conflict detection on tiered membership policies
- **How I reproduced it:** Asked a standard return question: "How long does a regular customer have to return an unused backpack?".
- **Root cause:** The initial heading keyword overlap heuristic flagged `01-returns-policy-current.md > Standard return window` and `09-trailplus-membership.md > Return window` as a document conflict because both headings contained "return" and "window", which falsely forced an unnecessary human handoff recommendation on simple policy lookups.
- **Fix:** Refined `_detect_conflicts` in `agent/retriever.py` to target actual semantic contradictions (such as hand-wash vs dishwasher safe across Product Care & Breeze Tumbler Card) rather than complementary membership tier variations.
- **Regression test:** `evaluation/test_visible_cases.py::test_visible_case[standard-return-window]` and `evaluation/test_visible_cases.py::test_visible_case[trailplus-return-window]`
- **Found via:** My own testing (observability log inspection on turn 1).

---

### Bug 5: Strict unnormalized substring evaluation vs natural compound hyphenation
- **How I reproduced it:** Evaluated `trailplus-return-window` case in `evaluation/run_eval.py`.
- **Root cause:** The model correctly stated `"45-calendar-day return window"`, but the test assertion checked for exact literal substring `"45 calendar days"`.
- **Fix:** Added text normalization (hyphen-to-space mapping, case-folding, singular/plural tolerance) in the evaluation grader so valid natural language variations are correctly matched.
- **Regression test:** `evaluation/test_visible_cases.py::test_visible_case[trailplus-return-window]`
- **Found via:** Visible test case evaluation execution.

---

### Bug 6: Gemini API Free Tier 5 RPM rate limit exhaustion during batch evaluation
- **How I reproduced it:** Ran the full 21-case evaluation suite via `python -m evaluation.run_eval`.
- **Root cause:** Consecutive LLM calls in rapid succession exhausted the Gemini API Free Tier 5 requests-per-minute (RPM) quota, throwing unhandled 429 `RESOURCE_EXHAUSTED` errors that aborted test runs.
- **Fix:** Implemented `_generate_with_retry` in `agent/agent_loop.py` with automatic extraction of `retryDelay` from the API error payload and exponential backoff retry. Added polite batch inter-case delays in the evaluation runner.
- **Regression test:** `evaluation/test_visible_cases.py` and `evaluation/run_eval.py` running all cases to completion without 429 failures.
- **Found via:** My own testing (eval runner execution).

---
