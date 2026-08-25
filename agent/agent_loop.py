"""Core agent loop: combines retrieval, function calling, safety enforcement, and session state."""

import re
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from agent.config import (
    GENERATION_MODEL,
    get_gemini_api_key,
)
from agent.logging_utils import log_turn
from agent.orders import get_order_store
from agent.prompts import SYSTEM_PROMPT, build_user_prompt_with_context
from agent.retriever import Retriever, format_context_for_prompt
from agent.safety import (
    check_action_hallucinations,
    check_sensitive_data_leaks,
    detect_handoff_signal,
    sanitize_response_text,
)
from agent.session import ConversationTurn, Session, get_session_manager
from agent.tools import execute_tool, get_agent_tools


@dataclass
class AgentResponse:
    """Structured response from the agent."""
    text: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    handoff_recommended: bool = False
    conflict_detected: bool = False
    session_id: str = ""


class SupportAgent:
    """The Aster & Row RAG Support Agent."""

    def __init__(self, retriever: Retriever | None = None):
        self.client = genai.Client(api_key=get_gemini_api_key())
        self.retriever = retriever or Retriever()
        self.session_manager = get_session_manager()
        self.order_store = get_order_store()

    def _build_retrieval_query(self, user_message: str, session: Session) -> str:
        """Create a contextualized retrieval query for follow-up turns."""
        if not session.turns:
            return user_message

        # Check if the user message looks like a short follow-up
        lower = user_message.lower().strip()
        is_short_followup = (
            len(user_message.split()) <= 6
            or lower.startswith(("what about", "how about", "and ", "why", "can i", "is it", "what if"))
        )

        if is_short_followup and session.turns:
            # Prepend the previous user message for topical context
            prev_turn = session.turns[-1]
            return f"{prev_turn.user_message} {user_message}"

        return user_message

    def _generate_with_retry(
        self, contents: list[types.Content], config: types.GenerateContentConfig, max_retries: int = 6
    ) -> Any:
        """Call Gemini generate_content with automatic exponential backoff on 429 quota exhaustion."""
        delay = 8.0
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_str = str(e)
                if ("RESOURCE_EXHAUSTED" in err_str or "429" in err_str) and attempt < max_retries - 1:
                    # Try to extract the suggested wait time from the error message
                    suggested_delay = None
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if not m:
                        m = re.search(r"retryDelay': '(\d+)s", err_str, re.IGNORECASE)
                    if m:
                        suggested_delay = float(m.group(1)) + 1.0

                    wait_time = suggested_delay if suggested_delay is not None else delay
                    time.sleep(wait_time)
                    delay = max(delay * 1.5, 12.0)
                else:
                    raise

    def chat(self, user_message: str, session_id: str | None = None) -> AgentResponse:
        """Process a single conversational turn."""
        start_time = time.time()
        session = self.session_manager.get_or_create(session_id)
        turn_index = len(session.turns) + 1

        # 1. Retrieval
        retrieval_query = self._build_retrieval_query(user_message, session)
        retrieved_results, conflict_detected = self.retriever.retrieve(retrieval_query)
        context_str = format_context_for_prompt(retrieved_results, conflict_detected)
        sources = [r.source_citation for r in retrieved_results]

        # 2. Build current turn user prompt
        prompt_with_context = build_user_prompt_with_context(user_message, context_str)

        # 3. Assemble conversation contents for Gemini
        contents: list[types.Content] = []
        for h in session.history:
            parts = []
            for p in h.get("parts", []):
                if "text" in p:
                    parts.append(types.Part.from_text(text=p["text"]))
            contents.append(types.Content(role=h["role"], parts=parts))

        # Add current user turn
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt_with_context)],
            )
        )

        # 4. Generate with Gemini + Tools
        tools = get_agent_tools()
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
            temperature=0.1,  # Low temperature for groundedness and consistency
        )

        tool_calls_record: list[dict[str, Any]] = []

        response = self._generate_with_retry(
            contents=contents,
            config=config,
        )

        # 5. Handle Tool / Function Calling loop
        function_calls = response.function_calls
        if function_calls:
            for call in function_calls:
                tool_name = call.name
                tool_args = dict(call.args) if call.args else {}
                tool_result = execute_tool(tool_name, tool_args)
                tool_calls_record.append({
                    "name": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                })

                # Append model's function call message to history
                contents.append(response.candidates[0].content)

                # Append function response
                tool_response_part = types.Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_result},
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[tool_response_part],
                    )
                )

            # Get final model response after tool execution
            response = self._generate_with_retry(
                contents=contents,
                config=config,
            )

        final_text = response.text or ""

        # 6. Deterministic Safety & Guardrail Inspections
        sensitive_violations = check_sensitive_data_leaks(final_text)
        if sensitive_violations:
            final_text = sanitize_response_text(final_text)

        action_violations = check_action_hallucinations(final_text)
        if action_violations:
            final_text += "\n\n(Note: As an automated assistant, I cannot directly perform account or order modifications. Please contact our support team to complete this request.)"

        # 7. Check if Handoff is Recommended
        handoff_recommended = detect_handoff_signal(final_text) or conflict_detected

        # Check if the lookup result itself had an exception or not found
        for tc in tool_calls_record:
            res = tc.get("result", {})
            if not res.get("success") or res.get("order", {}).get("status") == "exception":
                handoff_recommended = True

        # 8. Update Session History
        # We record clean user message in conversational history to keep context clean
        session.add_user_message(user_message)
        session.add_model_message(final_text)

        turn = ConversationTurn(
            user_message=user_message,
            agent_response=final_text,
            retrieved_sources=sources,
            tool_calls=tool_calls_record,
            handoff_recommended=handoff_recommended,
            conflict_detected=conflict_detected,
        )
        session.add_turn(turn)

        # 9. Observability logging
        elapsed_ms = (time.time() - start_time) * 1000
        retrieved_chunk_dicts = [
            {
                "filename": r.filename,
                "heading_path": r.heading_path,
                "score": r.score,
                "status": r.status,
            }
            for r in retrieved_results
        ]
        log_turn(
            session_id=session.session_id,
            turn_index=turn_index,
            user_message=user_message,
            retrieved_chunks=retrieved_chunk_dicts,
            tool_calls=tool_calls_record,
            final_response=final_text,
            handoff_recommended=handoff_recommended,
            conflict_detected=conflict_detected,
            latency_ms=elapsed_ms,
        )

        return AgentResponse(
            text=final_text,
            sources=sources,
            tool_calls=tool_calls_record,
            handoff_recommended=handoff_recommended,
            conflict_detected=conflict_detected,
            session_id=session.session_id,
        )
