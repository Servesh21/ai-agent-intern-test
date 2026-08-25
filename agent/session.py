"""Session and conversation history manager."""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    """Represents a single user-agent exchange."""
    user_message: str
    agent_response: str
    retrieved_sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    handoff_recommended: bool = False
    conflict_detected: bool = False


class Session:
    """A single conversation session maintaining isolated state."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.turns: list[ConversationTurn] = []
        # Raw Gemini Content objects for multi-turn model context
        self.history: list[dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        """Record user message in raw history."""
        self.history.append({
            "role": "user",
            "parts": [{"text": text}],
        })

    def add_model_message(self, text: str) -> None:
        """Record model response in raw history."""
        self.history.append({
            "role": "model",
            "parts": [{"text": text}],
        })

    def add_turn(self, turn: ConversationTurn) -> None:
        """Record a completed structured turn."""
        self.turns.append(turn)

    def get_last_n_turns(self, n: int = 5) -> list[ConversationTurn]:
        """Return the most recent N turns."""
        return self.turns[-n:]

    def clear(self) -> None:
        """Reset conversation history."""
        self.turns.clear()
        self.history.clear()


class SessionManager:
    """In-memory manager for isolated conversation sessions."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None = None) -> Session:
        """Get an existing session or create a new one."""
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> Session:
        """Reset an existing session."""
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def list_sessions(self) -> list[str]:
        """List active session IDs."""
        return list(self._sessions.keys())


# Singleton session manager
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create singleton SessionManager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
