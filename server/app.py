"""FastAPI backend providing a web UI and /api/chat endpoint."""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent_loop import SupportAgent
from agent.ingest import build_index

app = FastAPI(title="Aster & Row Support Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
agent: SupportAgent | None = None


@app.on_event("startup")
def startup_event():
    global agent
    build_index(force_rebuild=False)
    agent = SupportAgent()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    text: str
    sources: list[str]
    tool_calls: list[dict]
    handoff_recommended: bool
    conflict_detected: bool
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    resp = agent.chat(req.message, session_id=req.session_id)
    return ChatResponse(
        text=resp.text,
        sources=resp.sources,
        tool_calls=resp.tool_calls,
        handoff_recommended=resp.handoff_recommended,
        conflict_detected=resp.conflict_detected,
        session_id=resp.session_id,
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Aster & Row Support Agent API</h1><p>Visit /docs for API documentation.</p>")
