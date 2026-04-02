import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agents.agent import create_agent, run_query, _build_dynamic_context, SYSTEM_PROMPT, save_memory
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logging.root.addHandler(_handler)
logger = logging.getLogger("agent_health.api")
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = create_agent()
    await agent._ensure_initialized()
    logger.info("MCP servers connected, health agent ready")
    await MongoDB.ensure_indexes()
    yield
    await agent._disconnect_mcp()
    await MongoDB.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Health & Fitness Agent API",
    description="AI-powered everyday health and fitness companion with workout plans, nutrition guidance, and long-term progress tracking.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


# ── Request/Response models ──

class AskRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None

    model_config = {"json_schema_extra": {"examples": [
        {"query": "Create a 4-week beginner workout plan", "session_id": None}
    ]}}


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


class HealthProfileRequest(BaseModel):
    goals: str = Field(
        default="",
        description="Primary fitness goal: weight loss, muscle gain, endurance, general fitness, etc.",
    )
    fitness_level: str = Field(
        default="beginner",
        description="Self-assessed fitness level: beginner, intermediate, or advanced.",
    )
    available_equipment: list[str] = Field(
        default_factory=list,
        description="Equipment available (e.g. ['dumbbells', 'resistance bands', 'gym access']).",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions or preferences (e.g. ['vegetarian', 'gluten-free']).",
    )
    injuries_or_limitations: str = Field(
        default="",
        description="Any injuries, chronic pain, or physical limitations to account for.",
    )
    age: int | None = Field(default=None, description="User's age in years.")
    weight_kg: float | None = Field(default=None, description="Current weight in kilograms.")
    height_cm: float | None = Field(default=None, description="Height in centimetres.")
    sessions_per_week: int | None = Field(default=None, description="Preferred workout sessions per week.")
    minutes_per_session: int | None = Field(default=None, description="Minutes available per workout session.")


class HealthProfileResponse(BaseModel):
    user_id: str
    goals: str
    fitness_level: str
    available_equipment: list[str]
    dietary_restrictions: list[str]
    injuries_or_limitations: str
    age: int | None
    weight_kg: float | None
    height_cm: float | None
    sessions_per_week: int | None
    minutes_per_session: int | None


# ── Standard agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    logger.info(
        "POST /ask — session='%s' (%s), user='%s', query='%s'",
        session_id, "new" if is_new else "existing", user_id or "anonymous", body.query[:100],
    )

    result = await run_query(
        body.query,
        session_id=session_id,
        response_format=body.response_format,
        model_id=body.model_id,
        user_id=user_id,
    )
    response = result["response"]
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=body.query,
        response=response,
        steps=steps,
        user_id=user_id,
    )

    logger.info(
        "POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
        session_id, len(response),
        sum(1 for s in steps if s.get("action") == "tool_call"),
    )

    return AskResponse(session_id=session_id, query=body.query, response=response)


@app.post("/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(body: AskRequest, request: Request):
    """Stream the agent's response as Server-Sent Events (SSE)."""
    user_id = request.headers.get("X-User-Id") or None
    session_id = body.session_id or MongoDB.generate_session_id()
    logger.info(
        "POST /ask/stream — session='%s', user='%s', query='%s'",
        session_id, user_id or "anonymous", body.query[:100],
    )

    dynamic_context = await _build_dynamic_context(
        session_id, body.query, response_format=body.response_format, user_id=user_id
    )
    enriched_query = dynamic_context + body.query
    agent = create_agent()
    stream = agent.astream(
        enriched_query,
        session_id=session_id,
        system_prompt=SYSTEM_PROMPT,
        model_id=body.model_id,
    )

    _incoming_request_id = request.headers.get("X-Request-ID")

    async def event_stream():
        try:
            full_response: list[str] = []
            queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

            async def _stream_producer():
                try:
                    async for chunk in stream:
                        await queue.put(("chunk", chunk))
                    await queue.put(("done", None))
                except Exception as exc:
                    logger.error("Stream producer failed: %s", exc)
                    await queue.put(("error", str(exc)))

            async def _keepalive_producer():
                while True:
                    await asyncio.sleep(15)
                    await queue.put(("keepalive", None))

            producer_task = asyncio.create_task(_stream_producer())
            keepalive_task = asyncio.create_task(_keepalive_producer())

            try:
                while True:
                    kind, data = await queue.get()
                    if kind == "chunk":
                        full_response.append(data)
                        yield f"data: {json.dumps({'text': data})}\n\n"
                    elif kind == "keepalive":
                        yield ": keep-alive\n\n"
                    elif kind == "error":
                        error_msg = "An error occurred processing your request. Please try again or switch to a different model."
                        yield f"data: {json.dumps({'text': error_msg})}\n\n"
                        break
                    elif kind == "done":
                        break
            finally:
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass
                await producer_task

            response_text = "".join(full_response)

            if not response_text.strip():
                fallback = "Sorry, the model returned an empty response. Please try again or switch to a different model."
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                response_text = fallback

            try:
                save_memory(user_id=user_id or session_id, query=body.query, response=response_text)
                await MongoDB.save_conversation(
                    session_id=session_id,
                    query=body.query,
                    response=response_text,
                    steps=stream.steps if hasattr(stream, "steps") else [],
                    user_id=user_id,
                )
            except Exception as e:
                logger.error("Failed to save memory/conversation: %s", e)

            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/user/me", response_model=HistoryResponse)
async def get_history_by_user(http_request: Request):
    user_id = http_request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    logger.info("GET /history/user/me — user='%s'", user_id)
    history = await MongoDB.get_history_by_user(user_id)
    return HistoryResponse(session_id=user_id, history=history)


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


# ── Health profile endpoints ──

@app.post("/profile", response_model=HealthProfileResponse, status_code=status.HTTP_200_OK)
@limiter.limit("20/minute")
async def save_profile(body: HealthProfileRequest, request: Request):
    """Create or update a user's health profile.

    Requires the X-User-Id header to identify the user. The profile is automatically
    injected into every subsequent /ask request for that user.
    """
    user_id = request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-User-Id header is required to save a health profile.")

    profile_data = body.model_dump()
    await MongoDB.save_profile(user_id=user_id, profile=profile_data)
    logger.info("POST /profile — user='%s', goals='%s'", user_id, body.goals[:80] if body.goals else "")

    return HealthProfileResponse(user_id=user_id, **profile_data)


@app.get("/profile", response_model=HealthProfileResponse)
async def get_profile(request: Request):
    """Retrieve a user's stored health profile.

    Requires the X-User-Id header.
    """
    user_id = request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-User-Id header is required.")

    profile = await MongoDB.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="No health profile found for this user. Create one via POST /profile.")

    logger.info("GET /profile — user='%s'", user_id)
    return HealthProfileResponse(
        user_id=user_id,
        goals=profile.get("goals", ""),
        fitness_level=profile.get("fitness_level", "beginner"),
        available_equipment=profile.get("available_equipment", []),
        dietary_restrictions=profile.get("dietary_restrictions", []),
        injuries_or_limitations=profile.get("injuries_or_limitations", ""),
        age=profile.get("age"),
        weight_kg=profile.get("weight_kg"),
        height_cm=profile.get("height_cm"),
        sessions_per_week=profile.get("sessions_per_week"),
        minutes_per_session=profile.get("minutes_per_session"),
    )


# ── Plan export endpoint ──

@app.get("/export/plan/{session_id}")
async def export_plan(session_id: str):
    """Download the most recently generated fitness plan for a session."""
    file_meta = await MongoDB.get_latest_plan(session_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="No fitness plan found for this session. Ask the agent to generate one first.")

    result = await MongoDB.retrieve_file(file_meta["file_id"])
    if not result:
        raise HTTPException(status_code=404, detail="Plan file not found in storage.")

    data, meta = result
    filename = meta.get("filename", "fitness-plan.pdf")

    if filename.endswith(".pdf"):
        media_type = "application/pdf"
    else:
        media_type = "text/markdown"

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download any generated file by its file_id (from generate_fitness_plan tool output)."""
    result = await MongoDB.retrieve_file(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="File not found.")

    data, meta = result
    filename = meta.get("filename", "download")

    if filename.endswith(".pdf"):
        media_type = "application/pdf"
    elif filename.endswith(".md"):
        media_type = "text/markdown"
    else:
        media_type = "application/octet-stream"

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-health"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9005))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
