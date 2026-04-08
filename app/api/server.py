import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from agent.graph import graph
from config.settings import API_SECRET_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Agent",
    version="1.0.0",
    description="LangGraph-powered customer support with RAG, tools, and escalation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    if key != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return key


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    user_id: Optional[str] = Field(
        default=None,
        description="Stable identifier for the user/session. Auto-generated if not provided."
    )


class ChatResponse(BaseModel):
    request_id: str
    user_id: str
    response: str
    intent: str
    confidence: float
    latency_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    """Lightweight liveness probe — used by load balancers and uptime monitors."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["agent"], dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest, request: Request):
    """
    Send a message to the customer support agent.

    Requires header:  X-API-Key: <your-key>
    """
    request_id = str(uuid.uuid4())
    user_id = req.user_id or str(uuid.uuid4())

    logger.info("request_id=%s user_id=%s message_len=%d", request_id, user_id, len(req.message))
    t0 = time.monotonic()

    try:
        result = graph.invoke({
            "user_id":    user_id,
            "message":    req.message,
            "intent":     "",
            "confidence": 0.0,
            "response":   "",
            "history":    [],
        })
    except Exception as exc:
        logger.exception("Graph invocation failed: request_id=%s error=%s", request_id, exc)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

    latency = (time.monotonic() - t0) * 1000
    logger.info(
        "request_id=%s intent=%s confidence=%.2f latency_ms=%.1f",
        request_id, result["intent"], result["confidence"], latency,
    )

    return ChatResponse(
        request_id=request_id,
        user_id=user_id,
        response=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        latency_ms=round(latency, 1),
    )
