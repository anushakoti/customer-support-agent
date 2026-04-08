# Customer Support Agent

An AI-powered customer support agent built with **LangGraph**, **OpenAI**, and **FastAPI**.  
Handles FAQs via RAG, looks up orders and processes refunds via tools, and escalates to a human agent when confidence is low.

---

## Architecture

```
User Request
    │
    ▼
┌─────────────┐
│ Input Guard │  ← blocks prompt injection, PII, banned patterns
└──────┬──────┘
       │
    ▼
┌─────────────┐
│   Planner   │  ← LLM classifies intent + confidence (uses conversation history)
└──────┬──────┘
       │
  ┌────┴─────────────────┐
  │                       │                  │
  ▼                       ▼                  ▼
┌──────┐            ┌──────────┐       ┌──────────┐
│ Tool │            │   RAG    │       │ Escalate │
│Agent │            │  Agent   │       │          │
└──┬───┘            └────┬─────┘       └────┬─────┘
   │  order/refund       │  faq              │  unknown/low-confidence
   │                     │                  │
   └─────────────────────┴──────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ Output Guard │  ← scrubs PII, catches raw errors
                  └──────┬───────┘
                          │
                          ▼
                      Response
```

### Components

| Component | File | Responsibility |
|---|---|---|
| **Input Guardrails** | `app/agent/guardrails.py` | Regex-based prompt injection detection, message length limits, PII scrubbing on output |
| **Planner** | `app/agent/graph.py` | LLM-based intent classification into `order / refund / faq / unknown` |
| **Tool Agent** | `app/agent/tools.py` | Order status lookup and refund processing (stub → swap with real APIs) |
| **RAG Agent** | `app/rag/vector_store.py` | FAISS similarity search over a FAQ knowledge base |
| **Memory** | `app/agent/memory.py` | Per-user conversation history (in-process; swap Redis for multi-pod) |
| **API Server** | `app/api/server.py` | FastAPI with API key auth, request validation, health endpoint |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/anushakoti/customer-support-agent.git
cd customer-support-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
API_SECRET_KEY=your-strong-random-secret
LANGCHAIN_API_KEY=ls__...          # optional — enables LangSmith tracing
```

Generate a strong `API_SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Run

```bash
python run.py
```

The API is now available at `http://localhost:8000`.

---

## API Reference

### `POST /chat`

Send a message to the agent.

**Headers**

```
X-API-Key: <your API_SECRET_KEY>
Content-Type: application/json
```

**Request body**

```json
{
  "message": "Where is my order #1001?",
  "user_id": "user-abc123"   // optional — auto-generated if omitted
}
```

**Response**

```json
{
  "request_id": "f3a1c2d4-...",
  "user_id": "user-abc123",
  "response": "Order 1001 — Shipped, arriving by Friday.",
  "intent": "order",
  "confidence": 0.95,
  "latency_ms": 412.3
}
```

**Example curl**

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-secret" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I return an item?", "user_id": "u1"}'
```

---

### `GET /health`

Liveness probe for load balancers and uptime monitors.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Running Tests

Tests are fully mocked — no API keys needed, no network calls.

```bash
pytest app/tests/test_agent.py -v
```

**Test coverage includes:**

- Guardrail input blocking (banned words, prompt injection, length limit)
- Guardrail output scrubbing (raw error strings, PII patterns)
- Order ID extraction from freeform text
- Tool routing for known and unknown order IDs
- Refund eligibility logic
- Graph routing: order → tool, faq → rag, low-confidence → escalate
- Blocked input never reaching the planner (LLM not called)
- Missing order ID asks the user instead of failing

---

## Docker

```bash
# Build
docker build -t customer-support-agent .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e API_SECRET_KEY=your-secret \
  customer-support-agent
```

---

## Project Structure

```
.
├── app/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine
│   │   ├── guardrails.py     # Input / output safety
│   │   ├── memory.py         # Per-user conversation history
│   │   └── tools.py          # Order status & refund tools
│   ├── api/
│   │   └── server.py         # FastAPI app (auth, validation, health)
│   ├── config/
│   │   └── settings.py       # Env var loading + startup validation
│   ├── rag/
│   │   └── vector_store.py   # FAISS index build/load + search
│   └── tests/
│       └── test_agent.py     # Unit tests (all mocked)
├── .env.example              # Environment variable template
├── Dockerfile                # Multi-stage production build
├── requirements.txt          # Pinned dependencies
└── run.py                    # Dev server entry point
```

---

## Key Design Decisions

**Why LangGraph?**  
LangGraph gives explicit control over the agent's decision flow as a typed state machine. Compared to an agent loop, every routing decision is visible, testable, and reproducible — important for a support system where wrong escalations or missed refunds have real consequences.

**Why FAISS over a hosted vector DB?**  
FAISS runs in-process with zero extra infrastructure for a demo/interview context. The `vector_store.py` module persists the index to disk so it's not rebuilt on every restart. In production, swap for Pinecone, Weaviate, or pgvector backed by a real document store.

**Why in-process Memory?**  
`memory.py` uses a `deque(maxlen=20)` per user — simple and zero-dependency. The module exposes a singleton so it's easy to replace with a Redis-backed store by swapping one import without touching the graph.

**Why API key auth instead of OAuth?**  
API key auth is the simplest correct solution for a service-to-service API. For a customer-facing UI you'd add OAuth2/JWT, but the FastAPI `Depends` pattern used here makes it straightforward to swap.

---

## Extending the Agent

**Add a new intent** (e.g. `complaint`):

1. Add `"complaint"` to the intent classifier prompt in `graph.py`
2. Write a `handle_complaint(state)` node function
3. Add it to the graph with `builder.add_node` and `add_edge`
4. Return `"complaint"` from `router()` for the new intent
5. Add a test in `test_agent.py`

**Connect real order/payment APIs:**  
Replace the stub functions in `tools.py` — the interface and entity extraction stay the same.

**Scale memory across pods:**  
Replace the `memory` singleton in `memory.py` with a Redis client using the same `.get()` / `.add()` interface.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for LLM and embeddings |
| `API_SECRET_KEY` | ✅ | Bearer token for authenticating `/chat` requests |
| `LANGCHAIN_API_KEY` | ☑️ optional | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | ☑️ optional | Set `true` to enable LangSmith traces |
| `LANGCHAIN_PROJECT` | ☑️ optional | LangSmith project name |