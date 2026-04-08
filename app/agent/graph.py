import json
import logging
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI          # updated: langchain-openai package
from langchain_core.messages import HumanMessage # updated: langchain-core package

from agent.guardrails import Guardrails
from agent.memory import memory
from agent.tools import get_order_status, process_refund, extract_order_id
from rag.vector_store import search

logger = logging.getLogger(__name__)

guardrails = Guardrails()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    user_id: str                          # added: ties memory to a session
    message: str
    intent: str
    confidence: float
    response: str
    history: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def input_guard(state: AgentState) -> AgentState:
    if not guardrails.validate_input(state["message"]):
        logger.warning("Input blocked for user_id=%s", state.get("user_id"))
        state["intent"] = "blocked"
        state["response"] = "I'm not able to help with that request."
    return state


def planner(state: AgentState) -> AgentState:
    """Classify the user's intent via an LLM call."""
    if state.get("intent") == "blocked":
        return state

    # Include recent history so the classifier has conversation context
    history_text = ""
    if state.get("user_id"):
        past = memory.get(state["user_id"])
        if past:
            history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in past[-4:]
            )

    prompt = f"""You are an intent classifier for a customer support agent.

Recent conversation:
{history_text or "(none)"}

Latest message: {state['message']}

Respond with ONLY a JSON object — no markdown, no explanation:
{{"intent": "refund|order|faq|unknown", "confidence": <float 0.0-1.0>}}
"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(result.content)
        state["intent"] = parsed["intent"]
        state["confidence"] = float(parsed["confidence"])
        logger.info(
            "Planner result: intent=%s confidence=%.2f user_id=%s",
            state["intent"], state["confidence"], state.get("user_id"),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Planner failed to parse LLM response: %s | raw=%s", exc, getattr(result, "content", ""))
        state["intent"] = "unknown"
        state["confidence"] = 0.0

    return state


def tool_agent(state: AgentState) -> AgentState:
    """Handle order-status and refund intents using extracted entity IDs."""
    order_id = extract_order_id(state["message"])

    if not order_id:
        state["response"] = (
            "I'd be happy to help! Could you please share your order number "
            "(e.g. #1001) so I can look that up?"
        )
        return state

    if state["intent"] == "order":
        state["response"] = get_order_status(order_id)
    elif state["intent"] == "refund":
        state["response"] = process_refund(order_id)

    return state


def rag_agent(state: AgentState) -> AgentState:
    """Answer FAQ questions from the knowledge base."""
    result = search(state["message"])
    state["response"] = result
    return state


def escalate(state: AgentState) -> AgentState:
    """Route to a human agent."""
    reason = "low confidence" if state.get("confidence", 1) < 0.5 else state.get("intent", "unknown")
    logger.info("Escalating: reason=%s user_id=%s", reason, state.get("user_id"))
    state["response"] = (
        "I'm connecting you with a human agent who can better assist you. "
        "Please hold on for a moment."
    )
    return state


def output_guard(state: AgentState) -> AgentState:
    """Scrub PII / raw errors from the outgoing response, then persist to memory."""
    state["response"] = guardrails.validate_output(state["response"])

    # Persist both sides of the turn to memory
    user_id = state.get("user_id")
    if user_id:
        memory.add(user_id, "user", state["message"])
        memory.add(user_id, "assistant", state["response"])

    return state


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def router(state: AgentState) -> str:
    if state["intent"] == "blocked":
        return "escalate"
    if state["confidence"] < 0.5:
        return "escalate"
    if state["intent"] in ("order", "refund"):
        return "tool"
    if state["intent"] == "faq":
        return "rag"
    return "escalate"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("input_guard", input_guard)
builder.add_node("planner",     planner)
builder.add_node("tool",        tool_agent)
builder.add_node("rag",         rag_agent)
builder.add_node("escalate",    escalate)
builder.add_node("output_guard",output_guard)

builder.set_entry_point("input_guard")

builder.add_edge("input_guard", "planner")

builder.add_conditional_edges("planner", router, {
    "tool":     "tool",
    "rag":      "rag",
    "escalate": "escalate",
})

builder.add_edge("tool",     "output_guard")
builder.add_edge("rag",      "output_guard")
builder.add_edge("escalate", "output_guard")

builder.add_edge("output_guard", END)

graph = builder.compile()
