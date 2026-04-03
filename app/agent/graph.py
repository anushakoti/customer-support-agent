from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import operator
import json

from agent.guardrails import Guardrails
from agent.tools import get_order_status, process_refund
from rag.vector_store import search

guardrails = Guardrails()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class AgentState(TypedDict):
    message: str
    intent: str
    confidence: float
    response: str
    history: Annotated[list, operator.add]


# ------------------------
# Guardrail Nodes
# ------------------------
def input_guard(state: AgentState):
    if not guardrails.validate_input(state["message"]):
        state["intent"] = "blocked"
        state["response"] = "Request blocked due to policy"
    return state


def output_guard(state: AgentState):
    state["response"] = guardrails.validate_output(state["response"])
    return state


# ------------------------
# Planner (LLM)
# ------------------------
def planner(state: AgentState):
    if state.get("intent") == "blocked":
        return state

    prompt = f"""
    Classify intent:
    {state['message']}

    Return JSON:
    {{"intent":"refund|order|faq|unknown","confidence":0-1}}
    """

    result = llm.invoke([HumanMessage(content=prompt)])

    try:
        parsed = json.loads(result.content)
        state["intent"] = parsed["intent"]
        state["confidence"] = parsed["confidence"]
    except:
        state["intent"] = "unknown"
        state["confidence"] = 0.0

    return state


# ------------------------
# Tool Agent
# ------------------------
def tool_agent(state: AgentState):
    if state["intent"] == "order":
        state["response"] = get_order_status("123")
    elif state["intent"] == "refund":
        state["response"] = process_refund("123")
    return state


# ------------------------
# RAG Agent
# ------------------------
def rag_agent(state: AgentState):
    state["response"] = search(state["message"])
    return state


# ------------------------
# Escalation
# ------------------------
def escalate(state: AgentState):
    state["response"] = "Escalating to human agent"
    return state


# ------------------------
# Router
# ------------------------
def router(state: AgentState):
    if state["intent"] == "blocked":
        return "escalate"
    if state["confidence"] < 0.5:
        return "escalate"
    if state["intent"] in ["order", "refund"]:
        return "tool"
    if state["intent"] == "faq":
        return "rag"
    return "escalate"


# ------------------------
# Graph Build
# ------------------------
builder = StateGraph(AgentState)

builder.add_node("input_guard", input_guard)
builder.add_node("planner", planner)
builder.add_node("tool", tool_agent)
builder.add_node("rag", rag_agent)
builder.add_node("escalate", escalate)
builder.add_node("output_guard", output_guard)

builder.set_entry_point("input_guard")

builder.add_edge("input_guard", "planner")

builder.add_conditional_edges("planner", router, {
    "tool": "tool",
    "rag": "rag",
    "escalate": "escalate"
})

builder.add_edge("tool", "output_guard")
builder.add_edge("rag", "output_guard")
builder.add_edge("escalate", "output_guard")

builder.add_edge("output_guard", END)

graph = builder.compile()