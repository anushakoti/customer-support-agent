from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import operator
import json

from rag.vector_store import search

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
class AgentState(TypedDict):
    message: str
    intent: str
    confidence: float
    response: str
    history: Annotated[list, operator.add]

# ------------------
# Planner (LLM)
# ------------------
def planner(state: AgentState):
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
# ------------------
# Tool Agent
# ------------------
def tool_agent(state: AgentState):
    if state["intent"] == "order":
        state["response"] = "Order 123 is Shipped"
    elif state["intent"] == "refund":
        state["response"] = "Refund processed"
    return state

# ------------------
# RAG Agent (FAISS)
# ------------------
def rag_agent(state: AgentState):
    state["response"] = search(state["message"])
    return state

# ------------------
# Escalation
# ------------------
def escalate(state: AgentState):
    state["response"] = "Escalating to human"
    return state
# ------------------
# Router
# ------------------
def router(state: AgentState):
    if state["confidence"] < 0.5:
        return "escalate"
    if state["intent"] in ["order", "refund"]:
        return "tool"
    if state["intent"] == "faq":
        return "rag"
    return "escalate"
# ------------------
# Build Graph
# ------------------
builder = StateGraph(AgentState)

builder.add_node("planner", planner)
builder.add_node("tool", tool_agent)
builder.add_node("rag", rag_agent)
builder.add_node("escalate", escalate)

builder.set_entry_point("planner")

builder.add_conditional_edges("planner", router, {
    "tool": "tool",
    "rag": "rag",
    "escalate": "escalate"
})

builder.add_edge("tool", END)
builder.add_edge("rag", END)
builder.add_edge("escalate", END)
graph = builder.compile()