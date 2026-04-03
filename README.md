# Customer Support Agent (Production-Ready)

## Features
- LangGraph multi-agent system
- OpenAI LLM planner
- FAISS vector DB (RAG)
- LangSmith tracing
- Guardrails (input/output safety)

## Setup
pip install -r requirements.txt

export OPENAI_API_KEY=your_key
export LANGCHAIN_API_KEY=your_key

python run.py

## Architecture
User → Guardrails → Planner → Router → Tool/RAG/Escalation → Guardrails → Response