"""
# Customer Support Agent

## Features
- LangGraph multi-agent orchestration
- OpenAI LLM planner
- FAISS vector DB (RAG)
- LangSmith tracing (observability)

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
export LANGCHAIN_API_KEY=your_key
python run.py
```

## Architecture
User → Planner → Router → (Tool | RAG | Escalation)