from fastapi import FastAPI
from agent.graph import graph

app = FastAPI()

@app.post("/chat")
def chat(message: str):
    result = graph.invoke({
        "message": message,
        "intent": "",
        "confidence": 0.0,
        "response": "",
        "history": []
    })
    return {"response": result["response"]}