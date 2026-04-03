from agent.graph import graph

def test_order():
    res = graph.invoke({
        "message": "track my order",
        "intent": "",
        "confidence": 0.0,
        "response": "",
        "history": []
    })
    assert "Order" in res["response"]