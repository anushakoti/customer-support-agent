from agent.graph import graph

if __name__ == "__main__":
    while True:
        msg = input("You: ")
        res = graph.invoke({
            "message": msg,
            "intent": "",
            "confidence": 0.0,
            "response": "",
            "history": []
        })
        print("Agent:", res["response"])