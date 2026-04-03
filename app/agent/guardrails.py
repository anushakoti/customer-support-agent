class Guardrails:
    def validate_input(self, message: str) -> bool:
        banned = ["hack", "fraud", "bypass"]
        return not any(word in message.lower() for word in banned)

    def validate_output(self, response: str) -> str:
        if "error" in response.lower():
            return "Something went wrong. Escalating to human."
        return response