import re
import logging

logger = logging.getLogger(__name__)

# Patterns that indicate prompt injection or policy-violating intent
BANNED_PATTERNS = [
    r"\bhack\b",
    r"\bfraud\b",
    r"\bbypass\b",
    r"ignore (all |previous |your )?(instructions|rules|guidelines)",
    r"act as (an? )?(different|other|unrestricted|evil|DAN)",
    r"forget (your |all )?(instructions|training|rules)",
    r"you are now",
    r"jailbreak",
]

# Patterns that should never appear in output (card numbers, SSNs, etc.)
PII_PATTERNS = [
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # credit card
    r"\b\d{3}-\d{2}-\d{4}\b",                       # SSN
]

_compiled_banned = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]
_compiled_pii = [re.compile(p) for p in PII_PATTERNS]


class Guardrails:
    def validate_input(self, message: str) -> bool:
        """
        Returns True if the message is safe to process, False if it should be blocked.
        Logs every blocked attempt so you can monitor abuse patterns.
        """
        if len(message) > 2000:
            logger.warning("Input blocked: message exceeds 2000 characters.")
            return False

        for pattern in _compiled_banned:
            if pattern.search(message):
                logger.warning("Input blocked by guardrail pattern: %s", pattern.pattern)
                return False

        return True

    def validate_output(self, response: str) -> str:
        """
        Scrubs PII from outgoing responses and catches error strings that
        should never be surfaced raw to users.
        """
        # Scrub any PII that leaked into the response
        for pattern in _compiled_pii:
            if pattern.search(response):
                logger.error("PII detected in LLM output — redacting and escalating.")
                return "I'm unable to share that information. A human agent will follow up."

        # Never surface raw error strings
        if re.search(r"\b(traceback|exception|error|stack trace)\b", response, re.IGNORECASE):
            logger.error("Raw error string detected in LLM output.")
            return "Something went wrong on our end. A human agent will follow up shortly."

        return response
