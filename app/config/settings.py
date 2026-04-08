from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Fail fast at startup — do not wait for first request to discover missing keys
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is required but not set. Check your .env file.")

if not API_SECRET_KEY:
    raise EnvironmentError("API_SECRET_KEY is required but not set. Set it in your .env file.")

if not LANGCHAIN_API_KEY:
    logger.warning("LANGCHAIN_API_KEY not set — LangSmith tracing will be disabled.")
