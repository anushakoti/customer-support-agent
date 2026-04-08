import os
import logging
from pathlib import Path

from langchain_openai import OpenAIEmbeddings          # updated import
from langchain_community.vectorstores import FAISS     # updated import

logger = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).parent / "faiss_index"

# ---------------------------------------------------------------------------
# Knowledge base
# A realistic FAQ set. In production, load these from a database or CMS.
# ---------------------------------------------------------------------------

FAQ_DATA = [
    # Returns & Refunds
    "You can return most items within 30 days of delivery for a full refund.",
    "To start a return, go to My Orders, select the item, and click 'Return Item'.",
    "Refunds are processed within 3-5 business days after we receive the returned item.",
    "Sale items and digital downloads are non-refundable.",

    # Shipping
    "Standard shipping takes 3-5 business days.",
    "Express shipping takes 1-2 business days and costs $9.99.",
    "Free shipping is available on orders over $50.",
    "We ship to all 50 US states and to over 30 countries internationally.",

    # Orders
    "You can track your order using the tracking link sent to your email.",
    "To cancel an order, contact us within 1 hour of placing it.",
    "If your order arrives damaged, please take a photo and contact support within 48 hours.",

    # Account
    "To reset your password, click 'Forgot Password' on the login page.",
    "You can update your email address in Account Settings.",

    # Payments
    "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    "Your payment information is encrypted and never stored on our servers.",
]


def _build_index() -> FAISS:
    """Build the FAISS index from FAQ_DATA and persist it to disk."""
    logger.info("Building FAISS index from %d FAQ entries...", len(FAQ_DATA))
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(FAQ_DATA, embeddings)
    store.save_local(str(INDEX_PATH))
    logger.info("FAISS index saved to %s", INDEX_PATH)
    return store


def _load_or_build_index() -> FAISS:
    """
    Load a persisted FAISS index if it exists; otherwise build and save one.
    This avoids re-embedding the entire FAQ corpus on every cold start.
    """
    embeddings = OpenAIEmbeddings()
    if INDEX_PATH.exists():
        logger.info("Loading existing FAISS index from %s", INDEX_PATH)
        return FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    return _build_index()


# Initialise once at import time
vector_store = _load_or_build_index()


def search(query: str, k: int = 2) -> str:
    """
    Return the most relevant FAQ answer(s) for a query.

    Args:
        query: the user's question
        k:     number of results to retrieve and merge
    """
    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        return "I don't have information on that topic. Let me connect you with a human agent."

    # Return top result; if multiple, join for richer context
    answers = [doc.page_content for doc in docs]
    logger.debug("RAG retrieved %d docs for query: %s", len(answers), query)
    return " ".join(answers)
