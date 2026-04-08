import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Matches patterns like: order #1234, order 1234, #ORD-5678
ORDER_ID_PATTERN = re.compile(r"(?:order\s*#?|#)([A-Za-z0-9\-]+)", re.IGNORECASE)


def extract_order_id(message: str) -> Optional[str]:
    """
    Pull an order ID out of freeform user text.
    Returns None if no ID is found — caller should ask the user to provide it.
    """
    match = ORDER_ID_PATTERN.search(message)
    if match:
        return match.group(1)
    return None


def get_order_status(order_id: str) -> str:
    """
    Look up the status of an order.

    TODO: replace the stub below with a real DB / API call:
        response = requests.get(f"{ORDER_SERVICE_URL}/orders/{order_id}")
        response.raise_for_status()
        return response.json()["status"]
    """
    logger.info("Fetching order status for order_id=%s", order_id)

    # --- stub ---
    mock_orders = {
        "1001": "Shipped — arriving by Friday",
        "1002": "Processing — estimated ship date tomorrow",
        "1003": "Delivered on Monday",
    }
    status = mock_orders.get(order_id, f"Order {order_id} not found in our system.")
    return status


def process_refund(order_id: str) -> str:
    """
    Initiate a refund for an order.

    TODO: replace stub with real payments API call.
    """
    logger.info("Processing refund for order_id=%s", order_id)

    # --- stub ---
    mock_refundable = {"1001", "1002"}
    if order_id in mock_refundable:
        return f"Refund initiated for order {order_id}. You'll see the credit in 3–5 business days."
    return f"Order {order_id} is not eligible for a refund. A human agent will review your case."
