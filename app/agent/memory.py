import logging
from collections import deque
from typing import List, Dict

logger = logging.getLogger(__name__)

MAX_HISTORY_PER_USER = 20  # keep last N messages per user to cap memory usage


class Memory:
    """
    In-process conversation store.

    Keeps a capped history deque per user_id so long sessions don't
    grow unbounded. In production, swap self.store for Redis or a DB.
    """

    def __init__(self):
        self.store: Dict[str, deque] = {}

    def get(self, user_id: str) -> List[Dict]:
        """Return conversation history for user_id as a plain list."""
        return list(self.store.get(user_id, []))

    def add(self, user_id: str, role: str, content: str) -> None:
        """
        Append a message to the user's history.

        Args:
            user_id: unique identifier for the session/user
            role:    "user" or "assistant"
            content: message text
        """
        if user_id not in self.store:
            self.store[user_id] = deque(maxlen=MAX_HISTORY_PER_USER)
        self.store[user_id].append({"role": role, "content": content})
        logger.debug("Memory updated for user %s (%d messages)", user_id, len(self.store[user_id]))

    def clear(self, user_id: str) -> None:
        """Wipe history for a user (e.g. on session end)."""
        self.store.pop(user_id, None)


# Singleton — shared across requests in the same process.
# Replace with a Redis-backed store for multi-process / multi-pod deployments.
memory = Memory()
