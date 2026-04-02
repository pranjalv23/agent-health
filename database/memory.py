import logging
import os
from typing import Optional

from mem0 import MemoryClient

logger = logging.getLogger("agent_health.memory")

_client: Optional[MemoryClient] = None


def _get_client() -> MemoryClient:
    global _client
    if _client is None:
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY environment variable is not set.")
        logger.info("Initializing Mem0 client")
        _client = MemoryClient(api_key=api_key)
    return _client


def get_memories(user_id: str, query: str) -> list[str]:
    """Search Mem0 for facts relevant to the user and current query.

    Returns a list of plain-text memory strings to be injected into context.
    """
    try:
        client = _get_client()
        results = client.search(
            query=query,
            version="v2",
            filters={"user_id": user_id},
            limit=5,
        )
        memories = [
            r["memory"][:300]
            for r in results.get("results", [])
            if r.get("memory") and r.get("score", 0) >= 0.70
        ]
        if memories:
            logger.info("Retrieved %d memories for user='%s'", len(memories), user_id)
        else:
            logger.info("No relevant memories found for user='%s'", user_id)
        return memories
    except Exception as e:
        logger.warning("Failed to retrieve memories for user='%s': %s", user_id, e)
        return []


def save_memory(user_id: str, query: str, response: str) -> None:
    """Add the latest conversation turn to Mem0.

    Mem0 automatically extracts and stores relevant facts about the user
    (fitness goals, progress, preferences, injuries) from the exchange.
    """
    try:
        client = _get_client()
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        client.add(messages=messages, user_id=user_id)
        logger.info("Saved conversation to Mem0 for user='%s'", user_id)
    except Exception as e:
        logger.warning("Failed to save memory for user='%s': %s", user_id, e)
