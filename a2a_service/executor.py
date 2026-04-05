import logging

from agent_sdk.a2a.executor import BaseAgentExecutor
from agents.agent import run_query

logger = logging.getLogger("agent_health.a2a_executor")

class HealthExecutor(BaseAgentExecutor):
    """A2A executor that bridges incoming A2A tasks to the health agent."""
    def __init__(self):
        super().__init__(run_query_fn=run_query)
