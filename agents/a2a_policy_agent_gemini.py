import asyncio
import os 


from a2a.types import Message
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from dotenv import load_dotenv

from .policy_agent_logic_gemini import GeminiPolicyAgent

load_dotenv()

server = Server()
policy_agent = GeminiPolicyAgent()


@server.agent()
async def gemini_policy_agent(input: Message, context: RunContext):
    """Wrapper around the Gemini-backed policy agent."""
    prompt = get_message_text(input)
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, policy_agent.answer_query, prompt)
    yield AgentMessage(text=response)


def run() -> None:
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("POLICY_AGENT_PORT", 9999))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
