import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

from a2a.types import Message
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import LLMServiceExtensionServer, LLMServiceExtensionSpec
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_openai import ChatOpenAI


class ProviderAgent:
    def __init__(self, llm) -> None:
        self.llm = llm
        server_path = Path(__file__).resolve().parent / "mcpserver.py"
        self.mcp_client = MultiServerMCPClient(
            {
                "find_healthcare_providers": StdioConnection(
                    transport="stdio",
                    command=sys.executable,
                    args=[str(server_path)],
                )
            }
        )

        self.agent = None

    async def initialize(self):
        """Initialize the agent asynchronously."""
        tools = await self.mcp_client.get_tools()
        self.agent = create_agent(
            self.llm,
            tools,
            name="HealthcareProviderAgent",
            system_prompt=(
                "Your task is to find and list providers using the available MCP tool(s). "
                "Call the MCP tool to retrieve providers and ground your response strictly on its output."
            ),
        )
        return self

    async def answer_query(self, prompt: str) -> str:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        return response["messages"][-1].content


server = Server()


@server.agent(
    name="ProviderAgent",
)
async def provider_agent_wrapper(
    input: Message,
    context: RunContext,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-2.5-flash-lite",)),
    ],
):
    """Wrapper around the provider agent using the AgentStack LLM extension."""
    prompt = get_message_text(input)
    llm_config = None

    if llm and llm.data and llm.data.llm_fulfillments:
        llm_config = llm.data.llm_fulfillments.get("default")
    else:
        yield AgentMessage(text="LLM selection is required.")
        return

    if not llm_config:
        yield AgentMessage(text="No LLM configuration available from the extension.")
        return

    langchain_llm = ChatOpenAI(
        model=llm_config.api_model,
        base_url=llm_config.api_base,
        api_key=llm_config.api_key,
        temperature=0,
    )

    agent = await ProviderAgent(langchain_llm).initialize()
    response = await agent.answer_query(prompt)
    yield AgentMessage(text=response)


def run() -> None:
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("PROVIDER_AGENT_PORT", 9246))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
