import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

from a2a.types import Message
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import LLMServiceExtensionServer, LLMServiceExtensionSpec
from agentstack_sdk.a2a.extensions import PlatformApiExtensionServer, PlatformApiExtensionSpec
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_openai import ChatOpenAI


class ProviderAgent:
    # Create a Langchain agent to bring onto the AGent Stack Platform as an A2A Server
    def __init__(self, llm) -> None:
        # Store the LLM and prepare the MCP client for provider lookup
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
        # Fetch available MCP tools and build the LangChain agent around them
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

        # Invoke the agent with the user prompt and return the final content
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

# Create an instance of the server
server = Server()


@server.agent(
    # Add a name to the agent server so it can be discoverable on Agent Stack by name and called via handoff tool by the healthcare agent
    name="ProviderAgent",
)
async def provider_agent_wrapper(
    input: Message,
    context: RunContext,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-2.5-flash-lite",)),
    ],
    _: Annotated[PlatformApiExtensionServer, PlatformApiExtensionSpec()],
):
    """Wrapper around the provider agent using the AgentStack LLM extension."""
    # Pull the user's text prompt from the incoming message
    prompt = get_message_text(input)
    llm_config = None

    # Select the default LLM fulfillment from the extension
    if llm and llm.data and llm.data.llm_fulfillments:
        llm_config = llm.data.llm_fulfillments.get("default")
    else:
        yield AgentMessage(text="LLM selection is required.")
        return

    # Ensure we have a valid LLM config to create the client
    if not llm_config:
        yield AgentMessage(text="No LLM configuration available from the extension.")
        return

    # Build the LangChain OpenAI client using platform-provided credentials
    langchain_llm = ChatOpenAI(
        model=llm_config.api_model,
        base_url=llm_config.api_base,
        api_key=llm_config.api_key,
        temperature=0,
    )

    agent = await ProviderAgent(langchain_llm).initialize()
    response = await agent.answer_query(prompt)
    yield AgentMessage(text=response)

# Run the server
def run() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
