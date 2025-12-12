import base64
from pathlib import Path

from anthropic import AnthropicVertex
from anthropic.types import (
    Base64PDFSourceParam,
    DocumentBlockParam,
    MessageParam,
    TextBlockParam,
)
from dotenv import load_dotenv

from helpers import authenticate

class PolicyAgent:
    def __init__(self) -> None:
        load_dotenv()
        credentials, project_id = authenticate()
        self.client = AnthropicVertex(
            project_id=project_id,
            region="global",
            access_token=credentials.token,
        )
        with Path("../data/2026AnthemgHIPSBC.pdf").open("rb") as file:
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

    def answer_query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=1024,
            system="You are an expert insurance agent designed to assist with coverage queries. Use the provided documents to answer questions about insurance policies. If the information is not available in the documents, respond with 'I don't know'",
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        DocumentBlockParam(
                            type="document",
                            source=Base64PDFSourceParam(
                                type="base64",
                                media_type="application/pdf",
                                data=self.pdf_data,
                            ),
                        ),
                        TextBlockParam(
                            type="text",
                            text=prompt,
                        ),
                    ],
                )
            ],
        )
        
        return response.content[0].text.replace("$", r"\\$")
import base64
from pathlib import Path

from anthropic import AnthropicVertex
from anthropic.types import (
    Base64PDFSourceParam,
    DocumentBlockParam,
    MessageParam,
    TextBlockParam,
)
from dotenv import load_dotenv

from helpers import authenticate

class PolicyAgent:
    def __init__(self) -> None:
        load_dotenv()
        credentials, project_id = authenticate()
        self.client = AnthropicVertex(
            project_id=project_id,
            region="global",
            access_token=credentials.token,
        )
        with Path("../data/2026AnthemgHIPSBC.pdf").open("rb") as file:
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

    def answer_query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=1024,
            system="You are an expert insurance agent designed to assist with coverage queries. Use the provided documents to answer questions about insurance policies. If the information is not available in the documents, respond with 'I don't know'",
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        DocumentBlockParam(
                            type="document",
                            source=Base64PDFSourceParam(
                                type="base64",
                                media_type="application/pdf",
                                data=self.pdf_data,
                            ),
                        ),
                        TextBlockParam(
                            type="text",
                            text=prompt,
                        ),
                    ],
                )
            ],
        )
        
        return response.content[0].text.replace("$", r"\\$")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_openai import ChatOpenAI

from helpers import authenticate

class ProviderAgent:
    def __init__(self) -> None:
        credentials, project_id = authenticate()
        location = "us-central1"
        base_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"

        self.mcp_client = MultiServerMCPClient(
            {
                "find_healthcare_providers": StdioConnection(
                    transport="stdio",
                    command="uv",
                    args=["run", "mcpserver.py"],
                )
            }
        )

        self.credentials = credentials
        self.base_url = base_url
        self.agent = None

    async def initialize(self):
        """Initialize the agent asynchronously."""
        tools = await self.mcp_client.get_tools()
        self.agent = create_agent(
            ChatOpenAI(
                model="openai/gpt-oss-20b-maas",
                openai_api_key=self.credentials.token,
                openai_api_base=self.base_url,
            ),
            tools,
            name="HealthcareProviderAgent",
            system_prompt="Your task is to find and list providers using the find_healthcare_providers MCP Tool based on the users query. Only use providers based on the response from the tool.",
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
import base64
from pathlib import Path

from anthropic import AnthropicVertex
from anthropic.types import (
    Base64PDFSourceParam,
    DocumentBlockParam,
    MessageParam,
    TextBlockParam,
)
from dotenv import load_dotenv

from helpers import authenticate

class PolicyAgent:
    def __init__(self) -> None:
        load_dotenv()
        credentials, project_id = authenticate()
        self.client = AnthropicVertex(
            project_id=project_id,
            region="global",
            access_token=credentials.token,
        )
        with Path("../data/2026AnthemgHIPSBC.pdf").open("rb") as file:
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

    def answer_query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=1024,
            system="You are an expert insurance agent designed to assist with coverage queries. Use the provided documents to answer questions about insurance policies. If the information is not available in the documents, respond with 'I don't know'",
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        DocumentBlockParam(
                            type="document",
                            source=Base64PDFSourceParam(
                                type="base64",
                                media_type="application/pdf",
                                data=self.pdf_data,
                            ),
                        ),
                        TextBlockParam(
                            type="text",
                            text=prompt,
                        ),
                    ],
                )
            ],
        )
        
        return response.content[0].text.replace("$", r"\\$")
import base64
from pathlib import Path

from anthropic import AnthropicVertex
from anthropic.types import (
    Base64PDFSourceParam,
    DocumentBlockParam,
    MessageParam,
    TextBlockParam,
)
from dotenv import load_dotenv

from helpers import authenticate

class PolicyAgent:
    def __init__(self) -> None:
        load_dotenv()
        credentials, project_id = authenticate()
        self.client = AnthropicVertex(
            project_id=project_id,
            region="global",
            access_token=credentials.token,
        )
        with Path("../data/2026AnthemgHIPSBC.pdf").open("rb") as file:
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

    def answer_query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=1024,
            system="You are an expert insurance agent designed to assist with coverage queries. Use the provided documents to answer questions about insurance policies. If the information is not available in the documents, respond with 'I don't know'",
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        DocumentBlockParam(
                            type="document",
                            source=Base64PDFSourceParam(
                                type="base64",
                                media_type="application/pdf",
                                data=self.pdf_data,
                            ),
                        ),
                        TextBlockParam(
                            type="text",
                            text=prompt,
                        ),
                    ],
                )
            ],
        )
        
        return response.content[0].text.replace("$", r"\\$")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_openai import ChatOpenAI

from helpers import authenticate

class ProviderAgent:
    def __init__(self) -> None:
        credentials, project_id = authenticate()
        location = "us-central1"
        base_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"

        self.mcp_client = MultiServerMCPClient(
            {
                "find_healthcare_providers": StdioConnection(
                    transport="stdio",
                    command="uv",
                    args=["run", "mcpserver.py"],
                )
            }
        )

        self.credentials = credentials
        self.base_url = base_url
        self.agent = None

    async def initialize(self):
        """Initialize the agent asynchronously."""
        tools = await self.mcp_client.get_tools()
        self.agent = create_agent(
            ChatOpenAI(
                model="openai/gpt-oss-20b-maas",
                openai_api_key=self.credentials.token,
                openai_api_base=self.base_url,
            ),
            tools,
            name="HealthcareProviderAgent",
            system_prompt="Your task is to find and list providers using the find_healthcare_providers MCP Tool based on the users query. Only use providers based on the response from the tool.",
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
