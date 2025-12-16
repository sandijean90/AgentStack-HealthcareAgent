import asyncio
import os
from pathlib import Path
from typing import Annotated, Optional

from a2a.types import Message
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.extensions import LLMServiceExtensionServer, LLMServiceExtensionSpec
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import ChatModelParameters
from beeai_framework.backend.message import SystemMessage, UserMessage
from PyPDF2 import PdfReader


class PolicyAgent:
    """
    A policy agent that reads a benefits PDF and answers coverage questions.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        system_prompt: str = (
            "You are an expert insurance agent designed to assist with coverage queries. "
            "Use the provided documents to answer questions about insurance policies. "
            "If the information is not available in the documents, respond with \"I don't know\"."
        ),
    ) -> None:
        self.system_prompt = system_prompt

        self.pdf_path = Path(pdf_path) if pdf_path else Path(__file__).resolve().parent / "2026AnthemgHIPSBC.pdf"
        self.pdf_bytes = self._load_pdf(self.pdf_path)
        self.pdf_text = self._extract_pdf_text(self.pdf_path)

    @staticmethod
    def _load_pdf(path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {path.resolve()}")
        return path.read_bytes()

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """
        Extract text content from the PDF so the LLM can read it.
        Falls back to an empty string if extraction fails.
        """
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    def build_prompt(self, prompt: str) -> str:
        return (
            f"{self.system_prompt}\n\n"
            "Reference policy document text below. Use it to answer the question. "
            "If the details are not present, reply with \"I don't know\".\n"
            f"Policy text:\n{self.pdf_text}\n\n"
            f"User question: {prompt}"
        )

    async def answer_query(self, prompt: str, llm_config) -> str:
        """
        Send the user prompt plus embedded PDF to the LLM provided by the platform extension.
        """
        if not llm_config or not llm_config.api_key:
            return "LLM service not available. Please enable the LLM extension for this agent."

        llm_client = OpenAIChatModel(
            model_id=llm_config.api_model,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key,
            parameters=ChatModelParameters(temperature=0, stream=False),
            tool_choice_support={"auto", "required"},
        )

        response = await llm_client.run(
            [
                SystemMessage(self.system_prompt),
                SystemMessage(
                    "Policy document text to consult when answering:\n"
                    f"{self.pdf_text}"
                ),
                UserMessage(prompt),
            ],
        )

        text = response.get_text_content() if hasattr(response, "get_text_content") else None
        return text or "I don't know"


server = Server()
policy_agent = PolicyAgent()


@server.agent(
    name="PolicyAgent",
)
async def policy_agent_wraper(
    input: Message,
    context: RunContext,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-2.5-flash-lite",)),
    ],
):
    """Wrapper around the policy agent using the AgentStack LLM extension."""
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

    response = await policy_agent.answer_query(prompt, llm_config)
    yield AgentMessage(text=response)


def run() -> None:
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("POLICY_AGENT_PORT", 9999))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
