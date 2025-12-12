import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv

from policy_agent_logic_gemini import GeminiPolicyAgent


class GeminiPolicyAgentExecutor(AgentExecutor):
    """Executor that wraps the Gemini-backed policy agent."""

    def __init__(self) -> None:
        self.agent = GeminiPolicyAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        prompt = context.get_user_input()
        response = self.agent.answer_query(prompt)
        await event_queue.enqueue_event(new_agent_text_message(response))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def main() -> None:
    print("Running Gemini Policy Agent")
    load_dotenv()

    host = os.environ.get("AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("POLICY_AGENT_PORT", 9999))

    skill = AgentSkill(
        id="insurance_coverage_gemini",
        name="Insurance coverage (Gemini)",
        description="Provides information about insurance coverage options and details using Gemini.",
        tags=["insurance", "coverage"],
        examples=["What does my policy cover?", "Are mental health services included?"],
    )

    agent_card = AgentCard(
        name="InsurancePolicyCoverageAgentGemini",
        description="Provides information about insurance policy coverage options and details using Gemini.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=GeminiPolicyAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
