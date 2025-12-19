import json
import os
from typing import Annotated

from a2a.types import Message, Role
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext

from agentstack_sdk.a2a.extensions import (
    AgentDetail,
    AgentDetailContributor,
    AgentDetailTool,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModelParameters
from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.adapters.agentstack.agents import AgentStackAgent
from beeai_framework.adapters.agentstack.agents.types import AgentStackAgentStatus


server = Server()
memories: dict[str, UnconstrainedMemory] = {}


def get_memory(context: RunContext) -> UnconstrainedMemory:
    """Get or create session memory keyed by context id."""
    context_id = getattr(context, "context_id", getattr(context, "session_id", "default"))
    if context_id not in memories:
        memories[context_id] = UnconstrainedMemory()
    return memories[context_id]


def to_framework_message(message: Message):
    """Convert A2A Message to BeeAI Framework Message format."""
    message_text = get_message_text(message)
    if message.role == Role.agent:
        return AssistantMessage(message_text)
    return UserMessage(message_text)


def summarize_for_trajectory(data: object, limit: int = 400) -> str:
    """
    Convert tool inputs/outputs to a readable, bounded string for trajectory updates.
    """
    try:
        text = data if isinstance(data, str) else json.dumps(data, default=str)
    except Exception:
        text = str(data)

    return text if len(text) <= limit else f"{text[:limit]}... [truncated]"


@server.agent(
    name="Healthcare Concierge",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi there! I can help navigate benefits, providers, and coverage details.",
        input_placeholder="Ask a healthcare question...",
        programming_language="Python",
        framework="BeeAI",
        contributors=[
            AgentDetailContributor(
                name="Sandi Besen and Ken Ocheltree",
                email="name@example.com",
            )
        ],
        tools=[
            AgentDetailTool(
                name="Think",
                description="Plans the best approach before responding.",
            )
        ],
    ),
)
async def healthcare_concierge(
    message: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-2.5-flash-lite",)),
    ],
):
    """
    Healthcare concierge agent that answers insurance and provider questions.
    """

    yield trajectory.trajectory_metadata(
        title="Initializing Agent...",
        content="Setting up your Healthcare Concierge.",
    )

    memory = get_memory(context)

    # Load existing history into BeeAI memory
    history = [msg async for msg in context.load_history() if isinstance(msg, Message) and msg.parts]
    await memory.add_many(to_framework_message(item) for item in history)

    # Configure the LLM from extension fulfillment
    if not llm or not llm.data:
        yield trajectory.trajectory_metadata(title="LLM Error", content="LLM extension missing.")
        yield "LLM selection is required."
        return

    llm_config = llm.data.llm_fulfillments.get("default")
    if not llm_config:
        yield trajectory.trajectory_metadata(title="LLM Error", content="No LLM fulfillment available.")
        yield "No LLM configuration available from the extension."
        return


    llm_client = OpenAIChatModel(
        model_id=llm_config.api_model,
        base_url=llm_config.api_base,
        api_key=llm_config.api_key,
        parameters=ChatModelParameters(temperature=0, stream=True),
        tool_choice_support={"auto", "required"},
    )


    #Make the other AgentStack agents discoverable for the handoff tool
    agents = await AgentStackAgent.from_agent_stack(states={AgentStackAgentStatus.READY})
    #handoff_agents = {a.name: a for a in agents if a.name in {"PolicyAgent", "ResearchAgent", "ProviderAgent"}}
    print([a.name for a in agents])
    policy_handoff = HandoffTool(agents["PolicyAgent"])
    research_handoff = HandoffTool(agents["ResearchAgent"])
    provider_handoff = HandoffTool(agents["ProviderAgent"])
    #handoff_tools = [policy_handoff, research_handoff, provider_handoff]

    think_tool=ThinkTool()

    #ADD IN THE REAL INSTRUCTION WHEN ADDING IN THE HANDOFF TOOL
    instructions = (
        "You are a friendly healthcare concierge. "
        "Answer questions about plan coverage, in-network providers, and costs. "
        "Hand off your task to the PolicyAgent when there are specific questions pertaining to the user's policy details."
        "Hand off your task to the ResearchAgent when you need information about symptoms, health conditions, treatments, and procedures using up-to-date web resources."
        "Hand off your task to the ProviderAgent when you need information about the providers in network."
        "If unsure, ask clarifying questions before giving guidance."
    )

    agent = RequirementAgent(
        llm=llm_client,
        name="HealthcareConcierge",
        memory=memory,
        tools=[think_tool, policy_handoff, research_handoff, provider_handoff],
        requirements=[ConditionalRequirement(think_tool, force_at_step=1),
                      ConditionalRequirement(policy_handoff, min_invocations=1, max_invocations=1),
                      ConditionalRequirement(research_handoff, min_invocations=1, max_invocations=1),
                      ConditionalRequirement(provider_handoff, min_invocations=1, max_invocations=1),
                      ],
        role="Healthcare Concierge",
        instructions=instructions,
    )

    user_prompt = get_message_text(message)

    response_text = ""

    def handle_final_answer_stream(data, meta) -> None:
        nonlocal response_text
        if getattr(data, "delta", None):
            response_text += data.delta

    async for event, meta in agent.run(
        user_prompt,
        execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2),
    ).on("final_answer", handle_final_answer_stream):
        if meta.name == "final_answer":
            if getattr(event, "delta", None):
                yield event.delta
            elif getattr(event, "text", None):
                response_text += event.text
        elif meta.name == "success" and event.state.steps:
            step = event.state.steps[-1]
            if step.tool and step.tool.name == "think":
                thoughts = step.input.get("thoughts", "Planning response.")
                yield trajectory.trajectory_metadata(title="Thinking", content=thoughts[:200])
            elif step.tool:
                tool_name = step.tool.name
                if tool_name != "final_answer":
                    yield trajectory.trajectory_metadata(
                        title=f"{tool_name} (request)",
                        content=summarize_for_trajectory(step.input),
                    )

                    if getattr(step, "error", None):
                        yield trajectory.trajectory_metadata(
                            title=f"{tool_name} (error)",
                            content=step.error.explain(),
                        )
                    else:
                        output_text = (
                            step.output.get_text_content() if getattr(step, "output", None) else "No output"
                        )
                        yield trajectory.trajectory_metadata(
                            title=f"{tool_name} (response)",
                            content=summarize_for_trajectory(output_text),
                        )

    await context.store(AgentMessage(text=response_text))


def run() -> None:
    """Start the AgentStack server for the healthcare concierge."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
