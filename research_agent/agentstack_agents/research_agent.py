import os
from typing import Annotated, Any
import httpx
from textwrap import dedent
from pydantic import BaseModel, Field

from a2a.types import AgentSkill, Message

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentFinalAnswerEvent
from beeai_framework.tools import Tool, ToolRunOptions, JSONToolOutput
from beeai_framework.context import RunContext as BeeRunContext
from beeai_framework.emitter import Emitter

from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from agentstack_sdk.a2a.extensions.ui.agent_detail import EnvVar
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.a2a.extensions import (
    AgentDetail, AgentDetailTool,
    CitationExtensionServer, CitationExtensionSpec, 
    TrajectoryExtensionServer, TrajectoryExtensionSpec, 
    LLMServiceExtensionServer, LLMServiceExtensionSpec
)
from agentstack_sdk.server import Server
from streaming_citation_parser import StreamingCitationParser

server = Server()

class GoogleSearchToolInput(BaseModel):
    query: str = Field(description="Search query to find information")


class GoogleSearchTool(Tool[GoogleSearchToolInput, ToolRunOptions, JSONToolOutput]):
    name = "google_search"
    description = "Search Google using Serper API for current information"
    input_schema = GoogleSearchToolInput
    
    def __init__(self, api_key: str, options: dict[str, Any] | None = None):
        self.api_key = api_key
        super().__init__(options)
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "serper"], creator=self)
    
    async def _run(self, input: GoogleSearchToolInput, options: ToolRunOptions | None, context: BeeRunContext) -> JSONToolOutput:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": input.query, "num": 8},
                timeout=15.0
            )
            response.raise_for_status()
            return JSONToolOutput(response.json())


@server.agent(
    name="Research Agent",
    detail=AgentDetail(
        interaction_mode="multi-turn",
        variables=[
            EnvVar(
                name="SERPER_API_KEY",
                description="Serper API Key",
                required=True
            )
        ],
        user_greeting="Hi! I'm a Health Research Agent using Google search powered by Serper API.",
        version="1.0.0",
        tools=[
            AgentDetailTool(
                name="Google Search", 
                description="Intelligent web search powered by Google via Serper API. Automatically extracts optimal search terms from conversational queries."
            )
        ],
        framework="BeeAI Framework",
    ),
    skills=[
        AgentSkill(
            id="google-search-agent",
            name="Google Search Agent",
            description="Provides healthcare information about symptoms, health conditions, treatments, and procedures using up-to-date web resources.",
            tags=["Search", "Web", "Research"],
            examples=[
                "My shoulder is swollen and my arm hurts to move, what doctor should I see?",
                "I have a white rash and no feeling in a leg, who should I see?",
                "I have a rash and no fever, what kind of doctor should I see?",
            ]
        )
    ],
)
async def google_search_agent(
    input: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(
            suggested=("ibm-granite/granite-4.0-h-small",)
        )
    ],
):
    """Agent that provides information about health conditions, treatments, and procedures"""
    
    await context.store(input)

    user_query = ""
    for part in input.parts:
        if part.root.kind == "text":
            user_query = part.root.text
            break
    
    if not user_query:
        yield "Please provide a search query."
        return
    
    yield trajectory.trajectory_metadata(title="User Query", content=f"Received: '{user_query}'")
    
   
    api_key = os.getenv("SERPER_API_KEY","")    
    
    if not api_key:
        yield "No Serper API Key Provided"
        return
    
    yield trajectory.trajectory_metadata(title="Agent Setup", content="Initializing RequirementAgent with Google search")
    
    try:
        if not llm or not llm.data:
            raise ValueError("LLM service extension is required but not available")

        llm_config = llm.data.llm_fulfillments.get("default")
        if not llm_config:
            raise ValueError("No LLM fulfillment available")

        llm_client = AgentStackChatModel(parameters=ChatModelParameters(stream=True))
        llm_client.set_context(llm)
        
        agent = RequirementAgent(
            llm=llm_client,
            tools=[GoogleSearchTool(api_key)],
            instructions="You are a healthcare research agent tasked with providing information about health conditions. Use the google_search tool to find information on the web about options, symptoms, treatments, and procedures. Cite your sources in your responses. Output all of the information you find.",
        )
        
        search_results = None
        search_count = 0
        response_text = ""
        citation_parser = StreamingCitationParser()
        
        def handle_final_answer_stream(data: RequirementAgentFinalAnswerEvent, meta):
            nonlocal response_text
            if data.delta:
                response_text += data.delta
        
        async for event, meta in agent.run(user_query).on("final_answer", handle_final_answer_stream):
            if meta.name == "final_answer":
                if isinstance(event, RequirementAgentFinalAnswerEvent) and event.delta:
                    clean_text, new_citations = citation_parser.process_chunk(event.delta)
                    if clean_text:
                        yield clean_text
                    if new_citations:
                        yield citation.citation_metadata(citations=new_citations)
                continue
            
            if meta.name == "success" and event.state.steps:
                step = event.state.steps[-1]
                
                if step.tool and step.tool.name == "serper_search":
                    search_count += 1
                    search_query = step.input.get("query", "Unknown")
                    
                    yield trajectory.trajectory_metadata(
                        title=f"Search #{search_count}", 
                        content=f"Query: '{search_query}'"
                    )
                    
                    search_results = step.output.result
                    
                    if search_results:
                        num_results = len(search_results.get('organic', []))
                        yield trajectory.trajectory_metadata(
                            title=f"Results #{search_count}", 
                            content=f"Found {num_results} results"
                        )
        
        if final_text := citation_parser.finalize():
            yield final_text
        
        if citation_parser.citations:
            yield trajectory.trajectory_metadata(
                title="Complete", 
                content=f"Performed {search_count} search(es) with {len(citation_parser.citations)} citation(s)"
            )
        
        response_message = AgentMessage(
            text=response_text,
            metadata=(citation.citation_metadata(citations=citation_parser.citations) if citation_parser.citations else None)
        )
        await context.store(response_message)
    
    except Exception as e:
        yield trajectory.trajectory_metadata(title="Error", content=f"Exception: {str(e)}")
        error_msg = f"Error: {str(e)}"
        yield error_msg
        await context.store(AgentMessage(text=error_msg))


def run():
    server.run(
        host = os.environ.get("AGENT_HOST", "127.0.0.1"),
        port = int(os.environ.get("RESEARCH_AGENT_PORT", 9995)),
        context_store=PlatformContextStore()
    )


if __name__ == "__main__":
    run()
