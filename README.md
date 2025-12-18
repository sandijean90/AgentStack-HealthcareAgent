# AgentStack Healthcare Agents
This repo contains a small network of healthcare-focused A2A agents built on AgentStack. AgentStack manages these agents (and handles LLM inference for you through its LLM extension), and all of them run as A2A Servers so the Healthcare agent can call the others and hand off tasks. You can run everything locally with AgentStack or deploy to a hosted AgentStack for fully remote A2A communication. Install instructions for AgentStack are available in the quickstart: https://agentstack.beeai.dev/stable/introduction/quickstart.

## Agents in this repo
- `healthcare_agent/agentstack_agents/healthcare_agent.py`: Concierge/orchestrator that uses BeeAI’s RequirementAgent plus HandoffTool to route questions to the PolicyAgent, ResearchAgent, or ProviderAgent. Default host/port: `HOST` (default `127.0.0.1`) / `HEALTH_AGENT_PORT` (default `2345`).
- `policy_agent/agentstack_agents/policy_agent.py`: Answers coverage questions by reading `2026AnthemgHIPSBC.pdf` with the LLM provided by the AgentStack extension. Default host/port: `AGENT_HOST` / `POLICY_AGENT_PORT` (default `9999`).
- `provider_agent/agentstack_agents/provider_agent.py`: Finds providers via an MCP tool backed by `mcpserver.py` and returns results with an AgentStack-provisioned LLM. Default host/port: `AGENT_HOST` / `PROVIDER_AGENT_PORT` (default `9246`).
- `provider_agent/agentstack_agents/mcpserver.py`: FastMCP server that lists doctors from `doctors.json`; started on demand by the ProviderAgent.
- `research_agent/agentstack_agents/research_agent.py`: Health research agent that searches the web via Serper. Requests/uses `SERPER_API_KEY` through the Secrets extension and runs with the AgentStack LLM extension. Default host/port: `AGENT_HOST` / `RESEARCH_AGENT_PORT` (default `9998`).

## Local run (AgentStack-managed)
1) Install and start AgentStack using the quickstart (https://agentstack.beeai.dev/stable/introduction/quickstart), configuring your LLM provider as Gemini with the preferred model `gemini-2.5-flash-lite`.
2) In separate terminals, start each agent from its folder (ensures the correct `.venv` is used):
   - `cd policy_agent && uv run agentstack_agents/policy_agent.py`
   - `cd provider_agent && uv run agentstack_agents/provider_agent.py`
   - `cd research_agent && uv run agentstack_agents/research_agent.py`
   - `cd healthcare_agent && uv run agentstack_agents/healthcare_agent.py`
   (Set the host/port env vars noted above if you need non-defaults.)
3) Start the AgentStack UI in another terminal (per the quickstart) to see registered A2A servers by running agent list. You will see that the ResearchAgent is missing an enviorment variable.
4) In your CLI run: agentstack env add "Research Agent" SERPER_API_KEY="Keyvalue"
5) Use the UI: you can call each agent directly, or run the Healthcare agent, which will hand off to the Policy, Research, and Provider agents as needed.
