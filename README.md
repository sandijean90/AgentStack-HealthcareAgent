# AgentStack Healthcare Agents
This repo shows how agents built in different frameworks can be deployed on AgentStack. All agents run as A2A servers, so they can talk to each other whether you are running AgentStack locally or on a hosted deployment within your organization. Imagine your org’s hosted AgentStack where the HR agents can seamlessly call the Finance agents. This example does the same with healthcare agents. Install instructions for AgentStack are available in the quickstart: https://agentstack.beeai.dev/stable/introduction/quickstart.

## Agents in this repo
- `healthcare_agent/agentstack_agents/healthcare_agent.py`: Concierge/orchestrator that uses BeeAI’s RequirementAgent plus HandoffTool to route questions to the PolicyAgent, ResearchAgent, or ProviderAgent.
- `policy_agent/agentstack_agents/policy_agent.py`: Answers coverage questions by reading `2026AnthemgHIPSBC.pdf` with the LLM provided by the AgentStack extension.
- `provider_agent/agentstack_agents/provider_agent.py`: Finds providers via a custom MCP tool backed by `mcpserver.py` and returns results with an AgentStack-provisioned LLM.
- `provider_agent/agentstack_agents/mcpserver.py`: FastMCP server that lists doctors from `doctors.json`; started on demand by the ProviderAgent.
- `research_agent/agentstack_agents/research_agent.py`: Health research agent that searches the web via Serper. Requests/uses `SERPER_API_KEY` through the Secrets extension and runs with the AgentStack LLM extension.

## Local run (AgentStack-managed)
1) Install and start AgentStack using the quickstart (https://agentstack.beeai.dev/stable/introduction/quickstart), configuring your LLM provider as Gemini with the preferred model `gemini-2.5-flash-lite`.
> **Note for Windows users:** When you are running the agentstack platform start command the first time, select to configure the network as "nat" mode, not "mirrored" mode.  This network mode will allow the deployment of agents from github as directed in this repo. If the network mode selection is not seen, networkingmode can be changed to "nat" in the C:/Users/<your name>/.wslconfig file, and applied by shutting WSL down with "wsl --shutdown" and restarting with "agentstack platform start".
2) Add the agents through the AgentStack CLI (replace the release tag with the latest available on GitHub):
   ```bash
   agentstack add https://github.com/sandijean90/AgentStack-HealthcareAgent@release-0.0.16#path=/policy_agent
   agentstack add https://github.com/sandijean90/AgentStack-HealthcareAgent@release-0.0.16#path=/provider_agent
   agentstack add https://github.com/sandijean90/AgentStack-HealthcareAgent@release-0.0.16#path=/research_agent
   agentstack add https://github.com/sandijean90/AgentStack-HealthcareAgent@release-0.0.16#path=/healthcare_agent
   ```
   The platform builds and runs each agent for you—no need to start the servers manually.
3) You will see that the ResearchAgent is missing an environment variable. In your CLI run:
   ```bash
   agentstack env add "ResearchAgent" SERPER_API_KEY="Keyvalue"
   ```
4) Start the AgentStack UI:
   ```bash
   agentstack ui
   ```
5) Test the agents from the UI. Run them individually or run the Healthcare agent to see A2A handoffs across the Policy, Research, and Provider agents.

## Sample Questions to Ask Each Agent

### Helathcare Agent 
* I need mental health assistance and live in Austin Texas. Who can I see and what is covered by my policy?
* I am pregnant and need care in Miami, Florida. What are my options?

### Policy Agent
* Tell me about my policy.
* What is my coinsurance for office visits both in and out of network?

### Provider Agent
* What kind of doctors can I see in Houston Texas?
* I have a rash, who can I see in Los Angeles CA?

### Research Agent
* Tell me about the different types of diabetes.
* What can I do to reduce my cholesterol?

## Known Limitations
- The policy agent only has access to a summary of benefits with lmited information and can return "I don't know" (which is a valid response from this agent) depending on the question.
- For demo/illustrative purposes all agents are called (in a dynamic order) for each task. This may not be necessarily depending on the task and can be changed in the conditional requirements to yield better performance.
- The provider agent needs a very specifically formed tool call because of the expected input of the tool on the  mcp server. This can result in a malformed tool call depending on the LLM used and the strength of the system prompt. Future improvements can include a more flexible tool call.

