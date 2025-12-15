# AgentStack-HealthcareAgent
An healthcare based example of how to build and deploy an A2A Agent that calls other A2A Agents on the open source platform Agent Stack by IBM Research.

## Gemini policy agent server
- Ensure `GEMINI_API_KEY` is present in `.env` (and `AGENT_HOST` / `POLICY_AGENT_PORT` if you need non-default values). Dependency: `agentstack-sdk`.
- Start the server with `uv run python -m agents.a2a_policy_agent_gemini` from the repo root; it will expose the AgentStack serve endpoint on `http://0.0.0.0:9999` by default so AgentStack can discover it.
