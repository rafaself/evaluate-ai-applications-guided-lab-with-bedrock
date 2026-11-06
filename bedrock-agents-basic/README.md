# Bedrock agents basic — D&D Dungeon Master

This example keeps the OpenAI Agents SDK for its local tool-calling loop, but sends GPT-6 Luna requests to **Amazon Bedrock Runtime**, using AWS IAM credentials. No OpenAI API key or managed Bedrock Agent deployment is required.

## Setup

Requires Node.js 24 and AWS access to `us.openai.gpt-6-luna` in `us-east-1`. The agent uses the Runtime Responses API; the rubric grader uses Runtime Converse. IAM must permit the selected model/profile and any default-project authorization required by the Runtime Responses API.

From the parent `evaluate-ai-applications-guided-lab` directory:

```bash
nvm use 24
npm --prefix bedrock-agents-basic ci --ignore-scripts
npm --prefix bedrock-agents-basic run validate
npm --prefix bedrock-agents-basic run eval
```

The eval script loads the existing parent `.env`. It can contain `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, and `AWS_BEDROCK_REGION=us-east-1`. The standard AWS credential chain also supports named profiles and roles. Do not commit or share credentials.

Alternatively, run the local CLI from the parent directory:

```bash
./bedrock-agents-basic/node_modules/.bin/promptfoo eval \
  -c bedrock-agents-basic/promptfooconfig.yaml --env-file .env
```

Evaluation makes paid model and grading calls. Add `--no-cache` only to force fresh calls.

The Agents SDK stays on 0.14.1 because Promptfoo 0.125.1 requires that version. Its OpenAI client dependency is overridden to the same pinned 7.51.0 client used by the agent, which supports native Runtime endpoint selection and signing. The local tool round trip is checked against this combination.

## How it works

- `agents/dungeon-master-agent.ts` exports an Agents SDK agent whose `OpenAIResponsesModel` uses the OpenAI client's native Bedrock provider, the Runtime endpoint, and AWS SigV4 signing. The model is `us.openai.gpt-6-luna`.
- `tools/game-tools.ts` keeps the dice, inventory, character-stat, and scene tools. Tool execution stays local and the SDK handles subsequent model turns.
- `promptfooconfig.yaml` selects `openai:agents:dungeon-master` to run that local SDK agent. This provider name describes the runner, not the backend used by the exported model. Its rubric judge is explicitly `bedrock:converse:us.openai.gpt-6-luna`.
- Each test starts a fresh agent run. Inventory, character data, and scenes are fixtures; game state is not persisted.

The agent uses `store: false`. SDK tracing is disabled so this example does not export traces to OpenAI.

Official integration reference: [OpenAI on Amazon Bedrock](https://developers.openai.com/api/docs/guides/amazon-bedrock).
