# Bedrock agents basic — D&D Dungeon Master

This example uses the OpenAI Agents SDK with GPT-6 Luna on **Amazon Bedrock Runtime**. Game tools execute locally. No OpenAI API key or managed Bedrock Agent deployment is required.

## Setup and execution

Requires Node.js 24 and AWS access to `us.openai.gpt-6-luna` in `us-east-1`. The agent uses Runtime Responses; the rubric grader uses Runtime Converse. IAM must permit the model/profile and any default-project authorization required by Runtime Responses.

From the parent `evaluate-ai-applications-guided-lab` directory:

```bash
nvm use 24
npm --prefix bedrock-agents-basic ci --ignore-scripts
npm --prefix bedrock-agents-basic run validate
npm --prefix bedrock-agents-basic run eval
```

The eval script loads the parent `.env`. Use AWS access keys (and `AWS_SESSION_TOKEN` for temporary credentials), a named profile, or the standard AWS credential chain. `AWS_BEDROCK_REGION` defaults to `us-east-1`. Do not commit credentials. Evaluation makes paid model and grading calls.

## How tool verification works

`agents/dungeon-master-agent.ts` defines the agent and a small Promptfoo provider. After `run()`, the provider matches SDK tool-call items with their completed tool results by call ID and returns:

```typescript
{
  output: result.finalOutput,
  metadata: {
    toolCalls: [{ callId, name, arguments, result }]
  }
}
```

Promptfoo's JavaScript assertions inspect `context.metadata.toolCalls`. Missing metadata, a missing call, or incorrect required arguments fails the check; mentioning a tool in the answer does not count. No tracing server or additional port is needed. SDK export to OpenAI is disabled.

| Scenario | Required tool execution |
| --- | --- |
| Dragon attack, goblin attack, trap, ability check | `roll_dice` with `sides: 20`, `count: 1` |
| Character stats | `check_character_stats` with `playerId: player1` |
| Inventory or amulet | `check_inventory` with `playerId: player1` |
| Entering the crypt | `describe_scene` with `location: crypt` |
| Short rest | No `roll_dice` before the player agrees to spend Hit Dice |

Other arguments, call order, and additional relevant tools are allowed. Damage rolls are not mandatory when an attack misses. Existing response-content assertions and model rubrics remain in place. The checks verify calls and selected arguments; they do not validate every game mechanic or guarantee all returned values were used correctly in the final text.

Each test starts a fresh run. Inventory, character data, and scenes are fixtures; game state is not persisted. Tool metadata and token usage are saved with Promptfoo's results. The agent uses `store: false`.

The SDK and client versions are pinned; the OpenAI override keeps SDK packages on the same Runtime-capable client. Official integration reference: [OpenAI on Amazon Bedrock](https://developers.openai.com/api/docs/guides/amazon-bedrock).
