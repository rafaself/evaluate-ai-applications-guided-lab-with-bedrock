import { Agent, OpenAIResponsesModel, run, setTracingDisabled } from '@openai/agents';
import type { ApiProvider, CallApiContextParams, CallApiOptionsParams, ProviderOptions } from 'promptfoo';
import OpenAI from 'openai';
import { bedrock } from 'openai/providers/bedrock/aws';
import { defaultProvider } from '@aws-sdk/credential-provider-node';
import gameTools from '../tools/game-tools.js';

// Keep SDK tracing from exporting game data to the OpenAI tracing endpoint.
setTracingDisabled(true);

const region =
  process.env.AWS_BEDROCK_REGION ||
  process.env.AWS_REGION ||
  process.env.AWS_DEFAULT_REGION ||
  'us-east-1';
const client = new OpenAI({
  provider: bedrock({
    region,
    endpoint: 'runtime',
    credentialProvider: defaultProvider(),
  }),
});

const instructions = `You are an enthusiastic Dungeon Master running an epic fantasy D&D adventure.

Your role:
- Guide players through thrilling quests, combat encounters, and mysteries
- Use roll_dice for attack rolls, saving throws, ability checks, and damage (D&D 5e rules)
- Show each roll's purpose, dice notation, rolled values, modifier, and total in your answer. Include damage rolls when an attack hits.
- Use check_inventory to see what items, equipment, and gold players have
- Use check_character_stats to view player abilities, HP, AC, and level
- Use describe_scene to paint vivid, atmospheric pictures of locations and situations
- Be creative, dramatic, and keep the adventure exciting and immersive
- Embrace unexpected player actions - improvise and adapt the story
- Present meaningful choices - decisions should have consequences

Player defaults:
- If the user does not name a character or player ID, assume playerId "player1" (Thorin Ironforge)
- Do not ask which character to use when checking stats, inventory, equipment, or magic items unless the user explicitly mentions multiple characters
- When a player attacks, triggers a trap, makes an uncertain move, or examines an item, resolve the immediate outcome with the appropriate tools instead of asking whether you should roll
- For a short rest, explain recovery and ask whether and how many Hit Dice the player wants to spend. Do not spend them for the player.

When combat occurs:
- Roll initiative (d20) for turn order
- Roll the player's attack and damage dice using roll_dice without asking for confirmation
- Have enemies make attack rolls and saving throws
- Track HP and conditions
- Describe combat vividly but concisely

Keep responses punchy but atmospheric. Always roll dice for uncertain outcomes.
Reference character abilities and inventory items when relevant. Make every moment memorable!`
 
/**
 * Dungeon Master agent for epic D&D adventures
 */
const agent = new Agent({
  name: 'Dungeon Master',
  instructions,
  model: new OpenAIResponsesModel(client, 'us.openai.gpt-6-luna'),
  modelSettings: { store: false, maxTokens: 2048, reasoning: { effort: 'low' } },
  tools: gameTools,
});

/** Expose completed tool executions alongside the final answer for Promptfoo assertions. */
export default class DungeonMasterProvider implements ApiProvider {
  private readonly maxTurns: number;

  constructor(options: ProviderOptions = {}) {
    this.maxTurns = options.config?.maxTurns ?? 20;
  }

  id() {
    return 'bedrock:dungeon-master';
  }

  async callApi(prompt: string, context?: CallApiContextParams, options?: CallApiOptionsParams) {
    const result = await run(agent, prompt, {
      context: context?.vars,
      maxTurns: this.maxTurns,
      signal: options?.abortSignal,
    });

    const toolCalls = result.newItems.flatMap(item => {
      if (item.type !== 'tool_call_output_item' || item.rawItem.type !== 'function_call_result') return [];

      const completed = item.rawItem;

      if (completed.status !== 'completed') return [];

      const call = result.newItems.find(candidate =>
        candidate.type === 'tool_call_item' && candidate.rawItem.type === 'function_call' &&
        candidate.rawItem.callId === completed.callId);

      if (!call || call.type !== 'tool_call_item' || call.rawItem.type !== 'function_call') return [];

      return [{
        callId: completed.callId,
        name: call.rawItem.name,
        arguments: JSON.parse(call.rawItem.arguments),
        result: item.output,
      }];
    });

    const usage = result.runContext.usage;

    return {
      output: result.finalOutput ?? '',
      metadata: { toolCalls },
      tokenUsage: {
        prompt: usage.inputTokens,
        completion: usage.outputTokens,
        total: usage.totalTokens,
        numRequests: usage.requests,
      },
      cached: false,
    };
  }
}
