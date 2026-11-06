# Evaluate AI Applications — Promptfoo on Amazon Bedrock

Experiments adapted from Promptfoo examples to evaluate model responses and agent behavior using Amazon Bedrock.

| Example | Purpose |
| --- | --- |
| `getting-started/` | Translation checks with GPT-6.1 Sol and GPT-6 Luna |
| `compare-openai-models/` | Riddle responses with GPT-OSS, Luna, and Sol; content, grading, latency, and cost assertions |
| `eval-rag/` | RAG metrics using sample company policy documents |
| `bedrock-agents-basic/` | D&D agent with local game tools and a Bedrock-backed Agents SDK model |

## Setup

Use Node.js 24 and AWS credentials with access to the configured models. For a fresh clone, copy `.env.example` to `.env` and configure credentials locally. Do not commit the filled `.env`. Named profiles and other standard AWS credential sources can also be used.

```bash
nvm use 24
npm --prefix bedrock-agents-basic ci --ignore-scripts
```

## Run an evaluation

From the repository root:

```bash
npx promptfoo@0.125.1 eval -c getting-started/promptfooconfig.yaml --env-file .env
npx promptfoo@0.125.1 eval -c compare-openai-models/promptfooconfig.yaml --env-file .env --no-cache
npm --prefix bedrock-agents-basic run eval
```

Model calls and rubric grading incur API usage. The local response cache can reuse prior outputs; `--no-cache` forces fresh calls and is required for latency assertions. Cost and latency thresholds check completed calls; they are not spending caps or request timeouts.

The RAG example also needs suitable grading and embedding providers. Its model-graded metrics currently inherit Promptfoo defaults, so configure `defaultTest.options.provider.text` and `.embedding` for the services you intend to use before running it. Setting a response provider alone does not set those graders.

## Known limitations

- Promptfoo 0.125.1 lacks cost estimates for the Luna and GPT-6.1 Sol Bedrock Converse routes, so the comparison's `cost` assertions can error. Tracked upstream in [promptfoo/promptfoo#12174](https://github.com/promptfoo/promptfoo/issues/12174).
- The agent uses Bedrock Runtime Responses for model calls and Converse for rubric grading. IAM permissions can differ between APIs; see its README.
- The agent dependency audit reports 8 transitive vulnerabilities (5 high, 3 moderate) at publication. Review dependency updates before wider deployment.
- Sample game state and company policies are fixtures, not real services or production data.

The individual example READMEs retain additional upstream context. The root commands and Bedrock agent README describe this checkout's AWS configuration. Example source: [Promptfoo examples](https://github.com/promptfoo/promptfoo/tree/main/examples).
